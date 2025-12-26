# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Cross-attention module for Perceiver-based context adaptation.

Enables target patch tokens to query regional context latents for spatial enrichment.
"""

import logging
import os
import warnings

import torch
import torch.nn as nn
from einops import rearrange

from dinov2.layers.attention import MemEffAttention
from dinov2.layers.mlp import Mlp

logger = logging.getLogger("dinov2")

# Check xFormers availability for memory-efficient attention
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention
        XFORMERS_AVAILABLE = True
        logger.info("xFormers is available (CrossAttention)")
    else:
        logger.info("xFormers is disabled (CrossAttention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    logger.info("xFormers is not available (CrossAttention)")


class CrossAttention(nn.Module):
    """
    Cross-attention module where queries come from target tokens and
    keys/values come from regional context latents.

    This enables target patches to selectively attend to relevant regional context.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        dim_head: int = 64,
        qkv_bias: bool = False,
    ):
        """
        Args:
            dim: Embedding dimension for both query and context
            num_heads: Number of attention heads
            dim_head: Dimension per attention head
            qkv_bias: Whether to use bias in projections
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.hidden_dim = num_heads * dim_head

        # Separate projections for queries (from target) and keys/values (from context)
        self.to_q = nn.Linear(dim, self.hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, self.hidden_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(self.hidden_dim, dim)

        # Separate normalization for the two input domains
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Target patches [B, N_target, D] - patches to enrich
            context: Regional latents [B, N_context, D] - compressed regional context

        Returns:
            Attention output [B, N_target, D] - context-enriched target patches
        """
        B, N_q, D = query.shape
        N_c = context.shape[1]

        # Normalize inputs separately (cross-domain attention)
        query = self.query_norm(query)
        context = self.context_norm(context)

        # Queries from target patches
        q = self.to_q(query)  # [B, N_target, hidden_dim]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        q = q * self.scale

        # Keys and values from regional context
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # Compute cross-attention: target patches attend to regional latents
        if XFORMERS_AVAILABLE:
            # Use memory-efficient attention (Flash Attention)
            q = q.transpose(1, 2)  # [B, N_q, H, D]
            k = k.transpose(1, 2)  # [B, N_c, H, D]
            v = v.transpose(1, 2)  # [B, N_c, H, D]

            out = memory_efficient_attention(q, k, v)  # [B, N_q, H, D]
            out = out.reshape(B, N_q, self.hidden_dim)
        else:
            # Fallback to standard attention
            # Compute similarity: [B, H, N_target, N_context]
            sim = torch.einsum('bhqd,bhkd->bhqk', q, k)

            # Softmax over context dimension
            attn = sim.softmax(dim=-1)

            # Aggregate values
            out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')

        # Project back to original dimension
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """
    Transformer block with cross-attention followed by self-attention.

    Architecture: Cross-Attn → Self-Attn → FFN
    - Cross-attention: inject regional context into target
    - Self-attention: refine target patches among themselves
    - FFN: non-linear transformation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        drop_path: float = 0.0,
    ):
        """
        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            dim_head: Dimension per head
            mlp_ratio: Hidden dimension multiplier for FFN
            qkv_bias: Use bias in attention projections
            proj_bias: Use bias in output projections
            drop_path: Stochastic depth rate
        """
        super().__init__()

        # Cross-attention: target queries regional context
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
        )
        self.norm_cross = nn.LayerNorm(dim)

        # Self-attention: target patches refine among themselves
        self.self_attn = MemEffAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
        )
        self.norm_self = nn.LayerNorm(dim)

        # Feedforward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            bias=proj_bias,
        )
        self.norm_mlp = nn.LayerNorm(dim)

        # Stochastic depth (optional)
        self.drop_path = nn.Identity()  # TODO: Add DropPath if needed

    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: Target patches [B, N_target, D]
            context: Regional latents [B, N_context, D]

        Returns:
            Refined target patches [B, N_target, D]
        """
        # Cross-attention: inject regional context
        target = target + self.drop_path(self.cross_attn(self.norm_cross(target), context))

        # Self-attention: refine locally
        target = target + self.drop_path(self.self_attn(self.norm_self(target)))

        # FFN: non-linear transformation
        target = target + self.drop_path(self.mlp(self.norm_mlp(target)))

        return target


class CrossAttentionTransformer(nn.Module):
    """
    Stack of cross-attention blocks for fusing target patches with regional context.

    Input:
        - Target patches: [B, 256, 768] - patches from single tile
        - Regional latents: [B, 64, 768] - compressed regional context

    Output:
        - Enriched target: [B, 256, 768] - spatially-enriched patch embeddings
    """

    def __init__(
        self,
        dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        """
        Args:
            dim: Embedding dimension (768 for downprojected features)
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            dim_head: Dimension per head
            mlp_ratio: FFN hidden dimension multiplier
            qkv_bias: Use bias in attention
            proj_bias: Use bias in projections
        """
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Stack of cross-attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
            )
            for _ in range(depth)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: Target patches [B, N_target, D] - typically [B, 256, 768]
            context: Regional latents [B, N_context, D] - typically [B, 64, 768]

        Returns:
            Context-enriched target patches [B, N_target, D]
        """
        # Apply cross-attention blocks sequentially
        for block in self.blocks:
            target = block(target, context)

        # Final normalization
        return self.norm(target)
