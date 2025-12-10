# Context Adaptation Training for OpenMidnight
# Trains a context-ViT to process 3584x3584 regions using frozen OpenMidnight patch embeddings
# Losses: Contrastive (spatial + region-level) + MAE pixel reconstruction

import logging
import math
import os
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from PIL import Image
from openslide import OpenSlide
from omegaconf import OmegaConf
import wandb
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from dinov2.models.vision_transformer import DinoVisionTransformer, vit_giant2
from dinov2.layers import MemEffAttention, Mlp, PatchEmbed
from dinov2.layers.block import Block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("context_adaptation")


# ============================================================================
# Dataset: Region Dataset for 3584x3584 regions
# ============================================================================

class RegionDataset(Dataset):
    """Dataset that loads 3584x3584 regions from TCGA slides."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.region_size = cfg.data.region_size
        self.patch_size = cfg.data.patch_size
        self.patches_per_side = cfg.data.patches_per_side

        # Load sample list
        self.samples = []
        with open(cfg.data.sample_list, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split(" ")
                    path = parts[0]
                    x = int(parts[1])
                    y = int(parts[2])
                    level = int(parts[3])
                    self.samples.append((path, x, y, level))

        logger.info(f"Loaded {len(self.samples)} patch samples, will sample regions around them")

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, base_x, base_y, level = self.samples[idx]

        slide = OpenSlide(path)

        # Compute region top-left that contains the base patch
        # We want to sample a 3584x3584 region
        # Add some randomness to the region position
        offset_x = np.random.randint(0, self.patch_size) if np.random.rand() > 0.5 else 0
        offset_y = np.random.randint(0, self.patch_size) if np.random.rand() > 0.5 else 0

        region_x = max(0, base_x - offset_x - self.patch_size * (self.patches_per_side // 2))
        region_y = max(0, base_y - offset_y - self.patch_size * (self.patches_per_side // 2))

        # Avoid padding at slide edges by clamping the region within bounds
        level0_w, level0_h = slide.level_dimensions[0]
        downsample = slide.level_downsamples[level]
        region_size_level0 = int(round(self.region_size * downsample))

        if region_size_level0 > level0_w or region_size_level0 > level0_h:
            # Skip samples that cannot fit the desired region size
            return self.__getitem__((idx + 1) % len(self))

        max_x = max(level0_w - region_size_level0, 0)
        max_y = max(level0_h - region_size_level0, 0)
        orig_x, orig_y = region_x, region_y
        region_x = min(region_x, max_x)
        region_y = min(region_y, max_y)

        # Clamping applied silently; no logging

        # Read the full region
        region = slide.read_region(
            (region_x, region_y),
            level=level,
            size=(self.region_size, self.region_size)
        ).convert("RGB")

        # Extract individual patches as a grid
        patches = []
        patch_positions = []

        region_np = np.array(region)

        for i in range(self.patches_per_side):
            for j in range(self.patches_per_side):
                py = i * self.patch_size
                px = j * self.patch_size
                patch = region_np[py:py+self.patch_size, px:px+self.patch_size]
                patch_pil = Image.fromarray(patch)
                patch_tensor = self.normalize(self.to_tensor(patch_pil))
                patches.append(patch_tensor)
                patch_positions.append((i, j))

        # Stack all patches: [256, 3, 224, 224]
        patches_tensor = torch.stack(patches, dim=0)

        # Also return the full region for visualization
        region_tensor = self.normalize(self.to_tensor(region))

        return {
            "patches": patches_tensor,
            "region": region_tensor,
            "positions": torch.tensor(patch_positions),
            "path": path,
            "region_coords": (region_x, region_y),
            "idx": idx,
        }


# ============================================================================
# Models
# ============================================================================

class ContextViT(nn.Module):
    """
    Context ViT that processes patch embeddings from OpenMidnight.
    Takes 256 patch embeddings and performs self-attention across them.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        input_dim = cfg.patch_vit.embed_dim  # 1536 from OpenMidnight
        embed_dim = cfg.context_vit.embed_dim  # 768
        depth = cfg.context_vit.depth
        num_heads = cfg.context_vit.num_heads
        mlp_ratio = cfg.context_vit.mlp_ratio
        drop_path_rate = cfg.context_vit.drop_path_rate

        # Project from OpenMidnight embedding dim to context dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Learnable positional embeddings for the 16x16 grid
        num_patches = cfg.data.num_patches_per_region  # 256
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Mask token for masked patches (used for MAE)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                attn_class=MemEffAttention,
                ffn_layer=Mlp,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, patch_embeddings, mask=None):
        """
        Args:
            patch_embeddings: [B, 256, 1536] - embeddings from OpenMidnight
            mask: [B, 256] - bool mask, True = masked (for MAE)
        Returns:
            Dict with cls_token and patch_tokens
        """
        B, N, D = patch_embeddings.shape

        # Project to context dimension
        x = self.input_proj(patch_embeddings)  # [B, 256, 768]

        # Mask out tokens before attention to avoid leakage during MAE
        if mask is not None:
            mask_tokens = self.mask_token.expand(B, N, -1)
            x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

        # Add positional embeddings
        x = x + self.pos_embed

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 257, 768]

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return {
            "cls_token": x[:, 0],  # [B, 768]
            "patch_tokens": x[:, 1:],  # [B, 256, 768]
        }


class MAEDecoder(nn.Module):
    """
    Decoder for MAE pixel reconstruction.
    Takes context-ViT outputs and reconstructs masked patches.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        context_dim = cfg.context_vit.embed_dim  # 768
        decoder_dim = cfg.mae_decoder.embed_dim  # 512
        depth = cfg.mae_decoder.depth
        num_heads = cfg.mae_decoder.num_heads

        patch_size = cfg.data.patch_size  # 224

        # Mask token for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Project from context dim to decoder dim
        self.decoder_embed = nn.Linear(context_dim, decoder_dim)

        # Positional embeddings for decoder
        num_patches = cfg.data.num_patches_per_region
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                attn_class=MemEffAttention,
                ffn_layer=Mlp,
            )
            for _ in range(depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)

        # Predict patch pixels: decoder_dim -> 224*224*3
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * 3)

        # Initialize
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, context_tokens, mask):
        """
        Args:
            context_tokens: [B, 256, 768] - from context ViT
            mask: [B, 256] - bool, True = masked position to reconstruct
        Returns:
            reconstructed_patches: [B, num_masked, 224*224*3]
        """
        B, N, D = context_tokens.shape

        # Project to decoder dimension
        x = self.decoder_embed(context_tokens)  # [B, 256, 512]

        # Replace masked positions with mask token
        mask_tokens = self.mask_token.expand(B, N, -1)
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        x = torch.where(mask_expanded, mask_tokens, x)

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # Predict pixels for all positions
        pred = self.decoder_pred(x)  # [B, 256, 224*224*3]

        return pred


class ContextAdaptationModel(nn.Module):
    """
    Full model combining:
    - Frozen OpenMidnight (patch-ViT)
    - Trainable Context-ViT
    - MAE Decoder
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Load frozen OpenMidnight
        self.patch_vit = self._load_patch_vit(cfg)
        for param in self.patch_vit.parameters():
            param.requires_grad = False
        self.patch_vit.eval()

        # Context ViT
        self.context_vit = ContextViT(cfg)

        # MAE Decoder
        self.mae_decoder = MAEDecoder(cfg)

        # Projection head for contrastive loss
        self.contrastive_proj = nn.Sequential(
            nn.Linear(cfg.context_vit.embed_dim, cfg.context_vit.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.context_vit.embed_dim, 256),
        )

    def _load_patch_vit(self, cfg):
        """Load the frozen OpenMidnight checkpoint."""
        logger.info(f"Loading OpenMidnight from {cfg.patch_vit.checkpoint}")

        # Build ViT-Giant2 architecture
        patch_vit = vit_giant2(
            patch_size=cfg.patch_vit.patch_size,
            num_register_tokens=cfg.patch_vit.num_register_tokens,
            img_size=cfg.data.patch_size,  # 224
            block_chunks=0,  # No chunking for inference
        )

        # Load checkpoint
        ckpt = torch.load(cfg.patch_vit.checkpoint, map_location="cpu")

        # Handle different checkpoint formats
        if "teacher" in ckpt:
            state_dict = ckpt["teacher"]
        else:
            state_dict = ckpt

        # Clean up keys
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if "backbone" in k}

        # Load weights
        msg = patch_vit.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded OpenMidnight with message: {msg}")

        return patch_vit

    @torch.no_grad()
    def extract_patch_embeddings(self, patches):
        """
        Extract embeddings from all patches using frozen OpenMidnight.
        Args:
            patches: [B, 256, 3, 224, 224]
        Returns:
            embeddings: [B, 256, 1536]
        """
        B, N, C, H, W = patches.shape

        # Reshape to process all patches
        patches_flat = patches.view(B * N, C, H, W)

        # Process in smaller batches to avoid OOM
        batch_size = 64
        embeddings_list = []

        for i in range(0, B * N, batch_size):
            batch = patches_flat[i:i+batch_size]
            out = self.patch_vit(batch, is_training=True)
            # Use CLS token as patch embedding
            embeddings_list.append(out["x_norm_clstoken"])

        embeddings = torch.cat(embeddings_list, dim=0)  # [B*N, 1536]
        embeddings = embeddings.view(B, N, -1)  # [B, 256, 1536]

        return embeddings

    def forward(self, patches, mae_mask=None):
        """
        Args:
            patches: [B, 256, 3, 224, 224]
            mae_mask: [B, 256] - bool, True = masked for reconstruction

        Two-pass design to align training with downstream inference:
        1. Unmasked pass: context-ViT sees all patches → contrastive embeddings
           (matches downstream where we always have full context)
        2. Masked pass (if needed): context-ViT masks tokens → MAE reconstruction
           (prevents information leakage, forces learning from context)
        """
        # Get patch embeddings from frozen OpenMidnight
        patch_embeddings = self.extract_patch_embeddings(patches)

        # Pass 1: Unmasked forward for contrastive (matches downstream inference)
        context_out_full = self.context_vit(patch_embeddings, mask=None)

        # Contrastive embeddings from context-enriched tokens (post self-attention)
        contrastive_embeddings = self.contrastive_proj(context_out_full["patch_tokens"])
        contrastive_embeddings = F.normalize(contrastive_embeddings, dim=-1)

        # Pass 2: Masked forward for MAE (only when needed)
        mae_pred = None
        if mae_mask is not None and mae_mask.any():
            context_out_masked = self.context_vit(patch_embeddings, mask=mae_mask)
            mae_pred = self.mae_decoder(context_out_masked["patch_tokens"], mae_mask)

        return {
            "patch_embeddings": patch_embeddings,
            "context_cls": context_out_full["cls_token"],
            "context_patches": context_out_full["patch_tokens"],
            "contrastive_embeddings": contrastive_embeddings,
            "mae_pred": mae_pred,
        }


# ============================================================================
# Losses
# ============================================================================

def compute_contrastive_loss(embeddings, positions, cfg, region_indices=None, anchor_mask=None):
    """
    Contrastive loss with spatial awareness (vectorized for efficiency).
    - Neighboring patches (within radius) should be more similar
    - Patches in same region should be more similar than different regions
    - Uses all gathered samples as negatives; anchors are limited to the local batch

    Args:
        embeddings: [B, 256, D] - normalized embeddings
        positions: [B, 256, 2] - (row, col) positions in the 16x16 grid
        cfg: config
        region_indices: [B] - which region each sample belongs to (for inter-region contrast)
        anchor_mask: [B] bool mask indicating which regions to use as anchors (local batch)
    """
    B, N, D = embeddings.shape  # B = global batch across GPUs after all_gather
    temperature = cfg.loss.contrastive_temperature
    distance_scale = cfg.loss.spatial_distance_scale

    # Determine which samples to use as anchors
    if anchor_mask is None:
        anchor_indices = torch.arange(B, device=embeddings.device)
    else:
        anchor_indices = torch.nonzero(anchor_mask, as_tuple=True)[0]

    B_anchor = len(anchor_indices)

    # Flatten embeddings and positions for global similarity computation
    anchor_embeddings = embeddings[anchor_indices]  # [B_anchor, N, D]
    anchor_positions = positions[anchor_indices]  # [B_anchor, N, 2]

    anchor_flat = anchor_embeddings.view(B_anchor * N, D)  # [B_anchor*N, D]
    anchor_pos_flat = anchor_positions.view(B_anchor * N, 2)  # [B_anchor*N, 2]
    anchor_region_idx = anchor_indices.repeat_interleave(N)  # [B_anchor*N]

    all_embeddings_flat = embeddings.view(B * N, D)  # [B*N, D]
    all_positions_flat = positions.view(B * N, 2)  # [B*N, 2]
    all_region_idx = torch.arange(B, device=embeddings.device).repeat_interleave(N)  # [B*N]

    # Similarity of each anchor patch to all patches (including other regions)
    sim = torch.matmul(anchor_flat, all_embeddings_flat.t()) / temperature  # [B_anchor*N, B*N]

    # Spatial neighbor mask: same region (no hard radius cutoff)
    pos_diff = anchor_pos_flat.unsqueeze(1) - all_positions_flat.unsqueeze(0)  # [B_anchor*N, B*N, 2]
    pos_dist = torch.norm(pos_diff.float(), dim=-1)

    same_region = all_region_idx.unsqueeze(0) == anchor_region_idx.unsqueeze(1)  # [B_anchor*N, B*N]
    pos_mask = same_region  # positives are all patches in the same region (excluding self below)

    # Remove self matches
    global_patch_idx = torch.arange(B * N, device=embeddings.device)
    anchor_patch_idx = (anchor_indices.view(-1, 1) * N + torch.arange(N, device=embeddings.device).view(1, -1)).view(-1)
    self_mask = global_patch_idx.unsqueeze(0) == anchor_patch_idx.unsqueeze(1)
    pos_mask = pos_mask & ~self_mask

    # Negatives = all other regions
    neg_mask = ~same_region

    # Valid anchors need at least one positive and one negative
    valid_mask = pos_mask.any(dim=1) & neg_mask.any(dim=1)

    NEG_INF = -1e9
    sim_all = sim.masked_fill(self_mask, NEG_INF)

    # Distance-weighted positives: closer patches get higher weight
    pos_weight = torch.exp(-pos_dist / distance_scale)
    pos_weight = pos_weight * pos_mask.float()
    sim_pos = sim + torch.log(pos_weight + 1e-12)
    sim_pos = sim_pos.masked_fill(~pos_mask, NEG_INF)

    logsumexp_all = torch.logsumexp(sim_all, dim=1)  # [B_anchor*N]
    logsumexp_pos = torch.logsumexp(sim_pos, dim=1)  # [B_anchor*N]

    patch_loss = logsumexp_all - logsumexp_pos  # [B_anchor*N]
    patch_loss = patch_loss.masked_fill(~valid_mask, 0.0)

    num_valid = valid_mask.sum()
    if num_valid > 0:
        spatial_loss = patch_loss.sum() / num_valid
    else:
        spatial_loss = torch.tensor(0.0, device=embeddings.device)

    # Inter-region contrastive loss (if multiple regions in batch)
    region_loss = torch.tensor(0.0, device=embeddings.device)
    if B > 1:
        # Region-level embeddings (mean of patch embeddings)
        cls_embeddings = embeddings.mean(dim=1)  # [B, D]
        cls_embeddings = F.normalize(cls_embeddings, dim=-1)

        # Similarity of anchor regions to all regions
        anchor_cls = cls_embeddings[anchor_indices]  # [B_anchor, D]
        region_sim = torch.mm(anchor_cls, cls_embeddings.t()) / temperature  # [B_anchor, B]

        # Labels: each anchor should match itself in the global batch
        # anchor_indices tells us where each anchor is in the global batch
        labels = anchor_indices  # Correct: index into columns of region_sim
        region_loss = F.cross_entropy(region_sim, labels)

    return spatial_loss + region_loss


def compute_mae_loss(pred, target, mask, cfg):
    """
    MAE reconstruction loss (vectorized).

    Args:
        pred: [B, 256, 224*224*3] - predicted pixels
        target: [B, 256, 3, 224, 224] - target patches
        mask: [B, 256] - True = masked positions
    """
    if pred is None or not mask.any():
        return torch.tensor(0.0, device=target.device)

    B, N, C, H, W = target.shape

    # Reconstruct target following MAE: normalize each patch by its own mean/std in pixel space
    target_pixels = denormalize(target)  # [B, N, 3, 224, 224] in 0-1 space
    patch_mean = target_pixels.mean(dim=(2, 3, 4), keepdim=True)  # [B, N, 1, 1, 1]
    patch_std = target_pixels.flatten(2).std(dim=2, keepdim=True).view(B, N, 1, 1, 1)  # [B, N, 1, 1, 1]
    patch_std = patch_std + 1e-6
    target_norm = (target_pixels - patch_mean) / patch_std  # per-patch normalized target

    # Flatten target patches
    target_flat = target_norm.view(B, N, -1)  # [B, 256, 224*224*3]

    # Compute per-element squared error
    sq_error = (pred - target_flat) ** 2  # [B, 256, 224*224*3]

    # Mean over pixel dimension
    sq_error_per_patch = sq_error.mean(dim=-1)  # [B, 256]

    # Only average over masked positions
    mask_float = mask.float()
    num_masked = mask_float.sum()

    if num_masked > 0:
        loss = (sq_error_per_patch * mask_float).sum() / num_masked
    else:
        loss = torch.tensor(0.0, device=target.device)

    return loss


# ============================================================================
# Debug visualization
# ============================================================================

def denormalize(tensor):
    """Denormalize from ImageNet normalization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


@torch.no_grad()
def save_debug_images(batch, model_out, mae_mask, cfg, iteration, rank=0):
    """Save debug images for sanity checking."""
    if rank != 0:
        return

    debug_dir = Path(cfg.evaluation.debug_save_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Move tensors to CPU and convert to float32 for image saving
    if mae_mask is None:
        return

    patches = batch["patches"].detach().cpu().float()  # [B, 256, 3, 224, 224]
    region = batch["region"].detach().cpu().float()  # [B, 3, 3584, 3584]
    mae_mask_cpu = mae_mask.detach().cpu()

    B = patches.shape[0]

    for b in range(B):
        mask_b = mae_mask_cpu[b]  # [256]
        if not mask_b.any():
            continue  # Only save samples that were actually masked

        # 1. Save original region
        region_img = denormalize(region[b])
        save_image(region_img, debug_dir / f"iter{iteration}_sample{b}_region.png")

        # 2. Save grid of patches
        patches_denorm = denormalize(patches[b])  # [256, 3, 224, 224]
        # Reshape to 16x16 grid
        grid = make_grid(patches_denorm, nrow=16, padding=2)
        save_image(grid, debug_dir / f"iter{iteration}_sample{b}_patches_grid.png")

        # 3. Save masked patches visualization
        masked_patches_denorm = denormalize(patches[b]).clone()  # work in pixel space for visibility
        masked_patches_denorm[mask_b] = 0  # true black for masked patches
        grid_masked = make_grid(masked_patches_denorm, nrow=16, padding=2)
        save_image(grid_masked, debug_dir / f"iter{iteration}_sample{b}_masked.png")

        # 4. Save reconstruction if available
        if model_out["mae_pred"] is not None:
            pred = model_out["mae_pred"][b].detach().cpu().float()  # [256, 224*224*3]
            pred_patches = pred.view(256, 3, 224, 224)

            # Convert predicted normalized patches back to pixel space using per-patch mean/std
            patch_pixels = denormalize(patches[b]).clone()  # [256, 3, 224, 224]
            patch_mean = patch_pixels.mean(dim=(1, 2, 3), keepdim=True)  # [256, 1, 1, 1]
            patch_std = patch_pixels.flatten(1).std(dim=1, keepdim=True).view(256, 1, 1, 1) + 1e-6
            pred_pixels = pred_patches * patch_std + patch_mean

            # Create reconstruction visualization: original patches with reconstructed masked ones
            recon_vis = patches[b].clone()

            for i in range(256):
                if mask_b[i]:
                    # For masked patches, show reconstruction in pixel space
                    recon_vis[i] = pred_pixels[i].clamp(0, 1)
                else:
                    # For unmasked patches, show original pixels
                    recon_vis[i] = patch_pixels[i].clamp(0, 1)

            grid_recon = make_grid(recon_vis, nrow=16, padding=2)
            recon_path = debug_dir / f"iter{iteration}_sample{b}_reconstruction.png"
            save_image(grid_recon, recon_path)

            # Log reconstruction grid to wandb for visibility
            if wandb.run is not None:
                wandb.log(
                    {"images/reconstruction": wandb.Image(str(recon_path))},
                    step=iteration,
                )
        break  # Save only the first valid sample

    logger.info(f"Saved debug images to {debug_dir}")


# ============================================================================
# Training Loop
# ============================================================================

def train(cfg):
    # Setup distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_main = rank == 0

    if is_main:
        logger.info(f"Starting context adaptation training with {world_size} GPUs")
        logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Setup wandb
    if is_main:
        run_id_path = Path(cfg.train.output_dir) / "wandb_run_id.txt"
        run_id_path.parent.mkdir(parents=True, exist_ok=True)

        if run_id_path.exists():
            run_id = run_id_path.read_text().strip()
            resume_mode = "must"
        else:
            run_id = wandb.util.generate_id()
            run_id_path.write_text(run_id)
            resume_mode = "allow"

        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),
            id=run_id,
            resume=resume_mode,
        )

    # Create dataset and dataloader
    dataset = RegionDataset(cfg)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(dataloader)

    # Create model
    model = ContextAdaptationModel(cfg).to(device)

    # Wrap trainable parts in DDP (patch_vit is frozen, no need to wrap)
    model.context_vit = DDP(model.context_vit, device_ids=[local_rank])
    model.mae_decoder = DDP(model.mae_decoder, device_ids=[local_rank])
    model.contrastive_proj = DDP(model.contrastive_proj, device_ids=[local_rank])

    # Optimizer - only trainable parameters
    trainable_params = (
        list(model.context_vit.parameters()) +
        list(model.mae_decoder.parameters()) +
        list(model.contrastive_proj.parameters())
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )

    # Learning rate scheduler: cosine decay without warmup
    total_steps = cfg.train.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.optim.min_lr,
    )

    # Training loop
    iteration = 0

    for epoch in range(cfg.train.epochs):
        sampler.set_epoch(epoch)
        model.train()
        model.patch_vit.eval()  # Keep frozen model in eval mode

        for batch_idx, batch in enumerate(dataloader):
            patches = batch["patches"].to(device)  # [B, 256, 3, 224, 224]
            positions = batch["positions"].to(device)  # [B, 256, 2]

            B, N = patches.shape[:2]

            # Create MAE mask
            mae_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

            # For a fraction of samples, apply high masking ratio
            frac = float(cfg.loss.mae_num_masked_samples)
            num_masked_samples = math.ceil(frac * B)
            num_masked_samples = max(2, min(num_masked_samples, B)) if B > 1 else 1
            masked_sample_indices = torch.randperm(B)[:num_masked_samples]

            for b_idx in masked_sample_indices:
                num_to_mask = int(N * cfg.loss.mae_mask_ratio)
                mask_indices = torch.randperm(N)[:num_to_mask]
                mae_mask[b_idx, mask_indices] = True

            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_out = model(patches, mae_mask=mae_mask)

                # Gather contrastive inputs across all GPUs
                contrastive_embeddings = model_out["contrastive_embeddings"]
                positions_all = positions
                region_idx_all = batch["idx"].to(device)
                anchor_mask = None

                if world_size > 1:
                    gathered_embeddings = dist_fn.all_gather(contrastive_embeddings)
                    contrastive_embeddings = torch.cat(list(gathered_embeddings), dim=0)

                    positions_list = [torch.zeros_like(positions) for _ in range(world_size)]
                    dist.all_gather(positions_list, positions)
                    positions_all = torch.cat(positions_list, dim=0)

                    region_idx_list = [torch.zeros_like(region_idx_all) for _ in range(world_size)]
                    dist.all_gather(region_idx_list, region_idx_all)
                    region_idx_all = torch.cat(region_idx_list, dim=0)

                    anchor_mask = torch.zeros(contrastive_embeddings.shape[0], device=device, dtype=torch.bool)
                    start = rank * B
                    anchor_mask[start:start + B] = True

                # Compute losses
                contrastive_loss = compute_contrastive_loss(
                    contrastive_embeddings,
                    positions_all,
                    cfg,
                    region_indices=region_idx_all,
                    anchor_mask=anchor_mask,
                )

                mae_loss = compute_mae_loss(
                    model_out["mae_pred"],
                    patches,
                    mae_mask,
                    cfg,
                )

                total_loss = (
                    cfg.loss.contrastive_weight * contrastive_loss +
                    cfg.loss.mae_weight * mae_loss
                )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()

            grad_norm = None
            if cfg.optim.clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.optim.clip_grad)
            else:
                # Compute grad norm without clipping for logging
                norms = [p.grad.data.norm(2) for p in trainable_params if p.grad is not None]
                grad_norm = torch.norm(torch.stack(norms), 2) if norms else torch.tensor(0.0, device=device)

            optimizer.step()
            scheduler.step()

            # Save debug images every x iterations, starting at 0th iteration
            if cfg.evaluation.debug_save_images and iteration % 50 == 0:
                save_debug_images(batch, model_out, mae_mask, cfg, iteration, rank)

            # Logging
            if is_main and iteration % 25 == 0:
                lr = scheduler.get_last_lr()[0]
                grad_norm_val = grad_norm.item() if grad_norm is not None else 0.0
                logger.info(
                    f"Epoch {epoch} | Iter {iteration} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Contrastive: {contrastive_loss.item():.4f} | "
                    f"MAE: {mae_loss.item():.4f} | "
                    f"LR: {lr:.6f} | "
                    f"GradNorm: {grad_norm_val:.4f}"
                )

                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/contrastive_loss": contrastive_loss.item(),
                    "train/mae_loss": mae_loss.item(),
                    "train/lr": lr,
                    "train/grad_norm": grad_norm_val,
                    "train/epoch": epoch,
                }, step=iteration)

            iteration += 1

            if iteration >= total_steps:
                break

        # Save checkpoint
        if is_main and (epoch + 1) % cfg.evaluation.save_period_epochs == 0:
            ckpt_path = Path(cfg.train.output_dir) / f"checkpoint_epoch{epoch+1}.pth"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                "epoch": epoch,
                "iteration": iteration,
                "context_vit": model.context_vit.module.state_dict(),
                "mae_decoder": model.mae_decoder.module.state_dict(),
                "contrastive_proj": model.contrastive_proj.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)

            logger.info(f"Saved checkpoint to {ckpt_path}")

        if iteration >= total_steps:
            break

    # Final save
    if is_main:
        final_path = Path(cfg.train.output_dir) / "checkpoint_final.pth"
        torch.save({
            "epoch": epoch,
            "iteration": iteration,
            "context_vit": model.context_vit.module.state_dict(),
            "mae_decoder": model.mae_decoder.module.state_dict(),
            "contrastive_proj": model.contrastive_proj.module.state_dict(),
        }, final_path)
        logger.info(f"Training complete! Final checkpoint: {final_path}")
        wandb.finish()

    dist.destroy_process_group()


def main():
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "/home/paul/OpenMidnight/dinov2/configs/train/context_adaptation.yaml"

    cfg = OmegaConf.load(config_path)

    train(cfg)


if __name__ == "__main__":
    main()
