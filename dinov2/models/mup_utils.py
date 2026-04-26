# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""μP (maximal update parametrization) — thin wrappers over the `mup` library.

Basic usage (matches the mup README exactly):

    # 1. Build tiny reference backbones (never trained)
    base  = _build_ref_backbone(cfg, embed_dim=64,  num_heads=1)
    delta = _build_ref_backbone(cfg, embed_dim=128, num_heads=2)

    # 2. Persist shapes to disk (needed for checkpoint resume)
    make_base_shapes(base, delta, shapes_path)

    # 3. Apply μP to the actual backbone — rescales init and attaches
    #    p.infshape to every parameter.  Call BEFORE FSDP wrapping.
    #    Because the FSDP wrapper uses use_orig_params=True, the original
    #    parameter objects (and their infshape attributes) are preserved
    #    after wrapping, so MuAdamW can read them at optimizer construction.
    set_base_shapes(backbone, shapes_path)

    # 4. Build optimizer with MuAdamW (instead of plain AdamW)
    optimizer = MuAdamW(param_groups, lr=..., betas=...)

See: https://github.com/microsoft/mup
     Paper: https://arxiv.org/abs/2203.03466
"""
from __future__ import annotations

import logging
import os

import torch.nn as nn

logger = logging.getLogger("dinov2")

BASE_SHAPES_FILENAME = "mup_base_shapes.bsh"


def _depth_for_arch(arch: str) -> int:
    if "giant" in arch:
        return 40
    if "large" in arch:
        return 24
    if "base" in arch:
        return 12
    return 12  # small / default


def _build_ref_backbone(cfg, embed_dim: int, num_heads: int) -> nn.Module:
    """Build a minimal ViT backbone used only for shape registration.

    These models are never trained; we only extract parameter shape
    information from them (base and delta), following the mup README.
    """
    from dinov2.models.vision_transformer import DinoVisionTransformer

    return DinoVisionTransformer(
        img_size=224,
        patch_size=cfg.student.patch_size,
        embed_dim=embed_dim,
        depth=_depth_for_arch(cfg.student.arch),
        num_heads=num_heads,
        init_values=cfg.student.layerscale,
        ffn_layer=cfg.student.ffn_layer,
        block_chunks=0,       # no chunking for tiny reference models
        qkv_bias=cfg.student.qkv_bias,
        proj_bias=cfg.student.proj_bias,
        ffn_bias=cfg.student.ffn_bias,
        num_register_tokens=cfg.student.num_register_tokens,
        interpolate_offset=cfg.student.interpolate_offset,
        interpolate_antialias=cfg.student.interpolate_antialias,
    )


def setup_mup(backbone: nn.Module, cfg, output_dir: str) -> str:
    """Apply μP to *backbone* in-place and save base shapes for later reuse.

    Must be called **before** FSDP wrapping and **before** loading pretrained
    weights.  The FSDP wrapper already uses ``use_orig_params=True``, which
    preserves the original parameter objects (and their ``infshape``
    attributes) after wrapping, so ``MuAdamW`` can read them when the
    optimizer is built.

    Args:
        backbone: The student backbone (``model.student.backbone``).
        cfg: Training config (used to mirror the backbone architecture).
        output_dir: Directory where the ``.bsh`` shapes file is written.

    Returns:
        Path to the saved shapes file.
    """
    try:
        from mup import make_base_shapes, set_base_shapes
    except ImportError as exc:
        raise ImportError(
            "The 'mup' package is required for μP training. "
            "Add 'mup>=1.0.0' to pyproject.toml and reinstall."
        ) from exc

    # head_dim = 64 throughout the ViT family; keep it fixed so only
    # embed_dim varies between base and delta, matching the scaling axis.
    base  = _build_ref_backbone(cfg, embed_dim=64,  num_heads=1)
    delta = _build_ref_backbone(cfg, embed_dim=128, num_heads=2)

    os.makedirs(output_dir, exist_ok=True)
    shapes_path = os.path.join(output_dir, BASE_SHAPES_FILENAME)

    logger.info("μP: saving base shapes to %s", shapes_path)
    make_base_shapes(base, delta, shapes_path)
    del base, delta  # free memory immediately; we only needed the shapes

    logger.info("μP: calling set_base_shapes on backbone")
    set_base_shapes(backbone, shapes_path)
    # set_base_shapes rescales the freshly-initialized parameters to match μP
    # and attaches a `p.infshape` attribute to every parameter tensor.

    return shapes_path
