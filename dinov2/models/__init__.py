# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import sys
from pathlib import Path

from . import vision_transformer as vits


logger = logging.getLogger("dinov2")
_DINOV3_REPO = str(Path(__file__).resolve().parents[2].parent / "dinov3")


def build_model(args, only_teacher=False, only_student=False, img_size=224):
    if only_teacher and only_student:
        raise ValueError("only_teacher and only_student cannot both be True")
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = None
        if not only_student:
            teacher = vits.__dict__[args.arch](**vit_kwargs)
            if only_teacher:
                return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def _build_dinov3_model(args, only_teacher=False, only_student=False, img_size=224):
    if only_teacher and only_student:
        raise ValueError("only_teacher and only_student cannot both be True")
    if _DINOV3_REPO not in sys.path:
        sys.path.insert(0, _DINOV3_REPO)
    from dinov3.models import init_fp8
    from dinov3.models import vision_transformer as vits3

    arch = args.arch.removesuffix("_memeff")
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=args.patch_size,
        pos_embed_rope_base=args.pos_embed_rope_base,
        pos_embed_rope_min_period=args.pos_embed_rope_min_period,
        pos_embed_rope_max_period=args.pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,
        pos_embed_rope_dtype=args.pos_embed_rope_dtype,
        qkv_bias=args.qkv_bias,
        layerscale_init=args.layerscale,
        norm_layer=args.norm_layer,
        ffn_layer=args.ffn_layer,
        ffn_bias=args.ffn_bias,
        proj_bias=args.proj_bias,
        n_storage_tokens=args.n_storage_tokens,
        mask_k_bias=args.mask_k_bias,
        ffn_ratio=args.ffn_ratio,
        untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,
        untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,
    )
    teacher = None
    if not only_student:
        teacher = vits3.__dict__[arch](**vit_kwargs)
        teacher = init_fp8(teacher, args)
        if only_teacher:
            return teacher, teacher.embed_dim
    student = vits3.__dict__[arch](**vit_kwargs, drop_path_rate=args.drop_path_rate)
    student = init_fp8(student, args)
    embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False, only_student=False):
    if cfg.model.family == "dinov3":
        return _build_dinov3_model(
            cfg.student,
            only_teacher=only_teacher,
            only_student=only_student,
            img_size=cfg.crops.global_crops_size,
        )
    return build_model(
        cfg.student,
        only_teacher=only_teacher,
        only_student=only_student,
        img_size=cfg.crops.global_crops_size,
    )
