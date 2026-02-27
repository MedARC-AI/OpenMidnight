# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
import random
from functools import partial
import itertools
from io import BytesIO
from pathlib import Path
import gc
import contextlib
import glob

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.models import build_model_from_cfg
from datasets import IterableDatasetDict, load_dataset, DownloadConfig
from PIL import Image, ImageOps
import h5py
import numpy as np
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import folder
from tqdm import tqdm

import pyarrow
import pyarrow.dataset
import torch.distributed as dist
import torch.nn.functional as F

import pyarrow
import pyarrow.dataset
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")
import wandb

RUN_SCRIPT = os.environ.get("DINOV2_RUN_SCRIPT", "")
AUGMENTATION_FILE = Path(__file__).resolve().parents[1] / "data" / "augmentations.py"
VISION_TRANSFORMER_FILE = Path(__file__).resolve().parents[1] / "models" / "vision_transformer.py"
SSL_META_ARCH = Path(__file__).resolve().parents[1] / "train" / "ssl_meta_arch.py"

def _build_streaming_dataset(
    dataset_path: str,
    *,
    shuffle_buffer: int,
    base_seed: int = 0,
    fragment_prefetch_limit: int = 4,
    fragment_range_size: int = 32 << 20,
    epoch: int = 0,
):
    # Get current rank/size at call time (safe under elastic restarts)
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=fragment_prefetch_limit,
            range_size_limit=fragment_range_size,
        ),
    )

    ds = load_dataset(
        dataset_path,
        streaming=True,
        fragment_scan_options=fragment_scan_options,
    )["train"]

    # 1) shard first to avoid cross-rank duplication and wasted I/O
    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=global_rank)

    # 2) then shuffle; vary by epoch and rank
    seed = base_seed + epoch * 1_000_000 + global_rank * 10000
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return ds

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

_FULL_STATE_DICT_CFG = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

_HUB_ARCH_NAMES = {
    "vit_small": "vits",
    "vit_base": "vitb",
    "vit_large": "vitl",
    "vit_giant2": "vitg",
}


def _resolve_torchhub_name(cfg):
    hub_arch = _HUB_ARCH_NAMES.get(cfg.student.arch)
    if hub_arch is None:
        raise AssertionError(f"Unsupported student arch for pretrained load: {cfg.student.arch}")
    if cfg.student.patch_size != 14:
        raise AssertionError("Pretrained torch.hub weights are only defined for patch_size=14")
    if cfg.student.num_register_tokens not in (0, 4):
        raise AssertionError("Pretrained weights only support 0 or 4 register tokens")
    reg_suffix = "_reg" if cfg.student.num_register_tokens else ""
    return f"dinov2_{hub_arch}{cfg.student.patch_size}{reg_suffix}"


def _iter_vit_blocks(backbone):
    if backbone.chunked_blocks:
        for chunk in backbone.blocks:
            for blk in chunk:
                if not isinstance(blk, torch.nn.Identity):
                    yield blk
    else:
        for blk in backbone.blocks:
            yield blk


def _load_pretrained_backbone(cfg, model):
    def _mlp_kind(block):
        if hasattr(block.mlp, "fc1"):
            return "mlp"
        if hasattr(block.mlp, "w12"):
            return "swiglu"
        raise AssertionError("Unsupported FFN block type")

    hub_name = _resolve_torchhub_name(cfg)
    logger.info("Loading pretrained backbone from torch.hub: %s", hub_name)
    model_pretrained = torch.hub.load("facebookresearch/dinov2", hub_name)
    device = next(model.parameters()).device
    model_pretrained = model_pretrained.to(device)
    student_backbone = model.student.backbone
    teacher_backbone = model.teacher.backbone

    with torch.no_grad():
        if student_backbone.embed_dim != model_pretrained.embed_dim:
            raise AssertionError("Pretrained embed_dim mismatch")
        if student_backbone.n_blocks != model_pretrained.n_blocks:
            raise AssertionError("Pretrained depth mismatch")
        if student_backbone.num_register_tokens != model_pretrained.num_register_tokens:
            raise AssertionError("Pretrained register token count mismatch")

        student_backbone.patch_embed.proj.weight.copy_(model_pretrained.patch_embed.proj.weight)
        student_backbone.patch_embed.proj.bias.copy_(model_pretrained.patch_embed.proj.bias)
        student_backbone.cls_token.copy_(model_pretrained.cls_token)
        student_backbone.mask_token.copy_(model_pretrained.mask_token)
        if student_backbone.num_register_tokens:
            student_backbone.register_tokens.copy_(model_pretrained.register_tokens)

        pos_embed_pretrained = model_pretrained.pos_embed.detach()
        n_extra_tokens = 1
        cls_pos_embed = pos_embed_pretrained[:, :n_extra_tokens]
        patch_pos_embed = pos_embed_pretrained[:, n_extra_tokens:]

        orig_size = int(patch_pos_embed.shape[1] ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)

        target_h, target_w = student_backbone.patch_embed.patches_resolution
        resized_patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=model_pretrained.interpolate_antialias,
        )
        resized_patch_pos_embed = resized_patch_pos_embed.permute(0, 2, 3, 1).reshape(
            1, target_h * target_w, -1
        )
        new_pos_embed = torch.cat((cls_pos_embed, resized_patch_pos_embed), dim=1)

        student_backbone.pos_embed.copy_(new_pos_embed)
        teacher_backbone.pos_embed.copy_(new_pos_embed)

        student_blocks = list(_iter_vit_blocks(student_backbone))
        pretrained_blocks = list(_iter_vit_blocks(model_pretrained))
        if len(student_blocks) != len(pretrained_blocks):
            raise AssertionError("Pretrained block count mismatch")
        if student_blocks and _mlp_kind(student_blocks[0]) != _mlp_kind(pretrained_blocks[0]):
            raise AssertionError(
                f"FFN mismatch: cfg.student.ffn_layer builds {_mlp_kind(student_blocks[0])}, "
                f"but torch.hub {hub_name} uses {_mlp_kind(pretrained_blocks[0])}"
            )
        for dst, src in zip(student_blocks, pretrained_blocks):
            dst.load_state_dict(src.state_dict(), strict=True)

        student_backbone.norm.weight.copy_(model_pretrained.norm.weight)
        student_backbone.norm.bias.copy_(model_pretrained.norm.bias)


def _freeze_student_backbone_except_last_n(cfg, model):
    n_unfrozen = cfg.train.unfreeze_last_n_blocks
    student_backbone = model.student.backbone
    blocks = list(_iter_vit_blocks(student_backbone))
    total_blocks = len(blocks)
    if n_unfrozen < 1 or n_unfrozen > total_blocks:
        n_unfrozen = total_blocks
        print("cfg.train.unfreeze_last_n_blocks not set; setting to total blocks which is {total_blocks}")
    freeze_until = total_blocks - n_unfrozen
    if freeze_until == 0:
        return
    for p in student_backbone.parameters():
        p.requires_grad = False
    for blk in blocks[freeze_until:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in student_backbone.norm.parameters():
        p.requires_grad = True
    logger.info("Froze %d/%d student backbone blocks (trainable blocks: %d)", freeze_until, total_blocks, n_unfrozen)


def do_test(cfg, model, iteration): # save teacher checkpoint (used for eval only)
    # All ranks participate in FSDP state_dict() even with rank0_only=True
    is_main = distributed.is_main_process()
    iterstring = str(iteration)
    eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
    if is_main:
        os.makedirs(eval_dir, exist_ok=True)
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")

    if isinstance(model.teacher, FSDP):
        state_dict_module = model.teacher
    elif isinstance(model, FSDP):
        state_dict_module = model
    else:
        state_dict_module = None

    state_dict_ctx = (
        FSDP.state_dict_type(
            state_dict_module,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=_FULL_STATE_DICT_CFG,
        )
        if state_dict_module is not None
        else contextlib.nullcontext()
    )

    with torch.no_grad(), state_dict_ctx:
        teacher_sd = model.teacher.state_dict()

    if is_main:
        torch.save({"teacher": teacher_sd}, teacher_ckp_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del teacher_sd

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not is_main:
        return

    teacher, _ = build_model_from_cfg(cfg, only_teacher=True)
    teacher_state = torch.load(teacher_ckp_path, map_location="cpu")["teacher"]
    teacher_state = {k.replace("module.", ""): v for k, v in teacher_state.items()}
    teacher_state = {k.replace("backbone.", ""): v for k, v in teacher_state.items() if k.startswith("backbone.")}
    load_msg = teacher.load_state_dict(teacher_state, strict=False)
    logger.info("Loaded teacher for downstream eval with msg: %s", load_msg)
    teacher = teacher.cuda()
    teacher.eval()
    teacher.requires_grad_(False)
    device = next(teacher.parameters()).device
    step = iteration if isinstance(iteration, int) else int(str(iteration).split("_")[-1])

    def _resolve_eval_root(name, root_value):
        root_str = "" if root_value is None else str(root_value)
        if not root_str:
            logger.info("Skipping %s eval because %s_root is not set", name, name.lower())
            return None
        root_path = os.path.abspath(os.path.expanduser(root_str))
        if not os.path.isdir(root_path):
            logger.info("Skipping %s eval; dataset path missing: %s", name, root_path)
            return None
        return root_path

    bach_root = _resolve_eval_root("BACH", cfg.evaluation.bach_root)
    breakhis_root = _resolve_eval_root("BreakHis", cfg.evaluation.breakhis_root)
    pcam_root = _resolve_eval_root("PCam", cfg.evaluation.pcam_root)

    class _ResizeAndCrop(transforms.Compose):
        def __init__(self, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            ops = [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
            super().__init__(ops)

    transform = _ResizeAndCrop(
        size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    predict_batch_size = 64
    num_workers = 4
    train_batch_size = 256

    def _compute_embeddings(dataset, *, to_cpu=False):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=predict_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        feats = []
        targets = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                out = teacher(images, is_training=True)
                cls = out["x_norm_clstoken"].float()
                if to_cpu:
                    feats.append(cls.cpu())
                    targets.append(labels.cpu())
                else:
                    labels = labels.to(device, non_blocking=True)
                    feats.append(cls)
                    targets.append(labels)
        feats = torch.cat(feats, dim=0)
        targets = torch.cat(targets, dim=0)
        return feats, targets

    if bach_root is not None:
        _BACH_TRAIN_INDEX_RANGES = [
            (0, 41),
            (59, 60),
            (90, 139),
            (169, 240),
            (258, 260),
            (273, 345),
            (368, 400),
        ]
        _BACH_VAL_INDEX_RANGES = [
            (41, 59),
            (60, 90),
            (139, 169),
            (240, 258),
            (260, 273),
            (345, 368),
        ]
        _BACH_CLASS_TO_IDX = {"Benign": 0, "InSitu": 1, "Invasive": 2, "Normal": 3}

        class _BACHDataset(torch.utils.data.Dataset):
            def __init__(self, root, split, transform):
                self.root = os.path.abspath(os.path.expanduser(root))
                self.split = split
                self.transform = transform
                dataset_path = os.path.join(self.root, "ICIAR2018_BACH_Challenge", "Photos")
                self.samples = folder.make_dataset(
                    directory=dataset_path,
                    class_to_idx=_BACH_CLASS_TO_IDX,
                    extensions=(".tif",),
                )
                if len(self.samples) == 0:
                    raise RuntimeError(f"No BACH images found in {dataset_path}")
                if split == "train":
                    index_ranges = _BACH_TRAIN_INDEX_RANGES
                elif split == "val":
                    index_ranges = _BACH_VAL_INDEX_RANGES
                else:
                    raise ValueError("Invalid BACH split. Use 'train' or 'val'.")
                indices = []
                for start, end in index_ranges:
                    indices.extend(range(start, end))
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                image_path, target = self.samples[self.indices[idx]]
                image = Image.open(image_path).convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                target_tensor = torch.tensor(target, dtype=torch.long)
                return image, target_tensor

        train_ds = _BACHDataset(root=bach_root, split="train", transform=transform)
        val_ds = _BACHDataset(root=bach_root, split="val", transform=transform)

        train_feats, train_targets = _compute_embeddings(train_ds)
        val_feats, val_targets = _compute_embeddings(val_ds)

        in_features = train_feats.shape[-1]
        num_classes = 4
        head = torch.nn.Linear(in_features, num_classes, bias=True).to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-2)

        train_dataset = torch.utils.data.TensorDataset(train_feats, train_targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_dataset = torch.utils.data.TensorDataset(val_feats, val_targets)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            drop_last=False,
        )

        def _eval_head():
            head.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for feats_batch, targets_batch in val_loader:
                    feats_batch = feats_batch.to(device, non_blocking=True)
                    logits = head(feats_batch)
                    preds = logits.argmax(dim=1).cpu()
                    all_preds.append(preds)
                    all_targets.append(targets_batch.cpu())
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            plain_acc = float((preds == targets).float().mean().item())
            conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
            indices = targets * num_classes + preds
            bincount = torch.bincount(indices, minlength=num_classes * num_classes)
            conf = bincount.view(num_classes, num_classes)
            per_class = conf.diag().float() / conf.sum(dim=1).clamp_min(1)
            balanced_acc = float(per_class.mean().item())
            head.train()
            return plain_acc, balanced_acc

        max_steps = 12500  # eva uses 12500 steps with patience-based early stopping
        eval_every = 250
        patience = 1250
        steps = 0
        best_plain = -1.0
        best_balanced = -1.0
        best_state = None
        steps_since_improve = 0
        head.train()
        with tqdm(total=max_steps) as pbar:
            while steps < max_steps:
                for feats_batch, targets_batch in train_loader:
                    feats_batch = feats_batch.to(device, non_blocking=True)
                    targets_batch = targets_batch.to(device, non_blocking=True)
                    logits = head(feats_batch)
                    loss = criterion(logits, targets_batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    steps += 1
                    pbar.update(1)
                    if steps % eval_every == 0 or steps >= max_steps:
                        plain_acc, balanced_acc = _eval_head()
                        if plain_acc > best_plain:
                            best_plain = plain_acc
                            best_balanced = balanced_acc
                            best_state = {k: v.cpu() for k, v in head.state_dict().items()}
                            steps_since_improve = 0
                        else:
                            steps_since_improve += eval_every
                        if steps_since_improve >= patience:
                            steps = max_steps
                            break
                    if steps >= max_steps:
                        break

        if best_state is not None:
            head.load_state_dict(best_state)
            bach_acc_plain, bach_acc_balanced = best_plain, best_balanced
        else:
            bach_acc_plain, bach_acc_balanced = _eval_head()

        logger.info(
            "BACH val accuracy (linear probe): plain=%.4f balanced=%.4f",
            bach_acc_plain,
            bach_acc_balanced,
        )

        if wandb.run is not None and distributed.is_main_process():
            wandb.log(
                {
                    "val/BACH_BALANCED_ACCURACY": bach_acc_balanced,
                    "val/BACH_MULTICLASS_ACCURACY": bach_acc_plain,
                },
                step=step,
            )

    if breakhis_root is not None:
        _BREAKHIS_VAL_PATIENT_IDS = {
            "18842D",
            "19979",
            "15275",
            "15792",
            "16875",
            "3909",
            "5287",
            "16716",
            "2773",
            "5695",
            "16184CD",
            "23060CD",
            "21998CD",
            "21998EF",
        }
        _BREAKHIS_CLASSES = ["TA", "MC", "F", "DC"]
        _BREAKHIS_CLASS_TO_IDX = {label: index for index, label in enumerate(_BREAKHIS_CLASSES)}

        class _BreakHisDataset(torch.utils.data.Dataset):
            def __init__(self, root, split, transform):
                self.root = os.path.abspath(os.path.expanduser(root))
                self.split = split
                self.transform = transform
                dataset_path = os.path.join(self.root, "BreaKHis_v1", "histology_slides")
                pattern = os.path.join(dataset_path, "**", "40X", "*.png")
                self.image_files = sorted(glob.glob(pattern, recursive=True))
                if len(self.image_files) == 0:
                    raise RuntimeError(f"No BreakHis images found in {dataset_path}")
                indices = []
                for idx, image_file in enumerate(self.image_files):
                    class_name = os.path.basename(image_file).split("-")[0].split("_")[-1]
                    if class_name not in _BREAKHIS_CLASS_TO_IDX:
                        continue
                    patient_id = os.path.basename(image_file).split("-")[2]
                    if split == "train":
                        if patient_id not in _BREAKHIS_VAL_PATIENT_IDS:
                            indices.append(idx)
                    elif split == "val":
                        if patient_id in _BREAKHIS_VAL_PATIENT_IDS:
                            indices.append(idx)
                    else:
                        raise ValueError("Invalid BreakHis split. Use 'train' or 'val'.")
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                image_path = self.image_files[self.indices[idx]]
                image = Image.open(image_path).convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                class_name = os.path.basename(image_path).split("-")[0].split("_")[-1]
                target = _BREAKHIS_CLASS_TO_IDX[class_name]
                target_tensor = torch.tensor(target, dtype=torch.long)
                return image, target_tensor

        train_ds = _BreakHisDataset(root=breakhis_root, split="train", transform=transform)
        val_ds = _BreakHisDataset(root=breakhis_root, split="val", transform=transform)

        train_feats, train_targets = _compute_embeddings(train_ds)
        val_feats, val_targets = _compute_embeddings(val_ds)

        in_features = train_feats.shape[-1]
        num_classes = 4
        head = torch.nn.Linear(in_features, num_classes, bias=True).to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-2)

        train_dataset = torch.utils.data.TensorDataset(train_feats, train_targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_dataset = torch.utils.data.TensorDataset(val_feats, val_targets)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            drop_last=False,
        )

        def _eval_head():
            head.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for feats_batch, targets_batch in val_loader:
                    feats_batch = feats_batch.to(device, non_blocking=True)
                    logits = head(feats_batch)
                    preds = logits.argmax(dim=1).cpu()
                    all_preds.append(preds)
                    all_targets.append(targets_batch.cpu())
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            plain_acc = float((preds == targets).float().mean().item())
            conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
            indices = targets * num_classes + preds
            bincount = torch.bincount(indices, minlength=num_classes * num_classes)
            conf = bincount.view(num_classes, num_classes)
            per_class = conf.diag().float() / conf.sum(dim=1).clamp_min(1)
            balanced_acc = float(per_class.mean().item())
            head.train()
            return plain_acc, balanced_acc

        max_steps = 12500
        eval_every = 250
        patience = 500
        steps = 0
        best_plain = -1.0
        best_balanced = -1.0
        best_state = None
        steps_since_improve = 0
        head.train()
        while steps < max_steps:
            for feats_batch, targets_batch in train_loader:
                feats_batch = feats_batch.to(device, non_blocking=True)
                targets_batch = targets_batch.to(device, non_blocking=True)
                logits = head(feats_batch)
                loss = criterion(logits, targets_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
                if steps % eval_every == 0 or steps >= max_steps:
                    plain_acc, balanced_acc = _eval_head()
                    if plain_acc > best_plain:
                        best_plain = plain_acc
                        best_balanced = balanced_acc
                        best_state = {k: v.cpu() for k, v in head.state_dict().items()}
                        steps_since_improve = 0
                    else:
                        steps_since_improve += eval_every
                    if steps_since_improve >= patience:
                        steps = max_steps
                        break
                if steps >= max_steps:
                    break

        if best_state is not None:
            head.load_state_dict(best_state)
            breakhis_acc_plain, breakhis_acc_balanced = best_plain, best_balanced
        else:
            breakhis_acc_plain, breakhis_acc_balanced = _eval_head()

        logger.info(
            "BreakHis val accuracy (linear probe): plain=%.4f balanced=%.4f",
            breakhis_acc_plain,
            breakhis_acc_balanced,
        )

        if wandb.run is not None and distributed.is_main_process():
            wandb.log(
                {
                    "val/BREAKHIS_BALANCED_ACCURACY": breakhis_acc_balanced,
                    "val/BREAKHIS_MULTICLASS_ACCURACY": breakhis_acc_plain,
                },
                step=step,
            )

    if pcam_root is not None:
        def _pcam_h5_path(root, split, key):
            split_suffix = "valid" if split == "val" else split
            filename = f"camelyonpatch_level_2_split_{split_suffix}_{key}.h5"
            return os.path.join(root, filename)

        class _PCamDataset(torch.utils.data.Dataset):
            def __init__(self, root, split, transform):
                self.root = os.path.abspath(os.path.expanduser(root))
                self.split = split
                self.transform = transform
                self.x_path = _pcam_h5_path(self.root, split, "x")
                self.y_path = _pcam_h5_path(self.root, split, "y")
                if not os.path.isfile(self.x_path) or not os.path.isfile(self.y_path):
                    raise RuntimeError(f"Missing PatchCamelyon files for split {split} in {self.root}")
                with h5py.File(self.y_path, "r") as file:
                    self.length = len(file["y"])

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                with h5py.File(self.x_path, "r") as file:
                    image_array = file["x"][idx]
                with h5py.File(self.y_path, "r") as file:
                    target = file["y"][idx].squeeze()
                image = Image.fromarray(image_array)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                target_tensor = torch.tensor(float(target), dtype=torch.float32)
                return image, target_tensor

        train_ds = _PCamDataset(root=pcam_root, split="train", transform=transform)
        test_ds = _PCamDataset(root=pcam_root, split="test", transform=transform)

        num_samples_per_class = 10
        with h5py.File(_pcam_h5_path(pcam_root, "train", "y"), "r") as file:
            targets = file["y"][:].reshape(-1)
        class_indices = {}
        for idx, target in enumerate(targets):
            class_idx = int(target)
            if class_idx not in class_indices:
                class_indices[class_idx] = []
            class_indices[class_idx].append(idx)
        for class_idx, indices in class_indices.items():
            if len(indices) < num_samples_per_class:
                raise ValueError(
                    f"Class {class_idx} has only {len(indices)} samples, "
                    f"which is less than the required {num_samples_per_class} samples."
                )
        rng = np.random.default_rng(42)
        sampled_indices = []
        for class_idx in class_indices:
            sampled = rng.choice(
                class_indices[class_idx],
                size=num_samples_per_class,
                replace=False,
            ).tolist()
            sampled_indices.extend(sampled)
        rng.shuffle(sampled_indices)

        train_subset = torch.utils.data.Subset(train_ds, sampled_indices)

        train_feats, train_targets = _compute_embeddings(train_subset, to_cpu=True)
        test_feats, test_targets = _compute_embeddings(test_ds, to_cpu=True)

        in_features = train_feats.shape[-1]
        head = torch.nn.Linear(in_features, 1, bias=True).to(device)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-2)

        train_dataset = torch.utils.data.TensorDataset(train_feats, train_targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            drop_last=False,
        )

        test_dataset = torch.utils.data.TensorDataset(test_feats, test_targets)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            drop_last=False,
        )

        def _eval_head(loader):
            head.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for feats_batch, targets_batch in loader:
                    feats_batch = feats_batch.to(device, non_blocking=True)
                    logits = head(feats_batch).squeeze(1)
                    preds = (logits >= 0).to(torch.long).cpu()
                    all_preds.append(preds)
                    all_targets.append(targets_batch.to(torch.long).cpu())
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            plain_acc = float((preds == targets).float().mean().item())
            num_classes = 2
            indices = targets * num_classes + preds
            bincount = torch.bincount(indices, minlength=num_classes * num_classes)
            conf = bincount.view(num_classes, num_classes)
            per_class = conf.diag().float() / conf.sum(dim=1).clamp_min(1)
            balanced_acc = float(per_class.mean().item())
            head.train()
            return plain_acc, balanced_acc

        max_steps = 12500
        eval_every_epochs = 10
        patience_evals = 125
        eval_every = eval_every_epochs * len(train_loader)
        patience = patience_evals * eval_every
        steps = 0
        best_balanced = -1.0
        best_state = None
        steps_since_improve = 0
        head.train()
        while steps < max_steps:
            for feats_batch, targets_batch in train_loader:
                feats_batch = feats_batch.to(device, non_blocking=True)
                targets_batch = targets_batch.to(device, non_blocking=True)
                logits = head(feats_batch).squeeze(1)
                loss = criterion(logits, targets_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
                if steps % eval_every == 0 or steps >= max_steps:
                    _, balanced_acc = _eval_head(test_loader)
                    if balanced_acc > best_balanced:
                        best_balanced = balanced_acc
                        best_state = {k: v.cpu() for k, v in head.state_dict().items()}
                        steps_since_improve = 0
                    else:
                        steps_since_improve += eval_every
                    if steps_since_improve >= patience:
                        steps = max_steps
                        break
                if steps >= max_steps:
                    break

        if best_state is not None:
            head.load_state_dict(best_state)
        test_plain, test_balanced = _eval_head(test_loader)

        logger.info(
            "PCam test accuracy (linear probe, 10-shot): plain=%.4f balanced=%.4f",
            test_plain,
            test_balanced,
        )

        if wandb.run is not None and distributed.is_main_process():
            wandb.log(
                {
                    "val/PCAM_BINARY_ACCURACY": test_plain,
                    "val/PCAM_BALANCED_ACCURACY": test_balanced,
                },
                step=step,
            )


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if cfg.train.skip_checkpointer:
        print(f"\n\nSkipping FSDP checkpointer (cfg.train.skip_checkpointer={cfg.train.skip_checkpointer})\n\n")

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    
    from omegaconf import OmegaConf
    if distributed.is_main_process():
        run_id_path = Path(cfg.train.output_dir) / "wandb_run_id.txt"
        if resume and run_id_path.exists():
            run_id = run_id_path.read_text().strip()
            resume_mode = "must"
        else:
            run_id_path.parent.mkdir(parents=True, exist_ok=True)
            run_id = wandb.util.generate_id()
            run_id_path.write_text(run_id)
            resume_mode = "allow"
        run = wandb.init(
            project="tcga-finetuning",
            config=OmegaConf.to_container(cfg),
            id=run_id,
            resume=resume_mode,
        )
        repo_root = Path(__file__).resolve().parents[2]
        files_to_save = [
            Path(__file__).resolve(),
            AUGMENTATION_FILE,
            VISION_TRANSFORMER_FILE,
            SSL_META_ARCH,
            Path(CONFIG_FILE_PATH),
        ]
        run_script = os.environ.get("DINOV2_RUN_SCRIPT")
        if run_script:
            files_to_save.append(Path(run_script).resolve())
        for src in files_to_save:
            base_path = repo_root if src.is_relative_to(repo_root) else src.parent
            run.save(str(src), base_path=str(base_path), policy="now")
        logger.info("Trainable parameters: %s", trainable_params)
        logger.info("Total parameters: %s", total_params)
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters: {total_params}")

    # checkpointer
    if not cfg.train.skip_checkpointer:
        checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    else:
        start_iter = 0

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    early_stop_iter = cfg.optim.early_stop * OFFICIAL_EPOCH_LENGTH
    eta_target_iter = min(max_iter, early_stop_iter)

    if not cfg.train.skip_checkpointer:
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=5 * OFFICIAL_EPOCH_LENGTH,
            max_iter=max_iter,
            max_to_keep=1,
        )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    if cfg.train.streaming_from_hf:
        dataset_builder = partial(
            _build_streaming_dataset,
            dataset_path=str(cfg.train.streaming_dataset_path),
            shuffle_buffer=50000,                   
            base_seed=42, 
        )

        def decode_and_transform(item):
            image = Image.open(BytesIO(item["image_bytes"]))
            image = ImageOps.exif_transpose(image).convert("RGB")
            transformed = data_transform(image)
            slide_meta = (item["slide_path"], item["x"], item["y"], item["level"])
            return (transformed, None), slide_meta

        class _TransformedStreamingDataset(torch.utils.data.IterableDataset):
            def __init__(self, dataset_builder, transform, samples_per_epoch=None, reshuffle_every=0):
                self._dataset_builder = dataset_builder
                self._transform = transform
                self._samples_per_epoch = samples_per_epoch
                self._reshuffle_every = reshuffle_every  
                self._initialized = False
                self._epoch_seen = 0
                self._src_iter = None

            def _init_or_reshuffle(self, *, force: bool = False):
                if force or (not self._initialized) or (
                    self._reshuffle_every and (self._epoch_seen % self._reshuffle_every == 0)
                ):
                    src = self._dataset_builder(epoch=self._epoch_seen if self._reshuffle_every else 0)
                    worker_info = torch.utils.data.get_worker_info()
                    if worker_info is not None and worker_info.num_workers > 1:
                        src = src.shard(num_shards=worker_info.num_workers, index=worker_info.id)
                    self._src_iter = iter(src)
                    self._initialized = True

            def __iter__(self):
                while True:
                    self._init_or_reshuffle()

                    # Per-RANK quota
                    rank_quota = self._samples_per_epoch or (1 << 62)

                    worker_info = torch.utils.data.get_worker_info()
                    num_workers = worker_info.num_workers if worker_info is not None else 1
                    worker_id = worker_info.id if worker_info is not None else 0

                    # Split quota across workers (nearly even split)
                    base = rank_quota // num_workers
                    remainder = rank_quota % num_workers
                    local_quota = base + (1 if worker_id < remainder else 0)

                    produced = 0
                    while produced < local_quota:
                        try:
                            sample = next(self._src_iter)
                        except StopIteration:
                            # Refill the iterator; only reshuffle on epoch boundaries
                            self._init_or_reshuffle(force=True)
                            continue
                        yield self._transform(sample)
                        produced += 1

                    self._epoch_seen += 1

        # Define explicit per-epoch sample budget per rank to keep ranks in lock-step
        samples_per_epoch = cfg.train.batch_size_per_gpu * cfg.train.OFFICIAL_EPOCH_LENGTH
        dataset = _TransformedStreamingDataset(
            dataset_builder,
            decode_and_transform,
            samples_per_epoch=samples_per_epoch,
        )

        def _worker_init(_):
            torch.set_num_threads(1)
            os.environ.setdefault("OMP_NUM_THREADS", "1")

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            prefetch_factor=4,
            worker_init_fn=_worker_init,
        )
    else:
        from dinov2.data import SamplerType, make_data_loader, make_dataset
        sample_list_path = str(cfg.train.sample_list_path)
        if not sample_list_path:
            raise ValueError("cfg.train.sample_list_path must be set when streaming_from_hf is False")
        dataset_str = f"pathology:root=/data/TCGA/:sample_list_path={sample_list_path}"
        dataset = make_dataset(
            dataset_str=dataset_str,
            transform=data_transform,
            target_transform=lambda _: (),
        )
        sampler_type = SamplerType.SHARDED_INFINITE
        data_loader = make_data_loader(
            dataset=dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            shuffle=True,
            seed=0,
            sampler_type=sampler_type,
            sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
            drop_last=True,
            collate_fn=collate_fn,
        )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        eta_target_iter + 1,
        start_iter,
    ):
        if iteration >= early_stop_iter:
            logger.info("Early stopping at iteration {}".format(iteration))
            if cfg.evaluation.eval_period_iterations >= 0:
                do_test(cfg, model, f"training_{iteration}")
                torch.cuda.synchronize()
            if not cfg.train.skip_checkpointer:
                checkpointer.save(f"model_{iteration:07d}", iteration=iteration)
            break

        #Save instantly
        if cfg.evaluation.eval_period_iterations >= 0 and (iteration) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        if not cfg.train.skip_checkpointer:
            periodic_checkpointer.step(iteration)
        
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return
        
        nan_mask = torch.isnan(data["collated_global_crops"])
        nan_mask2 = torch.isnan(data["collated_local_crops"])
        if nan_mask.any():
            print("found nan in input data")
            print(data[indexes])
        

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if "total_loss" in loss_dict_reduced:
            losses_reduced = loss_dict_reduced.pop("total_loss")
        else:
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        if math.isnan(sum(loss_dict_reduced.values())):
            print(sum(loss_dict_reduced.values()))
            logger.info("NaN detected")
            print(data["indexes"])
            
            for name, param in model.named_parameters():
                if torch.isnan(param.data).any():
                    print(f"NaNs found in parameter: {name}")

            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
        
        if distributed.is_main_process():
            scalar_logs = {
                "Learning Rate": lr,
                "Momentum": mom,
                "Last Layer LR": last_layer_lr,
                "Total Loss": losses_reduced,
            }
            wandb.log({**scalar_logs, **loss_dict_reduced}, step=iteration)
    
        # Synchronize the GPU to ensure all operations are complete before measuring
        torch.cuda.synchronize()

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    print(cfg)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    #Load model here from pretrained.
    if cfg.train.use_pretrained:
        _load_pretrained_backbone(cfg, model)
    _freeze_student_backbone_except_last_n(cfg, model)

    model.prepare_for_distributed_training()
    logger.info("Model:\n{}".format(model))

    if args.eval_only and not cfg.train.skip_checkpointer:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    if not args.config_file:
        raise ValueError("config file path must be provided")
    CONFIG_FILE_PATH = os.path.abspath(args.config_file)
    main(args)
