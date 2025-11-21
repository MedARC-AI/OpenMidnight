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
import pickle
import numpy as np

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch
import timm
import timm.optim

from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch
from datasets import IterableDatasetDict, load_dataset, DownloadConfig
from PIL import Image, ImageOps
import torch.utils.data

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

def oom_observer(device, alloc, device_alloc, device_free):
    # snapshot right after an OOM happened
    print('saving allocated state during OOM')
    snapshot = torch.cuda.memory._snapshot()
    pickle.dump(snapshot, open('oom_snapshot.pickle', 'wb'))

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
    # SimDINO uses timm.optim for flexibility, fallback to original if not specified
    if hasattr(cfg.optim, 'opt') and cfg.optim.opt:
        opt_lower = cfg.optim.opt.lower()
        kwargs = getattr(cfg.optim, 'kwargs', {})
        return timm.optim.create_optimizer_v2(params_groups, opt=opt_lower, **kwargs)
    else:
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
    
    # SimDINO schedules
    clip_grad = dict(
        base_value=cfg.optim.clip_grad,
        final_value=getattr(cfg.optim, "clip_grad_end", cfg.optim.clip_grad),
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)
    clip_grad_schedule = CosineScheduler(**clip_grad)

    # REMOVED: Direct attribute access last_layer_lr_schedule.schedule[...] = 0
    # Logic moved to do_train loop to be robust against CosineScheduler implementation differences.

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        clip_grad_schedule
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

_FULL_STATE_DICT_CFG = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

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


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training
    single_gpu_run = distributed.get_global_size() <= 1
    if single_gpu_run: print("\n\nSINGLE GPU RUN, SKIPPING FSDP CHECKPOINTING\n\n")

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        clip_grad_schedule
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
            project="midnight-rep",
            config=OmegaConf.to_container(cfg),
            id=run_id,
            resume=resume_mode,
        )

    # checkpointer
    if not single_gpu_run:
        checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    else:
        start_iter = 0

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    early_stop_iter = cfg.optim.early_stop * OFFICIAL_EPOCH_LENGTH
    eta_target_iter = min(max_iter, early_stop_iter)
    
    # Calculate freeze iterations for last layer
    freeze_last_layer_iters = cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH

    if not single_gpu_run:
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=3 * OFFICIAL_EPOCH_LENGTH,
            max_iter=max_iter,
            max_to_keep=3,
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
            dataset_path="medarc/TCGA-12K-parquet",
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
        dataset = make_dataset(
            dataset_str="pathology:root=/data/TCGA/",
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
            sampler_advance=0,
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
            if not single_gpu_run:
                checkpointer.save(f"model_{iteration:07d}", iteration=iteration)
            break
        
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return
        
        nan_mask = torch.isnan(data["collated_global_crops"])
        nan_mask2 = torch.isnan(data["collated_local_crops"])
        if nan_mask.any():
            print("found nan in input data")
        

        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        
        # Logic to freeze last layer (set LR to 0)
        if iteration < freeze_last_layer_iters:
            last_layer_lr = 0
        else:
            last_layer_lr = last_layer_lr_schedule[iteration]
        
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # Dynamic Patch Sizing (SimDINO Feature)
        patch_size_seq = getattr(cfg.student, "patch_size_seq", None)
        if patch_size_seq:
            new_patch_size = np.random.choice(patch_size_seq)
            model.student.backbone.patch_embed.update_patch_size(new_patch_size)
            model.teacher.backbone.patch_embed.update_patch_size(new_patch_size)

        # compute losses

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients
        clip_grad_val = clip_grad_schedule[iteration]
        if fp16_scaler is not None:
            if clip_grad_val > 0:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(clip_grad_val)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if clip_grad_val > 0:
                for v in model.student.values():
                    v.clip_grad_norm_(clip_grad_val)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            print(sum(loss_dict_reduced.values()))
            logger.info("NaN detected")
            
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
        
        #Save instantly
        if cfg.evaluation.eval_period_iterations >= 0 and (iteration) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        if not single_gpu_run:
            periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    print(cfg)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    #Load model here from pretrained.
    if cfg.train.use_pretrained:

        if cfg.student.arch == "vit_giant2":
            print("loading pretrained DinoV2-giant") 
            model_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
        else:
            AssertionError("only giant pretrained is supported currently")

        model_pretrained = model_pretrained.to(torch.device("cuda"))
        model.student.backbone.patch_embed.proj.weight = model_pretrained.patch_embed.proj.weight
        model.student.backbone.patch_embed.proj.bias = model_pretrained.patch_embed.proj.bias
        model.student.backbone.cls_token = model_pretrained.cls_token
        model.student.backbone.register_tokens = model_pretrained.register_tokens
        model.student.backbone.mask_token = model_pretrained.mask_token

        print(model.state_dict().keys())
        print(model_pretrained.state_dict().keys())
        print(model_pretrained.pos_embed.shape) #1, 1360, 384. We lose pos embed because it was 518
        print(model.student.backbone.pos_embed.shape) #1, 257, 384

        # We need to make sure we grab *all* of the keys.
        # For each block, copy weights over.
        layers = []
        for layer in model_pretrained.blocks:
            layers.append(layer)
        i = 0
        for layer in model.student.backbone.blocks:
            for sublayer in layer:
                if type(sublayer) != torch.nn.Identity:
                    current = layers.pop(0)
                    sublayer.norm1.weight = current.norm1.weight
                    sublayer.norm1.bias = current.norm1.bias
                    sublayer.attn.qkv.weight = current.attn.qkv.weight
                    sublayer.attn.qkv.bias =  current.attn.qkv.bias
                    sublayer.attn.proj.weight = current.attn.proj.weight
                    sublayer.attn.proj.bias = current.attn.proj.bias
                    sublayer.norm2.weight = current.norm2.weight
                    sublayer.norm2.bias = current.norm2.bias
                    try:
                        sublayer.mlp.fc1.weight = current.mlp.fc1.weight
                        sublayer.mlp.fc2.weight = current.mlp.fc2.weight
                        sublayer.mlp.fc1.bias = current.mlp.fc1.bias
                        sublayer.mlp.fc2.bias = current.mlp.fc2.bias
                    except:
                        sublayer.mlp.w12.weight = current.mlp.w12.weight
                        sublayer.mlp.w12.bias = current.mlp.w12.bias
                        sublayer.mlp.w3.weight = current.mlp.w3.weight
                        sublayer.mlp.w3.bias = current.mlp.w3.bias
                    
                    # Compatibility with standard checkpoints:
                    # If current has ls1/ls2 gamma, assign to our layerscale
                    # NOTE: Our block expects ls1/ls2 to be LayerScale objects or Identity.
                    # If Identity, we can't load gamma. But we initialized with layerscale=1e-5 (default)
                    # so they should be LayerScale objects.
                    if hasattr(sublayer, 'ls1') and hasattr(current, 'ls1') and hasattr(sublayer.ls1, 'gamma'):
                        sublayer.ls1.gamma = current.ls1.gamma
                    if hasattr(sublayer, 'ls2') and hasattr(current, 'ls2') and hasattr(sublayer.ls2, 'gamma'):
                        sublayer.ls2.gamma = current.ls2.gamma


        model.student.backbone.norm.weight = model_pretrained.norm.weight
        model.student.backbone.norm.bias = model_pretrained.norm.bias 

    model.prepare_for_distributed_training()
    logger.info("Model:\n{}".format(model))

    if args.eval_only and not single_gpu_run:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    if getattr(args, 'debug', False) and distributed.is_main_process():
        logger.info("DEBUG MODE ENABLED: Recording Memory History")
        torch.cuda.memory._record_memory_history()
        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    if not args.config_file:
        raise ValueError("config file path must be provided")
    CONFIG_FILE_PATH = os.path.abspath(args.config_file)
    main(args)