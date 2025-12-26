# Perceiver Context Adaptation Training for OpenMidnight
# Trains a Perceiver-based context model using frozen OpenMidnight patch tokens
# Compresses 65,280 regional patches → 64 latents, then fuses with target via cross-attention
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
from dinov2.models.perceiver import PerceiverResampler
from dinov2.models.cross_attention import CrossAttentionTransformer
from dinov2.layers import MemEffAttention, Mlp, PatchEmbed
from dinov2.layers.block import Block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perceiver_context")


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

        # Load svs sample list
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
        # Add some randomness to the region position
        offset_x = np.random.randint(0, self.patch_size) if np.random.rand() > 0.5 else 0
        offset_y = np.random.randint(0, self.patch_size) if np.random.rand() > 0.5 else 0

        region_x = max(0, base_x - offset_x - self.patch_size * (self.patches_per_side // 2))
        region_y = max(0, base_y - offset_y - self.patch_size * (self.patches_per_side // 2))

        # Avoid padding at slide edges
        level0_w, level0_h = slide.level_dimensions[0]
        downsample = slide.level_downsamples[level]
        region_size_level0 = int(round(self.region_size * downsample))

        # Skip samples that cannot fit the desired region size
        if region_size_level0 > level0_w or region_size_level0 > level0_h:
            return self.__getitem__((idx + 1) % len(self))

        max_x = max(level0_w - region_size_level0, 0)
        max_y = max(level0_h - region_size_level0, 0)
        region_x = min(region_x, max_x)
        region_y = min(region_y, max_y)

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

        # Full region for visualization
        region_tensor = self.normalize(self.to_tensor(region))

        # NEW: Add slide_id for region-level contrastive
        slide_id = hash(path) % 10000

        return {
            "patches": patches_tensor,
            "region": region_tensor,
            "positions": torch.tensor(patch_positions),
            "slide_id": slide_id,  # NEW
            "path": path,
            "region_coords": (region_x, region_y),
            "idx": idx,
        }


# ============================================================================
# Models
# ============================================================================

class MAEDecoder(nn.Module):
    """
    Simple MAE decoder for pixel reconstruction.
    Takes context-enriched patch tokens and predicts masked patch pixels.
    """

    def __init__(self, cfg):
        super().__init__()
        context_dim = cfg.perceiver.dim  # 768
        decoder_dim = cfg.mae_decoder.embed_dim  # 512
        depth = cfg.mae_decoder.depth
        num_heads = cfg.mae_decoder.num_heads

        # Project from context dimension to decoder dimension
        self.proj = nn.Linear(context_dim, decoder_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                attn_class=MemEffAttention,
                ffn_layer=Mlp,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(decoder_dim)

        # Predict pixels
        patch_size = 224
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * 3)

        # Initialize
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, mask):
        """
        Args:
            x: [B, 256, context_dim] - enriched target patches
            mask: [B, 256] - True = masked positions

        Returns:
            pred: [B, 256, 224*224*3] - predicted pixels
        """
        B, N, _ = x.shape

        # Project to decoder dimension
        x = self.proj(x)  # [B, 256, decoder_dim]

        # Replace masked positions with mask token
        mask_tokens = self.mask_token.expand(B, N, -1)
        x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

        # Transform
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Predict pixels
        pred = self.pred(x)  # [B, 256, 224*224*3]

        return pred


class PerceiverContextModel(nn.Module):
    """
    Perceiver-based context adaptation model combining:
    - Frozen OpenMidnight (ViT-G/14) for patch token extraction
    - Downprojection (1536 → 768)
    - Regional Perceiver compressor (65,280 patches → 64 latents)
    - Cross-attention fusion (target patches × regional latents)
    - MAE decoder for pixel reconstruction
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Load frozen OpenMidnight
        self.patch_vit = self._load_patch_vit(cfg)
        self.train_patch_vit = False

        # Freeze by default
        for param in self.patch_vit.parameters():
            param.requires_grad = False
        self.patch_vit.eval()

        # Downproject from OpenMidnight dimension to working dimension
        input_dim = cfg.patch_vit.embed_dim  # 1536
        working_dim = cfg.perceiver.dim  # 768
        self.input_proj = nn.Linear(input_dim, working_dim)

        # Perceiver for regional context compression
        # Input: 65,280 patches (255 tiles × 256 patches)
        # Output: 64 latent tokens
        self.perceiver = PerceiverResampler(
            dim=working_dim,
            num_latents=cfg.perceiver.num_latents,  # 64
            num_layers=cfg.perceiver.num_layers,  # 6
            num_heads=cfg.perceiver.num_heads,  # 12
            dim_head=cfg.perceiver.dim_head,  # 64
            n_pos_embeddings=65280,  # 255 tiles × 256 patches
        )

        # Cross-attention transformer for target-region fusion
        # Fuses target patches (256) with regional latents (64)
        self.cross_attn = CrossAttentionTransformer(
            dim=working_dim,
            depth=cfg.cross_attention.depth,  # 4
            num_heads=cfg.cross_attention.num_heads,  # 12
        )

        # MAE decoder
        self.mae_decoder = MAEDecoder(cfg)

        # Contrastive projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(working_dim, working_dim),
            nn.GELU(),
            nn.Linear(working_dim, 256),
        )

    def _load_patch_vit(self, cfg):
        """Load the frozen OpenMidnight checkpoint."""
        logger.info(f"Loading OpenMidnight from {cfg.patch_vit.checkpoint}")

        # Build ViT-Giant2 architecture
        patch_vit = vit_giant2(
            patch_size=cfg.patch_vit.patch_size,
            num_register_tokens=cfg.patch_vit.num_register_tokens,
            img_size=cfg.data.patch_size,  # 224
            block_chunks=int(getattr(cfg.patch_vit, "block_chunks", 4)),
            ffn_layer=getattr(cfg.patch_vit, "ffn_layer", "swiglu"),
            init_values=float(getattr(cfg.patch_vit, "init_values", 1.0)),
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

    def extract_all_patch_tokens(self, patches):
        """
        Extract PATCH TOKENS (not CLS tokens) from all tiles using OpenMidnight.

        Args:
            patches: [B, 256, 3, 224, 224] - all tiles in region

        Returns:
            patch_tokens: [B, 256, 256, 1536] - patch tokens for each tile
        """
        B, N, C, H, W = patches.shape  # N=256 tiles

        # Flatten to process all tiles
        patches_flat = patches.view(B * N, C, H, W)

        # Process in batches to avoid OOM
        batch_size = 64
        tokens_list = []

        with torch.set_grad_enabled(self.train_patch_vit):
            for i in range(0, B * N, batch_size):
                batch = patches_flat[i:i+batch_size]
                out = self.patch_vit(batch, is_training=True)

                # KEY CHANGE: Use patch tokens instead of CLS token
                # Each 224×224 tile has 16×16 = 256 patches of size 14×14
                patch_tokens = out["x_norm_patchtokens"]  # [batch_size, 256, 1536]
                tokens_list.append(patch_tokens)

        # Concatenate and reshape
        all_tokens = torch.cat(tokens_list, dim=0)  # [B*256, 256, 1536]
        all_tokens = all_tokens.view(B, N, 256, -1)  # [B, 256 tiles, 256 patches, 1536]

        return all_tokens

    def forward(self, patches, positions):
        """
        Forward pass with Perceiver-based context adaptation.

        Args:
            patches: [B, 256, 3, 224, 224] - all tiles in region
            positions: [B, 256, 2] - (row, col) positions in 16×16 grid

        Returns:
            dict with:
                - target_embedding: [B, 768] - final embedding for contrastive
                - enriched_target: [B, 256, 768] - context-fused patches
                - mae_pred: [B, 256, 224*224*3] - pixel predictions (if training)
                - target_idx: [B] - which tile was selected as target
        """
        B = patches.shape[0]

        # Step 1: Extract patch tokens from all tiles
        all_patch_tokens = self.extract_all_patch_tokens(patches)  # [B, 256, 256, 1536]

        # Step 2: Downproject to working dimension
        all_patch_tokens = self.input_proj(all_patch_tokens)  # [B, 256, 256, 768]

        # Step 3: Randomly select target tile (one per sample)
        target_idx = torch.randint(0, 256, (B,), device=patches.device)

        # Extract target tile patches
        target_patches = all_patch_tokens[torch.arange(B), target_idx]  # [B, 256, 768]

        # Step 4: Extract regional patches (all except target)
        # Create mask to select all tiles except target
        regional_mask = torch.ones(B, 256, dtype=torch.bool, device=patches.device)
        regional_mask[torch.arange(B), target_idx] = False

        # Gather regional patches: 255 tiles × 256 patches = 65,280 patches
        regional_tiles = []
        for b in range(B):
            region_tiles_b = all_patch_tokens[b][regional_mask[b]]  # [255, 256, 768]
            regional_tiles.append(region_tiles_b)
        regional_tiles = torch.stack(regional_tiles, dim=0)  # [B, 255, 256, 768]

        # Flatten to patch level
        regional_patches = regional_tiles.reshape(B, 255 * 256, 768)  # [B, 65280, 768]

        # Step 5: Compress regional context with Perceiver
        regional_latents = self.perceiver(regional_patches)  # [B, 64, 768]

        # Step 6: Fuse target with regional context via cross-attention
        enriched_target = self.cross_attn(target_patches, regional_latents)  # [B, 256, 768]

        # Step 7: Generate final embedding (mean pooling)
        target_embedding = enriched_target.mean(dim=1)  # [B, 768]

        # Step 8: Contrastive projection
        contrastive_emb = self.contrastive_proj(target_embedding)  # [B, 256]
        contrastive_emb = F.normalize(contrastive_emb, dim=-1)

        # Step 9: MAE reconstruction (if training)
        mae_pred = None
        mae_mask = None
        if self.training:
            # Mask 75% of target patches
            mae_mask = torch.rand(B, 256, device=patches.device) < self.cfg.loss.mae_mask_ratio
            mae_pred = self.mae_decoder(enriched_target, mae_mask)

        return {
            "target_embedding": target_embedding,
            "contrastive_embeddings": contrastive_emb,
            "enriched_target": enriched_target,
            "mae_pred": mae_pred,
            "mae_mask": mae_mask,
            "target_idx": target_idx,
            "regional_latents": regional_latents,
        }


# ============================================================================
# Losses
# ============================================================================

def compute_contrastive_loss(embeddings, target_indices, positions, cfg, slide_ids=None, anchor_mask=None):
    """
    Contrastive loss for target tile embeddings with spatial awareness.

    Args:
        embeddings: [B, D] - normalized target tile embeddings
        target_indices: [B] - which tile was selected as target (0-255)
        positions: [B, 256, 2] - all tile positions
        cfg: config
        slide_ids: [B] - slide identifier for region-level contrast
        anchor_mask: [B] - bool mask for anchors (local batch)
    """
    B, D = embeddings.shape
    temperature = cfg.loss.contrastive_temperature
    distance_scale = cfg.loss.spatial_distance_scale

    # Get target positions
    target_positions = positions[torch.arange(B), target_indices]  # [B, 2]

    # Determine anchors
    if anchor_mask is None:
        anchor_indices = torch.arange(B, device=embeddings.device)
    else:
        anchor_indices = torch.nonzero(anchor_mask, as_tuple=True)[0]

    B_anchor = len(anchor_indices)

    anchor_emb = embeddings[anchor_indices]  # [B_anchor, D]
    anchor_pos = target_positions[anchor_indices]  # [B_anchor, 2]

    # Compute similarity matrix
    sim = anchor_emb @ embeddings.t() / temperature  # [B_anchor, B]

    # Spatial distance weighting
    pos_diff = anchor_pos.unsqueeze(1) - target_positions.unsqueeze(0)  # [B_anchor, B, 2]
    spatial_dist = torch.norm(pos_diff.float(), dim=-1)  # [B_anchor, B]

    # Positives: spatially close targets (within radius)
    pos_mask = (spatial_dist < distance_scale) & (spatial_dist > 0)

    # Distance-weighted positive similarity
    pos_weight = torch.exp(-spatial_dist / distance_scale)
    pos_weight = pos_weight * pos_mask.float()

    # Compute loss (InfoNCE with spatial weighting)
    exp_sim = torch.exp(sim)

    # Numerator: weighted positives
    pos_sim = (exp_sim * pos_weight).sum(dim=1) / (pos_weight.sum(dim=1) + 1e-8)

    # Denominator: all similarities
    neg_sim = exp_sim.sum(dim=1)

    loss = -torch.log(pos_sim / (neg_sim + 1e-8) + 1e-8).mean()

    return loss


def compute_mae_loss(pred, target_patches, mae_mask, target_idx, cfg):
    """
    MAE reconstruction loss for target tile only.

    Args:
        pred: [B, 256, 224*224*3] - predicted pixels for target patches
        target_patches: [B, 256, 3, 224, 224] - all patches
        mae_mask: [B, 256] - True = masked positions in target tile
        target_idx: [B] - which tile is the target
        cfg: config
    """
    if pred is None or mae_mask is None or not mae_mask.any():
        return torch.tensor(0.0, device=pred.device if pred is not None else target_patches.device)

    B = target_patches.shape[0]

    # Extract target tile patches
    target_tiles = target_patches[torch.arange(B), target_idx]  # [B, 3, 224, 224]

    # Denormalize and normalize per-patch (MAE style)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(target_tiles.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(target_tiles.device)
    target_pixels = target_tiles * std + mean  # [B, 3, 224, 224]

    # Expand to match patch structure
    # Note: Target tile itself contains 256 patches from ViT
    # But for MAE we reconstruct the original 224x224 pixels
    # So we need to treat the whole 224x224 as a single unit

    # Simplified: Reconstruct full 224x224 image
    target_flat = target_pixels.view(B, -1)  # [B, 224*224*3]

    # For now, use first position's prediction
    # TODO: Properly handle patch-level vs image-level reconstruction
    pred_flat = pred[:, 0, :]  # [B, 224*224*3] - use first patch's prediction

    # MSE loss
    mse = F.mse_loss(pred_flat, target_flat)

    return mse


def denormalize(tensor):
    """Denormalize from ImageNet normalization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


# ============================================================================
# Training
# ============================================================================

def concat_all_gather(tensor):
    """Gather tensors from all GPUs."""
    if not dist.is_initialized():
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor)
    output = torch.cat(tensors_gather, dim=0)
    return output


def train(cfg):
    """Main training loop."""

    # Initialize distributed training
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    logger.info(f"Rank {rank}/{world_size} initialized on {device}")

    # Initialize wandb (rank 0 only)
    if rank == 0 and cfg.wandb.project:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"perceiver_context_{cfg.train.output_dir.split('/')[-1]}",
        )

    # Create dataset and dataloader
    dataset = RegionDataset(cfg)

    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model = PerceiverContextModel(cfg).to(device)

    # Wrap in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optimizer
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)

    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.optim.base_lr * world_size,  # Linear scaling with batch size
        weight_decay=cfg.optim.weight_decay,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )

    # Learning rate scheduler (cosine)
    def lr_lambda(step):
        if step < 1000:  # Warmup
            return step / 1000
        # Cosine decay
        progress = (step - 1000) / (cfg.train.max_iters - 1000)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    model.train()
    iteration = 0

    logger.info(f"Starting training for {cfg.train.max_iters} iterations")

    while iteration < cfg.train.max_iters:
        if world_size > 1:
            sampler.set_epoch(iteration)

        for batch in dataloader:
            if iteration >= cfg.train.max_iters:
                break

            # Move to device
            patches = batch["patches"].to(device)  # [B, 256, 3, 224, 224]
            positions = batch["positions"].to(device)  # [B, 256, 2]
            slide_ids = batch["slide_id"].to(device) if "slide_id" in batch else None

            # Forward pass
            model_out = model(patches, positions)

            # Gather embeddings across GPUs for contrastive loss
            if world_size > 1:
                emb_gathered = concat_all_gather(model_out["contrastive_embeddings"])
                target_idx_gathered = concat_all_gather(model_out["target_idx"])
                positions_gathered = concat_all_gather(positions)
                slide_ids_gathered = concat_all_gather(slide_ids) if slide_ids is not None else None

                # Anchor mask: only local batch
                B_local = patches.shape[0]
                anchor_mask = torch.zeros(emb_gathered.shape[0], dtype=torch.bool, device=device)
                anchor_mask[rank * B_local:(rank + 1) * B_local] = True
            else:
                emb_gathered = model_out["contrastive_embeddings"]
                target_idx_gathered = model_out["target_idx"]
                positions_gathered = positions
                slide_ids_gathered = slide_ids
                anchor_mask = None

            # Contrastive loss
            contrastive_loss = compute_contrastive_loss(
                emb_gathered,
                target_idx_gathered,
                positions_gathered,
                cfg,
                slide_ids=slide_ids_gathered,
                anchor_mask=anchor_mask,
            )

            # MAE loss
            mae_loss = compute_mae_loss(
                model_out["mae_pred"],
                patches,
                model_out["mae_mask"],
                model_out["target_idx"],
                cfg,
            )

            # Total loss
            total_loss = (
                cfg.loss.contrastive_weight * contrastive_loss +
                cfg.loss.mae_weight * mae_loss
            )

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if cfg.optim.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.optim.clip_grad)

            optimizer.step()
            scheduler.step()

            # Logging
            if iteration % 10 == 0 and rank == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Iter {iteration}/{cfg.train.max_iters} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Contrastive: {contrastive_loss.item():.4f} | "
                    f"MAE: {mae_loss.item():.4f} | "
                    f"LR: {lr:.2e}"
                )

                if cfg.wandb.project:
                    wandb.log({
                        "train/total_loss": total_loss.item(),
                        "train/contrastive_loss": contrastive_loss.item(),
                        "train/mae_loss": mae_loss.item(),
                        "train/lr": lr,
                        "train/iteration": iteration,
                    })

            # Save checkpoint
            if iteration % cfg.evaluation.save_period_iters == 0 and rank == 0:
                output_dir = Path(cfg.train.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                checkpoint = {
                    "iteration": iteration,
                    "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                }

                torch.save(checkpoint, output_dir / f"checkpoint_iter{iteration}.pth")
                logger.info(f"Saved checkpoint at iteration {iteration}")

            iteration += 1

    logger.info("Training complete!")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys

    # Load config
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        cfg_path = "dinov2/configs/train/perceiver_context.yaml"

    cfg = OmegaConf.load(cfg_path)

    # Run training
    train(cfg)
