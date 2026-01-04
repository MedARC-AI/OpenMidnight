# Perceiver Contrastive Training for OpenMidnight
# Trains Perceiver-based context model using multi-target contrastive learning
# Feeds whole region (65,536 tokens) to Perceiver → 64 latents
# Samples K=8 targets, each attends to regional context
# Loss: InfoNCE contrastive (same region = positive, different region = negative)

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

from dinov2.models.vision_transformer import vit_giant2
from dinov2.models.perceiver import PerceiverResampler
from dinov2.models.cross_attention import CrossAttentionTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perceiver_contrastive")


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

        patches_tensor = torch.stack(patches, dim=0)
        region_tensor = self.normalize(self.to_tensor(region))
        slide_id = hash(path) % 10000

        return {
            "patches": patches_tensor,
            "region": region_tensor,
            "positions": torch.tensor(patch_positions),
            "slide_id": slide_id,
            "path": path,
            "region_coords": (region_x, region_y),
            "idx": idx,
        }


# ============================================================================
# Models
# ============================================================================

class PerceiverContextModel(nn.Module):
    """
    Perceiver-based context adaptation model for contrastive learning:
    - Frozen OpenMidnight (ViT-G/14) for patch token extraction
    - Downprojection (1536 → 768)
    - Regional Perceiver compressor (256 patches = 65,536 tokens → 64 latents)
    - Cross-attention fusion (K target patches × regional latents)
    - Contrastive projection head for InfoNCE loss
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
        # Input: 65,536 tokens (256 patches × 256 tokens per patch)
        # Output: 64 latent tokens
        self.perceiver = PerceiverResampler(
            dim=working_dim,
            num_latents=cfg.perceiver.num_latents,
            num_layers=cfg.perceiver.num_layers,
            num_heads=cfg.perceiver.num_heads,
            dim_head=cfg.perceiver.dim_head,
            n_pos_embeddings=65536,
        )

        # Cross-attention transformer for target-region fusion
        self.cross_attn = CrossAttentionTransformer(
            dim=working_dim,
            depth=cfg.cross_attention.depth,
            num_heads=cfg.cross_attention.num_heads,
        )

        # Contrastive projection head
        projection_dim = getattr(cfg.contrastive, 'projection_dim', 256)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(working_dim, working_dim),
            nn.GELU(),
            nn.Linear(working_dim, projection_dim),
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
        Extract patch tokens from all patches in region using OpenMidnight.

        Args:
            patches: [B, 256, 3, 224, 224] - 256 patches in region

        Returns:
            patch_tokens: [B, 256, 256, 1536] - 256 tokens per patch
        """
        B, N, C, H, W = patches.shape

        patches_flat = patches.view(B * N, C, H, W)

        batch_size = 64
        tokens_list = []

        with torch.set_grad_enabled(self.train_patch_vit):
            for i in range(0, B * N, batch_size):
                batch = patches_flat[i:i+batch_size]
                out = self.patch_vit(batch, is_training=True)

                patch_tokens = out["x_norm_patchtokens"]
                tokens_list.append(patch_tokens)

        all_tokens = torch.cat(tokens_list, dim=0)
        all_tokens = all_tokens.view(B, N, 256, -1)

        return all_tokens

    def forward(self, patches):
        """
        Forward pass with multi-target contrastive learning.

        Args:
            patches: [B, 256, 3, 224, 224] - 256 patches in region

        Returns:
            dict with:
                - contrastive_embeddings: [B, K, D] - K normalized embeddings for contrastive loss
                - target_indices: [B, K] - which patches were selected as targets
                - metrics: dict with monitoring stats
        """
        B = patches.shape[0]
        K = self.cfg.contrastive.num_targets

        metrics = {}

        # Step 1: Extract patch tokens from all patches in region
        all_patch_tokens = self.extract_all_patch_tokens(patches)  # [B, 256, 256, 1536]

        # Step 2: Downproject to working dimension
        all_patch_tokens = self.input_proj(all_patch_tokens)  # [B, 256, 256, 768]

        # Step 3: Feed whole region to Perceiver (no masking)
        regional_tokens = all_patch_tokens.reshape(B, 256 * 256, -1)  # [B, 65536, 768]
        regional_latents = self.perceiver(regional_tokens)  # [B, 64, 768]

        # Monitor: Perceiver latent similarity (collapse detection)
        with torch.no_grad():
            latents_norm = F.normalize(regional_latents, dim=-1)
            sim_matrix = latents_norm @ latents_norm.transpose(-1, -2)
            mask = ~torch.eye(64, dtype=torch.bool, device=sim_matrix.device)
            metrics["latent_similarity"] = sim_matrix[:, mask].mean().item()

        # Step 4: Sample K target patches randomly
        target_indices = torch.stack([
            torch.randperm(256, device=patches.device)[:K] for _ in range(B)
        ], dim=0)  # [B, K]

        # Step 5: Cross-attention for each target
        embeddings = []
        cross_attn_norms = []
        for k in range(K):
            target_idx_k = target_indices[:, k]  # [B]
            target_tokens = all_patch_tokens[torch.arange(B), target_idx_k]  # [B, 256, 768]
            enriched = self.cross_attn(target_tokens, regional_latents)  # [B, 256, 768]

            # Monitor: Cross-attention output norm
            with torch.no_grad():
                cross_attn_norms.append(enriched.norm(dim=-1).mean().item())

            emb = enriched.mean(dim=1)  # [B, 768]
            embeddings.append(emb)

        embeddings = torch.stack(embeddings, dim=1)  # [B, K, 768]
        metrics["cross_attn_output_norm"] = sum(cross_attn_norms) / len(cross_attn_norms)

        # Step 6: Project to contrastive space and normalize
        contrastive_emb = self.contrastive_proj(embeddings)  # [B, K, 256]
        contrastive_emb = F.normalize(contrastive_emb, dim=-1)

        # Monitor: Embedding norm sanity check
        with torch.no_grad():
            emb_norms = contrastive_emb.norm(dim=-1)
            metrics["emb_norm_mean"] = emb_norms.mean().item()
            metrics["emb_norm_std"] = emb_norms.std().item()

        return {
            "contrastive_embeddings": contrastive_emb,
            "target_indices": target_indices,
            "metrics": metrics,
        }


# ============================================================================
# Losses
# ============================================================================

def compute_contrastive_loss(embeddings, region_ids, cfg, anchor_mask=None):
    """
    InfoNCE contrastive loss for multi-target patch embeddings.
    Positive pairs: K targets from same region
    Negative pairs: targets from different regions

    Args:
        embeddings: [B, K, D] - K normalized target embeddings per region
        region_ids: [B] - region identifier (hash(path) % 10000)
        cfg: config
        anchor_mask: [B] - bool, True for local batch (distributed training)

    Returns:
        loss: scalar tensor
        metrics: dict with similarity and batch composition stats
    """
    B, K, D = embeddings.shape
    temperature = cfg.loss.contrastive_temperature

    # Determine anchors
    if anchor_mask is None:
        anchor_indices = torch.arange(B, device=embeddings.device)
    else:
        anchor_indices = torch.nonzero(anchor_mask, as_tuple=True)[0]

    # Flatten to patch level
    anchor_emb = embeddings[anchor_indices].reshape(-1, D)  # [B_anchor*K, D]
    all_emb = embeddings.reshape(B * K, D)  # [B*K, D]

    # Expand region IDs for each of K targets
    anchor_region_ids = region_ids[anchor_indices].repeat_interleave(K)
    all_region_ids = region_ids.repeat_interleave(K)

    # Similarity matrix (temperature-scaled)
    sim = (anchor_emb @ all_emb.t()) / temperature  # [B_anchor*K, B*K]

    # Positive mask: same region, different patch instance
    same_region = (all_region_ids.unsqueeze(0) == anchor_region_ids.unsqueeze(1))

    # Remove self-matches
    B_anchor = len(anchor_indices)
    anchor_idx = (anchor_indices.view(-1, 1) * K + torch.arange(K, device=embeddings.device).view(1, -1)).reshape(-1)
    global_idx = torch.arange(B * K, device=embeddings.device)
    self_mask = (global_idx.unsqueeze(0) == anchor_idx.unsqueeze(1))

    pos_mask = same_region & ~self_mask
    neg_mask = ~same_region & ~self_mask

    # InfoNCE with log-sum-exp (numerically stable)
    NEG_INF = -1e9
    sim_all = sim.masked_fill(self_mask, NEG_INF)
    sim_pos = sim.masked_fill(~pos_mask, NEG_INF)

    logsumexp_all = torch.logsumexp(sim_all, dim=1)
    logsumexp_pos = torch.logsumexp(sim_pos, dim=1)
    loss = logsumexp_all - logsumexp_pos

    # Only compute for samples with at least 1 positive
    valid_mask = pos_mask.any(dim=1)

    if not valid_mask.any():
        logger.warning(f"No positive pairs found in batch (B={B}, K={K})")
        metrics = {
            "pos_sim": 0.0, "neg_sim": 0.0, "similarity_gap": 0.0,
            "raw_pos_sim": 0.0, "raw_neg_sim": 0.0,
            "num_unique_regions": 0, "avg_positives": 0.0, "avg_negatives": 0.0,
        }
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True), metrics

    # Compute all metrics
    with torch.no_grad():
        # Temperature-scaled similarities
        pos_sims = sim[pos_mask].mean()
        neg_sims = sim[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=sim.device)

        # Raw (unscaled) similarities
        raw_sim = anchor_emb @ all_emb.t()
        raw_pos_sim = raw_sim[pos_mask].mean()
        raw_neg_sim = raw_sim[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=sim.device)

        # Batch composition
        num_unique_regions = len(torch.unique(region_ids))
        avg_positives = pos_mask.sum(dim=1).float().mean()
        avg_negatives = neg_mask.sum(dim=1).float().mean()

        metrics = {
            "pos_sim": pos_sims.item(),
            "neg_sim": neg_sims.item(),
            "similarity_gap": (pos_sims - neg_sims).item(),
            "raw_pos_sim": raw_pos_sim.item(),
            "raw_neg_sim": raw_neg_sim.item(),
            "num_unique_regions": num_unique_regions,
            "avg_positives": avg_positives.item(),
            "avg_negatives": avg_negatives.item(),
        }

    return loss[valid_mask].mean(), metrics


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


def compute_grad_norm(module):
    """Compute L2 norm of gradients for a module."""
    total_norm = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


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
            name=f"perceiver_contrastive_{cfg.train.output_dir.split('/')[-1]}",
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

    logger.info(f"Starting contrastive training for {cfg.train.max_iters} iterations (K={cfg.contrastive.num_targets})")

    while iteration < cfg.train.max_iters:
        if world_size > 1:
            sampler.set_epoch(iteration)

        for batch in dataloader:
            if iteration >= cfg.train.max_iters:
                break

            # Move to device
            patches = batch["patches"].to(device)  # [B, 256, 3, 224, 224]
            slide_ids = batch["slide_id"].to(device)  # [B]

            # Forward pass
            model_out = model(patches)
            embeddings = model_out["contrastive_embeddings"]  # [B, K, D]
            model_metrics = model_out["metrics"]

            # Gather across GPUs for distributed training
            if world_size > 1:
                gathered_embeddings = dist_fn.all_gather(embeddings)
                embeddings_all = torch.cat(list(gathered_embeddings), dim=0)

                slide_ids_list = [torch.zeros_like(slide_ids) for _ in range(world_size)]
                dist.all_gather(slide_ids_list, slide_ids)
                slide_ids_all = torch.cat(slide_ids_list, dim=0)

                # Anchor mask: only local batch computes gradients
                anchor_mask = torch.zeros(embeddings_all.shape[0], device=device, dtype=torch.bool)
                anchor_mask[rank * patches.shape[0] : (rank + 1) * patches.shape[0]] = True
            else:
                embeddings_all = embeddings
                slide_ids_all = slide_ids
                anchor_mask = None

            # Compute contrastive loss
            contrastive_loss, loss_metrics = compute_contrastive_loss(
                embeddings_all,
                slide_ids_all,
                cfg,
                anchor_mask=anchor_mask,
            )

            # Backward
            contrastive_loss.backward()

            # Compute gradient norms
            model_module = model.module if world_size > 1 else model
            grad_norms = {
                "input_proj": compute_grad_norm(model_module.input_proj),
                "perceiver": compute_grad_norm(model_module.perceiver),
                "cross_attn": compute_grad_norm(model_module.cross_attn),
                "contrastive_proj": compute_grad_norm(model_module.contrastive_proj),
            }

            # Gradient clipping
            if cfg.optim.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.optim.clip_grad)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if iteration % 10 == 0 and rank == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Iter {iteration}/{cfg.train.max_iters} | "
                    f"Loss: {contrastive_loss.item():.4f} | "
                    f"Pos: {loss_metrics['pos_sim']:.3f} | "
                    f"Neg: {loss_metrics['neg_sim']:.3f} | "
                    f"Gap: {loss_metrics['similarity_gap']:.3f} | "
                    f"Latent sim: {model_metrics['latent_similarity']:.3f} | "
                    f"LR: {lr:.2e}"
                )

                if cfg.wandb.project:
                    log_dict = {
                        "train/contrastive_loss": contrastive_loss.item(),
                        "train/pos_similarity": loss_metrics["pos_sim"],
                        "train/neg_similarity": loss_metrics["neg_sim"],
                        "train/similarity_gap": loss_metrics["similarity_gap"],
                        "train/raw_pos_similarity": loss_metrics["raw_pos_sim"],
                        "train/raw_neg_similarity": loss_metrics["raw_neg_sim"],
                        "train/num_unique_regions": loss_metrics["num_unique_regions"],
                        "train/avg_positives": loss_metrics["avg_positives"],
                        "train/avg_negatives": loss_metrics["avg_negatives"],
                        "train/latent_similarity": model_metrics["latent_similarity"],
                        "train/cross_attn_output_norm": model_metrics["cross_attn_output_norm"],
                        "train/grad_norm/input_proj": grad_norms["input_proj"],
                        "train/grad_norm/perceiver": grad_norms["perceiver"],
                        "train/grad_norm/cross_attn": grad_norms["cross_attn"],
                        "train/grad_norm/contrastive_proj": grad_norms["contrastive_proj"],
                        "train/lr": lr,
                        "train/iteration": iteration,
                    }

                    # Embedding norm check (every 100 iters)
                    if iteration % 100 == 0:
                        log_dict["train/emb_norm_mean"] = model_metrics["emb_norm_mean"]
                        log_dict["train/emb_norm_std"] = model_metrics["emb_norm_std"]

                    wandb.log(log_dict)

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
        cfg_path = "dinov2/configs/train/perceiver_contrastive.yaml"

    cfg = OmegaConf.load(cfg_path)

    # Run training
    train(cfg)
