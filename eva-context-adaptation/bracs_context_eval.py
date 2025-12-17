"""
BRACS context evaluation: True apples-to-apples comparison.

This script produces context-enriched embeddings for BRACS in a way that ensures
a fair comparison with the baseline (single-patch OpenMidnight CLS token).

KEY INSIGHT: BRACS dataset contains patch images at different dimensions.
Patches are classified in 7 classes, and the folder structure is organized 
by the first level as split folders and the second level as class folders.
BRACS doesn't naturally provide metadata to trace position of patches within
the source WSI. However, we have created a CSV metadata file containing this
information and other useful details.

NOTE: Data is not stored locally. Each time a WSI, patch image, or metadata.csv
is needed, it is downloaded from the AWS S3 bucket. This means that
that training may be slower due to heavy or repeated downloads.

For context evaluation, we:
1. Retrieve the patch metadata (centroid coordinates and WSI ID) from the CSV
2. Download the corresponding WSI from S3
3. Extract a square region of size patches_per_side × model_input_size
   centered on the patch centroid
4. Split the region into a patches_per_side × patches_per_side grid of individual patches
5. Resize each patch to model_input_size × model_input_size (matching OpenMidnight input)
6. Flatten the grid into a single tensor and compute the target index corresponding to the
   original patch position.
7. Run all 256 patches through OpenMidnight → context-ViT
8. Extract the context-enriched embedding for the CENTER patch position

For baseline comparison:
1. Download the single patch image from S3.
2. Resize to model_input_size × model_input_size (224×224).
3. Run through OpenMidnight only.
4. Save the CLS token.

This ensures both approaches use compatible data and differ only in whether
context is used.

Usage:
  python bracs_context_eval.py \
    --checkpoint /path/to/context_checkpoint.pth \
    --config /path/to/context_adaptation.yaml \
    --mode context \
    --output-root /path/to/output

  python bracs_context_eval.py \
    --baseline-checkpoint /path/to/openmidnight.pth \
    --mode baseline \
    --output-root /path/to/output_baseline
"""

import argparse
import csv
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Literal

import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Setup paths
_script_dir = Path(__file__).resolve().parent
_eva_probe_root = _script_dir.parent
_openmidnight_root = _eva_probe_root.parent
sys.path.insert(0, str(_openmidnight_root))
sys.path.insert(0, str(_eva_probe_root / "src"))

# Import after path setup
try:
    import openslide
except ImportError:
    openslide = None
    print("Warning: openslide not available, WSI reading will fail")

from omegaconf import OmegaConf


def load_openmidnight(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load the frozen OpenMidnight model."""
    from dinov2.models.vision_transformer import vit_giant2

    model = vit_giant2(
        patch_size=14,
        num_register_tokens=4,  # OpenMidnight uses 4 register tokens
        img_size=224,
        block_chunks=4,
        ffn_layer="swiglu",
        init_values=1.0,
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    if "teacher" in state:
        state = state["teacher"]
    state = {k.replace("backbone.", ""): v for k, v in state.items() if "backbone" in k}

    msg = model.load_state_dict(state, strict=False)
    print(f"Loaded OpenMidnight: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    model = model.to(device)
    model.eval()
    return model


def load_context_model(cfg_path: str, ckpt_path: str, device: torch.device):
    """Load the context adaptation model."""
    from dinov2.train.train_context import ContextAdaptationModel

    cfg = OmegaConf.load(cfg_path)
    model = ContextAdaptationModel(cfg).to(device)

    state = torch.load(ckpt_path, map_location="cpu")

    def strip_module(sd):
        return {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in sd.items()}

    if "patch_vit" in state:
        model.patch_vit.load_state_dict(strip_module(state["patch_vit"]), strict=False)
    model.context_vit.load_state_dict(strip_module(state["context_vit"]), strict=False)
    model.mae_decoder.load_state_dict(strip_module(state["mae_decoder"]), strict=False)
    model.contrastive_proj.load_state_dict(strip_module(state["contrastive_proj"]), strict=False)

    model.eval()
    return model, cfg

def download_with_awscli(bucket, key, dest, endpoint, profile=None):
    cmd = [
        "aws", "s3", "cp",
        f"s3://{os.path.join(bucket, key)}",
        dest,
        "--endpoint-url", endpoint
    ]
    if profile:
        cmd.extend(["--profile", profile])

    subprocess.run(cmd, check=True)

class BRACSContextDataset(Dataset):
    """
    PyTorch Dataset for BRACS patch-level embedding generation in baseline or context mode.

    This dataset supports two modes:
    - "baseline": returns a single image patch corresponding to a BRACS patch.
    - "context": returns a grid of spatially adjacent patches extracted from the WSI
      around the patch centroid, along with the index of the target patch within the grid.

    The dataset is designed to be wrapped by a DataLoader and to return samples in a
    format that matches the expected inputs of the embedding generation pipeline.

    Returns:
        Baseline mode:
            (patch: Tensor [3, H, W], label: int, index: int)

        Context mode:
            (region: Tensor [N, 3, H, W], target_index: int, label: int, index: int)

        where N = patches_per_side ** 2 and H = W = model_input_size.
    """

    def __init__(
        self,
        split: str,
        mode: Literal["baseline", "context"],
        aws_endpoint: str,
        aws_bracs_root: str = "/bracs",
        aws_bucket: str = "path-datasets",
        aws_profile: str = None,
        patches_per_side: int = 16,
        model_input_size: int = 224,
        max_samples: int = None,
    ):
        super().__init__()

        self.split = split
        self.mode = mode
        self.aws_endpoint = aws_endpoint
        self.aws_bracs_root = aws_bracs_root
        self.aws_bucket = aws_bucket
        self.aws_profile = aws_profile
        self.patches_per_side = patches_per_side
        self.model_input_size = model_input_size
        self.previous_wsi = ""  # Track previously downloaded WSI

        # Download patch metadata CSV
        tmp_csv_path = "/tmp/patch_metadata.csv"
        download_with_awscli(
            self.aws_bucket,
            os.path.join(self.aws_bracs_root, "patch_metadata.csv"),
            tmp_csv_path,
            self.aws_endpoint,
            self.aws_profile,
        )

        # Load metadata and filter by split
        self.meta_df = pd.read_csv(tmp_csv_path)
        self.meta_df = self.meta_df[self.meta_df["split"] == split].reset_index(drop=True)

        if max_samples is not None:
            # TODO: consider taking random samples here?
            self.meta_df = self.meta_df.head(max_samples).reset_index(drop=True)

        # Transform
        self.transform = v2.Compose([
            v2.Resize(size=model_input_size),
            v2.CenterCrop(size=model_input_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        # Clean up
        os.remove(tmp_csv_path)
    
    def __len__(self) -> int:
        return len(self.meta_df)
    
    def __getitem__(self, idx: int):
        row = self.meta_df.iloc[idx]

        label = int(row["label"])
        patch_name = row["patch_name"]

        if self.mode == "baseline":
            # Download patch image
            patch_key = os.path.join(
                self.aws_bracs_root, row["patch_fullpath"]
            )
            tmp_patch = f"/tmp/{patch_name}.png"

            download_with_awscli(
                self.aws_bucket,
                patch_key,
                tmp_patch,
                self.aws_endpoint,
                self.aws_profile,
            )

            img = Image.open(tmp_patch).convert("RGB")
            patch_tensor = self.transform(img)

            os.remove(tmp_patch)

            return patch_tensor, label, idx

        # Context mode
        else:
            # --- centroid info ---
            cx = float(row["centroid_x"])
            cy = float(row["centroid_y"])

            # --- download WSI ---
            wsi_id = "_".join(patch_name.split("_")[:2]) + ".svs"
            tmp_wsi = f"/tmp/{wsi_id}"

            # If already exists (from previous sample), reuse
            if tmp_wsi == self.previous_wsi:
                slide = openslide.OpenSlide(tmp_wsi)
            else:
                if os.path.exists(self.previous_wsi):
                    os.remove(self.previous_wsi)

                wsi_key = os.path.join(self.aws_bracs_root, "BRACS_WSI", wsi_id)
                download_with_awscli(
                    self.aws_bucket,
                    wsi_key,
                    tmp_wsi,
                    self.aws_endpoint,
                    self.aws_profile,
                )

                slide = openslide.OpenSlide(tmp_wsi)

            w, h = slide.dimensions
            region_size = self.patches_per_side * self.model_input_size
            half = region_size // 2

            left = int(max(cx - half, 0))
            top = int(max(cy - half, 0))
            right = int(min(cx + half, w))
            bottom = int(min(cy + half, h))

            region = slide.read_region(
                (left, top), 0, (right - left, bottom - top)
            ).convert("RGB")

            # Resize to exact grid size if needed
            if region.size != (region_size, region_size):
                region = region.resize((region_size, region_size), Image.BILINEAR)

            # --- relative centroid ---
            cx_rel = cx - left
            cy_rel = cy - top

            scale_x = region_size / (right - left)
            scale_y = region_size / (bottom - top)

            cx_rel *= scale_x
            cy_rel *= scale_y

            # --- grid index ---
            tile = self.model_input_size
            grid_col = int(cx_rel // tile)
            grid_row = int(cy_rel // tile)

            grid_col = min(max(grid_col, 0), self.patches_per_side - 1)
            grid_row = min(max(grid_row, 0), self.patches_per_side - 1)

            target_index = grid_row * self.patches_per_side + grid_col

            # --- extract tiles ---
            tiles = [
                self.transform(region.crop(
                    (c * tile, r * tile, c * tile + tile, r * tile + tile)
                ))
                for r in range(self.patches_per_side)
                for c in range(self.patches_per_side)
            ]

            region_tensor = torch.stack(tiles, dim=0)

            slide.close()
            self.previous_wsi = tmp_wsi

            return region_tensor, target_index, label, idx
    
    def get_patch_name(self, idx: int) -> str:
        return self.meta_df.iloc[idx]["patch_name"]
    
    def get_wsi_id(self, idx: int) -> str:
        patch_name = self.get_patch_name(idx)
        wsi_id = "_".join(patch_name.split("_")[:2])
        return wsi_id

def generate_embeddings(
    dataloader: DataLoader,
    output_root: Path,
    model,
    device: torch.device,
    patch_batch_size: int,
    use_amp: bool,
) -> List[Dict]:
    """Generate embeddings for the dataset."""
    split = dataloader.dataset.split
    mode = dataloader.dataset.mode

    emb_dir = output_root / "embeddings" / split
    emb_dir.mkdir(parents=True, exist_ok=True)

    records = []

    progress = tqdm(dataloader, desc=f"{split} ({mode})")

    if mode == "baseline":
        for batch_idx, (patches, labels, indices) in enumerate(progress):
            embeddings = forward_baseline(
                model, patches, dataloader.batch_size, device, use_amp
            )

            for i, (emb, lbl, ds_idx) in enumerate(zip(embeddings, labels, indices)):
                patch_name = dataloader.dataset.get_patch_name(ds_idx)
                emb_filename = f"{split}_{patch_name}_{ds_idx}.pt"
                torch.save(emb.float(), emb_dir / emb_filename)
                records.append({
                    "embeddings": str(emb_dir.relative_to(output_root) / emb_filename),
                    "target": lbl,
                    "split": split,
                    "wsi_id": dataloader.dataset.get_wsi_id(ds_idx),
                })

    # Context mode
    else:
        for batch_idx, (regions, target_indices, labels, indices) in enumerate(progress):
            embeddings = forward_context(
                model, regions, target_indices,
                patch_batch_size, device, use_amp
            )

            for i, (emb, lbl, ds_idx) in enumerate(zip(embeddings, labels, indices)):
                patch_name = dataloader.dataset.get_patch_name(ds_idx)
                emb_filename = f"{split}_{patch_name}_{ds_idx}.pt"
                torch.save(emb.float(), emb_dir / emb_filename)
                records.append({
                    "embeddings": str(emb_dir.relative_to(output_root) / emb_filename),
                    "target": lbl,
                    "split": split,
                    "wsi_id": dataloader.dataset.get_wsi_id(ds_idx),
                })

    progress.close()
    return records

def forward_baseline(
    model: nn.Module,
    patches: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """
    Run baseline OpenMidnight on patches and return CLS tokens.

    Args:
        patches: [N, 3, 224, 224]
    Returns:
        embeddings: [N, 1536]
    """
    model.eval()
    autocast_ctx = torch.cuda.amp.autocast if use_amp and device.type == "cuda" else nullcontext

    embeddings = []
    with torch.no_grad():
        with autocast_ctx():
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i + batch_size].to(device)
                out = model(batch, is_training=True)
                embeddings.append(out["x_norm_clstoken"].cpu())

    return torch.cat(embeddings, dim=0)


def forward_context(
    model,
    region_patches: torch.Tensor,
    target_indices: List[int],
    patch_batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """
    Run context model on regions and return context-enriched embeddings for target positions.

    Args:
        region_patches: [B, 256, 3, 224, 224]
        target_indices: [B] - flat index of target patch in each region
        patch_batch_size: batch size for OpenMidnight forward
    Returns:
        embeddings: [B, embed_dim]
    """
    model.eval()
    autocast_ctx = torch.cuda.amp.autocast if use_amp and device.type == "cuda" else nullcontext

    B, N, C, H, W = region_patches.shape
    patches_flat = region_patches.view(B * N, C, H, W).to(device)

    # Extract patch embeddings from OpenMidnight
    patch_embeddings = []
    with torch.no_grad():
        with autocast_ctx():
            for i in range(0, B * N, patch_batch_size):
                chunk = patches_flat[i:i + patch_batch_size]
                out = model.patch_vit(chunk, is_training=True)
                patch_embeddings.append(out["x_norm_clstoken"])

    patch_embeddings = torch.cat(patch_embeddings, dim=0).view(B, N, -1)  # [B, 256, 1536]

    # Run through context ViT
    with torch.no_grad():
        with autocast_ctx():
            context_out = model.context_vit(patch_embeddings, mask=None)
            context_tokens = context_out["patch_tokens"]  # [B, 256, embed_dim]

    # Extract embeddings for target positions
    embeddings = []
    for b, tgt_idx in enumerate(target_indices):
        embeddings.append(context_tokens[b, tgt_idx].cpu())

    return torch.stack(embeddings, dim=0)


def write_manifest(manifest_path: Path, records: List[Dict]) -> None:
    """Write manifest CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["embeddings", "target", "split", "wsi_id"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_eva_fit(
    eval_config: Path,
    output_root: Path,
    model_name: str,
    embed_dim: int,
    n_runs: int,
) -> None:
    """Run eva fit using the generated embeddings."""
    env = os.environ.copy()
    env["EMBEDDINGS_ROOT"] = str(output_root)
    env["MODEL_NAME"] = model_name
    env["IN_FEATURES"] = str(embed_dim)
    env["N_RUNS"] = str(n_runs)
    env["PYTHONPATH"] = f"{_openmidnight_root}:{env.get('PYTHONPATH', '')}"

    cmd = ["eva", "fit", "--config", str(eval_config)]
    print(f"\nRunning eva fit: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=str(_openmidnight_root), check=True)


def main():
    parser = argparse.ArgumentParser(description="BRACS context evaluation - fair comparison")

    # Mode selection
    parser.add_argument("--mode", required=True, choices=["baseline", "context"],
                        help="baseline = OpenMidnight CLS only; context = context-adapted embedding")
    parser.add_argument("--aws-endpoint", required=True,
                        help="AWS S3 cloudfare endpoint for downstream datasets")
    parser.add_argument("--aws-profile", default=None,
                        help="AWS profile for accessing S3")

    # Model paths
    parser.add_argument("--baseline-checkpoint",
                        default="/home/paul/OpenMidnight/checkpoints/teacher_epoch250000.pth",
                        help="Path to OpenMidnight checkpoint (for baseline mode)")
    parser.add_argument("--checkpoint",
                        help="Path to context adapter checkpoint (for context mode)")
    parser.add_argument("--config",
                        help="Path to context adapter config YAML (for context mode)")

    # Data paths
    parser.add_argument("--aws-bucket", default="path-datasets",
                        help="AWS S3 cloudfare bucket for downstream datasets")
    parser.add_argument("--aws-bracs-root", default="/bracs",
                        help="Path to BRACS data (BRACS_WSI, BRACS_RoI, patch_metadata.csv) in the bucket")
    parser.add_argument("--output-root", required=True,
                        help="Output directory for embeddings")

    # Processing options
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per split (for debugging)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding generation")
    parser.add_argument("--patch-batch-size", type=int, default=64,
                        help="Batch size for OpenMidnight forward (context mode)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")

    # Evaluation options
    parser.add_argument("--eval-config",
                        default=str(_openmidnight_root / "eval_configs/bracs_embeddings_only.yaml"))
    parser.add_argument("--model-name", default="bracs_eval",
                        help="Model name for eva logging")
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--skip-eva-fit", action="store_true")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "context":
        if not args.checkpoint or not args.config:
            parser.error("--checkpoint and --config required for context mode")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Load model
    if args.mode == "baseline":
        model = load_openmidnight(args.baseline_checkpoint, device)
        embed_dim = 1536
    else:
        model, cfg = load_context_model(args.config, args.checkpoint, device)
        embed_dim = int(cfg.context_vit.embed_dim)

    print(f"Embedding dimension: {embed_dim}")

    # Generate embeddings for each split
    output_root = Path(args.output_root)
    all_records = []

    for split in args.splits:
        print(f"\nProcessing {split} split...")

        dataset = BRACSContextDataset(
            split=split,
            mode=args.mode,
            aws_endpoint=args.aws_endpoint,
            aws_bracs_root=args.aws_bracs_root,
            aws_bucket=args.aws_bucket,
            aws_profile=args.aws_profile if args.aws_profile != "default" else None,
            max_samples=args.max_samples,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        records = generate_embeddings(
            dataloader=dataloader,
            output_root=output_root,
            model=model,
            device=device,
            patch_batch_size=args.patch_batch_size,
            use_amp=args.amp,
        )

        all_records.extend(records)
        dataset.close()

        print(f"  Generated {len(records)} embeddings")

    # Write manifest
    manifest_path = output_root / "manifest.csv"
    write_manifest(manifest_path, all_records)
    print(f"\nWrote manifest to {manifest_path} ({len(all_records)} total records)")

    # Stage for eva
    target_root = output_root / args.model_name / "bracs"
    target_root.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy(manifest_path, target_root / "manifest.csv")

    emb_src = output_root / "embeddings"
    emb_dst = target_root / "embeddings"
    if emb_dst.exists() or emb_dst.is_symlink():
        if emb_dst.is_symlink():
            emb_dst.unlink()
        else:
            shutil.rmtree(emb_dst)
    os.symlink(emb_src, emb_dst)

    # Run eva fit
    if not args.skip_eva_fit:
        run_eva_fit(
            eval_config=Path(args.eval_config),
            output_root=output_root,
            model_name=args.model_name,
            embed_dim=embed_dim,
            n_runs=args.n_runs,
        )


if __name__ == "__main__":
    main()
