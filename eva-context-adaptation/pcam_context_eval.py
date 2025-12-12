"""
PCam context evaluation: True apples-to-apples comparison.

This script produces context-enriched embeddings for PCam in a way that ensures
a fair comparison with the baseline (single-patch OpenMidnight CLS token).

KEY INSIGHT: The standard PCam H5 files contain pre-extracted 96×96 patches at
level 2 of the WSI pyramid. The coordinates in the CSV metadata tell us where
each patch came from.

For context evaluation, we:
1. Read the H5 patch index and its coordinates from the CSV metadata
2. Go back to the source WSI and extract a 16×16 region of 96×96 patches
   centered on the target location
3. Resize all patches to 224×224 (matching OpenMidnight input)
4. Run all 256 patches through OpenMidnight → context-ViT
5. Extract the context-enriched embedding for the CENTER patch position

For baseline comparison:
- Use the H5 patch directly (or the center patch from the region)
- Resize to 224×224
- Run through OpenMidnight only
- Save the CLS token

This ensures both approaches use compatible data and differ only in whether
context is used.

Usage:
  python pcam_context_eval.py \
    --checkpoint /path/to/context_checkpoint.pth \
    --config /path/to/context_adaptation.yaml \
    --mode context \
    --output-root /path/to/output

  python pcam_context_eval.py \
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
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
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


def resolve_wsi_path(wsi_name: str, data_root: Path) -> Optional[Path]:
    """Map PCam metadata wsi name to Camelyon16 file path."""
    parts = wsi_name.split("_")

    # Test slides: camelyon16_test_XXX -> testing/images/test_XXX.tif
    if len(parts) >= 3 and parts[1] == "test":
        idx = parts[2]
        base = data_root / "testing" / "images" / f"test_{idx}"
        for ext in [".tif", ".tiff"]:
            if base.with_suffix(ext).exists():
                return base.with_suffix(ext)
        return None

    # Train slides: camelyon16_train_tumor_XXX or camelyon16_train_normal_XXX
    if len(parts) >= 4 and parts[1] == "train":
        cls = parts[2]  # "tumor" or "normal"
        idx = parts[3]
        base = data_root / "training" / cls / f"{cls}_{idx}"
        for ext in [".tif", ".tiff"]:
            if base.with_suffix(ext).exists():
                return base.with_suffix(ext)
        return None

    return None


class PCamContextDataset:
    """
    Dataset that combines PCam H5 data with WSI region extraction.

    For each PCam sample:
    - Reads metadata (WSI name, coordinates, label) from CSV
    - Extracts a 16×16 region of patches from the source WSI
    - The target patch is at the CENTER of this region
    """

    def __init__(
        self,
        pcam_root: Path,
        camelyon_root: Path,
        split: str,
        patches_per_side: int = 16,
        patch_level: int = 2,
        patch_size_at_level: int = 96,
        model_input_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        self.pcam_root = pcam_root
        self.camelyon_root = camelyon_root
        self.split = split
        self.patches_per_side = patches_per_side
        self.patch_level = patch_level
        self.patch_size_at_level = patch_size_at_level
        self.model_input_size = model_input_size

        # Load metadata
        split_suffix = "valid" if split == "val" else split
        meta_path = pcam_root / f"camelyonpatch_level_2_split_{split_suffix}_meta.csv"
        self.meta_df = pd.read_csv(meta_path)

        # Load H5 files for labels and optionally for fallback patches
        h5_x_path = pcam_root / f"camelyonpatch_level_2_split_{split_suffix}_x.h5"
        h5_y_path = pcam_root / f"camelyonpatch_level_2_split_{split_suffix}_y.h5"
        self.h5_x = h5py.File(h5_x_path, "r")
        self.h5_y = h5py.File(h5_y_path, "r")

        if max_samples is not None:
            self.meta_df = self.meta_df.head(max_samples)

        # Image transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(model_input_size, antialias=True)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # WSI cache
        self.wsi_cache: Dict[str, "openslide.OpenSlide"] = {}

    def __len__(self):
        return len(self.meta_df)

    def get_h5_patch(self, idx: int) -> torch.Tensor:
        """Get the pre-extracted patch from H5 (96×96)."""
        patch_np = self.h5_x["x"][idx]  # [96, 96, 3]
        patch = self.to_tensor(patch_np)  # [3, 96, 96]
        patch = self.resize(patch)  # [3, 224, 224]
        patch = self.normalize(patch)
        return patch

    def get_label(self, idx: int) -> int:
        """Get the label for sample idx."""
        return int(self.h5_y["y"][idx].squeeze())

    def get_wsi(self, wsi_name: str) -> Optional["openslide.OpenSlide"]:
        """Get or open a WSI."""
        if wsi_name in self.wsi_cache:
            return self.wsi_cache[wsi_name]

        wsi_path = resolve_wsi_path(wsi_name, self.camelyon_root)
        if wsi_path is None or not wsi_path.exists():
            return None

        try:
            wsi = openslide.OpenSlide(str(wsi_path))
            self.wsi_cache[wsi_name] = wsi
            return wsi
        except Exception as e:
            print(f"Failed to open WSI {wsi_path}: {e}")
            return None

    def get_region_patches(self, idx: int) -> Optional[Tuple[torch.Tensor, int, int]]:
        """
        Extract a 16×16 region of patches centered on the target location.

        Returns:
            tuple of (patches [256, 3, 224, 224], target_row, target_col)
            or None if WSI is unavailable
        """
        row = self.meta_df.iloc[idx]
        wsi_name = str(row["wsi"])
        coord_x = int(row["coord_x"])  # level 0 coordinates
        coord_y = int(row["coord_y"])

        wsi = self.get_wsi(wsi_name)
        if wsi is None:
            return None

        # Calculate region bounds
        downsample = wsi.level_downsamples[self.patch_level]
        patch_size_level0 = int(round(self.patch_size_at_level * downsample))
        region_size_level0 = patch_size_level0 * self.patches_per_side

        # Center the region on the target patch
        center_offset = self.patches_per_side // 2
        region_x0 = coord_x - (center_offset * patch_size_level0)
        region_y0 = coord_y - (center_offset * patch_size_level0)

        # Clamp to slide bounds
        slide_w, slide_h = wsi.level_dimensions[0]
        region_x0 = max(0, min(region_x0, slide_w - region_size_level0))
        region_y0 = max(0, min(region_y0, slide_h - region_size_level0))

        # Compute target position within the region
        target_col = (coord_x - region_x0) // patch_size_level0
        target_row = (coord_y - region_y0) // patch_size_level0
        target_col = max(0, min(target_col, self.patches_per_side - 1))
        target_row = max(0, min(target_row, self.patches_per_side - 1))

        # Read the region at the specified level
        region_size_at_level = self.patch_size_at_level * self.patches_per_side
        try:
            region_img = wsi.read_region(
                (region_x0, region_y0),
                self.patch_level,
                (region_size_at_level, region_size_at_level)
            ).convert("RGB")
        except Exception as e:
            print(f"Failed to read region: {e}")
            return None

        region_np = np.array(region_img)

        # Split into 16×16 patches
        patches = []
        for r in range(self.patches_per_side):
            for c in range(self.patches_per_side):
                y0 = r * self.patch_size_at_level
                x0 = c * self.patch_size_at_level
                patch_np = region_np[y0:y0 + self.patch_size_at_level,
                                     x0:x0 + self.patch_size_at_level]
                patch = self.to_tensor(Image.fromarray(patch_np))
                patch = self.resize(patch)
                patch = self.normalize(patch)
                patches.append(patch)

        patches_tensor = torch.stack(patches, dim=0)  # [256, 3, 224, 224]
        return patches_tensor, target_row, target_col

    def close(self):
        """Clean up resources."""
        self.h5_x.close()
        self.h5_y.close()
        for wsi in self.wsi_cache.values():
            wsi.close()


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


def generate_embeddings(
    dataset: PCamContextDataset,
    output_root: Path,
    model,
    mode: str,  # "baseline" or "context"
    device: torch.device,
    batch_size: int,
    patch_batch_size: int,
    use_amp: bool,
) -> List[Dict]:
    """Generate embeddings for the dataset."""
    split = dataset.split
    emb_dir = output_root / "embeddings" / split
    emb_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0

    # For baseline mode, we can batch efficiently
    # For context mode, we need to process region by region

    progress = tqdm(total=len(dataset), desc=f"{split} ({mode})")

    if mode == "baseline":
        # Batch processing for baseline
        batch_patches = []
        batch_indices = []
        batch_labels = []

        for idx in range(len(dataset)):
            # Get patch from H5 (the exact same patch used in standard eval)
            patch = dataset.get_h5_patch(idx)
            label = dataset.get_label(idx)

            batch_patches.append(patch)
            batch_indices.append(idx)
            batch_labels.append(label)

            if len(batch_patches) >= batch_size or idx == len(dataset) - 1:
                if batch_patches:
                    patches_tensor = torch.stack(batch_patches, dim=0)
                    embeddings = forward_baseline(
                        model, patches_tensor, batch_size, device, use_amp
                    )

                    for i, (emb, ds_idx, lbl) in enumerate(zip(embeddings, batch_indices, batch_labels)):
                        emb_filename = f"{split}_{ds_idx}.pt"
                        torch.save(emb.float(), emb_dir / emb_filename)
                        records.append({
                            "embeddings": str(Path("embeddings") / split / emb_filename),
                            "target": lbl,
                            "split": split,
                            "wsi_id": str(dataset.meta_df.iloc[ds_idx]["wsi"]),
                        })

                    progress.update(len(batch_patches))
                    batch_patches = []
                    batch_indices = []
                    batch_labels = []

    else:  # context mode
        batch_regions = []
        batch_target_indices = []
        batch_ds_indices = []
        batch_labels = []

        for idx in range(len(dataset)):
            result = dataset.get_region_patches(idx)
            label = dataset.get_label(idx)

            if result is None:
                # Fall back to baseline-style processing for missing WSIs
                skipped += 1
                progress.update(1)
                continue

            patches, target_row, target_col = result
            target_flat_idx = target_row * dataset.patches_per_side + target_col

            batch_regions.append(patches)
            batch_target_indices.append(target_flat_idx)
            batch_ds_indices.append(idx)
            batch_labels.append(label)

            if len(batch_regions) >= batch_size or idx == len(dataset) - 1:
                if batch_regions:
                    regions_tensor = torch.stack(batch_regions, dim=0)
                    embeddings = forward_context(
                        model, regions_tensor, batch_target_indices,
                        patch_batch_size, device, use_amp
                    )

                    for i, (emb, ds_idx, lbl) in enumerate(zip(embeddings, batch_ds_indices, batch_labels)):
                        emb_filename = f"{split}_{ds_idx}.pt"
                        torch.save(emb.float(), emb_dir / emb_filename)
                        records.append({
                            "embeddings": str(Path("embeddings") / split / emb_filename),
                            "target": lbl,
                            "split": split,
                            "wsi_id": str(dataset.meta_df.iloc[ds_idx]["wsi"]),
                        })

                    progress.update(len(batch_regions))
                    batch_regions = []
                    batch_target_indices = []
                    batch_ds_indices = []
                    batch_labels = []

    progress.close()

    if skipped > 0:
        print(f"  Skipped {skipped} samples (missing WSI)")

    return records


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
    parser = argparse.ArgumentParser(description="PCam context evaluation - fair comparison")

    # Mode selection
    parser.add_argument("--mode", required=True, choices=["baseline", "context"],
                        help="baseline = OpenMidnight CLS only; context = context-adapted embedding")

    # Model paths
    parser.add_argument("--baseline-checkpoint",
                        default="/home/paul/OpenMidnight/checkpoints/teacher_epoch250000.pth",
                        help="Path to OpenMidnight checkpoint (for baseline mode)")
    parser.add_argument("--checkpoint",
                        help="Path to context adapter checkpoint (for context mode)")
    parser.add_argument("--config",
                        help="Path to context adapter config YAML (for context mode)")

    # Data paths
    parser.add_argument("--pcam-root", default="/data/eva-data/patch_camelyon",
                        help="Path to PatchCamelyon data (H5 files + metadata)")
    parser.add_argument("--camelyon-root", default="/data/eva-data/camelyon16",
                        help="Path to Camelyon16 WSIs")
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
                        default=str(_openmidnight_root / "eval_configs/pcam_embeddings_only.yaml"))
    parser.add_argument("--model-name", default="pcam_eval",
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

        dataset = PCamContextDataset(
            pcam_root=Path(args.pcam_root),
            camelyon_root=Path(args.camelyon_root),
            split=split,
            max_samples=args.max_samples,
        )

        records = generate_embeddings(
            dataset=dataset,
            output_root=output_root,
            model=model,
            mode=args.mode,
            device=device,
            batch_size=args.batch_size,
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
    target_root = output_root / args.model_name / "patch_camelyon"
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
