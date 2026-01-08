# EVA pathology datasets: effective MPP for linear probe patches

This file summarizes the *effective* microns-per-pixel (MPP) seen by the model inputs during EVA evaluation/linear probing. The numbers are derived from local configs and data in:
- /home/paul/OldMidnight/eval_configs
- /home/paul/OldMidnight/eva-probe
- /data/eva-data

## Method (what counts as "patch MPP")

- Patch-level datasets using `ResizeAndCrop(size=224)`:
  - Effective MPP is computed as `base_mpp * (shorter_side / 224)`.
  - If the original patch is already 224x224, effective MPP == base MPP.
- WSI patching datasets (Camelyon16, PANDASmall, CoNSeP):
  - EVA targets a `target_mpp` and reads from the closest WSI level, scaling the read size to match the desired physical field of view.
  - Effective MPP at model input was computed as `(scaled_width * level_mpp) / 224` using EVA's `get_closest_level` logic and actual slide metadata.
- MoNuSAC train uses `RandomResizedCrop` (default torchvision parameters). I simulated 5 crops per image with a fixed seed to estimate the distribution.

All MPP values are reported as um/px (microns per pixel).

## Summary table (effective MPP at model input)

Notes:
- For datasets with fixed sizes, min = median = mean = max.
- BreakHis has a documented MPP discrepancy; both interpretations are shown.
- MoNuSAC has split-specific behavior (train uses random crops; val/test uses ResizeAndCrop).

| Dataset | Effective MPP (min / median / mean / max) | Evidence / context |
| --- | --- | --- |
| BACH | 2.8800 / 2.8800 / 2.8800 / 2.8800 | Photos are 2048x1536 at 0.42 um/px, ResizeAndCrop to 224. |
| BRACS (RoI) | 0.1417 / 1.5301 / 1.7939 / 12.8717 | Variable-size RoIs at 0.25 um/px, ResizeAndCrop to 224. |
| BreakHis 40X (EVA doc mpp=0.25) | 0.5089 / 0.5134 / 0.5133 / 0.5134 | 700x456/460 images, ResizeAndCrop to 224. |
| BreakHis 40X (optics-derived mpp=0.492) | 1.0016 / 1.0104 / 1.0103 / 1.0104 | Same sizes, different base MPP. |
| CRC-100K | 0.5000 / 0.5000 / 0.5000 / 0.5000 | 224x224 at 0.5 um/px, ResizeAndCrop to 224. |
| Gleason (Arvaniti) | 0.7701 / 0.7701 / 0.7701 / 0.7701 | 750x750 at 0.23 um/px, ResizeAndCrop to 224. |
| MHIST | 2.0000 / 2.0000 / 2.0000 / 2.0000 | 224x224 at 2.0 um/px, ResizeAndCrop to 224. |
| PCam | 0.4166 / 0.4166 / 0.4166 / 0.4166 | 96x96 patches at 10x (0.972 um/px), ResizeAndCrop to 224. |
| PCam (10 shots) | 0.4166 / 0.4166 / 0.4166 / 0.4166 | Same PatchCamelyon images; sampler only reduces count. |
| Camelyon16 (small) | 0.2481 / 0.2496 / 0.2495 / 0.2498 | WSI patching with target_mpp=0.25, computed from slide metadata. |
| Panda (small) | 0.4962 / 0.4964 / 0.4965 / 0.4972 | WSI patching with target_mpp=0.5, computed from slide metadata. |
| CoNSeP | 0.2790 / 0.2790 / 0.2790 / 0.2790 | WSI-like PNGs, overwrite_mpp=0.25, patch=250, ResizeAndCrop to 224. |
| MoNuSAC (train, RandomResizedCrop sim) | 0.0309 / 0.3128 / 0.3860 / 1.5809 | 5 random crops per image, default RRC params. |
| MoNuSAC (val/test, ResizeAndCrop) | 0.0372 / 0.4185 / 0.4999 / 1.9831 | XML MPP + ResizeAndCrop on full images. |
| HEST | NA | HEST repo/data not present locally; cannot compute patch MPP here. |

## Dataset-by-dataset details

### BACH
- Effective MPP stats: min/median/mean/max = 2.8800 (n=400 images).
- Evidence: `/data/eva-data/bach/ICIAR2018_BACH_Challenge/Photos` images are 2048x1536; dataset docs list 0.42 um/px.
- Preprocessing: `eva.vision.datasets.BACH` + `ResizeAndCrop(size=224)` in `eval_configs/bach.yaml`.
- Nuances: EVA uses the Photos subset, not the WSIs.
- Nearest TCGA: TCGA-BRCA.

### BRACS (RoI)
- Effective MPP stats: min 0.1417, median 1.5301, mean 1.7939, max 12.8717 (n=4539 RoIs).
- Evidence: `/data/eva-data/bracs/BRACS_RoI/latest_version` PNG sizes are highly variable; BRACS docs state 0.25 um/px at 40x.
- Preprocessing: `eva.vision.datasets.BRACS` + `ResizeAndCrop(size=224)` in `eval_configs/bracs.yaml`.
- Nuances: RoIs are variable-size crops, so physical context varies widely.
- Nearest TCGA: TCGA-BRCA.

### BreakHis (40X)
- Effective MPP stats (EVA docs base 0.25): min 0.5089, median 0.5134, mean 0.5133, max 0.5134 (n=1995 images).
- Effective MPP stats (optics-derived base 0.492): min 1.0016, median 1.0104, mean 1.0103, max 1.0104.
- Evidence: `/data/eva-data/breakhis/BreaKHis_v1/histology_slides/breast` images are 700x456/460; `docs/datasets/breakhis.md` claims 0.25 um/px, while `README.txt` implies 0.492 um/px from camera optics.
- Preprocessing: `eva.vision.datasets.BreaKHis` defaults to 40X only + `ResizeAndCrop(size=224)` in `eval_configs/breakhist.yaml`.
- Nuances: This MPP discrepancy should be resolved before finalizing paper claims.
- Nearest TCGA: TCGA-BRCA.

### CRC-100K (NCT-CRC-HE-100K / CRC-VAL-HE-7K)
- Effective MPP stats: min/median/mean/max = 0.5000 (224x224 at 0.5 um/px; sample-checked 200 files).
- Evidence: dataset docs (Zenodo) state 224x224 at 0.5 um/px; local files in `/data/eva-data/crc` match.
- Preprocessing: `eva.vision.datasets.CRC` + `ResizeAndCrop(size=224)` in `eval_configs/crc.yaml`.
- Nearest TCGA: TCGA-COAD, TCGA-READ.

### Gleason (Arvaniti)
- Effective MPP stats: min/median/mean/max = 0.7701 (750x750 at 0.23 um/px; sample-checked 200 files).
- Evidence: patches in `/data/eva-data/arvaniti_gleason_patches`; base mpp from Arvaniti paper.
- Preprocessing: `eva.vision.datasets.GleasonArvaniti` + `ResizeAndCrop(size=224)` in `eval_configs/gleason_offline.yaml`.
- Nuances: TMA patches; EVA uses train/val split by TMA ID.
- Nearest TCGA: TCGA-PRAD.

### MHIST
- Effective MPP stats: min/median/mean/max = 2.0000 (224x224 at 2.0 um/px; sample-checked 200 files).
- Evidence: `docs/datasets/mhist.md` lists 5x (2.0 um/px) and 224x224; images in `/data/eva-data/mhist/images`.
- Preprocessing: `eva.vision.datasets.MHIST` + `ResizeAndCrop(size=224)` in `eval_configs/mhist.yaml`.
- Nearest TCGA: TCGA-COAD, TCGA-READ.

### PCam (PatchCamelyon) + PCam (10 shots)
- Effective MPP stats: min/median/mean/max = 0.4166.
- Evidence: H5 patches are 96x96 (`/data/eva-data/patch_camelyon/*.h5`); PCam docs list 0.243 um/px at 40x, downsampled to 10x (0.972 um/px).
- Preprocessing: `eva.vision.datasets.PatchCamelyon` + `ResizeAndCrop(size=224)` in `eval_configs/pcam_10.yaml` (also used for full PCam).
- Nuances: PCam(10 shots) only changes the sampler (`num_samples: 10`), not the patch MPP.
- Nearest TCGA: TCGA-BRCA (lymph node metastasis of breast cancer).

### Camelyon16 (small)
- Effective MPP stats: min 0.2481, median 0.2496, mean 0.2495, max 0.2498 (n=399 slides).
- Evidence: computed from OpenSlide metadata in `/data/eva-data/camelyon16` using EVA's `get_closest_level` patching logic.
- Preprocessing: `eva.vision.datasets.Camelyon16` with `ForegroundGridSampler(max_samples=10000)`, `target_mpp=0.25`, `width=height=224`, and `ResizeAndCrop(size=224)` in `eval_configs/cam16_small.yaml`.
- Nuances: a few slides have nonstandard native MPP, but patching still targets 0.25 um/px.
- Nearest TCGA: TCGA-BRCA.

### Panda (small)
- Effective MPP stats: min 0.4962, median 0.4964, mean 0.4965, max 0.4972 (n=10616 slides).
- Evidence: computed from OpenSlide metadata in `/data/eva-data/panda/prostate-cancer-grade-assessment/train_images` using EVA's patching logic.
- Preprocessing: `eva.vision.datasets.PANDASmall` with `ForegroundGridSampler(max_samples=200)`, `target_mpp=0.5`, `width=height=224`, and `ResizeAndCrop(size=224)` in `eval_configs/panda_small.yaml`.
- Nuances: PANDASmall uses a subset of slides, but MPP distribution is effectively the same as full PANDA.
- Nearest TCGA: TCGA-PRAD.

### CoNSeP
- Effective MPP stats: min/median/mean/max = 0.2790.
- Evidence: EVA uses `overwrite_mpp=0.25` and `target_mpp=0.25` with patch size 250 in `eva-probe/src/eva/vision/data/datasets/segmentation/consep.py`; PNGs are in `/data/eva-data/consep`.
- Preprocessing: `eva.vision.datasets.CoNSeP` + `ResizeAndCrop(size=224)` in `eval_configs/consep.yaml`.
- Nuances: 0.25 um/px is an EVA assumption (not embedded in the PNGs).
- Nearest TCGA: TCGA-COAD, TCGA-READ.

### MoNuSAC
- Effective MPP stats (train, RandomResizedCrop sim): min 0.0309, median 0.3128, mean 0.3860, max 1.5809 (5 crops per image, fixed seed).
- Effective MPP stats (val/test, ResizeAndCrop): min 0.0372, median 0.4185, mean 0.4999, max 1.9831 (n=85 images).
- Evidence: XML `MicronsPerPixel` attributes in `/data/eva-data/monusac` + image sizes.
- Preprocessing: train uses `RandomResizedCrop(size=224)`; val/test uses `ResizeAndCrop(size=224)` in `eval_configs/monusac.yaml`.
- Nuances: multi-organ dataset (TCGA-derived); train MPP distribution changes per epoch due to random crops.
- Nearest TCGA: TCGA-BRCA, TCGA-KIRC/KIRP, TCGA-LIHC, TCGA-PRAD (multi-organ).

### HEST
- HEST data/configs are not present under `/home/paul/OldMidnight` or `/data/eva-data` on this machine.
- To compute MPP, we need the HEST repo and its preprocessed patch metadata (see README instructions in `/home/paul/OldMidnight/README.md`).
- Nearest TCGA: HEST is a multi-dataset benchmark; TCGA mapping depends on the specific HEST dataset/task.

## Extra notes / preprocessing nuances worth remembering

- `ResizeAndCrop` scales the *shorter side* to 224 then center-crops; this shrinks the physical field of view for smaller source patches (e.g., PCam 96x96).
- WSI patching uses `target_mpp` and chooses the closest available WSI level, so the effective MPP is *slightly below* the target due to integer rounding.
- For RandomResizedCrop (MoNuSAC train), the MPP varies substantially; reported stats are based on simulation, not a fixed deterministic transform.
- If you want a machine-readable export of the MPP stats (CSV/JSON) for paper tables/plots, I can generate it.
