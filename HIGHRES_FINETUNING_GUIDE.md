# High-Resolution Post-Training for OpenMidnight — Handoff Guide

> **Handoff context**: This document summarizes everything tried so far on the high-resolution post-training task. The current state is: we've reproduced *some* of the segmentation gains the Midnight paper reports, but classification scores are below paper numbers and we don't fully understand why. Treat all "interpretations" below as hypotheses, not gold truth.

---

## 1. Goal

Reproduce the Midnight paper's Section 2 high-resolution post-training: starting from a Phase 1 OpenMidnight checkpoint (224px, trained on TCGA), continue training at higher resolution so the ViT sees more tokens per tile and learns finer detail. The paper reports significant segmentation improvements (+CoNSeP, +MoNuSAC) at the cost of small classification regressions on coarse-magnification tasks.

Implementation lives in PR [MedARC-AI/OpenMidnight#68](https://github.com/MedARC-AI/OpenMidnight/pull/68).

---

## 2. Method (4 lines)

The pretrained model sees 224px tiles → 256 ViT tokens. For high-res finetuning, we read the **same physical tissue region** but at 2x resolution (448px tiles, halved µm/px). DINO crops at 392px give the ViT **784 tokens** instead of 256. Train for ~120k optimizer steps from the Phase 1 checkpoint with reduced LR (1e-4); segmentation should improve.

---

## 3. Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Tile size | **448px** (not 512) | OpenMidnight Phase 1 uses 224px. 2x = 448. Preserves physical-area invariant. Midnight paper uses 256→512 (also 2x). |
| Crop sizes | 392px global, 168px local | Paper's exact values. 392/14 = 28 → 784 tokens. |
| Magnifications | [1.0, 0.5, 0.25, 0.125] µm/px | Paper's targets. Synthesized from SVS levels via read+resize. |
| Skip rule | Skip target if native_mpp > 2× target_mpp | Avoids excessive upscaling. 0.125 µm/px is skipped for 20x slides. |
| Checkpoint loading | Backbone + DINO/iBOT heads | Preserves all Phase 1 training. (PR #66 only loads backbone — we believe that's a bug.) |
| pos_embed | Interpolated 16×16 → 28×28 at load time | The ViT also has dynamic interpolation in its forward pass, so this is technically redundant but harmless. |
| LR scaling | Includes grad accumulation in effective batch | `effective_batch = batch/gpu × gpus × accum_steps`, then sqrt-scaled vs 1024. |
| FSDP no_sync | Applied during accumulation micro-steps | Avoids redundant all-reduce on non-final micro-steps (added after PR review). |

---

## 4. How tile extraction works (the non-obvious part)

SVS pyramid levels are 4x apart (e.g., 0.25, 1.0, 4.0 µm/px), so there's no native level at every target µm/px. We **synthesize** the desired magnification by reading the right number of pixels from the closest higher-res level and resizing.

Formula: `read_size = 448 × target_mpp / level_mpp`

Concrete example for a 40x slide (native 0.25 µm/px at level 0):

```
Target 1.0 µm/px → physical area = 448 µm
  Level 0 is at 0.25 µm/px → need 448/0.25 = 1792 pixels
  Read 1792px at level 0 → DOWNSAMPLE to 448px

Target 0.5 µm/px → physical area = 224 µm
  Read 224/0.25 = 896px at level 0 → DOWNSAMPLE to 448px

Target 0.25 µm/px → physical area = 112 µm
  Read 112/0.25 = 448px at level 0 → use DIRECTLY

Target 0.125 µm/px → physical area = 56 µm
  Read 56/0.25 = 224px at level 0 → UPSCALE to 448px
  SKIPPED: requires >2x upscale from native
```

The sample list stores `path x y level read_size` (5 fields). At training time, `SlideDataset.__getitem__` reads `read_size` pixels and resizes to `patch_size_pixels` (448).

Physical-area invariant: `448 × target_mpp = 224 × (2 × target_mpp)` — matches Phase 1 at each magnification.

---

## 5. Files in PR #68

- `prepatching_scripts/create_sample_dataset_txt_highres.py` — µm/px-aware sample list generator with multiprocessing (`--workers N`)
- `dinov2/data/datasets/slide_dataset.py` — parses `read_size` field, resizes to `patch_size_pixels`
- `dinov2/data/loaders.py` — passes `patch_size_pixels` through dataset string
- `dinov2/train/train.py` — `_load_from_teacher_checkpoint()` (backbone + heads + pos_embed interp), gradient accumulation loop with `no_sync()`, eval transform uses `cfg.crops.global_crops_size`
- `dinov2/train/ssl_meta_arch.py` — `loss_scale` parameter for gradient accumulation
- `dinov2/utils/config.py` — LR scaling factors in accumulation steps
- `dinov2/configs/train/vitg14_reg4_highres.yaml` — Phase 2 config
- `run_highres_finetune.sh` — launch script

---

## 6. Eva eval pipeline modifications

We did make **one** modification to eva-probe to support our setup:

**File**: `eva-probe/run_evals.sh`

```diff
-EVAL_CONFIG_DIR="${REPO_ROOT}/eval_configs"
+EVAL_CONFIG_DIR="${EVAL_CONFIG_DIR:-${REPO_ROOT}/eval_configs}"
```

This lets us override the eval config dir via env var (used for scaling-law experiments on a different branch — not strictly needed for high-res evals).

For high-res evals, we control behavior **only** through env vars (no code changes inside eva-probe):

- `RESIZE_DIM=392` for Phase 2 checkpoints (392×392 input), `RESIZE_DIM=224` for Phase 1
- `PATIENCE=999999` to disable early stopping (matches the paper's eval protocol)
- `CHECKPOINT_PATH=<path>` set by `run_evals.sh`

**Note**: The eval configs use `ResizeAndCrop` which does `Resize(shorter_edge→size)` then `CenterCrop(size)`. The Midnight paper says they "resized to 392×392" which could be a direct stretch instead. For non-square inputs like BreakHis (700×460), this could differ slightly, but it's standard CV practice.

---

## 7. How to run

```bash
# 1. Generate sample list (full 25M list already exists at /data/rdatchane/sample_dataset_highres_25M.txt)
uv run python3 prepatching_scripts/create_sample_dataset_txt_highres.py \
  --target_patches 25000000 --output sample_dataset_highres_25M.txt --workers 20

# 2. Submit training (8 GPUs, 1 node)
sbatch run_highres_noaccum.sbatch  # latest config, no grad accum

# 3. Monitor
tail -f slurms/<job_id>.out

# 4. Submit eval (1 GPU, runs all 12 tasks sequentially)
sbatch run_eval_phase2_avg_25M_noes.sbatch
```

---

## 8. All experiments run

### Training runs

| Run | Starting checkpoint | Sample list | Batch/GPU | Accum | Real optimizer steps | Notes |
|---|---|---|---|---|---|---|
| `phase2_averaged_2M` | Averaged (87.5k-137.5k) | 2M | 6 | 4 | 30k | First successful run |
| `phase2_nonavg_2M` | Non-averaged (137.5k) | 2M | 6 | 4 | 30k | Sister run |
| `phase2_averaged_25M` | Averaged | 25M_v2 | 6 | 4 | 30k | Same dynamics as 2M |
| `phase2_nonavg_25M` | Non-averaged | 25M_v2 | 6 | 4 | 30k | Sister run |
| `dinov2base_25M` | Meta DINOv2 (ImageNet, no pathology) | 25M_v2 | 6 | 4 | 30k | Sanity check — confirms Phase 1 matters |
| `phase1_continuation_sanity` | Averaged via our `_load_from_teacher_checkpoint()` at 224px | original sample_dataset_30 | 48 | 1 | ran briefly | Sanity test for checkpoint-loading bug. First attempt crashed (warmup_epochs=10 > epochs=5). Re-submitted with epochs=15. |
| `noaccum_averaged` | Averaged | 25M_v2 | 12 | 1 | 120k | **Crashed at start** — averaged checkpoint path was deleted from cluster |
| `noaccum_omckpt` | `/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth` | 25M_v2 | 12 | 1 | 120k | Was at 95k/120k last checked. **Need to verify final state.** |

### Eva benchmark results

Results are stored as TSV files in `eval_results/<run_name>/fast_eval_<timestamp>/summary.tsv`. Aggregated CSV at `eval_results_summary.csv`.

Best eval row so far: `phase2_averaged_20k_25M_noes` (no early stopping):

| Task | Type | Phase 1 (avg) | Phase 2 (ours, 25M, no-ES) | Midnight-92k/392 |
|---|---|---|---|---|
| BACH | Classif. | 0.902 | **0.906** | 0.904 |
| BRACS | Classif. | 0.643 | 0.628 | **0.646** |
| BreakHis | Classif. | **0.886** | 0.756 | 0.802 |
| CRC | Classif. | 0.965 | **0.965** | 0.966 |
| Gleason | Classif. | **0.811** | 0.776 | 0.807 |
| MHIST | Classif. | **0.813** | 0.803 | 0.828 |
| PCam | Classif. | 0.924 | 0.925 | **0.951** |
| **CoNSeP** | **Segm.** | 0.633 | **0.661** | 0.662 |
| **MoNuSAC** | **Segm.** | 0.646 | **0.699** | 0.708 |
| Cam16 | Slide | 0.842 | 0.844 | **0.868** |
| Panda | Slide | **0.646** | 0.607 | 0.651 |

Highlights:
- Beating Midnight on BACH (+0.002)
- Tying Midnight on CRC, CoNSeP
- Clear segmentation gain vs Phase 1 (+3pp CoNSeP, +5pp MoNuSAC)
- Underperforming on most classification tasks compared to Phase 1, especially BreakHis

---

## 9. Concerns and points of doubt (interpretations, not facts)

### Implementation concerns

1. **Initial loss is suspiciously high (~14 vs Phase 1 final ~9.6).**
   - Our explanation: 3x more tokens (256→784) means iBOT loss covers more masked patches, plus interpolated pos_embed needs to adapt.
   - But a 47% jump feels large. Even after 30k steps loss only drops to ~11.2, never recovering to Phase 1's 9.6.
   - We ran a partial sanity check (continue Phase 1 at 224px from teacher checkpoint via our loader) — this was meant to isolate "is loading buggy" from "is the resolution change the cause", but the run was short and didn't fully resolve the question.
   - **Recommended next step**: re-run that sanity check carefully and confirm initial loss at 224px matches ~9.6.

2. **The `_load_from_teacher_checkpoint()` only loads model weights — not optimizer state, sinkhorn centers, or teacher EMA momentum schedule.**
   - Sinkhorn-Knopp centering is computed per-batch (stateless) — should be fine.
   - Optimizer state restart (no Adam momentum/variance) might cause unstable early training.
   - Teacher EMA momentum schedule restarts at base value (0.994) — Phase 1 finished at a different value.
   - **Concern**: paper might have continued from full training state. We didn't.

3. **Gradient accumulation step counting bug (now fixed for new runs).**
   - In old runs (2M / 25M_v2 with `accum=4`): `early_stop_iter = epochs * OFFICIAL_EPOCH_LENGTH = 96 * 1250 = 120000`. But `iteration` counter increments only every 4 micro-steps. So: 120k data loads = 30k optimizer steps.
   - **What we thought**: 120k steps. **What actually happened**: 30k steps. We trained 4x less than intended.
   - For new runs (`noaccum_*`): `accum=1`, so 1:1 between data loads and optimizer steps. ETA at 95k/120k was correctly 5h.
   - **All eval results above use only 30k-step models.**

4. **Effective batch is much smaller than the paper's.**
   - Paper: 48 GPUs × 6/GPU × 4 accum = **1152 effective batch**.
   - Old runs: 8 × 6 × 4 = **192**. New noaccum: 8 × 12 × 1 = **96**.
   - LR scaling (`base_lr * sqrt(eff_batch/1024)`) handles this in theory, but small batches still mean noisier gradients per step.
   - **Concern**: with 8 GPUs we cannot match the paper's effective batch unless we crank `accum_steps` up significantly.

5. **Warmup duration may be too short.**
   - Old runs: 2 epochs (~2500 micro-steps, ~0.5% of intended 120k optimizer steps).
   - New noaccum runs: 5 epochs (~6250 steps, ~5% of training). DINOv2 standard is 5-10%.

6. **Checkpoint we use as "Phase 1" may itself differ from what the paper used.**
   - Paper trained Midnight-12k on **only TCGA** for ~250k iterations (their "Midnight-12k" model).
   - We use `/data/ratna/retrain/eval/averaged_87500_to_137500/teacher_checkpoint.pth` — this is a **Polyak-averaged** model across checkpoints from 87.5k to 137.5k of an OpenMidnight Phase 1 run.
   - **Concern**: averaging across checkpoints might smooth training noise but also change the loss landscape relative to a single non-averaged checkpoint. We did test both (`averaged` vs `nonavg`) — averaged was slightly better.

### Methodology / evaluation concerns

7. **PCam_10 fails on every eval run.** Likely a data download / config issue in `eva-probe`, not model-related. Worth investigating because PCam_10 is one of the few tasks the paper reports specifically benefiting from high-res post-training.

8. **BACH fails on the non-averaged Phase 1 eval and some Phase 2 evals.** Same likely cause as PCam_10.

9. **eva uses `Resize(shorter_edge)` + `CenterCrop`**, not direct `Resize(H, W)`. For non-square images (BreakHis 700×460), this drops some pixels at the edges. Paper says "resized to 392×392" — ambiguous whether they stretch or crop. Probably not the dominant factor in our gaps but worth verifying.

10. **Sample list might still be sub-optimal.**
    - The 25M_v2 sample list targets [1.0, 0.5, 0.25] µm/px (0.125 skipped for almost all slides). Paper uses [1.0, 0.5, 0.25, 0.125] but most TCGA scans are 40x or 20x — they probably also skip 0.125 in practice.
    - Distribution across magnifications is not exactly uniform — depends on slide-by-slide tissue-check pass rate.

### Things we haven't tried

- **Lower base LR** (e.g., 5e-5 instead of 1e-4) — finetuning often needs less LR than initial training.
- **Layer-wise learning rate decay** (LLRD) — DINOv2 standard for finetuning. Currently `layerwise_decay: 1.0` (off).
- **Loading optimizer state** from Phase 1 (would need an FSDP checkpoint, not just the teacher checkpoint).
- **Resuming the EMA teacher momentum schedule** from where Phase 1 left off.
- **A direct comparison run at 224px** (`global_crops_size=224`, batch=48, no accum) — would isolate "checkpoint loading is correct" from "resolution change works".
- **An eval at 392px without any high-res finetuning** — just feed the Phase 1 checkpoint 392px tiles. Would tell us how much the training itself helps vs just feeding bigger images.

---

## 10. Cluster info

- **TCGA SVS files**: `/block/TCGA/` (11,438 files)
- **Eva data**: `/block/eva-data/{bach,breakhis,patch_camelyon,...}`
- **Phase 1 checkpoints — averaged**: was at `/data/ratna/retrain/eval/averaged_87500_to_137500/teacher_checkpoint.pth`. **This path no longer exists** as of our last check. Variants exist at `/data/ratna/aug/<exp>/eval/averaged_87500_to_137500/teacher_checkpoint.pth` but these are augmentation-ablation runs, not the original. Ratna needs to clarify where the original moved.
- **Phase 1 checkpoint — non-averaged 137.5k**: `/data/ratna/retrain/eval/training_137500/teacher_checkpoint.pth` (also gone).
- **Other Phase 1 checkpoints**:
  - `/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth` ← **what `noaccum_omckpt` uses**
  - `/data/OpenMidnight_ckpts/OM_replication_interpolationfix/eval/training_45000/teacher_checkpoint.pth`
  - `/data/OpenMidnight_ckpts/halfTCGA/training_250000/teacher_checkpoint.pth`
  - `/data/OpenMidnight_ckpts/10percentTCGA/eval/training_120000/teacher_checkpoint.pth`
- **Sample lists**:
  - `/data/rdatchane/sample_dataset_highres_25M.txt` ← **full 25M, copied from `sample_dataset_highres_25M_v2.txt`**
  - `/admin/home/rdatchane/OpenMidnight/sample_dataset_highres_25M_v2.txt` (same content)
  - `sample_dataset_highres_2M.txt` (older, smaller)
  - `sample_dataset_highres_1M.txt` (oldest)
  - **Note**: `sample_dataset_highres_25M.txt` (without `_v2` suffix) in the repo root is incomplete (~5.3M patches — generation was interrupted). Don't use it.
- **W&B project**: `tcga-finetuning`
  - Run ID for `noaccum_omckpt`: `s9fcc22r`
- **Training output dirs**: `output_vitg14_highres*` and `output_vitg14_highres_noaccum*`. Note: only teacher checkpoints are saved (every `eval_period_iterations=5000`), and the `PeriodicCheckpointer` keeps only the most recent FSDP checkpoint.

---

## 11. Suggested debug priorities

If I were picking up this task fresh, the order I'd attack it:

1. **Verify checkpoint loading is bit-exact**: write a script that loads the Phase 1 checkpoint into a fresh model, computes embeddings on a fixed batch, and compares to the same embeddings from the original `do_test()` saved checkpoint. They should be identical (modulo pos_embed if interpolated). This rules out load bugs definitively.

2. **Run the 224px-continuation sanity check to completion** and confirm initial loss is ~9.6, not higher. If it's higher, we have a bug in our loader. If it's ~9.6, the high-loss issue is purely from the resolution change.

3. **Run the `noaccum_omckpt` 120k-step model through full eva benchmark** (with `RESIZE_DIM=392 PATIENCE=999999`). This is the first run where we actually trained for the paper's intended number of optimizer steps. If results still plateau early, the issue is not undertraining.

4. **Investigate why classification tasks regress so much** vs Phase 1. The paper reports degradation on Camelyon16/HEST but not BreakHis/Gleason/MHIST as severely as we see. Could be:
   - LR too high
   - Warmup too short
   - Effective batch too small
   - Something specific to our smaller training dataset (12k slides vs paper's 92k)

5. **Test eval at 392px without any high-res finetuning.** Just take the Phase 1 checkpoint, evaluate at 392px (relying on dynamic pos_embed interpolation in the ViT). If that already gets most of the segmentation gains, the finetuning step might not be doing as much as we think.
