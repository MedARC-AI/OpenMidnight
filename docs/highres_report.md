# High-Resolution Post-Training for OpenMidnight

*Internal report — May 2026. Rohit Datchanamourty.*

This documents what we tried, what worked, what didn't, and what to test next on reproducing the high-resolution post-training step from the Midnight paper (Kaiko.ai, §2) on the OpenMidnight Phase 1 checkpoint. Two companion docs stay around as historical record: `HIGHRES_FINETUNING_GUIDE.md` (original handoff) and `HIGHRES_EXPERIMENTS.md` (live tracker).

---

## TL;DR

We did **not** see the paper's claimed ~+1pp average gain. Our best run lands at **+0.5pp average over 12 EVA tasks** vs the appropriate baseline (Phase 1 evaluated at 392 px). Against `Midnight-12k`, the paper's TCGA-only variant which is our right comparison point at matched data scale, we're within ±1pp on 8/12 tasks and beat it on Gleason, CoNSeP, MoNuSAC, BACH, MHIST. Against `Midnight-92k/392`, the paper's flagship with 8× more data, we're typically 1-2pp behind on slide-level tasks.

The most useful single finding from this work is that **roughly half of the paper's reported segmentation gain reproduces just by evaluating Phase 1 at 392 px with dynamic positional-embedding interpolation** — no training needed. CoNSeP goes 0.623 → 0.646 and MoNuSAC 0.649 → 0.680 from changing only the eval resolution. The high-res *training* step then adds another +0.9pp on top. Anyone replicating should measure this baseline first.

The most likely cause of the smaller-than-paper gain: data scale (12k TCGA WSIs vs the paper's Midnight-92k 92k WSIs). The paper's matching 12k variant isn't directly comparable since they don't report the high-res-trained version of it.

---

## 1. What the paper does, what we did

The paper continues training a finished SSL ViT-g14 backbone at 2× pixel density and 0.5× µm/px, preserving physical tissue area per tile but quadrupling tokens per crop (16² → 28² globals, 7² → 12² locals). Trained for 120k optimizer steps at effective batch 1152 on 48 H100s.

Our setup, condensed:

| Parameter | Midnight-92k/392 | Ours (Node 1, best run) |
|---|---|---|
| Starting checkpoint | Midnight-92k @ 224 (92k WSIs) | OpenMidnight Phase 1 @ 224 (12k TCGA WSIs) |
| Tile size (pre-crop) | 512 px | 448 px |
| Magnifications sampled | 1.0/0.5/0.25/0.125 µm/px | same |
| Global / local crop | 392 / 168 | same |
| Tokens per global / local | 28×28 / 12×12 | same |
| Optimizer steps | 120k | 30k |
| GPUs | 48 H100 | 8 H100 (1 node) |
| Per-GPU batch | 6 | 12 |
| Gradient accumulation | ×4 | ×4 (Node 1/2) or ×12 (warmstart_5k, warmstart_short) |
| Effective batch | 1152 | 384 (Node 1/2) or 1152 (warmstart runs) |
| Base LR | 1e-4 (sqrt-scaled) | same |

Tile-sampling pipeline reads `448 × target_mpp / level_mpp` pixels from the closest SVS pyramid level and resizes to 448 px. The skip rule drops 0.125 µm/px for slides whose native MPP is more than 2× the target (i.e. all 20× scans), giving a non-uniform but reproducible distribution across magnifications. The physical-area invariant `448 × target_mpp = 224 × 2 × target_mpp` is verified end-to-end in `highres_verification.ipynb`.

Sample-list generation (`prepatching_scripts/create_sample_dataset_txt_highres.py`):

```python
TARGET_MPPS = [1.0, 0.5, 0.25, 0.125]
TILE_SIZE = 448
MAX_UPSCALE = 2.0

for target_mpp in TARGET_MPPS:
    if native_mpp > MAX_UPSCALE * target_mpp:
        continue  # skip if we'd have to upscale >2x from the finest level

    # Pick the finest pyramid level that's still coarser-or-equal to target.
    target_ds = target_mpp / native_mpp
    best_level = 0
    for l in range(slide.level_count):
        if slide.level_downsamples[l] <= target_ds:
            best_level = l

    # Read `read_size` px from `best_level`, then resize to TILE_SIZE px.
    level_mpp = native_mpp * slide.level_downsamples[best_level]
    read_size = int(round(TILE_SIZE * target_mpp / level_mpp))
    # ...emit sample-list line: "{path} {x} {y} {best_level} {read_size}"
```

Per-tile consumption at training time (`dinov2/data/datasets/slide_dataset.py`):

```python
def __getitem__(self, index):
    path, x, y, level, read_size = self.image_files[index].split(" ")
    slide = OpenSlide(path)
    patch = slide.read_region((int(x), int(y)), level=int(level),
                              size=(int(read_size), int(read_size)))
    img = patch.convert("RGB")
    # Resize the synthesized magnification to TILE_SIZE px before augmentation.
    if int(read_size) != self.patch_size_pixels:
        img = img.resize((self.patch_size_pixels, self.patch_size_pixels),
                         Image.BICUBIC)
    return self.transforms(img, None), index
```

![High-res tile examples at 1.0 / 0.5 / 0.25 / 0.125 µm/px, same tissue region; token-grid overlay (16×16 vs 28×28)](placeholder_tiles_and_tokens.png)

**Schedule warm-start.** The single biggest implementation choice. DINOv2's trainer rebuilds all schedules from iter 0 on every fresh run: teacher temp warms 0.04 → 0.07 over 31% of training; weight decay warms 0.04 → 0.2; teacher momentum 0.994 → 1.0; AdamW state resets. Phase 1 ended near the terminal values of all of these, so a naive reload destabilizes the student for tens of thousands of steps. We override at iter 0 — the load-bearing overrides live in `dinov2/configs/train/vitg14_reg4_highres_warmstart*.yaml`:

```yaml
teacher:
  momentum_teacher: 0.9995          # default 0.994 → 1.0; Phase 1 ended near 1.0
  warmup_teacher_temp: 0.07         # default warms 0.04 → 0.07 over 30 epochs
  teacher_temp: 0.07
  warmup_teacher_temp_epochs: 0     # skip the temp warmup entirely
optim:
  weight_decay: 0.2                 # default warms 0.04 → 0.2; start at terminal
  weight_decay_end: 0.2
  freeze_last_layer_epochs: 0       # student head is already trained
  layerwise_decay: 0.9              # DINOv2 finetuning standard (was 1.0)
```

The LR keeps its 5-epoch warmup because AdamW second moments do start fresh on every run.

**Other implementation bits.** FSDP `no_sync()` during accumulation micro-steps to avoid redundant all-reduces. LR scaled by `sqrt(eff_batch / 1024)` from the configured `base_lr`. The pos_embed gets bicubic-interpolated 16×16 → 28×28 inside `_load_from_teacher_checkpoint` (`dinov2/train/train.py`) before the state-dict load:

```python
pos_embed = backbone_state["pos_embed"]
cls_pos = pos_embed[:, :1]                              # CLS token, kept as-is
patch_pos = pos_embed[:, 1:]                            # (1, 256, dim)
orig = int(patch_pos.shape[1] ** 0.5)                   # 16
target_h, target_w = student_backbone.patch_embed.patches_resolution  # (28, 28)

patch_pos = patch_pos.reshape(1, orig, orig, -1).permute(0, 3, 1, 2)
patch_pos = F.interpolate(patch_pos, size=(target_h, target_w),
                          mode="bicubic", align_corners=False)
patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, -1)
backbone_state["pos_embed"] = torch.cat((cls_pos, patch_pos), dim=1)
```

The ViT's `interpolate_pos_encoding` does the same thing dynamically on every forward pass, so this load-time step is technically redundant in steady state — but loading already-interpolated weights matters during the first few warmup iterations before the model has trained on 28×28 positions.

**Evaluation.** Two protocols measured. Default DINOv2 hub returns CLS only (1536-dim); paper specifies CLS + mean-pooled patch tokens (3072-dim) for downstream linear probes. The wrapper (`eva-probe/.../pathology/openmidnight.py`) toggles via env var:

```python
def openmidnight_dinov2_vitg14_reg(include_patch_tokens=False, **kwargs):
    base = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg",
                          pretrained=True)
    def _forward(self, x):
        feats = self.forward_features(x)
        cls = feats["x_norm_clstoken"]                      # (B, 1536)
        if include_patch_tokens:
            patch_mean = feats["x_norm_patchtokens"].mean(1)  # (B, 1536)
            return torch.cat([cls, patch_mean], dim=-1)       # (B, 3072)
        return cls
    base.forward = _forward.__get__(base, type(base))
    return base
```

The wrapper is wired into each eval config via `ModelFromFunction`, with all knobs exposed as env vars so a single set of configs serves every protocol variant:

```yaml
# eval_configs/bach.yaml (and 9 other classification + slide configs)
backbone:
  class_path: eva.core.models.wrappers.ModelFromFunction
  init_args:
    path: eva.vision.models.networks.backbones.pathology.openmidnight
          .openmidnight_dinov2_vitg14_reg
    arguments:
      include_patch_tokens:  ${oc.env:INCLUDE_PATCH_TOKENS, false}
      interpolate_offset:    ${oc.env:INTERPOLATE_OFFSET, 0.1}
      interpolate_antialias: ${oc.env:INTERPOLATE_ANTIALIAS, false}
    checkpoint_path: ${oc.env:CHECKPOINT_PATH, ...}
```

Segmentation tasks bypass this and use spatial patch features (1536-dim) into `ConvDecoderWithImage`. Input resize switched from 224 → 392 for all high-res evals (via env var `RESIZE_DIM`).

---

## 2. Bugs we found, and what they invalidate

Several latent bugs in train and eval code surfaced over the course of this work. Worth listing because most pre-fix run results — including everything in the original handoff doc's results table — are wrong.

| # | Bug | Location | Effect |
|---|---|---|---|
| 1 | Eval check fires per micro-step | `train.py:1222` | With `accum=N`, in-training `do_test()` runs N× per period. With accum=12, the first warmstart attempt (51408) burned ~108 min on iter-0 evals before any optimizer step. |
| 2 | `log_every` exits at yield count | `helpers.py:136` + `train.py:1209` | Training loop breaks at `(eta_target_iter+1)` *data-loader yields*. With accum=12 that's only 5k opt steps, not 60k. **All warmstart and ablation runs from May 10–11 (51417, 51439, 51440, 51443, 51444) "completed" but trained for 1/N of their target. They appear in W&B as full runs.** |
| 3 | `IN_FEATURES=3072` leaks into seg | `eval_configs/consep.yaml`, `monusac.yaml` | Seg decoder expects 1536-dim spatial features, fails with `expected 1536, got 3072 channels`. |
| 4 | `PATIENCE=999999` wrong for seg `fit` | sbatch env | Disables early stopping, forces decoder to train full 12500 steps → time-limit. |
| 5 | `PYTORCH_CUDA_ALLOC_CONF` not set | sbatch env | Memory fragmentation OOM around iter ~25k. Killed ablation_lr5e-5 mid-run (51409). |
| 6 | `${PYTHONPATH}` unbound under `set -u` | run_*.sbatch | Sbatch dies 1s after launch. Took out one round of Node 1/Node 2 submissions (51487, 51488). |

All fixed. The post-fix training runs are 51493 (Node 1) and 51494 (Node 2); the post-fix evals start from 51486.

**Limitation:** bugs 1 and 2 mean we have *no* checkpoint trained to convergence at paper's effective batch. The warmstart_5k checkpoint is the closest, and at 5k opt steps (still in LR warmup) it carries genuinely partial signal.

---

## 3. What we ran

Three classes of experiments:

**Baselines.** Phase 1 evaluated unchanged at both 224 and 392 px input, with CLS+Mean. Closes the question of how much of the eventual high-res "gain" comes from the eval transform alone.

**Trainings.** One 5k-step run at paper-matching effective batch (warmstart_5k, ID 51417 — meant to be 60k but log_every-truncated). Two 30k-step runs at 1/3 effective batch: Node 1 (51493, drop_path=0.4) and Node 2 (51494, drop_path=0.2). Earlier this week, four single-knob ablations (lr=5e-5, drop_path=0.2, global_crops_scale=[0.5,1.0], iBOT mask=[0.1,0.3]) were also run at accum=4 but all truncated to ~7.5k opt steps and clustered within 0.05 of each other on total loss — no knob differentiated under truncation.

**Eval matrix.** Each of the three usable training checkpoints (warmstart_5k, Node 1, Node 2) evaluated under both CLS+Mean and CLS-only protocols, giving 6 full EVA runs. P1 segmentation re-evaluated at both 224 and 392 px to close the seg baseline gap.

**Prior session (pre-fix), kept for historical narrative only.** ~8 training runs were attempted before this week from a prior Claude Code session, results in `HIGHRES_FINETUNING_GUIDE.md` §8. They used pre-fix train code (bugs 1 and 2 both active) and CLS-only eval (the wrapper didn't exist yet). Their numbers can't be directly compared to anything below; the best of them (phase2_averaged_25M_noes) reported CoNSeP +3pp / MoNuSAC +5pp vs P1@224 but BreakHis −13pp.

![Loss curves across all post-fix trainings (W&B), indexed by optimizer step](placeholder_loss_curves.png)

---

## 4. Results

### 4.1 Headline table

CSV form: `highres_comparison.csv`. Numbers are `val/MulticlassAccuracy` or `test/BinaryBalancedAccuracy` for classification, `MonaiDiceScore` for segmentation, `test/MulticlassAccuracy` for slide-level tasks. All non-seg tasks use CLS+Mean (3072-dim); seg uses spatial patch features (1536-dim).

| Model | pc10 | bach | brcs | bkhs | crc | glsn | mhst | pc | c16 | pnd | cnsp | mnsc | HEST |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **P1 @ 224, ours** | 0.821 | 0.917 | 0.645 | 0.908 | 0.967 | 0.824 | 0.832 | 0.933 | 0.835 | 0.652 | 0.623 | 0.649 | — |
| **P1 @ 392, ours** | 0.769 | 0.917 | 0.655 | 0.842 | 0.966 | 0.816 | 0.829 | 0.920 | 0.814 | 0.660 | 0.646 | 0.680 | — |
| **Node 1 @ 392, ours** | 0.809 | 0.929 | 0.636 | 0.838 | 0.966 | 0.821 | 0.822 | 0.935 | 0.846 | 0.633 | 0.655 | 0.690 | — |
| Midnight-12k (paper) | 0.803 | 0.907 | 0.639 | 0.840 | 0.967 | 0.790 | 0.815 | 0.931 | 0.869 | 0.656 | 0.625 | 0.664 | 0.412 |
| Midnight-92k (paper) | 0.882 | 0.889 | 0.615 | 0.793 | 0.967 | 0.823 | 0.831 | 0.948 | 0.872 | 0.643 | 0.629 | 0.656 | 0.425 |
| Midnight-92k/392 (paper) | 0.900 | 0.904 | 0.646 | 0.802 | 0.966 | 0.807 | 0.828 | 0.951 | 0.868 | 0.651 | 0.662 | 0.708 | 0.415 |

### 4.2 The eval-at-392 effect

Feeding 392 px input through Phase 1 (no training) via dynamic pos_embed interpolation gives most of the segmentation gain the paper attributes to high-res training:

| Task | P1@224 | P1@392 | Δ from eval change alone | Δ Node 1 over P1@392 |
|---|---|---|---|---|
| CoNSeP | 0.623 | 0.646 | **+2.3** | +0.9 |
| MoNuSAC | 0.649 | 0.680 | **+3.2** | +1.0 |

The training step adds another ~1pp on top of each, consistent with what we see in the paper's deltas vs `Midnight-92k`. So the paper's "segmentation gain from high-res post-training" is roughly 65–75% an eval-protocol effect and 25–35% an actual training effect. This was not obvious from the paper text and is the single most useful finding from this work.

For classification, the eval-at-392 change goes the *other* way:

| Task | P1@224 | P1@392 | Δ from eval change alone |
|---|---|---|---|
| BreakHis | 0.908 | 0.842 | −6.6 |
| PCam_10 | 0.821 | 0.769 | −5.2 |
| PCam | 0.933 | 0.920 | −1.3 |

In the original handoff doc, BreakHis going 0.886 → 0.756 in `phase2_averaged_25M_noes` was attributed to bad finetuning. Most of that was just feeding 392 px through Phase 1 hurting BreakHis, plus the training step not fully recovering it.

### 4.3 Did the training help?

Apples-to-apples: Node 1 (high-res-trained, evaluated at 392) vs Phase 1 (not trained, evaluated at 392). Same input resolution, same eval protocol.

| Task | P1@392 | Node 1 | Δ |
|---|---|---|---|
| pc10 | 0.769 | 0.809 | **+4.0** |
| bach | 0.917 | 0.929 | +1.3 |
| brcs | 0.655 | 0.636 | −1.9 |
| bkhs | 0.842 | 0.838 | −0.4 |
| crc | 0.966 | 0.966 | 0 |
| glsn | 0.816 | 0.821 | +0.5 |
| mhst | 0.829 | 0.822 | −0.7 |
| pc | 0.920 | 0.935 | +1.5 |
| c16 | 0.814 | 0.846 | **+3.2** |
| pnd | 0.660 | 0.633 | −2.7 |
| cnsp | 0.646 | 0.655 | +0.9 |
| mnsc | 0.680 | 0.690 | +1.0 |
| **Avg** | — | — | **+0.5** |

Net positive but smaller than the paper's reported ~+1pp average. The strongest gains are on PCam_10 (+4) and Cam16 (+3.2), both of which got hit hardest by the eval-at-392 transform — the training is recovering, not adding new capability. BACH (+1.3) and the two seg tasks (+1 each) are the only places where we see clean additive gain. Three regressions: BRACS, MHIST, Panda (all in the −0.7 to −2.7 range).

![Per-task bars: P1@224 vs P1@392 vs Node 1](placeholder_eval_bars.png)

### 4.4 5k paper-batch ≈ 30k third-batch

The single most surprising result of the campaign. `warmstart_5k` (5000 opt steps at eff batch 1152, log_every-truncated mid-LR-warmup) lands within ±2pp of Node 1 (30000 opt steps at eff batch 384, fully cosine-decayed) on every single task. 7/12 tasks within ±1pp:

| Task | warmstart_5k | Node 1 | Δ |
|---|---|---|---|
| bach | 0.936 | 0.929 | −0.7 |
| brcs | 0.653 | 0.636 | −1.7 |
| bkhs | 0.824 | 0.838 | +1.4 |
| crc | 0.967 | 0.966 | flat |
| glsn | 0.828 | 0.821 | −0.7 |
| mhst | 0.822 | 0.822 | flat |
| pc10 | 0.813 | 0.809 | flat |
| pc | 0.926 | 0.935 | +0.9 |
| c16 | 0.847 | 0.846 | flat |
| pnd | 0.661 | 0.633 | **−2.8** |
| cnsp | 0.657 | 0.655 | flat |
| mnsc | 0.691 | 0.690 | flat |

Either signal saturates very early in this fine-tuning regime, or paper batch size is doing the work and additional small-batch steps are a wash. Either way the practical implication is clear: a multi-node run at paper batch for *5–10k* iters (not 60k or 120k) plausibly captures most of the achievable result for ~1/12 the wall-clock.

### 4.5 CLS+Mean vs CLS-only

On the same three checkpoints we ran both embedding protocols. Average absolute Δ across 10 non-seg tasks: ~0.9pp, mixed sign. Task-specific patterns:

- CLS-only wins on BACH (+1.4-1.5pp consistently across all three checkpoints) and PCam (+0.9pp)
- CLS+Mean wins on Panda (+2pp consistently)
- Everything else: within ±1pp

No systematic advantage either way. The paper's CLS+Mean prescription doesn't matter much for this checkpoint family — possibly because Midnight-92k benefits from the patch-mean signal while a 12k-trained model doesn't carry the same fidelity in patch tokens.

### 4.6 Drop_path 0.4 vs 0.2

Node 1 (drop_path=0.4, matching Phase 1) and Node 2 (drop_path=0.2, the DINOv2 finetuning default). Average |Δ| across 12 tasks: 0.5pp, no consistent direction. 30k steps is too short to see the regularization difference, or the knob simply doesn't matter for this stage.

### 4.7 Against the paper's results

`Midnight-12k` (TCGA-only, 224 px) is the data-scale-matched comparison point — same WSI count as ours, but Kaiko's pretraining recipe rather than OpenMidnight's. Node 1 vs Midnight-12k:

| Task | Midnight-12k | Node 1 | Δ |
|---|---|---|---|
| pc10 | 0.803 | 0.809 | +0.6 |
| bach | 0.907 | 0.929 | **+2.2** |
| brcs | 0.639 | 0.636 | −0.3 |
| bkhs | 0.840 | 0.838 | −0.2 |
| crc | 0.967 | 0.966 | 0 |
| glsn | 0.790 | 0.821 | **+3.1** |
| mhst | 0.815 | 0.822 | +0.7 |
| pc | 0.931 | 0.935 | +0.4 |
| c16 | 0.869 | 0.846 | −2.3 |
| pnd | 0.656 | 0.633 | −2.3 |
| cnsp | 0.625 | 0.655 | **+3.0** |
| mnsc | 0.664 | 0.690 | **+2.6** |

Within ±1pp on 8/12 tasks. We beat Midnight-12k on BACH, Gleason, CoNSeP, MoNuSAC. Lose on Cam16 and Panda — both slide-level tasks, both plausibly benefiting from data diversity we can't match. This is the closest thing to validation that the pipeline is qualitatively correct.

Against `Midnight-92k/392` (8× more data, paper's flagship): mixed but expected. We beat them on BACH (+2.5), BreakHis (+3.6), Gleason (+1.4); lose on Cam16 (−2.2), Panda (−1.8), CoNSeP (−0.7), MoNuSAC (−1.8). The data-scale gap shows up where we'd expect.

---

## 5. Discussion

**Why didn't we hit +1pp?** Most likely: data scale. The paper trains Midnight-92k on roughly 8× more WSIs than we have, including ~80k slides from a proprietary NKI set that adds substantially more morphological variety than TCGA. They claim +1pp from high-res post-training going Midnight-92k → Midnight-92k/392; we measured +0.5pp going OM Phase 1 → Node 1. They don't report Midnight-12k → Midnight-12k/392, which is the comparison we actually need. Our number isn't necessarily inconsistent with theirs at matched data scale — we just have nothing to check it against.

Second contender: tile-sampling distribution. Our 25M sample list is non-uniform across magnifications because the skip rule rejects 0.125 µm/px for ~40% of slides that are 20× scans. The paper doesn't report their distribution; if their NKI scans are predominantly 40×, theirs would be more balanced at the finest magnifications. We could rebalance via per-magnification quotas but haven't tried it.

**Reframing what the paper actually does.** The most useful single observation: most of the segmentation gain that the paper attributes to high-res training is just an eval-protocol effect — feeding the same Phase 1 model 392 px input gets you 65–75% of the way there for free. The training step adds a real but smaller bump. This isn't called out anywhere in the paper text. Anyone reproducing should compute the eval-at-392 baseline first because it's nearly zero cost.

**Things that didn't move the needle.** drop_path. CLS+Mean. 5× more training steps at 1/3 batch. The four single-knob ablations. None of these created visible differentiation in our setup. Either they genuinely don't matter much or our signal-to-noise floor is too high to detect their effect.

**What we couldn't verify.** HEST gene regression (never set up). Paper's full 120k iters at full effective batch (single-node bottleneck). Starting from the released Midnight-12k checkpoint instead of our Phase 1 (would isolate "starting checkpoint" from "everything else"). Smaller backbones (ViT-B/L would let us iterate at full iter count for less compute).

---

## 6. Open questions / directions

Roughly in order of expected information per compute-day:

1. **Match paper compute exactly.** Multi-node run at eff batch 1152 from OM Phase 1, full 60–120k iters. The cleanest test of whether our single-node setup is just compute-bottlenecked. If the answer is yes (i.e., scaling to multi-node closes the gap to paper Δ), conclude that the recipe is fine and data-scale + compute are the actual constraints.
2. **Start from Midnight-12k.** The released Kaiko Midnight-12k checkpoint on HuggingFace, same TCGA-only pretraining data, but their recipe. Same high-res post-training on top would isolate "pretraining recipe quality" from "high-res execution".
3. **Tile-sampling distribution.** Force a more uniform per-magnification budget (e.g., 5M tiles at each of the four µm/px values). Cheap to retry once.
4. **HEST evaluation.** Set up the gene-expression regression task; we have no data on the one task the paper itself flags as regressing under high-res. Without it, our "average gain" is on 12 tasks vs paper's 13.
5. **Smaller backbone.** ViT-B/14 with the same recipe runs roughly 3× faster per step on the same hardware. Lets us reach full iter count on one node, which would also let us settle (4) by running ablations.

---

## Appendix A — File map

- Training config: `dinov2/configs/train/vitg14_reg4_highres_warmstart_accum4.yaml` (Node 1), `..._dp02.yaml` (Node 2), `..._warmstart.yaml` (warmstart_5k, eff batch 1152).
- Sbatch launchers: `run_highres_warmstart_accum4.sbatch`, `..._dp02.sbatch`, `run_highres_warmstart.sbatch`.
- Eval sbatches: `run_eval_phase1_omckpt_at{224,392}.sbatch`, `run_eval_phase1_seg_at{224,392}.sbatch`, `run_eval_phase2_warmstart_accum4{,_dp02,_cls,_dp02_cls}.sbatch`, `run_eval_phase2_warmstart_5k{,_cls}.sbatch`.
- EVA backbone wrapper (CLS+Mean toggle): `eva-probe/src/eva/vision/models/networks/backbones/pathology/openmidnight.py`.
- EVA configs: `eval_configs/{bach,bracs,breakhist,crc,gleason,mhist,pcam,pcam_10,cam16_small,panda_small,consep,monusac}.yaml`. Seg configs hardcode `in_features: 1536`.
- Data pipeline: `prepatching_scripts/create_sample_dataset_txt_highres.py`, `dinov2/data/datasets/slide_dataset.py`, `dinov2/data/loaders.py`.
- Training core: `dinov2/train/train.py` (`_load_from_teacher_checkpoint` for schedule warm-start + pos_embed interp; bug fixes at lines 1209 and 1222).
- Bug-fix diffs are in this branch's git history; the relevant commits are post-`f832c7c`.

## Appendix B — Reproduction recipe

```bash
# 1. Generate 25M-tile sample list (only if it doesn't exist already)
uv run python3 prepatching_scripts/create_sample_dataset_txt_highres.py \
    --target_patches 25000000 --workers 20 \
    --output sample_dataset_highres_25M.txt

# 2. Launch a training run
sbatch run_highres_warmstart_accum4.sbatch       # Node 1 config; 24h on 1 node

# 3. After training, launch full EVA at 392 px with CLS+Mean
sbatch run_eval_phase2_warmstart_accum4.sbatch   # 24h on 1 GPU

# 4. Optional: same eval at CLS-only for A/B
sbatch run_eval_phase2_warmstart_accum4_cls.sbatch
```

Reads results from `eval_results/<run_name>/fast_eval_<timestamp>/summary.tsv`. Aggregated across runs in `eval_results_summary.csv`.

Phase 1 starting checkpoint: `/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth`. TCGA SVS files: `/block/TCGA/`. EVA data roots: `/block/eva-data/{bach,bracs,breakhis,crc,arvaniti_gleason_patches,mhist,patch_camelyon,consep,monusac,camelyon16,panda/prostate-cancer-grade-assessment}`.
