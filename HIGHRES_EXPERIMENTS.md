# High-res post-training experiments tracker

Started 2026-05-10. Branch `feat/highres-finetuning-v2`.

Eval pipeline switched to CLS+Mean concat (3072-dim) per Midnight paper protocol.
Wrapper: `eva-probe/src/eva/vision/models/networks/backbones/pathology/openmidnight.py`.
Activated via env vars `INCLUDE_PATCH_TOKENS=true` + `IN_FEATURES=3072`.

Phase 1 starting checkpoint: `/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth`.

Eff batch formula: `8 GPUs × batch_per_gpu × accum_steps`. Scaled LR: `base_lr × sqrt(eff_batch / 1024)`.

## Baselines (Step 0)

| Run | Checkpoint | Eval @ | Embedding | sbatch | Job ID | Output dir | Status |
|---|---|---|---|---|---|---|---|
| `phase1_omckpt_at224_clsmean` | omckpt | 224 | CLS+Mean (3072) | run_eval_phase1_omckpt_at224.sbatch | 51413 | eval_results/phase1_omckpt_at224_clsmean | running (n-5) |
| `phase1_omckpt_at392_clsmean` | omckpt | 392 | CLS+Mean (3072) | run_eval_phase1_omckpt_at392.sbatch | 51414 | eval_results/phase1_omckpt_at392_clsmean | running (n-5) |

## Headline training (Step 2)

| Run | Config diffs vs noaccum_omckpt | Eff batch | Iters | Output dir | Job ID | Status |
|---|---|---|---|---|---|---|
| `warmstart` (cancelled) | same | 1152 | 60k | output_vitg14_highres_warmstart | 51408 | cancelled — eval-loop bug (see Notes) |
| `warmstart` (restarted) | same + train.py eval-gate fix | 1152 | 60k | output_vitg14_highres_warmstart | 51417 | running (resubmitted) |

Eval after training: `run_eval_phase2_warmstart.sbatch` → `eval_results/phase2_warmstart_60k_clsmean`.

## Contingent ablations (Step 4)

Each adds ONE knob change on top of warmstart base. Run with `accum=4` (eff batch 384) and 24k optimizer steps (~5h each on 1 node) for fast diagnostic signal — not directly comparable to the headline warmstart numbers since batch differs. Submitted sequentially via SLURM `afterany` deps on a second node.

| Run | Knob change | Iters | Output dir | Job ID | Depends on | Status |
|---|---|---|---|---|---|---|
**Two parallel chains (split 20:14 to use a 2nd dedicated node):**
- Chain A: 51439 droppath02 → 51440 cropscale
- Chain B: 51443 ibotmask → 51444 lr5e-5

Each ~12-13h wallclock at accum=4. Both chains run in parallel → all 4 done in ~25h instead of ~50h.

| Run | Knob change | Iters | Output dir | Job ID | Depends on | Status |
|---|---|---|---|---|---|---|
| `ablation_lr5e-5` (1st try) | base_lr 1e-4 → 5e-5 | 30k | output_vitg14_highres_ablation_lr5e-5 | 51409 | — | **FAILED at iter ~25k (CUDA OOM, rank 7)** |
| `ablation_droppath02` | drop_path_rate 0.4 → 0.2 | 30k | output_vitg14_highres_ablation_droppath02 | 51439 | none | running (n-6) |
| `ablation_cropscale` | global_crops_scale [0.32,1.0] → [0.5,1.0] | 30k | output_vitg14_highres_ablation_cropscale | 51440 | afterany:51439 | pending |
| `ablation_ibotmask` | ibot mask_ratio_min_max [0.1,0.45] → [0.1,0.3] | 30k | output_vitg14_highres_ablation_ibotmask | 51443 | none | pending (parallel chain) |
| `ablation_lr5e-5` (resub) | base_lr 1e-4 → 5e-5 | 30k | output_vitg14_highres_ablation_lr5e-5 (overwrites) | 51444 | afterany:51443 | pending |

## Reference points

- Phase 1 averaged @ 224, CLS-only (legacy): see `eval_results_summary.csv` row `phase1_averaged`.
- Old phase 2 @ 30k @ 392 CLS-only (the regressing eval): see `phase2_averaged_25M_noes`.
- Paper Midnight-92k/392 column (CLS+Mean @ 392): see HIGHRES_FINETUNING_GUIDE.md §8.

## Notes

- 2026-05-10: Branch checkpoint slate is clean (all Phase 2 outputs deleted). Re-establishing baselines + warmstart from scratch.
- 2026-05-10: First eval submission (51406/51407) failed because `eva` CLI wasn't installed in the auto-created uv venv. Fixed by `uv pip install -e ./eva-probe`; resubmitted as 51413/51414.
- Warmstart at iter 0 confirmed scaled `lr=1.06e-4` (= 1e-4 × √(1152/1024)) and all schedule warm-start values applied.
- 2026-05-10: Found a pre-existing eval-loop bug in `dinov2/train/train.py:1222`. With `gradient_accumulation_steps>1`, the eval check fires once per micro-step (not per optimizer step), causing `accum_steps` redundant `do_test()` calls per evaluation period. With accum=12 and 9 min/eval that was ~108 min wasted at every (iter % 5000 == 0) hit. Fixed by gating the eval+checkpointer block on `micro_step % accum_steps == 0`. Cancelled warmstart 51408 (still on iter 0, no training progress lost) and resubmitted as 51417. Pending ablations (51410/51411/51412) automatically pick up the fix when their dependency clears. Ablation lr5e-5 (51409) left running — has real training progress and only loses ~36min/eval-period (accum=4).
- 2026-05-10 15:25: warmstart 51417 verified — exactly 1 BACH eval at iter 0, training reached iter 10 in ~9 min (vs ~108 min stuck on iter 0 with the bug). Total loss at iter 20 = 14.29 (dominated by local_crops loss at 10.96 — expected because 168px local crop has 3× more tokens than Phase 1's 98px). Wall-clock projection for 60k iters: ~12h.
- 2026-05-10 19:42: ablation_lr5e-5 (51409) FAILED with CUDA OOM at iter ~25000 (rank 7: tried 638 MiB, 636 MiB free — fragmentation). Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to all training sbatch scripts. Cancelled droppath02 (51410, only 2.7% done) + pending cropscale/ibotmask (51411/51412); resubmitted as 51439→51440→51441→51442 (afterany chain), with lr5e-5 retry queued at the end. Warmstart 51417 left running (42% done at accum=12, different memory profile, hasn't OOMed).
- 2026-05-11 morning: all 5 trainings "COMPLETED" — BUT only ~5000 opt steps each (warmstart) / ~7500 (ablations), not the configured 60k/30k. **Root cause: `dinov2/logging/helpers.py:136-137` breaks the data-loader iteration when `i >= n_iterations`, where `i` counts data-loader yields (= micro-steps) and `n_iterations` was passed as `eta_target_iter+1` (un-multiplied by accum_steps). So with accum=12 the loop exited at ~5000 opt steps (in mid-warmup, lr ≈ 8.5e-5). Confirmed by "Training Total time" log being printed instead of "Early stopping at iteration".** Fixed in `dinov2/train/train.py:1205` by multiplying `n_iterations` arg by `accum_steps`.
- 2026-05-11 morning: 2nd eval bug — setting `IN_FEATURES=3072` globally in the sbatch also leaks into consep/monusac (segmentation) which expect 1536-dim spatial features. Caused `RuntimeError: expected 3072 channels, got 1536` for those two tasks in 51413/51414. Fixed by hardcoding `in_features: 1536` in `eval_configs/consep.yaml` and `eval_configs/monusac.yaml`.
- 2026-05-11: even with severely truncated training, in-training evals at iter 5000 show consistent gains across all 5 runs. PCam_10: 0.770→0.802-0.823 (+3-5pp). BACH bal: 0.917→0.933-0.939 (+1.6-2.2pp). Submitted 51486 to run full 12-task EVA on warmstart_5000 checkpoint.
- 2026-05-11: baselines 51413 (P1@224) and 51414 (P1@392) showed eval-at-392 alone hurts classification significantly: BreakHis −6.6pp, PCam_10 −5.2pp, PCam −1.3pp at 392 vs 224. The high-res post-training mostly recovers these (warmstart_5000 in-training PCam_10 = 0.822 ≈ P1@224's 0.821). So ~half of previously-attributed "Phase 2 regression" was just eval-pipeline artifact.
- 2026-05-11 afternoon: launched 2 fresh parallel warmstart runs with corrected `log_every` bug:
  - 51487/51488 FAILED in 1s — my generator dropped the `:+:` shell-safety from `${PYTHONPATH:+:${PYTHONPATH}}`. With `set -u`, bare `${PYTHONPATH}` on an unset var = unbound-variable error. Fixed line 24 in both sbatches.
  - **51493** Node 1 (resub): warmstart accum=4, 30k opt steps, drop_path=0.4 (default) — running on n-8, ~24h
  - **51494** Node 2 (resub): warmstart accum=4, 30k opt steps, drop_path=0.2 (finetuning standard) — running on n-6, ~24h
  - Full EVA queued `afterok` for each (51495, 51496).
  - Plus quick consep/monusac re-eval on P1@224 (51491) and P1@392 (51492) to close the 2/12 gap from the IN_FEATURES bug.
- 2026-05-12: submitted 3 CLS-only re-evals (51606 warmstart_5k, 51607 Node 1, 51608 Node 2) at IN_FEATURES=1536 / INCLUDE_PATCH_TOKENS=false. A/B vs the existing CLS+Mean evals (51486, 51495, 51496) on the same checkpoints — lets us quantify whether CLS+Mean (3072) is actually helping.
- 2026-05-12: seg re-evals 51548/51549 COMPLETED. Phase 1 baseline MonaiDiceScore:
  - CoNSeP: 0.6226 (@224) → **0.6461 (@392)**, **+2.35pp from eval-at-392 alone, no training**.
  - MoNuSAC: 0.6488 (@224) → **0.6804 (@392)**, **+3.16pp from eval-at-392 alone**.
  - Paper Midnight-92k/392: CoNSeP 0.662 / MoNuSAC 0.708. Our P1@392 is already within 1.6-2.8pp of paper's *post-training* segmentation numbers without any high-res training.
  - **Reframes the question**: most of paper's segmentation "gain" is from eval-at-392 with dynamic pos_embed interp, not from training. Training likely adds only a marginal additional bump.
  - Note: PATIENCE=999999 (no-early-stop) was wrong for seg `fit` mode; default 1250 is correct. PATIENCE=999999 only makes sense for classification linear-probes.
