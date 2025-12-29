# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import time

import torch
import torch.nn.functional as F

import dinov2.distributed as distributed
import wandb


class JEPALogger:
    def __init__(self, cfg):
        self.eval_period = cfg.evaluation.eval_period_iterations
        self.end = time.time()
        self.max_rankme_samples = 2048
        self.max_hist_samples = 50000

    def start_iter(self, iteration: int):
        data_time = time.time() - self.end
        log_heavy = self.eval_period > 0 and iteration % self.eval_period == 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return time.time(), data_time, log_heavy

    def finish_iter(self):
        self.end = time.time()

    def _module_norm(self, module, use_grad: bool):
        total = None
        for p in module.parameters():
            t = p.grad if use_grad else p
            if t is None:
                continue
            t = t.detach()
            if t.is_sparse:
                t = t.coalesce().values()
            t = t.float()
            norm_sq = t.pow(2).sum()
            total = norm_sq if total is None else total + norm_sq
        if total is None:
            return torch.zeros((), device=next(module.parameters()).device)
        return total.sqrt()

    def module_norms(self, backbone, projector, use_grad: bool):
        return self._module_norm(backbone, use_grad), self._module_norm(projector, use_grad)

    def compute_proj_stats(self, proj, centers, n_global_crops: int, n_local_crops: int, log_heavy: bool):
        heavy_stats = None
        proj_hist = None
        with torch.no_grad():
            proj_detach = proj.detach()
            centers_detach = centers.detach()
            pred_loss_global = (centers_detach - proj_detach[:n_global_crops]).square().mean()
            if n_local_crops > 0:
                pred_loss_local = (centers_detach - proj_detach[n_global_crops:]).square().mean()
            else:
                pred_loss_local = proj_detach.new_tensor(float("nan"))
            diff = centers_detach.unsqueeze(0) - proj_detach
            view_mse = diff.square().mean(dim=(1, 2))
            pred_loss_view_std = view_mse.std(unbiased=False)
            if n_global_crops >= 2:
                cos_sim_global_global = F.cosine_similarity(proj_detach[0], proj_detach[1], dim=-1).mean()
            else:
                cos_sim_global_global = proj_detach.new_tensor(float("nan"))
            cos_sim_view_center = F.cosine_similarity(
                proj_detach,
                centers_detach.unsqueeze(0),
                dim=-1,
            ).mean()
            proj_flat = proj_detach.flatten(0, 1).float()
            proj_mean = proj_flat.mean(dim=0)
            proj_mean_abs = proj_mean.abs().mean()
            proj_std_mean = proj_flat.std(dim=0, unbiased=False).mean()
            proj_norms = proj_flat.norm(dim=-1)
            proj_norm_mean = proj_norms.mean()
            proj_norm_std = proj_norms.std(unbiased=False)

            if log_heavy and distributed.is_main_process():
                centered = proj_flat - proj_mean
                denom = max(1, proj_flat.shape[0] - 1)
                cov = centered.t().mm(centered) / denom
                diag = cov.diag()
                cov_diag_mean = diag.mean()
                cov_diag_std = diag.std(unbiased=False)
                cov_sq_sum = cov.pow(2).sum()
                diag_sq_sum = diag.pow(2).sum()
                offdiag_den = cov.numel() - diag.numel()
                if offdiag_den > 0:
                    cov_offdiag_rms = ((cov_sq_sum - diag_sq_sum) / offdiag_den).sqrt()
                else:
                    cov_offdiag_rms = cov.new_tensor(0.0)
                rank_proj = proj_flat
                if rank_proj.shape[0] > self.max_rankme_samples:
                    rank_proj = rank_proj[: self.max_rankme_samples]
                s = torch.linalg.svdvals(rank_proj)
                p = (s / s.sum()) + 1e-5
                rankme_proj = torch.exp(-(p * p.log()).sum())
                hist_values = proj_flat.flatten()
                if hist_values.numel() > self.max_hist_samples:
                    hist_values = hist_values[: self.max_hist_samples]
                proj_hist = wandb.Histogram(hist_values.detach().cpu().numpy())
                heavy_stats = {
                    "train/cov_diag_mean": cov_diag_mean,
                    "train/cov_diag_std": cov_diag_std,
                    "train/cov_offdiag_rms": cov_offdiag_rms,
                    "train/rankme_proj": rankme_proj,
                }

        metrics_tensors = {
            "train/pred_loss_global": pred_loss_global.detach(),
            "train/pred_loss_local": pred_loss_local.detach(),
            "train/pred_loss_view_std": pred_loss_view_std.detach(),
            "train/cos_sim_global_global": cos_sim_global_global.detach(),
            "train/cos_sim_view_center": cos_sim_view_center.detach(),
            "train/proj_mean_abs": proj_mean_abs.detach(),
            "train/proj_std_mean": proj_std_mean.detach(),
            "train/proj_norm_mean": proj_norm_mean.detach(),
            "train/proj_norm_std": proj_norm_std.detach(),
        }
        return metrics_tensors, heavy_stats, proj_hist

    def log_step(
        self,
        *,
        iteration: int,
        loss_dict,
        metrics_tensors,
        heavy_stats,
        proj_hist,
        optimizer,
        lr: float,
        wd: float,
        amp_scale,
        grad_norm_backbone,
        grad_norm_projector,
        param_norm_backbone,
        param_norm_projector,
        batch_size: int,
        n_global_crops: int,
        n_local_crops: int,
        iter_start: float,
        data_time: float,
    ):
        iter_time = time.time() - iter_start
        world_size = distributed.get_global_size()
        num_views = n_global_crops + n_local_crops
        images_per_sec = (batch_size * num_views * world_size) / max(iter_time, 1e-6)
        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        else:
            max_mem_mb = 0.0
        lr_values = [group["lr"] for group in optimizer.param_groups]
        lr_min = min(lr_values)
        lr_max = max(lr_values)
        metrics_device = next(iter(metrics_tensors.values())).device
        metrics_tensors.update(
            {
                "train/grad_norm_backbone": grad_norm_backbone.detach(),
                "train/grad_norm_projector": grad_norm_projector.detach(),
                "train/param_norm_backbone": param_norm_backbone.detach(),
                "train/param_norm_projector": param_norm_projector.detach(),
                "train/amp_scale": torch.as_tensor(amp_scale, device=metrics_device),
                "train/lr_min": torch.as_tensor(lr_min, device=metrics_device),
                "train/lr_max": torch.as_tensor(lr_max, device=metrics_device),
                "train/iter_time": torch.as_tensor(iter_time, device=metrics_device),
                "train/data_time": torch.as_tensor(data_time, device=metrics_device),
                "train/images_per_sec": torch.as_tensor(images_per_sec, device=metrics_device),
                "train/max_mem_mb": torch.as_tensor(max_mem_mb, device=metrics_device),
            }
        )

        if world_size > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
            loss_dict_reduced = {k: v.item() / world_size for k, v in loss_dict.items()}
            for v in metrics_tensors.values():
                torch.distributed.all_reduce(v)
            metrics_reduced = {k: v.item() / world_size for k, v in metrics_tensors.items()}
        else:
            loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
            metrics_reduced = {k: v.item() for k, v in metrics_tensors.items()}

        if distributed.is_main_process():
            sigreg_over_pred = loss_dict_reduced["sigreg_loss"] / (loss_dict_reduced["pred_loss"] + 1e-8)
            wandb_payload = {
                "Learning Rate": lr,
                "Weight Decay": wd,
                **loss_dict_reduced,
                "train/lejepa_loss": loss_dict_reduced["lejepa_loss"],
                "train/sigreg_over_pred": sigreg_over_pred,
                **metrics_reduced,
            }
            if heavy_stats is not None:
                wandb_payload.update({k: v.item() for k, v in heavy_stats.items()})
                wandb_payload["train/proj_hist"] = proj_hist
            wandb.log(wandb_payload, step=iteration)

        return loss_dict_reduced, metrics_reduced
