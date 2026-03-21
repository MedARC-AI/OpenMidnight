"""
Gram (patch-similarity) loss from the DINOv3 training recipe.

Reference: MedARC-AI/path-fm-dinov3 (dinov3/loss/gram_loss.py)

The loss measures the MSE between the pairwise cosine-similarity matrices
("Gram matrices") of the student's and the teacher's patch-token features.
This encourages the student to reproduce the full relational structure of
patch embeddings captured by the teacher, beyond just the CLS token.

Usage (EMA-teacher mode, the default in this codebase):
    In forward_backward(), pass the teacher's x_norm_patchtokens as
    target_feats and the student's x_norm_patchtokens as output_feats.

Shapes
------
    img_level=True  (default) : (B, N, D)  — one Gram matrix per image
    img_level=False           : (B*N, D)   — one Gram matrix per batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """
    MSE between normalised pairwise patch-similarity matrices of student and
    teacher, as introduced in the DINOv3 training objective.

    Args:
        apply_norm:               L2-normalise patch features before computing
                                  similarities (recommended; matches DINOv3).
        remove_neg:               Zero-clip negative similarities in *both*
                                  student and teacher matrices.
        remove_only_teacher_neg:  Zero-clip teacher negatives; clip student
                                  only where the teacher was also negative.
                                  Mutually exclusive with remove_neg.
    """

    def __init__(
        self,
        apply_norm: bool = True,
        remove_neg: bool = False,
        remove_only_teacher_neg: bool = False,
    ):
        super().__init__()
        assert not (remove_neg and remove_only_teacher_neg), (
            "remove_neg and remove_only_teacher_neg are mutually exclusive"
        )
        self.mse = nn.MSELoss()
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

    def forward(
        self,
        output_feats: torch.Tensor,
        target_feats: torch.Tensor,
        img_level: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            output_feats: Student patch features.
                          Shape (B, N, D) if img_level else (B*N, D).
            target_feats: Teacher patch features (no gradient flows).
                          Same shape as output_feats.
            img_level:    If True compute one Gram per image; if False compute
                          one Gram across the whole local batch.

        Returns:
            Scalar MSE loss averaged over all similarity pairs.
        """
        # Always use fp32 for stability (patch tokens may be fp16)
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # ── Teacher Gram matrix ───────────────────────────────────────────────
        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)
        if not img_level and target_feats.dim() == 3:
            target_feats = target_feats.flatten(0, 1)      # (B*N, D)
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # ── Student Gram matrix ───────────────────────────────────────────────
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)
        if not img_level and output_feats.dim() == 3:
            output_feats = output_feats.flatten(0, 1)
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        # ── Optional negative clipping ─────────────────────────────────────────
        if self.remove_neg:
            target_sim = target_sim.clamp(min=0.0)
            student_sim = student_sim.clamp(min=0.0)
        elif self.remove_only_teacher_neg:
            neg_mask = target_sim < 0
            target_sim = target_sim.clamp(min=0.0)
            student_sim[neg_mask] = student_sim[neg_mask].clamp(min=0.0)

        return self.mse(student_sim, target_sim)
