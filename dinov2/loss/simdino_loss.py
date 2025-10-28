# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimDINOLoss(nn.Module):
    """
    SimDINO loss function, as described in https://arxiv.org/abs/2502.10385.

    This loss is composed of two parts:
    1. A squared Euclidean distance term to enforce similarity between student and teacher representations.
    2. A coding rate regularization term on the student's output to prevent representation collapse.
    """
    def __init__(self, out_dim, gamma=0.1, eps=1e-6):
        """
        Args:
            out_dim (int): The dimensionality of the feature representations.
            gamma (float): The weight for the coding rate regularization term.
            eps (float): A small epsilon value for numerical stability in the coding rate calculation.
        """
        super().__init__()
        self.out_dim = out_dim
        self.gamma = gamma
        self.eps = eps

    def forward(self, student_cls_tokens, teacher_cls_tokens):
        """
        Calculates the SimDINO loss.

        Args:
            student_cls_tokens (torch.Tensor): Output class tokens from the student model. Shape: (B, D)
            teacher_cls_tokens (torch.Tensor): Output class tokens from the teacher model. Shape: (B, D)

        Returns:
            torch.Tensor: The computed SimDINO loss.
        """

        # Ensure inputs are normalized as per the paper's methodology
        student_cls_tokens = F.normalize(student_cls_tokens, p=2, dim=-1)
        teacher_cls_tokens = F.normalize(teacher_cls_tokens, p=2, dim=-1)

        # 1. Squared Euclidean Distance Loss
        # For normalized vectors, 0.5 * ||x - y||^2 = 1 - x^T * y
        # We can also compute it directly.
        distance_loss = F.mse_loss(student_cls_tokens, teacher_cls_tokens, reduction="mean") * self.out_dim / 2

        # 2. Coding Rate Regularization
        # This term encourages the features to be spread out, preventing collapse.
        B, D = student_cls_tokens.shape

        # Center the student tokens
        student_centered = student_cls_tokens - student_cls_tokens.mean(dim=0, keepdim=True)

        # Calculate the covariance matrix
        # torch.cov expects (features, observations), so we transpose
        covariance_matrix = torch.cov(student_centered.T)

        # Calculate the coding rate term: R(Z) = 0.5 * logdet(I + (D / ε^2) * Cov(Z))
        identity = torch.eye(D, device=student_cls_tokens.device)
        coding_rate_loss = 0.5 * torch.logdet(identity + (D / (self.eps**2)) * covariance_matrix)

        # Final SimDINO Loss
        total_loss = distance_loss - self.gamma * coding_rate_loss

        return total_loss