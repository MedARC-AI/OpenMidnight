# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimDINOLoss(nn.Module):
    def __init__(self, out_dim, gamma=0.1, eps=1e-6):
        super().__init__()
        self.out_dim = out_dim
        self.gamma = gamma
        self.eps = eps

    def forward(self, student_cls_tokens, teacher_cls_tokens):
        student_cls_tokens = F.normalize(student_cls_tokens, p=2, dim=-1)
        teacher_cls_tokens = F.normalize(teacher_cls_tokens, p=2, dim=-1)

        distance_loss = F.mse_loss(student_cls_tokens, teacher_cls_tokens, reduction="mean") * self.out_dim / 2

        B, D = student_cls_tokens.shape

        student_centered = student_cls_tokens - student_cls_tokens.mean(dim=0, keepdim=True)

        covariance_matrix = torch.cov(student_centered.T)

        identity = torch.eye(D, device=student_cls_tokens.device)
        coding_rate_loss = 0.5 * torch.logdet(identity + (D / (self.eps**2)) * covariance_matrix)

        total_loss = distance_loss - self.gamma * coding_rate_loss

        return total_loss
