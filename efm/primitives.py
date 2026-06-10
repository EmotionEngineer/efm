from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from efm.utils import init_thresholds


def log_margin(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.log(u + eps) - torch.log(v + eps)

class DualHeadMixin:
    n_classes: int

    def make_head(self, n_rules: int, n_classes: int) -> nn.Linear:
        self.n_classes = n_classes
        return nn.Linear(n_rules, max(n_classes, 1))

    def supervised_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.n_classes > 1:
            return F.cross_entropy(logits, y.long())
        return F.mse_loss(logits.view(-1), y.float())

class AxisAlignedMarginRules(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_rules: int = 64,
        beta: float = 6.0,
        kappa_gating: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_rules = n_rules
        self.beta = beta
        self.kappa_gating = kappa_gating

        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.ineq = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.esign = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(math.log(6.0)))
        self.t = nn.Parameter(torch.zeros(n_rules))

    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds(self.th, X)

    def literals(self, x: torch.Tensor):
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        direction = torch.tanh(self.ineq)[None, :, :]
        mask = torch.sigmoid(self.mask_logit)[None, :, :]
        sign = torch.tanh(self.esign)[None, :, :]

        if self.kappa_gating:
            # Kappa Gating behavior (gated kappa inside sigmoid, preserving u + v = 1)
            c = torch.sigmoid(kappa * mask * direction * (x[:, None, :] - self.th[None, :, :]))
            a = sign * (2.0 * c - 1.0)
            u = 0.5 * (1.0 + sign) * c + 0.5 * (1.0 - sign) * (1.0 - c)
            v = 1.0 - u
        else:
            # Original behavior (mask scaling on final subsets)
            c = torch.sigmoid(kappa * direction * (x[:, None, :] - self.th[None, :, :]))
            a = mask * sign * (2.0 * c - 1.0)
            u = mask * (0.5 * (1.0 + sign) * c + 0.5 * (1.0 - sign) * (1.0 - c))
            v = mask - u

        return u, v, a, mask

    def margin(self, x: torch.Tensor) -> torch.Tensor:
        u, v, _, _ = self.literals(x)
        return log_margin(u, v)

    def fire(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.beta * (evidence - self.t[None, :]))
