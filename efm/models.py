from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from efm.aggregators import EvidenceAggregator, build_aggregator
from efm.primitives import AxisAlignedMarginRules, DualHeadMixin


class EFM(DualHeadMixin, nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_rules: int = 64,
        n_classes: int = 1,
        aggregator: str | EvidenceAggregator = "student_t",
        beta: float = 6.0,
        kappa_gating: bool = False,
        **agg_kwargs,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_rules = n_rules

        self.rules = AxisAlignedMarginRules(
            input_dim, n_rules, beta=beta, kappa_gating=kappa_gating
        )
        self.aggregator = build_aggregator(aggregator, input_dim, n_rules, **agg_kwargs)
        self.head = self.make_head(n_rules, n_classes)

    @property
    def aggregator_name(self) -> str:
        return self.aggregator.name

    def init_from_data(self, X: np.ndarray) -> None:
        self.rules.init_from_data(X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.rules.margin(x)
        evidence = self.aggregator(m)
        z = self.rules.fire(evidence)
        return self.head(z)

    def loss_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.supervised_loss(self.forward(x), y)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, n_rules={self.n_rules}, "
            f"aggregator={self.aggregator_name!r}, n_classes={self.n_classes}"
        )
