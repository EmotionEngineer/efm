from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceAggregator(nn.Module):
    name: str = "base"

    def __init__(self, input_dim: int, n_rules: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_rules = n_rules

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"name={self.name}, input_dim={self.input_dim}, n_rules={self.n_rules}"

class GaussianTemplate(EvidenceAggregator):
    name = "gaussian"

    def __init__(self, input_dim: int, n_rules: int) -> None:
        super().__init__(input_dim, n_rules)
        self.loc = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.log_scale = nn.Parameter(torch.ones(n_rules, input_dim))

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.log_scale)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        sigma2 = self.scale[None, :, :] ** 2 + 1e-6
        return -((m - self.loc[None, :, :]) ** 2 / (2.0 * sigma2)).sum(dim=2)

class StudentTTemplate(EvidenceAggregator):
    name = "student_t"

    def __init__(self, input_dim: int, n_rules: int, nu_init: float = 5.0) -> None:
        super().__init__(input_dim, n_rules)
        self.mu = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.log_s = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.log_nu = nn.Parameter(torch.ones(n_rules) * math.log(nu_init))

    @property
    def nu(self) -> torch.Tensor:
        return F.softplus(self.log_nu).clamp(1.0, 50.0)

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.log_s)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        nu = self.nu
        s2 = self.scale ** 2 + 1e-6
        return (
            -((nu[None, :, None] + 1.0) / 2.0)
            * torch.log1p(
                (m - self.mu[None, :, :]) ** 2 / (nu[None, :, None] * s2[None, :, :])
            )
        ).sum(dim=2)

class HyperplaneArrangement(EvidenceAggregator):
    name = "hyperplane"

    def __init__(self, input_dim: int, n_rules: int, n_planes: int = 8) -> None:
        super().__init__(input_dim, n_rules)
        self.n_planes = n_planes
        self.W_plane = nn.Parameter(torch.randn(n_rules, n_planes, input_dim) * 0.1)
        self.b_plane = nn.Parameter(torch.zeros(n_rules, n_planes))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum("brd,rpd->brp", m, self.W_plane) + self.b_plane[None, :, :]
        return F.relu(proj).sum(dim=2)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", n_planes={self.n_planes}"

class StateCoupledRecurrence(EvidenceAggregator):
    name = "state_coupled"

    def __init__(self, input_dim: int, n_rules: int, steps: int = 3) -> None:
        super().__init__(input_dim, n_rules)
        self.steps = steps
        self.A = nn.Parameter(
            torch.eye(input_dim).unsqueeze(0).repeat(n_rules, 1, 1) * 0.9
        )
        self.B = nn.Parameter(torch.zeros(n_rules, input_dim))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        state = m
        acc = torch.zeros(m.shape[0], self.n_rules, device=m.device)
        for _ in range(self.steps):
            acc = acc + state.abs().sum(dim=2)
            state = torch.einsum("rid,brd->bri", self.A, state) + self.B[None, :, :]
        return acc

    def extra_repr(self) -> str:
        return super().extra_repr() + f", steps={self.steps}"

AGGREGATORS: dict[str, type[EvidenceAggregator]] = {
    "gaussian": GaussianTemplate,
    "student_t": StudentTTemplate,
    "hyperplane": HyperplaneArrangement,
    "state_coupled": StateCoupledRecurrence,
}

_ALIASES: dict[str, str] = {
    "gmte": "gaussian",
    "smte": "student_t",
    "hyp": "hyperplane",
    "sc": "state_coupled",
}

def normalize_aggregator_name(spec: str) -> str:
    key = str(spec).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in AGGREGATORS:
        valid = sorted(set(AGGREGATORS) | set(_ALIASES))
        raise ValueError(f"Unknown aggregator '{spec}'. Choose from {valid}.")
    return key

def build_aggregator(spec, input_dim: int, n_rules: int, **kwargs) -> EvidenceAggregator:
    if isinstance(spec, EvidenceAggregator):
        return spec

    name = normalize_aggregator_name(spec)
    cls = AGGREGATORS[name]

    import inspect
    accepted = set(inspect.signature(cls.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(input_dim, n_rules, **filtered)
