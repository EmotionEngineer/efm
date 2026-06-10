from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(arr, device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype, device=device)

@torch.no_grad()
def init_thresholds(
    th: torch.Tensor,
    X: np.ndarray,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    max_q: int = 9,
) -> None:
    X = np.asarray(X, dtype=np.float32)
    R, D = th.shape
    for j in range(D):
        qn = min(R, max_q)
        qs = np.linspace(q_lo, q_hi, num=qn, dtype=np.float32)
        vals = np.quantile(X[:, j], qs).astype(np.float32)
        th[:, j].copy_(
            torch.tensor(np.resize(vals, R), dtype=torch.float32, device=th.device)
        )
