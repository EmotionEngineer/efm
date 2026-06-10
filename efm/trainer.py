from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from efm.utils import default_device, to_tensor


class TrainConfig:
    def __init__(
        self,
        epochs: int = 200,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        patience: int = 40,
        mask_l1: float = 1e-4,
        grad_clip: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.mask_l1 = mask_l1
        self.grad_clip = grad_clip
        self.device = device or default_device()

class EFMTrainer:
    def __init__(self, model: nn.Module, config: TrainConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def fit(self, X_train, y_train, X_val, y_val, task: str = "regression"):
        cfg = self.config
        y_dtype = torch.float32 if task == "regression" else torch.long

        Xt = to_tensor(X_train, cfg.device)
        Xv = to_tensor(X_val, cfg.device)
        yt = to_tensor(y_train, cfg.device, dtype=y_dtype)
        yv = to_tensor(y_val, cfg.device, dtype=y_dtype)

        opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2
        )

        best_val = math.inf
        best_state: dict | None = None
        wait = 0
        self.history_ = {"train_loss": [], "val_loss": []}

        for _ in range(cfg.epochs):
            self.model.train()
            perm = torch.randperm(Xt.shape[0], device=cfg.device)
            epoch_losses: list[float] = []

            for i in range(0, Xt.shape[0], cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                xb, yb = Xt[idx], yt[idx]

                opt.zero_grad()
                loss = self.model.loss_batch(xb, yb)

                if cfg.mask_l1 > 0 and hasattr(self.model, "rules"):
                    mask = torch.sigmoid(self.model.rules.mask_logit)
                    loss = loss + cfg.mask_l1 * mask.sum()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                opt.step()
                epoch_losses.append(float(loss.detach()))

            sched.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = float(self.model.loss_batch(Xv, yv).detach())

            self.history_["train_loss"].append(float(np.mean(epoch_losses)))
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= cfg.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history_

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        return self.model(to_tensor(X, self.config.device)).cpu().numpy()

    @torch.no_grad()
    def predict(self, X, task: str = "regression", y_scaler=None) -> np.ndarray:
        out = self.predict_proba(X)
        if task == "regression":
            if y_scaler is not None:
                return y_scaler.inverse_transform(out.reshape(-1, 1)).ravel()
            return out.ravel()
        return np.argmax(out, axis=1)
