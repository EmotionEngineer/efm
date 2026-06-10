from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class RuleExplainer:
    def __init__(self, model, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names

    def _rule_importance(self) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            w = self.model.head.weight.detach().cpu().numpy()
            importance = np.abs(w).sum(axis=0)
        return np.argsort(importance)[::-1], importance

    def _extract_raw(self) -> dict:
        rules = self.model.rules
        agg = self.model.aggregator
        with torch.no_grad():
            raw = {
                "mask": torch.sigmoid(rules.mask_logit).cpu().numpy(),
                "threshold": rules.th.cpu().numpy(),
                "direction": torch.tanh(rules.ineq).cpu().numpy(),
                "sign": torch.tanh(rules.esign).cpu().numpy(),
            }
            if hasattr(agg, "log_nu"):
                raw["nu"] = F.softplus(agg.log_nu).cpu().numpy()
            if hasattr(agg, "mu"):
                raw["template"] = agg.mu.cpu().numpy()
            elif hasattr(agg, "loc"):
                raw["template"] = agg.loc.cpu().numpy()
        return raw

    def get_rules(self, top_k: int = 10, mask_threshold: float | None = None) -> list[dict]:
        idx, importance = self._rule_importance()
        raw = self._extract_raw()
        rules = []

        for r in idx[:top_k]:
            mask_r = raw["mask"][r]
            cutoff = (
                mask_threshold
                if mask_threshold is not None
                else max(0.01, 0.25 * float(mask_r.max()))
            )
            active = np.where(mask_r > cutoff)[0]

            conditions = []
            for j in active[:20]:
                feat = str(self.feature_names[j]) if j < len(self.feature_names) else f"f{j}"
                conditions.append({
                    "feature": feat,
                    "operator": ">" if raw["direction"][r, j] > 0 else "<",
                    "threshold": float(raw["threshold"][r, j]),
                    "evidence_sign": "+" if raw["sign"][r, j] > 0 else "−",
                    "mask_weight": float(mask_r[j]),
                })

            rule = {
                "rule_id": int(r),
                "importance": float(importance[r]),
                "conditions": conditions,
                "n_active": int(len(active)),
            }
            if "nu" in raw:
                rule["nu"] = float(raw["nu"][r])
            rules.append(rule)

        return rules

    def format_rules(self, top_k: int = 10, mask_threshold: float | None = None) -> str:
        rules = self.get_rules(top_k=top_k, mask_threshold=mask_threshold)
        agg_name = getattr(self.model, "aggregator_name", "?")
        lines = [
            f"EFM Rule Report  [aggregator={agg_name}]  "
            f"(top {min(top_k, len(rules))} rules by importance)",
        ]
        for i, rule in enumerate(rules, 1):
            nu_str = f"  ν={rule['nu']:.2f}" if "nu" in rule else ""
            lines.append(
                f"\nRule #{i:02d}  [id={rule['rule_id']}, "
                f"importance={rule['importance']:.4f}, "
                f"n_conds={rule['n_active']}{nu_str}]"
            )
            if not rule["conditions"]:
                lines.append("    (no active literals)")
            else:
                for c in rule["conditions"]:
                    lines.append(
                        f"    {c['feature']:30s} {c['operator']} {c['threshold']:+.3f}"
                        f"  [ev={c['evidence_sign']}, mask={c['mask_weight']:.2f}]"
                    )
        return "\n".join(lines)

    def plot_structure(self, top_k: int = 10, max_features: int = 30):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        idx, _ = self._rule_importance()
        raw = self._extract_raw()
        top_idx = idx[:top_k]

        mask_view = raw["mask"][top_idx]
        sign_view = raw["sign"][top_idx]

        feat_idx = np.argsort(mask_view.mean(0))[::-1][:max_features]
        mask_view = mask_view[:, feat_idx]
        sign_view = sign_view[:, feat_idx]
        feat_labels = [
            str(self.feature_names[j]) if j < len(self.feature_names) else f"f{j}"
            for j in feat_idx
        ]
        signed_mask = mask_view * np.sign(sign_view)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(max(8, len(feat_labels) * 0.35 + 2), max(4, top_k * 0.4 + 1.5)),
        )
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        im0 = axes[0].imshow(mask_view, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axes[0].set_title("mask weights", fontsize=9)
        im1 = axes[1].imshow(signed_mask, aspect="auto", cmap="coolwarm", norm=norm)
        axes[1].set_title("signed mask (mask × sign)", fontsize=9)

        for ax in axes:
            ax.set_xticks(range(len(feat_labels)))
            ax.set_xticklabels(feat_labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(top_idx)))
            ax.set_yticklabels([f"r{r}" for r in top_idx], fontsize=8)

        fig.colorbar(im0, ax=axes[0], shrink=0.8)
        fig.colorbar(im1, ax=axes[1], shrink=0.8)
        agg_name = getattr(self.model, "aggregator_name", "")
        fig.suptitle(f"EFM Rule Structure  [{agg_name}]", fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig
