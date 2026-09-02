"""Closed-form glass-box rule books (no PyTorch).

These estimators sit next to the differentiable EFM wrappers. They print
axis-aligned ``IF … THEN affine`` rules (or a two-piece linear GAM) and
fit coefficients with spectral-floored ridge (SFR).

Naming
------
* ``AffineRule*`` — boosted affine rectangles, jointly refit (legacy: BAFL).
* ``ResidualRule*`` — residual CART indicators stacked onto an affine book,
  then path-smoothed. ``n_rounds=2`` or ``3`` is the only post-depth knob
  (legacy: RT-Post2 / RT-Post3).
* ``PiecewiseGAM*`` — two linear pieces per coordinate (legacy: PWL-GAM).
* ``SplitLinear*`` — one direction ``z = vᵀx``, two affines (legacy: TwoLaw).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


def mad_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust location/scale: median and 1.4826·MAD per column."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.where(mad < 1e-8, 1.0, mad)
    return med, 1.4826 * mad


def squash(X: np.ndarray, med: np.ndarray, scale: np.ndarray, bound: float = 3.0) -> np.ndarray:
    """MAD-standardize and clip to ``[-bound, bound]``."""
    return np.clip((np.asarray(X, float) - med) / scale, -bound, bound)


def sfr(D: np.ndarray, y: np.ndarray, lam: float = 1.0, eps: float = 2e-3) -> np.ndarray:
    """Spectral-floored ridge: drop modes with σ < eps·σ_max, then σ/(σ²+λ)."""
    D = np.asarray(D, float)
    y = np.asarray(y, float).ravel()
    if D.ndim == 1:
        D = D[:, None]
    n, p = D.shape
    if n == 0 or p == 0:
        return np.zeros(p)
    try:
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
    except np.linalg.LinAlgError:
        gram = D.T @ D + lam * np.eye(p)
        return np.linalg.lstsq(gram, D.T @ y, rcond=None)[0]
    if s.size == 0 or s[0] < 1e-15:
        return np.zeros(p)
    w = np.where(s >= eps * s[0], s / (s * s + lam), 0.0)
    return (Vt.T * w) @ (U.T @ y)


def y_clamp_bounds(y: np.ndarray) -> tuple[float, float]:
    q05, q95 = np.percentile(y, [5, 95])
    iqr = max(float(np.subtract(*np.percentile(y, [75, 25]))), 1e-8)
    return float(q05 - 4 * iqr), float(q95 + 4 * iqr)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _to_array(X) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        return np.asarray(X.to_numpy(), float)
    return np.asarray(X, float)


# ---------------------------------------------------------------------------
# Affine model tree (axis-aligned; SFR leaf)
# ---------------------------------------------------------------------------


def _affine_fit(X: np.ndarray, r: np.ndarray, idx: np.ndarray, lam: float = 2.0) -> np.ndarray:
    if len(idx) < 8:
        return np.zeros(X.shape[1] + 1)
    D = np.c_[np.ones(len(idx)), X[idx]]
    return sfr(D, r[idx], lam=lam, eps=5e-3)


def _sse_affine(X: np.ndarray, r: np.ndarray, idx: np.ndarray) -> float:
    c = _affine_fit(X, r, idx)
    e = r[idx] - np.c_[np.ones(len(idx)), X[idx]] @ c
    return float(np.dot(e, e))


class AffineTree:
    """Depth-limited axis-aligned tree with an affine model in each leaf."""

    def __init__(
        self,
        max_depth: int = 2,
        min_leaf: int = 80,
        max_leaves: int = 4,
        random_state: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_leaves = max_leaves
        self.random_state = random_state

    def fit(self, X: np.ndarray, r: np.ndarray) -> AffineTree:
        self.tree_ = self._grow(X, r, np.arange(len(X)), 0, 1)
        self._fit_leaves(X, r)
        return self

    def _grow(self, X, r, idx, depth, n_leaves):
        if depth >= self.max_depth or n_leaves >= self.max_leaves or len(idx) < 2 * self.min_leaf:
            return {"kind": "leaf", "idx": idx}
        sse0 = _sse_affine(X, r, idx)
        best = None
        for j in range(X.shape[1]):
            col = X[idx, j]
            for t in np.quantile(col, [0.25, 0.4, 0.5, 0.6, 0.75]):
                left, right = idx[col <= t], idx[col > t]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                sse = _sse_affine(X, r, left) + _sse_affine(X, r, right)
                if best is None or sse < best[0]:
                    best = (sse, j, float(t), left, right)
        if best is None or sse0 - best[0] < 0.005 * (sse0 + 1e-8):
            return {"kind": "leaf", "idx": idx}
        _, j, t, left, right = best
        return {
            "kind": "split",
            "j": j,
            "t": t,
            "L": self._grow(X, r, left, depth + 1, n_leaves + 1),
            "R": self._grow(X, r, right, depth + 1, n_leaves + 1),
        }

    def _fit_leaves(self, X, r):
        def walk(node):
            if node["kind"] == "leaf":
                node["coef"] = _affine_fit(X, r, node["idx"])
            else:
                walk(node["L"])
                walk(node["R"])

        walk(self.tree_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        def ev(node, Xs):
            if node["kind"] == "leaf":
                return np.c_[np.ones(len(Xs)), Xs] @ node["coef"]
            mask = Xs[:, node["j"]] <= node["t"]
            out = np.zeros(len(Xs))
            if mask.any():
                out[mask] = ev(node["L"], Xs[mask])
            if (~mask).any():
                out[~mask] = ev(node["R"], Xs[~mask])
            return out

        return ev(self.tree_, X)


def leaf_assign(tree: dict, X: np.ndarray) -> tuple[np.ndarray, int]:
    lab = np.zeros(len(X), dtype=int)

    def walk(node, mask, k):
        if not mask.any():
            return k
        if node["kind"] == "leaf":
            lab[mask] = k
            node["_id"] = k
            return k + 1
        m = X[:, node["j"]] <= node["t"]
        k = walk(node["L"], mask & m, k)
        return walk(node["R"], mask & (~m), k)

    nlab = walk(tree, np.ones(len(X), dtype=bool), 0)
    return lab, nlab


def _rules_from_tree(tree: dict, names: list[str] | None = None) -> list[str]:
    lines: list[str] = []

    def walk(node, conds):
        if node["kind"] == "leaf":
            c = node["coef"]
            order = np.argsort(-np.abs(c[1:]))[:4]
            terms = " + ".join(
                f"{c[j + 1]:+.3f}*{(names[j] if names else f'x{j}')}" for j in order
            )
            prem = " AND ".join(conds) if conds else "TRUE"
            lines.append(f"IF {prem} THEN {c[0]:+.3f} {terms}")
            return
        nm = names[node["j"]] if names else f"x{node['j']}"
        walk(node["L"], conds + [f"{nm} <= {node['t']:.3f}"])
        walk(node["R"], conds + [f"{nm} >  {node['t']:.3f}"])

    walk(tree, [])
    return lines


def _soft_rule_design(Xs: np.ndarray, trees: list, nlabs: list[int], temperature: float) -> np.ndarray:
    """Path membership as a product of steep logistics (almost-hard leaves)."""
    parts = []
    for tree, nl in zip(trees, nlabs):
        paths: list[list] = []

        def walk(node, ineq):
            if node["kind"] == "leaf":
                paths.append(ineq)
                return
            walk(node["L"], ineq + [(node["j"], node["t"], +1)])
            walk(node["R"], ineq + [(node["j"], node["t"], -1)])

        walk(tree.tree_, [])
        nlab = min(nl, len(paths))
        for k in range(nlab):
            g = np.ones(len(Xs))
            for j, thr, sgn in paths[k]:
                z = sgn * (thr - Xs[:, j]) / max(temperature, 1e-4)
                g = g * _sigmoid(z)
            parts += [g[:, None], g[:, None] * Xs]
    return np.concatenate(parts, 1) if parts else np.zeros((len(Xs), 0))


# ---------------------------------------------------------------------------
# Affine rule book
# ---------------------------------------------------------------------------


class _AffineCore:
    """Boosted affine rectangles, then one SFR on the joint design."""

    def __init__(
        self,
        n_trees: int = 15,
        shrinkage: float = 0.2,
        max_depth: int = 2,
        max_leaves: int = 4,
        min_leaf: int = 80,
        random_state: int = 0,
    ) -> None:
        self.n_trees = n_trees
        self.shrinkage = shrinkage
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_leaf = min_leaf
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> _AffineCore:
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        self.med_, self.sc_ = mad_scale(X)
        Xs = squash(X, self.med_, self.sc_)
        self.ylo, self.yhi = y_clamp_bounds(y)
        self.f0_ = Ridge(1.5).fit(Xs, y)
        resid = y - self.f0_.predict(Xs)
        self.trees_: list[AffineTree] = []
        for m in range(self.n_trees):
            tree = AffineTree(
                self.max_depth, self.min_leaf, self.max_leaves, self.random_state + m
            )
            tree.fit(Xs, resid)
            self.trees_.append(tree)
            resid = resid - self.shrinkage * tree.predict(Xs)
        D, self.nlab_ = self._design(Xs, collect=True)
        self.coef_ = sfr(D, y - self.f0_.predict(Xs), lam=4.0, eps=8e-3)
        return self

    def _design(self, Xs: np.ndarray, collect: bool = False):
        parts = [np.ones((len(Xs), 1)), Xs]
        nlab_list: list[int] = []
        stored = getattr(self, "nlab_", None)
        for i, tree in enumerate(self.trees_):
            lab, nlab = leaf_assign(tree.tree_, Xs)
            if stored is not None:
                nlab = stored[i]
            nlab_list.append(nlab)
            for k in range(nlab):
                mask = (lab == k).astype(float)
                parts.append(mask[:, None])
                parts.append(mask[:, None] * Xs)
        D = np.concatenate(parts, 1)
        return (D, nlab_list) if collect else D

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = squash(X, self.med_, self.sc_)
        D = self._design(Xs)
        if D.shape[1] != len(self.coef_):
            out = self.f0_.predict(Xs)
            for tree in self.trees_:
                out = out + self.shrinkage * tree.predict(Xs)
            return np.clip(out, self.ylo, self.yhi)
        return np.clip(self.f0_.predict(Xs) + D @ self.coef_, self.ylo, self.yhi)

    def n_rules(self) -> int:
        return int(sum(self.nlab_))

    def explain(self, feature_names=None, max_rules: int = 24) -> str:
        lines = [f"AffineRuleBook  intercept ridge + {self.n_rules()} affine rectangles (joint SFR)"]
        for i, tree in enumerate(self.trees_):
            for rule in _rules_from_tree(tree.tree_, feature_names):
                lines.append(f"  [tree {i}] {rule}")
        return "\n".join(lines[: max_rules + 1])


class _LogisticAffineCore(_AffineCore):
    """Newton working residual + joint IRLS; ŷ = σ(η)."""

    def __init__(self, n_trees: int = 10, shrinkage: float = 0.4, irls: int = 6, **kw: Any) -> None:
        super().__init__(n_trees=n_trees, shrinkage=shrinkage, **kw)
        self.irls = irls

    def fit(self, X: np.ndarray, y: np.ndarray) -> _LogisticAffineCore:
        X = np.asarray(X, float)
        y = (np.asarray(y).ravel() == np.max(y)).astype(float)
        self.med_, self.sc_ = mad_scale(X)
        Xs = squash(X, self.med_, self.sc_)
        pbar = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        self.f0_ = float(np.log(pbar / (1 - pbar)))
        eta = np.full(len(y), self.f0_)
        self.trees_, self.nlab_ = [], []
        for m in range(self.n_trees):
            p = np.clip(_sigmoid(eta), 1e-6, 1 - 1e-6)
            w = p * (1 - p)
            z = (y - p) / w
            tree = AffineTree(
                self.max_depth, self.min_leaf, self.max_leaves, self.random_state + m
            )
            tree.fit(Xs, z * np.sqrt(w))
            self.trees_.append(tree)
            _, nlab = leaf_assign(tree.tree_, Xs)
            self.nlab_.append(nlab)
            eta = eta + self.shrinkage * tree.predict(Xs)
        D = self._design(Xs)
        eta = np.full(len(y), self.f0_)
        coef = np.zeros(D.shape[1])
        for _ in range(self.irls):
            p = np.clip(_sigmoid(eta), 1e-6, 1 - 1e-6)
            w = p * (1 - p)
            zwork = eta - self.f0_ + (y - p) / w
            sw = np.sqrt(w)
            coef = sfr(D * sw[:, None], zwork * sw, lam=3.0, eps=6e-3)
            eta = self.f0_ + D @ coef
        self.coef_ = coef
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xs = squash(X, self.med_, self.sc_)
        D = self._design(Xs)
        if D.shape[1] != len(self.coef_):
            return np.full(len(Xs), self.f0_)
        return self.f0_ + D @ self.coef_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _leaf_indicators(tree: DecisionTreeRegressor, X: np.ndarray, ids: np.ndarray) -> np.ndarray:
    lab = tree.apply(X)
    mp = {int(v): i for i, v in enumerate(ids)}
    Z = np.zeros((len(X), len(ids)))
    for i, v in enumerate(lab):
        j = mp.get(int(v))
        if j is not None:
            Z[i, j] = 1.0
    return Z


# ---------------------------------------------------------------------------
# sklearn estimators
# ---------------------------------------------------------------------------


class AffineRuleRegressor(BaseEstimator, RegressorMixin):
    """Jointly refit book of boosted affine rectangles.

    Parameters
    ----------
    n_trees : int
        Boosting rounds that propose rectangles.
    min_leaf : int
        Minimum samples per affine leaf.
    """

    def __init__(
        self,
        n_trees: int = 15,
        min_leaf: int = 80,
        random_state: int | None = 42,
    ) -> None:
        self.n_trees = n_trees
        self.min_leaf = min_leaf
        self.random_state = random_state

    def fit(self, X, y) -> AffineRuleRegressor:
        X = _to_array(X)
        y = np.asarray(y, float).ravel()
        self.n_features_in_ = X.shape[1]
        self.core_ = _AffineCore(
            n_trees=self.n_trees,
            min_leaf=self.min_leaf,
            random_state=int(self.random_state or 0),
        )
        self.core_.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        return self.core_.predict(_to_array(X))

    def explain(self, feature_names=None, max_rules: int = 24) -> str:
        check_is_fitted(self)
        return self.core_.explain(feature_names, max_rules=max_rules)


class AffineRuleClassifier(BaseEstimator, ClassifierMixin):
    """Logistic affine rectangles with joint IRLS."""

    def __init__(
        self,
        n_trees: int = 10,
        min_leaf: int = 40,
        random_state: int | None = 42,
    ) -> None:
        self.n_trees = n_trees
        self.min_leaf = min_leaf
        self.random_state = random_state

    def fit(self, X, y) -> AffineRuleClassifier:
        X = _to_array(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        yb = (y == self.classes_.max()).astype(float)
        self.core_ = _LogisticAffineCore(
            n_trees=self.n_trees,
            min_leaf=self.min_leaf,
            max_depth=2,
            max_leaves=4,
            random_state=int(self.random_state or 0),
        )
        self.core_.fit(X, yb)
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)
        return self.core_.predict_proba(_to_array(X))

    def predict(self, X) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]

    def explain(self, feature_names=None, max_rules: int = 16) -> str:
        check_is_fitted(self)
        return self.core_.explain(feature_names, max_rules=max_rules)


class ResidualRuleRegressor(BaseEstimator, RegressorMixin):
    """Affine rule book + residual CART rounds + path smoothing.

    ``n_rounds=2`` and ``n_rounds=3`` are the two flagship depths. The
    printed book stays one jointly refit SFR; CART only proposes regions.
    """

    def __init__(
        self,
        n_rounds: int = 2,
        max_leaves: int = 10,
        n_trees: int = 12,
        smooth: float = 0.08,
        random_state: int | None = 42,
    ) -> None:
        self.n_rounds = n_rounds
        self.max_leaves = max_leaves
        self.n_trees = n_trees
        self.smooth = smooth
        self.random_state = random_state

    def fit(self, X, y) -> ResidualRuleRegressor:
        X = _to_array(X)
        y = np.asarray(y, float).ravel()
        self.n_features_in_ = X.shape[1]
        seed = int(self.random_state or 0)
        self.carts_: list[DecisionTreeRegressor] = []
        self.leaf_ids_: list[np.ndarray] = []
        pred = None
        Xe = X
        min_leaf = max(40, len(X) // 80)
        for rd in range(self.n_rounds):
            if pred is None:
                core = _AffineCore(n_trees=self.n_trees, min_leaf=80, random_state=seed)
                core.fit(X, y)
                pred = core.predict(X)
            resid = y - pred
            cart = DecisionTreeRegressor(
                max_leaf_nodes=self.max_leaves,
                min_samples_leaf=min_leaf,
                random_state=seed + rd,
            )
            cart.fit(X, resid)
            self.carts_.append(cart)
            extras = []
            self.leaf_ids_ = []
            for tree in self.carts_:
                ids = np.unique(tree.apply(X))
                self.leaf_ids_.append(ids)
                extras.append(_leaf_indicators(tree, X, ids))
            Xe = np.c_[X, *extras]
            core = _AffineCore(
                n_trees=self.n_trees, min_leaf=80, random_state=seed + 50 * (rd + 1)
            )
            core.fit(Xe, y)
            pred = core.predict(Xe)
        self.core_ = core
        # Path-smooth the last affine book on the expanded design.
        Xs = squash(Xe, core.med_, core.sc_)
        G = np.c_[np.ones((len(Xs), 1)), Xs]
        R = _soft_rule_design(Xs, core.trees_, core.nlab_, temperature=self.smooth)
        D = np.c_[G, R] if R.size else G
        self.smooth_coef_ = sfr(D, y - core.f0_.predict(Xs), lam=4.0, eps=8e-3)
        self.ylo, self.yhi = core.ylo, core.yhi
        return self

    def _expand(self, X: np.ndarray) -> np.ndarray:
        extras = [
            _leaf_indicators(tree, X, ids) for tree, ids in zip(self.carts_, self.leaf_ids_)
        ]
        return np.c_[X, *extras] if extras else X

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = _to_array(X)
        Xe = self._expand(X)
        core = self.core_
        Xs = squash(Xe, core.med_, core.sc_)
        G = np.c_[np.ones((len(Xs), 1)), Xs]
        R = _soft_rule_design(Xs, core.trees_, core.nlab_, temperature=self.smooth)
        D = np.c_[G, R] if R.size else G
        if D.shape[1] != len(self.smooth_coef_):
            return core.predict(Xe)
        return np.clip(core.f0_.predict(Xs) + D @ self.smooth_coef_, self.ylo, self.yhi)

    def explain(self, feature_names=None, max_rules: int = 24) -> str:
        check_is_fitted(self)
        head = (
            f"ResidualRuleBook  n_rounds={self.n_rounds}, "
            f"path-smooth T={self.smooth}\n"
        )
        return head + self.core_.explain(
            self._expanded_names(feature_names), max_rules=max_rules
        )

    def _expanded_names(self, feature_names=None) -> list[str]:
        d = int(self.n_features_in_)
        names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
        names = names[:d]
        for rd, ids in enumerate(getattr(self, "leaf_ids_", [])):
            for k in range(len(ids)):
                names.append(f"cart{rd}_L{k}")
        return names


class ResidualRuleClassifier(BaseEstimator, ClassifierMixin):
    """Same residual expansion as :class:`ResidualRuleRegressor` for binary y.

    Path smoothing is skipped: mixing hard rectangles with a sigmoid link
    is enough, and a second SFR on mixed PWL/rect designs wrecks calibration.
    """

    def __init__(
        self,
        n_rounds: int = 2,
        max_leaves: int = 10,
        n_trees: int = 8,
        random_state: int | None = 42,
    ) -> None:
        self.n_rounds = n_rounds
        self.max_leaves = max_leaves
        self.n_trees = n_trees
        self.random_state = random_state

    def fit(self, X, y) -> ResidualRuleClassifier:
        X = _to_array(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        yb = (y == self.classes_.max()).astype(float)
        seed = int(self.random_state or 0)
        self.carts_: list[DecisionTreeRegressor] = []
        self.leaf_ids_: list[np.ndarray] = []
        pred = None
        min_leaf = max(40, len(X) // 80)
        for rd in range(self.n_rounds):
            if pred is None:
                core = _LogisticAffineCore(
                    n_trees=self.n_trees, min_leaf=40, random_state=seed
                )
                core.fit(X, yb)
                pred = core.predict_proba(X)[:, 1]
            resid = yb - pred
            cart = DecisionTreeRegressor(
                max_leaf_nodes=self.max_leaves,
                min_samples_leaf=min_leaf,
                random_state=seed + rd,
            )
            cart.fit(X, resid)
            self.carts_.append(cart)
            extras = []
            self.leaf_ids_ = []
            for tree in self.carts_:
                ids = np.unique(tree.apply(X))
                self.leaf_ids_.append(ids)
                extras.append(_leaf_indicators(tree, X, ids))
            Xe = np.c_[X, *extras]
            core = _LogisticAffineCore(
                n_trees=self.n_trees, min_leaf=40, random_state=seed + 50 * (rd + 1)
            )
            core.fit(Xe, yb)
            pred = core.predict_proba(Xe)[:, 1]
        self.core_ = core
        return self

    def _expand(self, X: np.ndarray) -> np.ndarray:
        extras = [
            _leaf_indicators(tree, X, ids) for tree, ids in zip(self.carts_, self.leaf_ids_)
        ]
        return np.c_[X, *extras] if extras else X

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)
        return self.core_.predict_proba(self._expand(_to_array(X)))

    def predict(self, X) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]

    def _expanded_names(self, feature_names=None) -> list[str]:
        d = int(getattr(self, "n_features_in_", 0) or (feature_names and len(feature_names) or 0))
        if not d and hasattr(self, "carts_") and self.carts_:
            d = int(self.carts_[0].n_features_in_)
        names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
        names = names[:d]
        for rd, ids in enumerate(getattr(self, "leaf_ids_", [])):
            for k in range(len(ids)):
                names.append(f"cart{rd}_L{k}")
        return names

    def explain(self, feature_names=None, max_rules: int = 16) -> str:
        check_is_fitted(self)
        return (
            f"ResidualRuleBook  n_rounds={self.n_rounds} (classifier)\n"
            + self.core_.explain(self._expanded_names(feature_names), max_rules=max_rules)
        )


# ---------------------------------------------------------------------------
# Piecewise GAM and split-linear
# ---------------------------------------------------------------------------


def _pwl_design(Xs: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    parts = [np.ones((len(Xs), 1)), Xs]
    for j, t in enumerate(cuts):
        left = (Xs[:, j] <= t).astype(float)[:, None]
        right = 1.0 - left
        xj = Xs[:, j : j + 1]
        parts += [left, left * xj, right, right * xj]
    return np.concatenate(parts, 1)


def _irls_or_sfr(D: np.ndarray, y: np.ndarray, task: str, lam: float = 2.0):
    if task != "clf":
        return sfr(D, y, lam=lam, eps=4e-3), None
    pbar = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    f0 = float(np.log(pbar / (1 - pbar)))
    eta = np.full(len(y), f0)
    coef = np.zeros(D.shape[1])
    for _ in range(6):
        p = np.clip(_sigmoid(eta), 1e-6, 1 - 1e-6)
        w = p * (1 - p)
        z = eta - f0 + (y - p) / w
        sw = np.sqrt(w)
        coef = sfr(D * sw[:, None], z * sw, lam=lam, eps=5e-3)
        eta = f0 + D @ coef
    return coef, f0


def _pwl_explain(cuts, coef, feature_names=None, intercept: float = 0.0, link: str = "") -> str:
    d = len(cuts)
    names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
    names = (names + [f"x{j}" for j in range(len(names), d)])[:d]
    c = np.asarray(coef, float).ravel()
    b0 = float(intercept) + (float(c[0]) if c.size else 0.0)
    lin = c[1 : 1 + d] if c.size > 1 else np.zeros(d)
    rest = c[1 + d :]
    lines = [
        f"PiecewiseGAM  {link + ' ' if link else ''}two linear pieces / feature (joint SFR)  b0={b0:+.3f}"
    ]
    mag = []
    for j in range(d):
        sl = float(lin[j]) if j < len(lin) else 0.0
        block = rest[4 * j : 4 * j + 4] if rest.size >= 4 * (j + 1) else np.zeros(4)
        # left: aL + bL x ; right: aR + bR x  (plus global sl * x)
        aL, bL, aR, bR = (float(block[k]) if k < len(block) else 0.0 for k in range(4))
        left_s, right_s = sl + bL, sl + bR
        mag.append((abs(left_s) + abs(right_s) + abs(aL) + abs(aR), j, aL, left_s, aR, right_s, cuts[j]))
    mag.sort(reverse=True)
    for _, j, aL, sL, aR, sR, t in mag[:8]:
        nm = names[j]
        lines.append(
            f"  {nm}:  IF {nm}<= {t:+.3f} THEN {aL:+.3f}{sL:+.3f}*{nm}  "
            f"ELSE {aR:+.3f}{sR:+.3f}*{nm}"
        )
    return "\n".join(lines)


class PiecewiseGAMRegressor(BaseEstimator, RegressorMixin):
    """Additive two-piece linear model per coordinate, jointly SFR-fit."""

    def __init__(self, random_state: int | None = 42) -> None:
        self.random_state = random_state

    def fit(self, X, y) -> PiecewiseGAMRegressor:
        X, y = _to_array(X), np.asarray(y, float).ravel()
        self.n_features_in_ = X.shape[1]
        self.med_, self.sc_ = mad_scale(X)
        Xs = squash(X, self.med_, self.sc_)
        self.ylo, self.yhi = y_clamp_bounds(y)
        self.cuts_ = np.median(Xs, 0)
        D = _pwl_design(Xs, self.cuts_)
        self.coef_, _ = _irls_or_sfr(D, y, "reg", lam=1.5)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        Xs = squash(_to_array(X), self.med_, self.sc_)
        return np.clip(_pwl_design(Xs, self.cuts_) @ self.coef_, self.ylo, self.yhi)

    def explain(self, feature_names=None, **kwargs) -> str:
        check_is_fitted(self)
        return _pwl_explain(self.cuts_, self.coef_, feature_names, intercept=0.0, link="")


class PiecewiseGAMClassifier(BaseEstimator, ClassifierMixin):
    """Logistic piecewise-linear GAM (IRLS + SFR)."""

    def __init__(self, random_state: int | None = 42) -> None:
        self.random_state = random_state

    def fit(self, X, y) -> PiecewiseGAMClassifier:
        X = _to_array(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        yb = (y == self.classes_.max()).astype(float)
        self.med_, self.sc_ = mad_scale(X)
        Xs = squash(X, self.med_, self.sc_)
        self.cuts_ = np.median(Xs, 0)
        D = _pwl_design(Xs, self.cuts_)
        self.coef_, self.f0_ = _irls_or_sfr(D, yb, "clf", lam=1.5)
        return self

    def decision_function(self, X) -> np.ndarray:
        check_is_fitted(self)
        Xs = squash(_to_array(X), self.med_, self.sc_)
        return self.f0_ + _pwl_design(Xs, self.cuts_) @ self.coef_

    def predict_proba(self, X) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]

    def explain(self, feature_names=None, **kwargs) -> str:
        check_is_fitted(self)
        return _pwl_explain(self.cuts_, self.coef_, feature_names, intercept=self.f0_, link="σ")


def _two_design(Xs: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    z = Xs @ v
    left = (z <= t).astype(float)[:, None]
    right = 1.0 - left
    return np.concatenate([np.ones((len(Xs), 1)), Xs, left, left * Xs, right, right * Xs], 1)


class SplitLinearRegressor(BaseEstimator, RegressorMixin):
    """One direction ``z = vᵀx̃`` and two jointly fit affines."""

    def __init__(self, random_state: int | None = 42) -> None:
        self.random_state = random_state

    def fit(self, X, y) -> SplitLinearRegressor:
        X, y = _to_array(X), np.asarray(y, float).ravel()
        self.n_features_in_ = X.shape[1]
        self.med_, self.sc_ = mad_scale(X)
        Xs = squash(X, self.med_, self.sc_)
        self.ylo, self.yhi = y_clamp_bounds(y)
        self.f0_ = Ridge(1.5).fit(Xs, y)
        r = y - self.f0_.predict(Xs)
        v = Ridge(1.0).fit(Xs, r).coef_.ravel()
        nrm = np.linalg.norm(v)
        self.v_ = v / nrm if nrm > 1e-12 else np.eye(len(v))[0]
        self.t_ = float(np.median(Xs @ self.v_))
        self.coef_ = sfr(_two_design(Xs, self.v_, self.t_), r, lam=3.0, eps=6e-3)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        Xs = squash(_to_array(X), self.med_, self.sc_)
        eta = self.f0_.predict(Xs) + _two_design(Xs, self.v_, self.t_) @ self.coef_
        return np.clip(eta, self.ylo, self.yhi)

    def explain(self, feature_names=None, k: int = 6, **kwargs) -> str:
        check_is_fitted(self)
        names = feature_names or [f"x{j}" for j in range(len(self.v_))]
        top = np.argsort(-np.abs(self.v_))[:k]
        z = " + ".join(f"{self.v_[j]:+.3f}*{names[j]}" for j in top)
        return (
            f"SplitLinear  z = {z}   t = {self.t_:.3f}\n"
            "  IF z <= t THEN affine_0(x)  ELSE affine_1(x)  (joint SFR)"
        )


class SplitLinearClassifier(BaseEstimator, ClassifierMixin):
    """Logistic two-affine split on ``z = vᵀx̃``."""

    def __init__(self, random_state: int | None = 42) -> None:
        self.random_state = random_state

    def fit(self, X, y) -> SplitLinearClassifier:
        X = _to_array(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        yb = (y == self.classes_.max()).astype(float)
        self.med_, self.sc_ = mad_scale(X)
        Xs = squash(X, self.med_, self.sc_)
        self.f0_lin_ = Ridge(1.5).fit(Xs, yb)
        r = yb - self.f0_lin_.predict(Xs)
        v = Ridge(1.0).fit(Xs, r).coef_.ravel()
        nrm = np.linalg.norm(v)
        self.v_ = v / nrm if nrm > 1e-12 else np.eye(len(v))[0]
        self.t_ = float(np.median(Xs @ self.v_))
        D = _two_design(Xs, self.v_, self.t_)
        self.coef_, self.f0_ = _irls_or_sfr(D, yb, "clf", lam=2.5)
        return self

    def decision_function(self, X) -> np.ndarray:
        check_is_fitted(self)
        Xs = squash(_to_array(X), self.med_, self.sc_)
        return self.f0_ + _two_design(Xs, self.v_, self.t_) @ self.coef_

    def predict_proba(self, X) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]

    def explain(self, feature_names=None, k: int = 6, **kwargs) -> str:
        check_is_fitted(self)
        names = feature_names or [f"x{j}" for j in range(len(self.v_))]
        top = np.argsort(-np.abs(self.v_))[:k]
        z = " + ".join(f"{self.v_[j]:+.3f}*{names[j]}" for j in top)
        return (
            f"SplitLinear  σ(two affines on z = {z}, t = {self.t_:.3f})"
        )
