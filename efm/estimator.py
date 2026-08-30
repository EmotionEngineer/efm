from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from efm.aggregators import CLOSED_FORM, normalize_aggregator_name
from efm.explain import RuleExplainer
from efm.models import EFM
from efm.trainer import EFMTrainer, TrainConfig
from efm.utils import default_device, set_seed


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # type: ignore[call-arg]

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", ohe)]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")

class _EFMBase(BaseEstimator):
    def __init__(
        self,
        aggregator: str = "student_t",
        n_rules: int = 64,
        n_planes: int = 8,
        steps: int = 3,
        epochs: int = 200,
        lr: float = 3e-3,
        batch_size: int = 512,
        patience: int = 40,
        mask_l1: float = 1e-4,
        weight_decay: float = 1e-4,
        random_state: int | None = 42,
        device: str | None = None,
        val_fraction: float = 0.15,
        kappa_gating: bool = False,
        n_rounds: int = 2,
        n_trees: int = 15,
        min_leaf: int = 80,
        smooth: float = 0.08,
    ) -> None:
        self.aggregator = aggregator
        self.n_rules = n_rules
        self.n_planes = n_planes
        self.steps = steps
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.mask_l1 = mask_l1
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.val_fraction = val_fraction
        self.kappa_gating = kappa_gating
        self.n_rounds = n_rounds
        self.n_trees = n_trees
        self.min_leaf = min_leaf
        self.smooth = smooth

    def _resolve_device(self):
        import torch
        return torch.device(self.device) if self.device else default_device()

    def _to_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        return pd.DataFrame(X)

    def _build_config(self) -> TrainConfig:
        return TrainConfig(
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            patience=self.patience,
            mask_l1=self.mask_l1,
            device=self._resolve_device(),
        )

    def _closed_name(self) -> str | None:
        name = normalize_aggregator_name(self.aggregator)
        return name if name in CLOSED_FORM else None

    def _build_closed(self, task: str):
        from efm.rulebook import (
            AffineRuleClassifier,
            AffineRuleRegressor,
            PiecewiseGAMClassifier,
            PiecewiseGAMRegressor,
            ResidualRuleClassifier,
            ResidualRuleRegressor,
            SplitLinearClassifier,
            SplitLinearRegressor,
        )

        name = self._closed_name()
        seed = self.random_state
        if name == "affine":
            if task == "regression":
                return AffineRuleRegressor(
                    n_trees=self.n_trees, min_leaf=self.min_leaf, random_state=seed
                )
            return AffineRuleClassifier(
                n_trees=self.n_trees, min_leaf=min(self.min_leaf, 40), random_state=seed
            )
        if name == "residual":
            if task == "regression":
                return ResidualRuleRegressor(
                    n_rounds=self.n_rounds,
                    n_trees=self.n_trees,
                    smooth=self.smooth,
                    random_state=seed,
                )
            return ResidualRuleClassifier(
                n_rounds=self.n_rounds,
                n_trees=min(self.n_trees, 8),
                random_state=seed,
            )
        if name == "piecewise":
            cls = PiecewiseGAMRegressor if task == "regression" else PiecewiseGAMClassifier
            return cls(random_state=seed)
        if name == "split_linear":
            cls = SplitLinearRegressor if task == "regression" else SplitLinearClassifier
            return cls(random_state=seed)
        raise ValueError(name)

    def _fit_closed(self, X_t, y, task: str):
        self.rulebook_ = self._build_closed(task)
        self.rulebook_.fit(X_t, y)
        return self

    def _build_module(self, input_dim: int, n_classes: int) -> EFM:
        normalize_aggregator_name(self.aggregator)
        return EFM(
            input_dim=input_dim,
            n_rules=self.n_rules,
            n_classes=n_classes,
            aggregator=self.aggregator,
            n_planes=self.n_planes,
            steps=self.steps,
            kappa_gating=self.kappa_gating,
        )

    def explain(self, feature_names=None, top_k: int = 10, mask_threshold=None) -> str:
        check_is_fitted(self)
        names = feature_names or getattr(self, "feature_names_in_", None)
        if hasattr(self, "rulebook_"):
            return self.rulebook_.explain(feature_names=names, max_rules=top_k)
        return RuleExplainer(self.module_, names).format_rules(
            top_k=top_k, mask_threshold=mask_threshold
        )

    def plot_rules(self, feature_names=None, top_k: int = 10):
        check_is_fitted(self)
        if hasattr(self, "rulebook_"):
            raise AttributeError(
                "plot_rules is defined for differentiable aggregators; "
                "closed-form books use explain()."
            )
        names = feature_names or self.feature_names_in_
        return RuleExplainer(self.module_, names).plot_structure(top_k=top_k)

class EFMRegressor(_EFMBase, RegressorMixin):
    def fit(self, X, y, X_val=None, y_val=None) -> EFMRegressor:
        if self.random_state is not None:
            set_seed(self.random_state)

        X_df = self._to_df(X)
        y_arr = np.asarray(y, dtype=np.float64)

        self.preprocessor_ = _build_preprocessor(X_df)

        if X_val is None:
            from sklearn.model_selection import train_test_split
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_df, y_arr, test_size=self.val_fraction, random_state=self.random_state
            )
        else:
            X_tr, X_va = X_df, self._to_df(X_val)
            y_tr, y_va = y_arr, np.asarray(y_val, dtype=np.float64)

        X_tr_t = self.preprocessor_.fit_transform(X_tr).astype(np.float32)
        X_va_t = self.preprocessor_.transform(X_va).astype(np.float32)
        self.feature_names_in_: list[str] = list(X_df.columns)

        if self._closed_name() is not None:
            X_all = self.preprocessor_.transform(X_df).astype(np.float32)
            return self._fit_closed(X_all, y_arr, task="regression")

        self.y_scaler_ = StandardScaler()
        y_tr_s = self.y_scaler_.fit_transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32)
        y_va_s = self.y_scaler_.transform(y_va.reshape(-1, 1)).ravel().astype(np.float32)

        self.module_ = self._build_module(X_tr_t.shape[1], n_classes=1)
        self.module_.init_from_data(X_tr_t)

        self.trainer_ = EFMTrainer(self.module_, self._build_config())
        self.history_ = self.trainer_.fit(X_tr_t, y_tr_s, X_va_t, y_va_s, task="regression")
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X_t = self.preprocessor_.transform(self._to_df(X)).astype(np.float32)
        if hasattr(self, "rulebook_"):
            return self.rulebook_.predict(X_t)
        return self.trainer_.predict(X_t, task="regression", y_scaler=self.y_scaler_)

class EFMClassifier(_EFMBase, ClassifierMixin):
    def fit(self, X, y, X_val=None, y_val=None) -> EFMClassifier:
        if self.random_state is not None:
            set_seed(self.random_state)

        X_df = self._to_df(X)

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = int(len(self.classes_))

        self.preprocessor_ = _build_preprocessor(X_df)

        if X_val is None:
            from sklearn.model_selection import train_test_split
            try:
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X_df, y_enc, test_size=self.val_fraction,
                    random_state=self.random_state, stratify=y_enc,
                )
            except ValueError:
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X_df, y_enc, test_size=self.val_fraction,
                    random_state=self.random_state,
                )
        else:
            X_tr, X_va = X_df, self._to_df(X_val)
            y_tr = y_enc
            y_va = self.label_encoder_.transform(y_val)

        X_tr_t = self.preprocessor_.fit_transform(X_tr).astype(np.float32)
        X_va_t = self.preprocessor_.transform(X_va).astype(np.float32)
        self.feature_names_in_: list[str] = list(X_df.columns)

        if self._closed_name() is not None:
            if n_classes > 2 and self._closed_name() in ("affine", "residual"):
                raise ValueError(
                    "aggregator='affine' and 'residual' currently support binary "
                    "classification only; use a differentiable aggregator or "
                    "'piecewise' / 'split_linear'."
                )
            X_all = self.preprocessor_.transform(X_df).astype(np.float32)
            return self._fit_closed(X_all, y_enc, task="classification")

        self.module_ = self._build_module(X_tr_t.shape[1], n_classes=n_classes)
        self.module_.init_from_data(X_tr_t)

        self.trainer_ = EFMTrainer(self.module_, self._build_config())
        self.history_ = self.trainer_.fit(
            X_tr_t, y_tr.astype(np.int64), X_va_t, y_va.astype(np.int64),
            task="classification",
        )
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X_t = self.preprocessor_.transform(self._to_df(X)).astype(np.float32)
        if hasattr(self, "rulebook_"):
            idx = self.rulebook_.predict(X_t).astype(int)
        else:
            idx = self.trainer_.predict(X_t, task="classification")
        return self.label_encoder_.inverse_transform(idx.astype(int))

    def predict_proba(self, X) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        check_is_fitted(self)
        X_t = self.preprocessor_.transform(self._to_df(X)).astype(np.float32)
        if hasattr(self, "rulebook_"):
            return self.rulebook_.predict_proba(X_t)
        logits = self.trainer_.predict_proba(X_t)
        return F.softmax(torch.tensor(logits), dim=1).numpy()
