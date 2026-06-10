from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split

from efm import (
    AGGREGATORS,
    EFM,
    EFMClassifier,
    EFMRegressor,
    RuleExplainer,
    build_aggregator,
    normalize_aggregator_name,
    set_seed,
)
from efm.aggregators import StudentTTemplate
from efm.primitives import AxisAlignedMarginRules, log_margin
from efm.trainer import EFMTrainer, TrainConfig
from efm.utils import macro_f1, rmse

AGG_NAMES = ["gaussian", "student_t", "hyperplane", "state_coupled"]
ALIASES = {"GMTE": "gaussian", "SMTE": "student_t", "HYP": "hyperplane", "SC": "state_coupled"}

@pytest.fixture(scope="module")
def regression_data():
    b = load_diabetes(as_frame=True)
    X = b.data.to_numpy(dtype=np.float32)
    y = b.target.to_numpy(dtype=np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=0)

@pytest.fixture(scope="module")
def wine_data():
    b = load_wine(as_frame=True)
    return b.data, b.target

class TestLogMargin:
    def test_shape_preserved(self):
        u, v = torch.rand(8, 16, 10), torch.rand(8, 16, 10)
        assert log_margin(u, v).shape == u.shape

    def test_symmetric_boundary(self):
        t = torch.full((4,), 0.5)
        assert torch.allclose(log_margin(t, t), torch.zeros(4), atol=1e-4)

class TestRuleLayer:
    def test_firing_range(self):
        layer = AxisAlignedMarginRules(input_dim=5, n_rules=8)
        z = layer.fire(layer.margin(torch.randn(16, 5)).sum(-1))
        assert z.shape == (16, 8) and z.min() > 0 and z.max() < 1

    def test_margin_shape(self):
        layer = AxisAlignedMarginRules(input_dim=4, n_rules=6)
        m = layer.margin(torch.randn(10, 4))
        assert m.shape == (10, 6, 4)

class TestAggregatorRegistry:
    @pytest.mark.parametrize("name", AGG_NAMES)
    def test_all_registered(self, name):
        assert name in AGGREGATORS

    @pytest.mark.parametrize("alias,canonical", list(ALIASES.items()))
    def test_alias_normalization(self, alias, canonical):
        assert normalize_aggregator_name(alias) == canonical
        assert normalize_aggregator_name(alias.lower()) == canonical

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            normalize_aggregator_name("does_not_exist")

    @pytest.mark.parametrize("name", AGG_NAMES)
    def test_builder_output_shape(self, name):
        agg = build_aggregator(name, input_dim=8, n_rules=16)
        m = torch.randn(4, 16, 8)
        assert agg(m).shape == (4, 16)

    def test_builder_passes_instance_through(self):
        inst = StudentTTemplate(8, 16)
        assert build_aggregator(inst, 8, 16) is inst

    def test_builder_forwards_only_relevant_kwargs(self):
        agg = build_aggregator("gaussian", 8, 16, n_planes=99, steps=99)
        assert agg(torch.randn(2, 16, 8)).shape == (2, 16)

@pytest.mark.parametrize("agg", AGG_NAMES)
class TestUnifiedEFM:
    B, D, R = 32, 8, 16

    def test_regression_shape(self, agg):
        model = EFM(self.D, n_rules=self.R, n_classes=1, aggregator=agg)
        assert model(torch.randn(self.B, self.D)).shape == (self.B, 1)

    def test_classification_shape(self, agg):
        model = EFM(self.D, n_rules=self.R, n_classes=3, aggregator=agg)
        assert model(torch.randn(self.B, self.D)).shape == (self.B, 3)

    def test_aggregator_name(self, agg):
        model = EFM(self.D, n_rules=self.R, aggregator=agg)
        assert model.aggregator_name == agg

    def test_alias_construction(self, agg):
        alias = {v: k for k, v in ALIASES.items()}[agg]
        model = EFM(self.D, n_rules=self.R, aggregator=alias)
        assert model.aggregator_name == agg

    def test_loss_scalar_and_grads(self, agg):
        model = EFM(self.D, n_rules=self.R, n_classes=1, aggregator=agg)
        loss = model.loss_batch(torch.randn(self.B, self.D), torch.randn(self.B))
        assert loss.ndim == 0 and not torch.isnan(loss)
        loss.backward()
        for n, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {n}"

    def test_init_from_data(self, agg):
        X = np.random.randn(80, self.D).astype(np.float32)
        EFM(self.D, n_rules=self.R, aggregator=agg).init_from_data(X)

def test_student_t_extra_kwargs():
    model = EFM(6, n_rules=8, aggregator="student_t", nu_init=10.0)
    assert (model.aggregator.nu > 0).all()

def test_hyperplane_n_planes():
    model = EFM(6, n_rules=8, aggregator="hyperplane", n_planes=4)
    assert model.aggregator.n_planes == 4

def test_state_coupled_steps():
    model = EFM(6, n_rules=8, aggregator="state_coupled", steps=5)
    assert model.aggregator.steps == 5

class TestTrainer:
    def test_regression_runs(self, regression_data):
        set_seed(0)
        X_tr, X_te, y_tr, y_te = regression_data
        split = int(0.8 * len(X_tr))

        from sklearn.preprocessing import StandardScaler
        ys = StandardScaler()
        y_t = ys.fit_transform(y_tr[:split].reshape(-1, 1)).ravel().astype(np.float32)
        y_v = ys.transform(y_tr[split:].reshape(-1, 1)).ravel().astype(np.float32)

        model = EFM(X_tr.shape[1], n_rules=16, aggregator="student_t")
        model.init_from_data(X_tr[:split])
        trainer = EFMTrainer(model, TrainConfig(epochs=20, patience=10,
                                                batch_size=64, device=torch.device("cpu")))
        hist = trainer.fit(X_tr[:split], y_t, X_tr[split:], y_v, task="regression")
        assert len(hist["val_loss"]) > 0

class TestEFMRegressor:
    @pytest.mark.parametrize("agg", AGG_NAMES)
    def test_fit_predict(self, agg, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        model = EFMRegressor(aggregator=agg, n_rules=16, epochs=20,
                             random_state=0, device="cpu")
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        assert pred.shape == y_te.shape
        assert math.isfinite(rmse(y_te, pred))

    def test_explain(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        model = EFMRegressor(aggregator="student_t", n_rules=8, epochs=5,
                             random_state=0, device="cpu")
        model.fit(X_tr, y_tr)
        report = model.explain(top_k=3)
        assert "Rule #01" in report and "student_t" in report

class TestEFMClassifier:
    @pytest.mark.parametrize("agg", AGG_NAMES)
    def test_fit_predict(self, agg, wine_data):
        X, y = wine_data
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        model = EFMClassifier(aggregator=agg, n_rules=16, epochs=15,
                              random_state=0, device="cpu")
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        assert pred.shape == y_te.shape
        assert 0 <= macro_f1(y_te, pred) <= 1

    def test_predict_proba_sums_to_one(self, wine_data):
        X, y = wine_data
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        model = EFMClassifier(aggregator="gaussian", n_rules=8, epochs=5,
                              random_state=0, device="cpu")
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_classes_preserved(self, wine_data):
        X, y = wine_data
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        model = EFMClassifier(aggregator="hyperplane", n_rules=8, epochs=5,
                              random_state=0, device="cpu")
        model.fit(X_tr, y_tr)
        assert set(np.unique(model.predict(X_te))).issubset(set(np.unique(y_te)))

class TestRuleExplainer:
    def test_get_rules_student_t_has_nu(self, regression_data):
        X_tr, _, _, _ = regression_data
        model = EFM(X_tr.shape[1], n_rules=8, aggregator="student_t")
        model.init_from_data(X_tr)
        names = [f"f{j}" for j in range(X_tr.shape[1])]
        rules = RuleExplainer(model, names).get_rules(top_k=5)
        assert len(rules) <= 5 and "nu" in rules[0]

    def test_get_rules_gaussian_no_nu(self, regression_data):
        X_tr, _, _, _ = regression_data
        model = EFM(X_tr.shape[1], n_rules=8, aggregator="gaussian")
        model.init_from_data(X_tr)
        names = [f"f{j}" for j in range(X_tr.shape[1])]
        rules = RuleExplainer(model, names).get_rules(top_k=5)
        assert "nu" not in rules[0]

    def test_plot_returns_figure(self, regression_data):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure
        X_tr, _, _, _ = regression_data
        model = EFM(X_tr.shape[1], n_rules=8, aggregator="state_coupled")
        model.init_from_data(X_tr)
        names = [f"f{j}" for j in range(X_tr.shape[1])]
        fig = RuleExplainer(model, names).plot_structure(top_k=4)
        assert isinstance(fig, matplotlib.figure.Figure)
