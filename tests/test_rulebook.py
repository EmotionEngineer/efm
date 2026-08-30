from __future__ import annotations

import numpy as np
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split

from efm import EFMClassifier, EFMRegressor, normalize_aggregator_name
from efm.aggregators import CLOSED_FORM
from efm.utils import macro_f1, rmse


def _reg_split():
    b = load_diabetes()
    return train_test_split(b.data, b.target, test_size=0.3, random_state=0)


def _clf_split():
    b = load_wine()
    y = (b.target == 0).astype(int)
    return train_test_split(b.data, y, test_size=0.3, random_state=0, stratify=y)


def test_closed_form_aliases():
    assert normalize_aggregator_name("residual") in CLOSED_FORM
    assert normalize_aggregator_name("RT-Post") == "residual"
    assert normalize_aggregator_name("pwl") == "piecewise"
    assert normalize_aggregator_name("twolaw") == "split_linear"


def test_residual_regressor_rounds():
    X_tr, X_te, y_tr, y_te = _reg_split()
    m2 = EFMRegressor(aggregator="residual", n_rounds=2, n_trees=4, random_state=0)
    m3 = EFMRegressor(aggregator="residual", n_rounds=3, n_trees=4, random_state=0)
    a = m2.fit(X_tr, y_tr).predict(X_te)
    b = m3.fit(X_tr, y_tr).predict(X_te)
    assert a.shape == b.shape == y_te.shape
    assert np.isfinite(rmse(y_te, a))
    assert "n_rounds=2" in m2.explain()


def test_affine_and_piecewise_and_split():
    X_tr, X_te, y_tr, y_te = _reg_split()
    for agg in ("affine", "piecewise", "split_linear"):
        m = EFMRegressor(aggregator=agg, n_trees=4, min_leaf=20, random_state=0)
        pred = m.fit(X_tr, y_tr).predict(X_te)
        assert np.isfinite(rmse(y_te, pred))
        assert m.explain()


def test_piecewise_classifier_proba():
    X_tr, X_te, y_tr, y_te = _clf_split()
    m = EFMClassifier(aggregator="piecewise", random_state=0)
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-5)
    assert 0 <= macro_f1(y_te, m.predict(X_te)) <= 1


def test_residual_classifier_binary():
    X_tr, X_te, y_tr, y_te = _clf_split()
    m = EFMClassifier(aggregator="residual", n_rounds=2, n_trees=4, random_state=0)
    m.fit(X_tr, y_tr)
    assert m.predict(X_te).shape == y_te.shape
    assert m.predict_proba(X_te).shape[1] == 2
