# EFM — Explainable Fuzzy Machines

**EFM** is a lightweight PyTorch and scikit-learn library for tabular machine learning.

The public API is two estimators. A string `aggregator` selects the internals: differentiable fuzzy rules (GPU, gradient descent) or a closed-form rule book (CPU, spectral-floored ridge). Both print `IF-THEN` logic.

```python
from efm import EFMRegressor, EFMClassifier

EFMRegressor(aggregator="gaussian")          # Gaussian template matching
EFMRegressor(aggregator="student_t")         # Heavy-tailed templates
EFMRegressor(aggregator="residual")          # Affine leaves + residual CART
EFMRegressor(aggregator="residual", n_rounds=3)
EFMClassifier(aggregator="hyperplane")       # Piecewise-linear evidence
EFMClassifier(aggregator="piecewise")        # Two-piece linear GAM
EFMClassifier(aggregator="state_coupled")    # Recurrent rule state
```

---

## Why EFM?

Tree ensembles are accurate and opaque. GAMs such as EBM give shape plots, not thresholds. EFM learns explicit, human-readable rules.

> **Ground truth:** `IF x0 > 0.5 AND x1 < 0 THEN +3.0 …`

**EBM** — importances and pairwise heatmaps, no symbolic cuts.

```text
x0 (0.38) | x1 (0.36) | x0 × x1 (0.26)
```

**RuleFit** — thresholds, but dense overlapping trees.

```text
x0 <= 0.171 and x1 > 0.690 | x0 > 1.661 and x1 > -0.191 and x4 <= 0.859
```

**EFM** — sparse literals after L1 masking (differentiable) or a jointly refit affine book (closed-form).

```text
Rule 1: x0 > +0.460 AND x1 < -0.021
Rule 2: x2 > +1.035 AND x5 < +0.249
```

---

## Differentiable path

Fuzzy aggregators (`student_t`, `gaussian`, `hyperplane`, `state_coupled`) train an `nn.Module`:

```text
x
│
├─ fuzzy rule layer     thresholds, inequality signs, feature masks
├─ log-margin tensor
├─ evidence aggregator
├─ rule firing strengths
└─ linear head          regression value or class logits
```

Training uses AdamW, a validation split, and early stopping. Typical fit time is seconds on a GPU.

---

## Closed-form path

These aggregators skip PyTorch. Coefficients are a single spectral-floored ridge (SFR) fit.

| `aggregator` | What you get |
| --- | --- |
| `"affine"` | Boosted affine rectangles, jointly refit |
| `"residual"` | Affine book + residual CART × `n_rounds` + path smoothing |
| `"piecewise"` | Additive two-piece linear GAM |
| `"split_linear"` | `IF vᵀx ≤ t THEN affine ELSE affine` |

`n_rounds=2` (default) or `3` is the only extra knob for `"residual"`. Prefer `"piecewise"` on additive classification; `"residual"` / `"affine"` on region-heavy tables. Do not mix `"piecewise"` and `"residual"` in one coefficient vector.

---

## Aggregators

| Aggregator | Kind | Idea |
| --- | --- | --- |
| `"student_t"` | fuzzy | Heavy-tailed template matching in log-margin space |
| `"gaussian"` | fuzzy | Gaussian template matching |
| `"hyperplane"` | fuzzy | Sum of learned ReLU hyperplanes |
| `"state_coupled"` | fuzzy | Short linear recurrence over rule state |
| `"affine"` | closed | Jointly refit affine rectangles |
| `"residual"` | closed | Residual CART on an affine book |
| `"piecewise"` | closed | Two linear pieces per coordinate |
| `"split_linear"` | closed | One split direction, two affines |

Aliases (`"SMTE"`, `"GMTE"`, `"HYP"`, `"SC"`) still resolve to the fuzzy names.

---

## Installation

```bash
pip install efm
```

From a checkout:

```bash
git clone https://github.com/EmotionEngineer/efm.git
cd efm
pip install -e .
pip install -e ".[dev]"   # pytest, ruff, mypy
pytest tests -v
```

---

## Quick start: regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from efm import EFMRegressor

data = load_diabetes(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=0,
)

model = EFMRegressor(
    aggregator="student_t",
    n_rules=64,
    epochs=200,
    random_state=0,
)
model.fit(X_train, y_train)
print(model.predict(X_test)[:5])
print(model.explain(top_k=5))
```

No GPU, no epoch loop:

```python
model = EFMRegressor(aggregator="residual", n_rounds=2, random_state=0)
model.fit(X_train, y_train)
print(model.explain(top_k=8))
```

---

## Quick start: classification

`kappa_gating=True` scales the steepness of each fuzzy boundary so L1 masking can drop noise features instead of cancelling inside the log-ratio.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from efm import EFMClassifier

X_raw, y_raw = make_classification(
    n_samples=1000, n_features=20, n_informative=4, n_redundant=0, random_state=0,
)
X = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(20)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_raw, test_size=0.2, stratify=y_raw, random_state=0,
)

model = EFMClassifier(
    aggregator="student_t",
    n_rules=32,
    epochs=150,
    mask_l1=1e-3,
    kappa_gating=True,
    random_state=0,
)
model.fit(X_train, y_train)
print(model.predict(X_test)[:5])
print(model.explain(top_k=3, mask_threshold=0.10))
```

Additive closed-form GAM:

```python
model = EFMClassifier(aggregator="piecewise", random_state=0)
model.fit(X_train, y_train)
print(model.explain())
```

---

## Bare PyTorch

Sklearn wrappers handle preprocessing, a validation split, and target scaling. The module itself is `EFM` (fuzzy aggregators only).

```python
import torch
from efm import EFM

model = EFM(
    input_dim=10,
    n_rules=32,
    n_classes=3,
    aggregator="student_t",
    kappa_gating=True,
)
x = torch.randn(8, 10)
y = torch.randint(0, 3, size=(8,))
loss = model.loss_batch(x, y)
loss.backward()
```

---

## Explanations

```python
print(model.explain(top_k=10, mask_threshold=0.10))
fig = model.plot_rules(top_k=10)          # fuzzy aggregators
fig.savefig("rules.png", dpi=150)
```

```text
Rule #01  [id=12, importance=0.8342, n_conds=3, ν=3.91]
    bmi                            > +0.418  [ev=+, mask=0.62]
    bp                             > -0.055  [ev=+, mask=0.44]
    s5                             > +0.276  [ev=−, mask=0.31]
```

- Thresholds are in the **transformed** space (standardized numerics, one-hot categoricals).
- Pass `model.preprocessor_.get_feature_names_out()` if you need those names explicitly.
- Closed-form aggregators implement `explain()` only (`plot_rules` is fuzzy-specific).

---

## Parameters

Shared:

| Parameter | Description |
| --- | --- |
| `aggregator` | See the table above |
| `random_state` | Seed |
| `val_fraction` | Hold-out fraction when `X_val` is omitted |

Fuzzy:

| Parameter | Description |
| --- | --- |
| `n_rules` | Number of fuzzy rules |
| `n_planes` | Planes per rule (`hyperplane`) |
| `steps` | Recurrence depth (`state_coupled`) |
| `epochs`, `lr`, `batch_size`, `patience` | AdamW training |
| `mask_l1`, `weight_decay` | Sparsity and decay |
| `device` | `"cpu"`, `"cuda"`, or `None` (auto) |
| `kappa_gating` | Steeper literals so masks prune noise. Default `False` |

Closed-form:

| Parameter | Description |
| --- | --- |
| `n_rounds` | Residual CART rounds for `"residual"` (`2` or `3`) |
| `n_trees` | Affine boosting rounds (`"affine"`, `"residual"`) |
| `smooth` | Path-membership temperature (`"residual"`, default `0.08`) |
| `min_leaf` | Minimum samples per affine leaf |

---

## Validation split

```python
model.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)
```

If `X_val` is omitted, a split of size `val_fraction` is taken from the training data (fuzzy path). Closed-form books fit on the full training matrix.

---

## Saving

```python
import joblib
joblib.dump(model, "efm_model.joblib")
```

Fuzzy module weights:

```python
torch.save(model.module_.state_dict(), "efm_state.pt")
```

---

## Development

```bash
pytest tests -v
ruff check efm tests
python -m build
```

---

## Limitations

- With `kappa_gating=False`, masks sit inside the fuzzy subset and can train slowly on small data. Use `kappa_gating=True` when feature selection must be sharp.
- High-cardinality categoricals expand into many one-hot columns.
- Explanation thresholds are post-preprocessing.
- `"affine"` and `"residual"` classifiers are **binary** in the library. `"piecewise"` and `"split_linear"` classifiers are binary too (`ŷ = σ(·)` vs the max label). For multiclass use a fuzzy aggregator, or wrap a closed book in one-vs-rest (see `examples/synthetic.ipynb`).

---

## License

MIT.
