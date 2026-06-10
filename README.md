# EFM — Explainable Fuzzy Machines

**EFM** is a lightweight PyTorch and scikit-learn library for tabular machine learning, built entirely from differentiable fuzzy rules.

While tree ensembles act as black boxes, and GAMs (like EBM) rely on additive shape functions, EFM uses gradient descent to learn explicit, human-readable `IF-THEN` rules. The fitted model exposes its internal logic—including explicit thresholds, feature masks, and evidence signs—while training in seconds on a GPU.

```python
from efm import EFMRegressor, EFMClassifier

EFMRegressor(aggregator="gaussian")        # Gaussian template matching
EFMRegressor(aggregator="student_t")       # Heavy-tailed template matching
EFMClassifier(aggregator="hyperplane")     # Piecewise-linear evidence
EFMClassifier(aggregator="state_coupled")  # Recurrent state-coupled evidence
```

---

## Why EFM?

EFM occupies a unique space on the Pareto frontier of accuracy and interpretability. To illustrate, here is how different explainable models attempt to recover a known synthetic ground-truth rule:

> **Ground Truth:** `IF x0 > 0.5 AND x1 < 0 THEN +3.0 ...`

**1. Explainable Boosting Machines (EBM)**
EBM provides excellent predictive accuracy by learning additive shape functions, but it does not produce symbolic logic or thresholds.
```text
# EBM Output (Feature Importances & Pairs)
x0 (0.38) | x1 (0.36) | x0 × x1 (0.26)
```

**2. RuleFit**
RuleFit extracts rules from shallow decision trees. While it provides thresholds, the resulting rules are often highly overlapping, dense, visually overwhelming, and slow to train.
```text
# RuleFit Output (Messy, overlapping rules)
x0 <= 0.171 and x1 > 0.690 | x0 > 1.661 and x1 > -0.191 and x4 <= 0.859 | x4
```

**3. EFM (Explainable Fuzzy Machines)**
EFM uses L1-masked gradient descent to prune irrelevant literals, resulting in clean, sparse, and mutually exclusive fuzzy rules.
```text
# EFM Output (Clean, sparse, readable thresholds)
Rule 1: x0 > +0.460 AND x1 < -0.021
Rule 2: x2 > +1.035 AND x5 < +0.249
```

---

## Model Sketch

An EFM model has three main parts:

1. **Axis-aligned fuzzy literals**
   Each rule compares each feature against a learned threshold using a soft inequality.
2. **Evidence aggregation**
   The per-feature log margins are reduced to one evidence score per rule.
3. **Linear prediction head**
   Rule firing strengths are passed to a linear head for regression or classification.

In rough form:

```text
x
│
├─ fuzzy rule layer
│     feature thresholds
│     inequality directions
│     feature masks
│
├─ log-margin tensor
│
├─ evidence aggregator
│     gaussian | student_t | hyperplane | state_coupled
│
├─ rule firing strengths
│
└─ linear output head
      regression value or class logits
```

---

## Aggregators

| Aggregator | Legacy alias | Main idea |
| --- | --- | --- |
| `"student_t"` | `"SMTE"` | Heavy-tailed template matching in log-margin space |
| `"gaussian"` | `"GMTE"` | Gaussian template matching in log-margin space |
| `"hyperplane"` | `"HYP"` | Sum of learned ReLU hyperplane responses |
| `"state_coupled"` | `"SC"` | A short learned linear recurrence over rule state |

The same `EFM` class is used in all cases. Only the aggregator changes.

---

## Installation

Install the latest stable release directly from PyPI:

```bash
pip install efm
```

**From Source:**
If you want to install from a local checkout:
```bash
git clone https://github.com/EmotionEngineer/efm.git
cd efm
pip install -e .
```

For development (includes testing tools):
```bash
pip install -e ".[dev]"
pytest tests -v
```

---

## Quick Start: Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from efm import EFMRegressor

data = load_diabetes(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=0,
)

model = EFMRegressor(
    aggregator="student_t",
    n_rules=64,
    epochs=200,
    random_state=0,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print(pred[:5])
print(model.explain(top_k=5))
```

---

## Quick Start: Classification

By setting `kappa_gating=True`, the model scales the steepness ($\kappa$) of the fuzzy boundary directly. This prevents mask parameters from cancelling out inside the log-ratio, allowing the L1 penalty to prune the 16 noise features.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from efm import EFMClassifier

# Generate classification data with 20 features (only 4 are actually useful)
X_raw, y_raw = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=4,
    n_redundant=0,
    random_state=0
)

# Convert to DataFrame to assign clean feature names
feature_names = [f"feat_{i}" for i in range(20)]
X = pd.DataFrame(X_raw, columns=feature_names)
y = pd.Series(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# Initialize and train with Kappa Gating enabled for clean feature pruning
model = EFMClassifier(
    aggregator="student_t",
    n_rules=32,
    epochs=150,
    mask_l1=1e-3,
    kappa_gating=True,  # Enables robust gradient flow for rule-masking
    random_state=0
)

model.fit(X_train, y_train)

labels = model.predict(X_test)
proba = model.predict_proba(X_test)

print(labels[:5])
print(model.explain(top_k=3, mask_threshold=0.10))
```

---

## Bare PyTorch Usage

The scikit-learn estimators handle preprocessing, validation splitting, target scaling, and training. If you want direct access to the module, use `EFM`.

```python
import torch
from efm import EFM

model = EFM(
    input_dim=10,
    n_rules=32,
    n_classes=3,
    aggregator="student_t",
    kappa_gating=True,  # Optional: enable kappa gating for raw PyTorch usage
)

x = torch.randn(8, 10)
y = torch.randint(0, 3, size=(8,))

logits = model(x)
loss = model.loss_batch(x, y)

loss.backward()

print(logits.shape)
```

---

## Explaining a Fitted Model

The scikit-learn wrappers expose two convenience methods:

```python
# Print the top rules as text
print(model.explain(top_k=10, mask_threshold=0.10))

# Plot the internal structures (masks, locations, scales)
fig = model.plot_rules(top_k=10)
fig.savefig("rules.png", dpi=150)
```

The text report lists the most influential rules according to the absolute weights in the output head. Each rule includes active literals, learned thresholds, evidence signs, and mask weights.

Example shape of the output:

```text
Rule #01  [id=12, importance=0.8342, n_conds=3, ν=3.91]
    bmi                            > +0.418  [ev=+, mask=0.62]
    bp                             > -0.055  [ev=+, mask=0.44]
    s5                             > +0.276  [ev=−, mask=0.31]
```

Notes:
- Thresholds are shown in the transformed feature space used by the model.
- Numeric features are standardized by the estimator wrapper.
- Categorical features are one-hot encoded.
- For exact post-transform feature names, pass names from the fitted preprocessor:

```python
names = model.preprocessor_.get_feature_names_out()
print(model.explain(feature_names=list(names), top_k=10))
```

---

## Main Estimator Parameters

| Parameter | Description |
| --- | --- |
| `aggregator` | One of `"student_t"`, `"gaussian"`, `"hyperplane"`, `"state_coupled"` |
| `n_rules` | Number of fuzzy rules |
| `n_planes` | Number of planes per rule for the hyperplane aggregator |
| `steps` | Recurrence depth for the state-coupled aggregator |
| `epochs` | Maximum training epochs |
| `lr` | AdamW learning rate |
| `batch_size` | Mini-batch size |
| `patience` | Early-stopping patience |
| `mask_l1` | L1 penalty on feature masks to encourage sparsity |
| `weight_decay` | AdamW weight decay |
| `random_state` | Seed for reproducibility |
| `device` | `"cpu"`, `"cuda"`, or `None` for automatic selection |
| `val_fraction` | Validation split fraction when no validation set is supplied |
| `kappa_gating` | If `True`, enables Kappa Gating inside literals to bypass log-ratio cancellation, ensuring robust feature selection gradients. Default is `False`. |

---

## Custom Validation Data

```python
model.fit(
    X_train,
    y_train,
    X_val=X_valid,
    y_val=y_valid,
)
```

If no validation set is passed, the estimator creates one internally.

---

## Saving and Loading

The sklearn-style estimators can usually be saved with `joblib`:

```python
import joblib

joblib.dump(model, "efm_model.joblib")
loaded = joblib.load("efm_model.joblib")

pred = loaded.predict(X_test)
```

For lower-level PyTorch workflows, save the module state dict:

```python
torch.save(model.module_.state_dict(), "efm_state.pt")
```

---

## Development

Run the test suite:

```bash
pytest tests -v
```

Run linting:

```bash
ruff check efm tests
```

Build a source and wheel distribution:

```bash
python -m build
```

---

## Current Limitations

EFM is research-oriented code. A few practical details matter:

- Under the default configuration (`kappa_gating=False`), the masks resides inside the fuzzy subset definitions, which can make them slow to train on small datasets due to log-ratio cancellation. For exact, robust feature selection on small or easy datasets, set `kappa_gating=True`.
- High-cardinality categorical columns can create many one-hot features.
- Thresholds in explanations are reported after preprocessing.

---

## License

MIT.
