# Getting Started

This tutorial walks you through sklearn-meta's core workflows, starting simple and building up to stacking ensembles. By the end you'll know which tool to reach for and when.

---

## Installation

```bash
pip install -e .
```

Optional boosting libraries:

```bash
pip install xgboost lightgbm catboost
```

---

## Sample Data

All examples below use this setup:

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
```

---

## 1. Tune a Model

`tune()` finds the best hyperparameters for a single model using cross-validated Bayesian optimization (Optuna):

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn_meta import tune

result = tune(
    RandomForestClassifier,
    X_train, y_train,
    params={
        "n_estimators": (50, 500),
        "max_depth": (3, 20),
    },
    fixed_params={"random_state": 42},
    n_trials=50,
    metric="roc_auc",
)

print(result.best_params_)    # {'n_estimators': 312, 'max_depth': 8}
print(result.best_score_)     # 0.9523
```

The `params` dict defines the search space. Values can be:

| Format | Meaning | Example |
|--------|---------|---------|
| `(low, high)` | Numeric range (int if both ints, float otherwise) | `(50, 500)` |
| `(low, high, {"log": True})` | Log-scaled range | `(0.01, 0.3, {"log": True})` |
| `[a, b, c]` | Categorical choices | `["sqrt", "log2"]` |

`tune()` returns a `TrainingRun` -- see [Inspecting Results](#5-inspecting-results) below.

---

## 2. Cross-Validate Without Tuning

`cross_validate()` evaluates a model with fixed parameters. No hyperparameter search, just CV scoring and out-of-fold predictions:

```python
from sklearn_meta import cross_validate

result = cross_validate(
    RandomForestClassifier,
    X_train, y_train,
    fixed_params={"n_estimators": 200, "max_depth": 10, "random_state": 42},
    metric="roc_auc",
)

print(result.best_score_)        # mean CV score
print(result.oof_predictions_)   # out-of-fold predictions for every training sample
```

This is useful for getting an unbiased score for a known configuration, or generating OOF predictions for downstream use.

---

## 3. Compare Models

`compare()` tunes multiple models and ranks them:

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_meta import compare

result = compare(
    {
        "rf": (RandomForestClassifier, {"n_estimators": (50, 500), "max_depth": (3, 20)}),
        "gbm": (GradientBoostingClassifier, {"learning_rate": (0.01, 0.3, {"log": True})}),
        "lr": LogisticRegression,  # no tuning, just CV with defaults
    },
    X_train, y_train,
    metric="roc_auc",
    n_trials=50,
)

print(result.leaderboard)   # sorted DataFrame: model, score, time
print(result.best_name)     # "gbm"
result.best_run.predict(X_test)
```

Models without a `params` dict are cross-validated with their defaults (no tuning). You can also pass estimator instances: `"lr": LogisticRegression(C=0.5)`.

---

## 4. Build a Stacking Ensemble

`stack()` trains base models and a meta-learner that combines their predictions, with automatic out-of-fold handling to prevent data leakage:

```python
from sklearn_meta import stack

result = stack(
    base_models={
        "rf": (RandomForestClassifier, {"n_estimators": (50, 500)}),
        "gbm": (GradientBoostingClassifier, {"learning_rate": (0.01, 0.3, {"log": True})}),
    },
    meta_model=LogisticRegression,
    X=X_train, y=y_train,
    metric="roc_auc",
    n_trials=50,
)

predictions = result.predict(X_test)
print(result.best_score_)   # meta-learner's CV score
```

Base models are tuned first, then their out-of-fold predictions become features for the meta-learner. For classifier stacks, `stack()` uses base-model probabilities by default when they are available; pass `stack_output="prediction"` to stack class labels instead. See [Stacking](stacking.md) for details on how this works and when to use it.

---

## 5. Inspecting Results

All four helpers return a `TrainingRun` (or `ComparisonResult` for `compare()`). Here's what you can do with a `TrainingRun`:

```python
# Predictions
predictions = result.predict(X_test)
# For classifiers, predict() returns final class labels

# Best hyperparameters and score
result.best_params_
result.best_score_

# Out-of-fold predictions (every training sample predicted by a model that never saw it)
result.oof_predictions_

# Feature importances (for models that support it)
result.feature_importances_

# Formatted summary
print(result.summary())
```

For multi-model graphs (stacking), these shortcuts resolve to the leaf node (meta-learner). Access individual models via `result.node_results`:

```python
rf_result = result.node_results["rf"]
rf_result.best_params
rf_result.mean_score
rf_result.oof_predictions
rf_result.models  # list of fitted models, one per fold
```

---

## 6. Saving and Loading

Save a full training run (fold models, OOF predictions, config):

```python
result.save("./my_run")

from sklearn_meta import TrainingRun
restored = TrainingRun.load("./my_run")
```

For deployment, save just the lightweight inference graph:

```python
inference = result.compile_inference()
inference.save("./my_model")

from sklearn_meta import InferenceGraph
loaded = InferenceGraph.load("./my_model")
predictions = loaded.predict(X_test)
```

---

## 7. When You Need More Control

The convenience helpers cover the most common workflows. When you need custom graph architectures, feature selection, reparameterization, or plugins, use the explicit API:

```python
from sklearn_meta import GraphBuilder, RunConfigBuilder, fit

# 1. Define the model graph
graph = (
    GraphBuilder("my_pipeline")
    .add_model("rf", RandomForestClassifier)
        .int_param("n_estimators", 50, 500)
        .int_param("max_depth", 3, 20)
        .fixed_params(random_state=42)
    .build()
)

# 2. Configure the run
config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=50, metric="roc_auc")
    .feature_selection(method="shadow")         # not available via tune()
    .reparameterization(enabled=True)           # not available via tune()
    .build()
)

# 3. Fit
result = fit(graph, X_train, y_train, config)
```

This produces the same `TrainingRun` as the convenience helpers. The explicit API gives you access to:

- **Feature selection** -- shadow features, permutation importance ([docs](feature-selection.md))
- **Reparameterization** -- orthogonal parameter transforms for faster convergence ([docs](reparameterization.md))
- **Custom graph architectures** -- multi-level stacking, distillation, transforms ([docs](model-graphs.md))
- **Plugins** -- model-specific hooks for XGBoost, early stopping, etc. ([docs](plugins.md))
- **Estimator scaling** -- tune with fewer trees, train with more ([docs](tuning.md#estimator-scaling))

### DataView

The convenience helpers accept raw `X`/`y` arrays. The explicit API does too (via `fit(graph, X, y, config)`), but for advanced use cases you can construct a `DataView` directly:

```python
from sklearn_meta import DataView

# Basic usage
data = DataView.from_Xy(X_train, y_train)

# With groups for group CV
data = DataView.from_Xy(X_train, y_train, groups=patient_ids)

# With auxiliary channels (e.g., base margins for XGBoost)
data = DataView.from_Xy(X_train, y_train, base_margin=margin_array)
```

`DataView` is immutable -- every transformation returns a new instance:

```python
# Restrict to specific features
data2 = data.select_features(["feat_1", "feat_3"])

# Subset to specific rows
data3 = data.select_rows(train_indices)

# Add overlay columns (e.g., upstream OOF predictions for stacking)
data4 = data.with_overlay("pred_rf", rf_oof_predictions)
data5 = data.with_overlays({"pred_rf": rf_preds, "pred_xgb": xgb_preds})

# Materialize into concrete arrays for inspection
batch = data.materialize()
batch.X              # Feature DataFrame
batch.y              # Target array
batch.feature_names  # List of feature column names
```

You generally don't need `DataView` directly -- `fit()` and the convenience helpers handle data wrapping internally. It's useful when you need feature subsetting, overlays, or auxiliary channels.

### Choosing Between Helpers and the Explicit API

| Use case | Approach |
|----------|----------|
| Tune one model | `tune()` |
| Evaluate a fixed config | `cross_validate()` |
| Compare several models | `compare()` |
| Simple two-level stacking | `stack()` |
| Feature selection or reparameterization | `GraphBuilder` + `RunConfigBuilder` |
| Multi-level stacking or custom DAGs | `GraphBuilder` + `RunConfigBuilder` |
| Plugins (XGBoost, early stopping) | `GraphBuilder` + `RunConfigBuilder` |

---

## What's Next

- **[Tuning](tuning.md)** -- Metrics, early stopping, estimator scaling, and more
- **[Stacking](stacking.md)** -- How stacking works and building custom architectures
- **[Feature Selection](feature-selection.md)** -- Automated feature pruning
- **[Search Spaces](search-spaces.md)** -- Advanced parameter definitions
