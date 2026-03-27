# sklearn-meta

A Python library for automated ML pipelines with hyperparameter optimization, model stacking, feature selection, and knowledge distillation.

## Installation

```bash
pip install -e .
```

Optional boosting libraries:

```bash
pip install xgboost lightgbm catboost
```

## Quick Start

Tune a model in a few lines:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn_meta import tune

result = tune(
    RandomForestClassifier,
    X_train, y_train,
    params={"n_estimators": (50, 500), "max_depth": (3, 20)},
    metric="roc_auc",
)
result.best_params_        # best hyperparameters
result.best_score_         # mean CV score
result.predict(X_test)     # predictions from all fold models
```

This runs 100 Optuna trials with 5-fold stratified CV, then returns a `TrainingRun` with fitted fold models ready for inference.

## Common Workflows

### Compare models

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_meta import compare

result = compare(
    {
        "rf": (RandomForestClassifier, {"n_estimators": (50, 500)}),
        "gbm": (GradientBoostingClassifier, {"learning_rate": (0.01, 0.3, {"log": True})}),
        "lr": LogisticRegression,
    },
    X_train, y_train,
    metric="roc_auc",
)
print(result.leaderboard)
```

### Stack models

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
)
result.predict(X_test)
```

### Cross-validate without tuning

```python
from sklearn_meta import cross_validate

result = cross_validate(
    RandomForestClassifier,
    X_train, y_train,
    fixed_params={"n_estimators": 200, "random_state": 42},
    metric="roc_auc",
)
result.best_score_
result.oof_predictions_
```

## When You Need More Control

The convenience helpers (`tune`, `cross_validate`, `stack`, `compare`) cover common workflows. For feature selection, reparameterization, plugins, or custom graph architectures, use the explicit API:

```python
from sklearn_meta import GraphBuilder, RunConfigBuilder, fit

graph = (
    GraphBuilder("my_pipeline")
    .add_model("rf", RandomForestClassifier)
        .int_param("n_estimators", 50, 500)
        .int_param("max_depth", 3, 20)
        .fixed_params(random_state=42)
    .build()
)

config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=100, metric="roc_auc")
    .feature_selection(method="shadow")
    .reparameterization(enabled=True, use_prebaked=True)
    .build()
)

result = fit(graph, X_train, y_train, config)
```

Both approaches return the same `TrainingRun` object.

## Working with Results

```python
# Predictions
result.predict(X_test)
result.predict_proba(X_test)

# Inspection
result.best_params_
result.best_score_
result.oof_predictions_
result.feature_importances_
print(result.summary())

# Per-node results (for multi-model graphs)
result.node_results["rf"].best_params
result.node_results["rf"].mean_score

# Save / load
result.save("./my_run")
from sklearn_meta import TrainingRun
restored = TrainingRun.load("./my_run")

# Lightweight inference-only artifact
inference = result.compile_inference()
inference.save("./my_model")
from sklearn_meta import InferenceGraph
loaded = InferenceGraph.load("./my_model")
```

## Documentation

### Tutorial

1. [Getting Started](docs/getting-started.md) -- From first tune to stacking ensembles
2. [Tuning](docs/tuning.md) -- Hyperparameter optimization in depth
3. [Stacking](docs/stacking.md) -- Multi-model ensembles
4. [Feature Selection](docs/feature-selection.md) -- Automated feature pruning

### Reference

- [Search Spaces](docs/search-spaces.md) -- Parameter types, conditional params, shorthand
- [Cross-Validation](docs/cross-validation.md) -- CV strategies, nested CV, grouped splits
- [Model Graphs](docs/model-graphs.md) -- Custom DAG architectures
- [Reparameterization](docs/reparameterization.md) -- Orthogonal parameter transforms
- [Plugins](docs/plugins.md) -- Model-specific lifecycle hooks
- [API Reference](docs/api-reference.md) -- Complete class and function reference
- [Joint Quantile Regression](docs/joint-quantile-regression.md) -- Multivariate targets with uncertainty

## Upgrading to 0.2

`0.2.0` is intentionally breaking. The old `sklearn_meta.api` and `sklearn_meta.core.*` modules were removed. See [CHANGELOG.md](CHANGELOG.md) for migration guidance.

## Project Structure

```
sklearn_meta/
├── __init__.py            # Public API and fit() function
├── convenience.py         # tune(), cross_validate(), stack(), compare()
├── spec/                  # Graph & node specs (pure data, no runtime)
│   ├── builder.py         # GraphBuilder & NodeBuilder fluent API
│   ├── graph.py           # GraphSpec
│   ├── node.py            # NodeSpec
│   └── dependency.py      # DependencyEdge, DependencyType
├── data/                  # DatasetRecord, DataView, MaterializedBatch
├── runtime/               # RunConfig, CVConfig, TuningConfig, RuntimeServices
├── engine/                # GraphRunner, CVEngine, trainers
├── artifacts/             # TrainingRun, InferenceGraph
├── search/                # SearchSpace, OptunaBackend
├── meta/                  # Reparameterization transforms
├── selection/             # Feature selection (shadow, permutation, threshold)
├── plugins/               # Model-specific plugins (XGBoost, joint quantile)
├── execution/             # Parallel execution backends
├── persistence/           # Fit caching
└── audit/                 # Logging
```

## Running Tests

```bash
pip install -e .[dev]
python3 -m ruff check .
python3 -m pytest
```

## License

MIT License
