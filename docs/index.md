# sklearn-meta

A Python library for automated ML pipelines with hyperparameter optimization, model stacking, and feature selection.

---

## Install

```bash
pip install -e .
```

Optional boosting libraries:

```bash
pip install xgboost lightgbm catboost
```

---

## Quick Example

Tune a random forest in a few lines:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn_meta import tune

result = tune(
    RandomForestClassifier,
    X_train, y_train,
    params={"n_estimators": (50, 500), "max_depth": (3, 20)},
    metric="roc_auc",
)
result.best_params_
result.predict(X_test)
```

sklearn-meta also provides `cross_validate()`, `stack()`, and `compare()` for other common workflows, and a full `GraphBuilder` API when you need more control. See [Getting Started](getting-started.md) for a walkthrough.

---

## Documentation

### Tutorial

Work through these in order to learn sklearn-meta:

1. [Getting Started](getting-started.md) -- From first tune to stacking ensembles
2. [Tuning](tuning.md) -- Hyperparameter optimization in depth
3. [Stacking](stacking.md) -- Multi-model ensembles
4. [Feature Selection](feature-selection.md) -- Automated feature pruning

### Reference

Look up specific topics:

- [Search Spaces](search-spaces.md) -- Parameter types, conditional params, shorthand notation
- [Cross-Validation](cross-validation.md) -- CV strategies, nested CV, grouped splits
- [Model Graphs](model-graphs.md) -- Custom DAG architectures with GraphBuilder
- [Reparameterization](reparameterization.md) -- Orthogonal parameter transforms
- [Plugins](plugins.md) -- Extending sklearn-meta with model-specific behavior
- [Joint Quantile Regression](joint-quantile-regression.md) -- Multivariate targets with uncertainty
- [API Reference](api-reference.md) -- Complete class and function reference

---

## How It Works

sklearn-meta has three phases:

1. **Define** a model graph -- what models to train and how they connect
2. **Configure** the run -- CV strategy, tuning trials, feature selection
3. **Fit** -- sklearn-meta handles tuning, cross-validation, and training

The convenience helpers (`tune`, `cross_validate`, `stack`, `compare`) handle all three phases in a single call. For custom architectures, use `GraphBuilder` + `RunConfigBuilder` + `fit()` to control each phase separately.

```python
# Convenience helper (one call)
result = tune(MyModel, X, y, params={...}, metric="roc_auc")

# Explicit API (full control)
graph = GraphBuilder("pipeline").add_model("m", MyModel).param(...).build()
config = RunConfigBuilder().cv(n_splits=5).tuning(n_trials=50, metric="roc_auc").build()
result = fit(graph, X, y, config)
```

Both approaches return the same `TrainingRun` object with `.predict()`, `.best_params_`, `.best_score_`, and more.

---

## License

MIT License -- see [LICENSE](../LICENSE) for details.
