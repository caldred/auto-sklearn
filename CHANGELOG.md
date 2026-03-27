# Changelog

## 0.2.0

Breaking release that reorganizes the package around a new public surface.

### What changed

- `sklearn_meta.api` and `sklearn_meta.core.*` were removed.
- Core concepts split into `spec`, `data`, `runtime`, `engine`, and `artifacts` packages.
- Training now flows through explicit graph, data, config, and runner objects.
- Inference artifacts are compiled from training runs.
- New convenience helpers: `tune()`, `cross_validate()`, `stack()`, `compare()`.

### Migrating from 0.1

Replace the old `GraphBuilder(...).fit(...)` pattern with the new explicit flow:

```python
from sklearn_meta import GraphBuilder, RunConfigBuilder, fit

graph = GraphBuilder("example").add_model("m", SomeEstimator).build()
config = RunConfigBuilder().cv(n_splits=5).tuning(n_trials=50, metric="roc_auc").build()
result = fit(graph, X_train, y_train, config)
```

Or use the convenience helpers for common workflows:

```python
from sklearn_meta import tune
result = tune(SomeEstimator, X_train, y_train, params={...}, metric="roc_auc")
```

Projects that still depend on removed imports (`sklearn_meta.api`, `sklearn_meta.core.*`) should pin to `v0.1.0` or `release/0.1.x` until migrated.
