# Tuning

sklearn-meta uses Optuna's TPE sampler to find optimal hyperparameters via cross-validated Bayesian optimization.

---

## Quick Start

For a single model, `tune()` handles everything:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn_meta import tune

result = tune(
    RandomForestClassifier,
    X_train, y_train,
    params={"n_estimators": (50, 500), "max_depth": (3, 20)},
    fixed_params={"random_state": 42},
    n_trials=100,
    metric="roc_auc",
)

result.best_params_        # best hyperparameters
result.best_score_         # mean CV score
result.predict(X_test)     # predictions from all fold models
```

### tune() Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `params` | Search space dict (see [Search Spaces](search-spaces.md)) | required |
| `fixed_params` | Non-tuned parameters | `None` |
| `n_trials` | Number of Optuna trials | `100` |
| `metric` | Scoring metric | `"neg_mean_squared_error"` |
| `cv` | Number of folds or a `CVConfig` | `5` |
| `strategy` | CV strategy (`"stratified"`, `"random"`, `"group"`, `"time_series"`) | auto-detected |
| `groups` | Group labels for group CV | `None` |
| `verbosity` | Log level (0=silent, 1=summary, 2=detailed) | `1` |

---

## Using the Explicit API

When you need feature selection, reparameterization, plugins, or fine-grained control, use `GraphBuilder` + `RunConfigBuilder`:

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
    .tuning(
        n_trials=100,
        metric="roc_auc",
        early_stopping_rounds=20,
        show_progress=True,
    )
    .feature_selection(method="shadow")
    .reparameterization(enabled=True, use_prebaked=True)
    .verbosity(2)
    .build()
)

result = fit(graph, X_train, y_train, config)
```

---

## Metrics

### Built-in Metrics

| Metric | Task | Direction |
|--------|------|-----------|
| `accuracy` | Classification | Maximize |
| `roc_auc` | Binary classification | Maximize |
| `f1` | Classification | Maximize |
| `precision` | Classification | Maximize |
| `recall` | Classification | Maximize |
| `neg_log_loss` | Classification | Maximize |
| `neg_mean_squared_error` | Regression | Maximize |
| `neg_root_mean_squared_error` | Regression | Maximize |
| `neg_mean_absolute_error` | Regression | Maximize |
| `r2` | Regression | Maximize |

The score direction (`greater_is_better`) is inferred automatically for all standard sklearn scorer names. Loss-style scorers are already negated (for example `neg_log_loss` and `neg_mean_squared_error`), so they are still maximized.

### Custom Metrics

```python
from sklearn.metrics import make_scorer

def custom_metric(y_true, y_pred):
    return score

config = (
    RunConfigBuilder()
    .tuning(
        metric=make_scorer(custom_metric),
        greater_is_better=True,  # required for custom metrics
    )
    .build()
)
```

---

## Early Stopping and Timeouts

Stop tuning early when there's no improvement, or set a wall-clock limit:

```python
config = (
    RunConfigBuilder()
    .tuning(
        n_trials=1000,
        metric="roc_auc",
        early_stopping_rounds=20,  # stop if no improvement for 20 trials
        timeout=3600,              # stop after 1 hour
    )
    .build()
)
```

---

## Estimator Scaling

For boosting models (XGBoost, LightGBM, GradientBoosting), tune with fewer estimators for speed, then use more in the final CV pass:

```python
config = (
    RunConfigBuilder()
    .tuning(n_trials=100, metric="roc_auc")
    .estimator_scaling(
        tuning_n_estimators=100,     # fast trials
        final_n_estimators=1000,     # full power for final training
    )
    .build()
)
```

To automatically search for the best `n_estimators`:

```python
config = (
    RunConfigBuilder()
    .tuning(n_trials=100, metric="roc_auc")
    .estimator_scaling(
        tuning_n_estimators=100,
        scaling_search=True,
        scaling_factors=[2, 5, 10, 20],
    )
    .build()
)
```

---

## Progress Monitoring

```python
config = (
    RunConfigBuilder()
    .tuning(n_trials=100, show_progress=True)
    .verbosity(2)  # 0=silent, 1=summary, 2=detailed
    .build()
)
```

---

## Parallel Execution

By default, nodes within the same graph layer are tuned sequentially. To parallelize, pass a `LocalExecutor` via `RuntimeServices`:

```python
from sklearn_meta import GraphRunner, RuntimeServices
from sklearn_meta.execution.local import LocalExecutor

services = RuntimeServices(executor=LocalExecutor(n_jobs=4))
runner = GraphRunner(services)
result = runner.fit(graph, data, config)
```

Use parallel execution when you have multiple independent base models in the same layer and sufficient CPU/memory. Avoid it when models already use internal parallelism (`n_jobs=-1`), since they'll contend for the same cores.

### Training Dispatch

For more control over where and how node training runs, use a `TrainingDispatcher`. The `LocalTrainingDispatcher` runs node-training jobs in the local process, using live in-memory objects for single-worker execution and serialized jobs when parallelizing via an executor. Implement the `TrainingDispatcher` protocol to run jobs on remote workers or custom infrastructure.

```python
from sklearn_meta import GraphRunner, RuntimeServices, LocalTrainingDispatcher

services = RuntimeServices(training_dispatcher=LocalTrainingDispatcher())
result = GraphRunner(services).fit(graph, data, config)
```

When a dispatcher is configured, the runner uses `graph.get_training_layers()` to find nodes that can train concurrently (ignoring non-blocking edges), then batches dispatchable nodes per layer. Conditional and distilled nodes are always trained locally.

Use `validate_dispatchable()` to check compatibility before dispatch:

```python
from sklearn_meta import validate_dispatchable

warnings = validate_dispatchable(graph, config)
for w in warnings:
    print(f"{w.node_name}: {w.message}")
```

---

## Custom Search Backend

The default search backend is `OptunaBackend` with a TPE sampler. To customize it (e.g., different sampler, pruner, or parallelism):

```python
import optuna
from sklearn_meta import OptunaBackend, RuntimeServices, GraphRunner

backend = OptunaBackend(
    direction="maximize",
    random_state=42,
    sampler=optuna.samplers.CmaEsSampler(seed=42),
    pruner=optuna.pruners.MedianPruner(),
    n_jobs=4,
    show_progress_bar=True,
)

services = RuntimeServices(search_backend=backend)
result = GraphRunner(services).fit(graph, data, config)
```

---

## Fit Caching

Cache fitted models to avoid redundant computation when re-running experiments:

```python
from sklearn_meta import FitCache, RuntimeServices, GraphRunner

cache = FitCache(cache_dir="./cache", max_size_mb=1000.0)
services = RuntimeServices(fit_cache=cache)
result = GraphRunner(services).fit(graph, data, config)
```

The cache is keyed on the model class, hyperparameters, and training data hash. Subsequent runs with the same configuration skip fitting and return cached results.

---

## How Tuning Works

Under the hood, `tune()` and the explicit API both follow the same pipeline:

1. **For each trial**, Optuna's TPE sampler suggests hyperparameters
2. **Cross-validate** the model with those hyperparameters
3. **Report** the mean CV score back to Optuna
4. Repeat until `n_trials`, `timeout`, or `early_stopping_rounds` is reached
5. **Final training**: retrain on all folds with the best hyperparameters

For stacking graphs, nodes are tuned **layer by layer**: base models first, then meta-learners using the base models' out-of-fold predictions. See [Stacking](stacking.md) for details.

---

## TuningConfig Reference

For direct `RunConfig` construction:

```python
from sklearn_meta import RunConfig, CVConfig, CVStrategy, TuningConfig

config = RunConfig(
    cv=CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42),
    tuning=TuningConfig(
        n_trials=100,
        timeout=3600,
        early_stopping_rounds=20,
        metric="roc_auc",
        show_progress=True,
    ),
    verbosity=2,
)
```

| TuningConfig Field | Description | Default |
|-----------|-------------|---------|
| `n_trials` | Maximum number of trials | `100` |
| `timeout` | Time limit in seconds | `None` |
| `early_stopping_rounds` | Stop if no improvement for N trials | `None` |
| `metric` | Scoring metric name | `"neg_mean_squared_error"` |
| `greater_is_better` | Direction (auto-inferred for standard metrics) | `None` |
| `strategy` | Graph traversal strategy (`LAYER_BY_LAYER`, `GREEDY`, `NONE`) | `LAYER_BY_LAYER` |
| `show_progress` | Display progress bar | `False` |

---

## Best Practices

1. **Start small** -- Use 20 trials during development, scale up for production.
2. **Use log scale for learning rates and regularization** -- `(0.001, 0.1, {"log": True})` explores orders of magnitude evenly.
3. **Set reasonable bounds** -- `"max_depth": (3, 15)` not `(1, 100)`. Tighter bounds mean fewer wasted trials.
4. **Use early stopping** -- `early_stopping_rounds=20` prevents spending time after convergence.
5. **Use estimator scaling for boosting models** -- Tune fast, train with more estimators.
6. **Try reparameterization** -- For models with correlated parameters (e.g., learning_rate vs n_estimators), see [Reparameterization](reparameterization.md).

---

## Next Steps

- [Stacking](stacking.md) -- Combine multiple models
- [Feature Selection](feature-selection.md) -- Prune noisy features
- [Reparameterization](reparameterization.md) -- Faster optimization via parameter transforms
- [Search Spaces](search-spaces.md) -- Conditional parameters, space operations
