# Reparameterization

Reparameterization transforms correlated hyperparameters into orthogonal search dimensions, enabling faster and more efficient optimization.

---

## The Problem: Correlated Parameters

Many hyperparameters are inherently correlated. In gradient boosting, for example, a lower learning rate needs more estimators to converge, while a higher learning rate needs fewer. Optimizers searching this space waste trials on invalid combinations (e.g., high learning rate + many estimators = overfitting).

## The Solution: Orthogonal Transforms

Reparameterization maps correlated parameters into independent dimensions. For a `learning_rate` / `n_estimators` pair, the transform produces:

- **log_product** (`log(lr) + log(n)`) -- total learning budget
- **log_ratio** (`log(lr) - log(n)`) -- balance between the two

Because the transformed dimensions are independent, the optimizer explores meaningful regions and converges faster with fewer wasted trials.

---

## Quick Start

Define the graph structure with `GraphBuilder`, then pass a `RunConfig` with reparameterization settings to `GraphRunner`.

```python
from sklearn_meta import GraphBuilder, GraphRunner, DataView, RunConfigBuilder
import xgboost as xgb
import pandas as pd

# 1. Build the graph spec (structure only)
graph = (
    GraphBuilder("my_pipeline")
    .add_model("xgb", xgb.XGBClassifier)
        .param("learning_rate", 0.01, 0.3, log=True)
        .int_param("n_estimators", 50, 500)
        .int_param("max_depth", 3, 10)
        .param("subsample", 0.6, 1.0)
        .fixed_params(random_state=42, eval_metric="logloss")
    .build()
)

# 2. Prepare data
data = DataView.from_Xy(X=pd.DataFrame(X), y=pd.Series(y))

# 3. Configure the run with reparameterization
config = (
    RunConfigBuilder()
    .cv(n_splits=5)
    .tuning(n_trials=50, metric="roc_auc")
    .reparameterization(enabled=True, use_prebaked=True)
    .build()
)

# 4. Run
run = GraphRunner.from_config(config).fit(graph, data, config)

# 5. Inspect results
node_result = run.node_results["xgb"]
print(f"Best params: {node_result.best_params}")

# 6. Build inference graph for prediction
inference = run.compile_inference()
predictions = inference.predict(X_test)
```

With `use_prebaked=True`, sklearn-meta automatically applies known reparameterizations for the model type (e.g., `learning_rate x n_estimators` for XGBoost).

---

## Prebaked Reparameterizations

These are applied automatically when `use_prebaked=True`.

| Model | Key | Parameters | Transform |
|-------|-----|------------|-----------|
| XGBoost / LightGBM / GradientBoosting | `xgb_learning_budget` | learning_rate, n_estimators | LogProduct |
| XGBoost | `xgb_regularization` | reg_alpha, reg_lambda | Ratio |
| LightGBM | `lgbm_regularization` | reg_alpha, reg_lambda | Ratio |
| LightGBM / CatBoost | `gbm_depth_leaves` | max_depth, num_leaves | LogProduct |
| MLP / Neural | `nn_learning_epochs` | learning_rate, epochs | LogProduct |
| MLP / Neural | `nn_dropout` | dropout1, dropout2 | Linear |
| MLP / Neural / Torch | `nn_weight_decay_dropout` | weight_decay, dropout | Ratio |
| ElasticNet / SGD / Linear | `elastic_net` | alpha/C, l1_ratio | Ratio |
| RandomForest / ExtraTrees | `rf_complexity` | max_depth, min_samples_split | LogProduct |
| RandomForest / ExtraTrees / Bagging | `rf_sampling` | max_features, max_samples | LogProduct |
| SVC / SVR / SVM | `svm_kernel` | C, gamma | LogProduct |
| CatBoost | `catboost_regularization` | l2_leaf_reg, random_strength | Ratio |

---

## Available Reparameterization Classes

All classes live in `sklearn_meta.meta.reparameterization` and provide `.forward(params)` / `.inverse(params)` methods.

### LogProductReparameterization

For parameters with an inverse relationship (e.g., learning rate x iterations):

```python
from sklearn_meta.meta.reparameterization import LogProductReparameterization

reparam = LogProductReparameterization(
    name="learning_budget",
    param1="learning_rate",
    param2="n_estimators",
)
```

### RatioReparameterization

For parameters that should sum to a constant (e.g., regularization weights):

```python
from sklearn_meta.meta.reparameterization import RatioReparameterization

reparam = RatioReparameterization(
    name="regularization",
    param1="l1_ratio",
    param2="l2_ratio",
)
```

### LinearReparameterization

For weighted combinations of multiple parameters:

```python
from sklearn_meta.meta.reparameterization import LinearReparameterization

reparam = LinearReparameterization(
    name="complexity",
    params=["depth", "leaves", "samples"],
    weights=[1.0, 0.5, 0.1],
)
```

---

## Custom Reparameterization Example

Pass custom reparameterization objects via `ReparameterizationConfig`:

```python
from sklearn_meta.meta.reparameterization import LogProductReparameterization
from sklearn_meta.runtime.config import ReparameterizationConfig, RunConfig, CVConfig, TuningConfig

reparam = LogProductReparameterization(
    name="learning_budget",
    param1="learning_rate",
    param2="n_estimators",
)

config = RunConfig(
    cv=CVConfig(n_splits=5),
    tuning=TuningConfig(n_trials=50, metric="roc_auc"),
    reparameterization=ReparameterizationConfig(
        enabled=True,
        use_prebaked=False,
        custom_reparameterizations=(reparam,),
    ),
)
```

To combine custom reparameterizations with prebaked ones, set `use_prebaked=True` alongside `custom_reparameterizations`.

---

## Best Practices

1. **Start with prebaked.** Use `use_prebaked=True` when available -- prebaked configs encode domain knowledge about known parameter correlations.

2. **Only reparameterize known correlations.** Applying transforms to uncorrelated parameters adds complexity without benefit.

3. **Prefer log-scale ranges for LogProduct.** The transform is most effective when the underlying parameters span multiplicative ranges (e.g., `learning_rate` from 0.001 to 0.3 with `log=True`).

---

## Next Steps

- [Search Spaces](search-spaces.md) -- Define parameter ranges
- [Tuning](tuning.md) -- Optimization configuration
- [Cross-Validation](cross-validation.md) -- CV strategies
- [Model Graphs](model-graphs.md) -- Multi-model pipelines
