# Feature Selection

sklearn-meta can automatically prune noisy features using shadow features, permutation importance, or importance thresholds. Feature selection is configured as part of the run config and runs transparently during training.

---

## Quick Start

Enable feature selection via `RunConfigBuilder`:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn_meta import GraphBuilder, RunConfigBuilder, fit

graph = (
    GraphBuilder("my_pipeline")
    .add_model("rf", RandomForestClassifier)
        .int_param("n_estimators", 50, 500)
        .int_param("max_depth", 3, 20)
    .build()
)

config = (
    RunConfigBuilder()
    .cv(n_splits=5)
    .tuning(n_trials=50, metric="roc_auc")
    .feature_selection(method="shadow", n_shadows=5)
    .build()
)

result = fit(graph, X_train, y_train, config)

# See which features were kept
print(result.node_results["rf"].selected_features)
```

Feature selection is not available through the convenience helpers (`tune()`, etc.) -- use the explicit API.

---

## Methods

### Shadow Features (default)

Compares each real feature against a synthetic noise baseline. If a feature can't beat its paired shadow, it's unlikely to generalize.

```python
.feature_selection(method="shadow", n_shadows=5, threshold_mult=1.414)
```

The most statistically rigorous method. Works best with tree-based models that have native feature importances.

### Permutation Importance

Shuffles each feature and measures the drop in score. Model-agnostic -- works with any estimator.

```python
.feature_selection(method="permutation", threshold_mult=1.414)
```

### Importance Threshold

Drops features below a percentile of the model's native feature importances. Fastest method.

```python
.feature_selection(method="threshold", threshold_percentile=10)
```

---

## Feature Groups

When multiple columns represent one logical feature (e.g., one-hot encoded categories), group them so they're selected or dropped atomically:

```python
.feature_selection(
    method="shadow",
    feature_groups={
        "city_ohe": ["city_sf", "city_ny", "city_la"],
        "device_ohe": ["device_mobile", "device_desktop"],
    },
)
```

Groups are never partially selected -- all columns in a group are kept or dropped together.

---

## Retune After Pruning

By default (`retune_after_pruning=True`), sklearn-meta runs a two-phase optimization:

1. **Phase 1**: Tune on all features, run feature selection with the best params
2. **Phase 2**: Retune on selected features only, with a narrowed search space

This compensates for the fact that optimal hyperparameters often shift after noisy features are removed -- typically the model can afford less regularization.

---

## FeatureSelectionConfig Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | `"shadow"`, `"permutation"`, or `"threshold"` | `"shadow"` |
| `n_shadows` | Number of shadow rounds | `5` |
| `threshold_mult` | Multiplier for shadow threshold (higher = stricter) | `1.414` |
| `threshold_percentile` | Percentile cutoff for permutation/threshold methods | `10.0` |
| `retune_after_pruning` | Retune hyperparameters after selection | `True` |
| `min_features` | Minimum features to keep | `1` |
| `max_features` | Maximum features to keep | `None` |
| `feature_groups` | `{group_name: [columns]}` for atomic selection | `None` |
| `random_state` | Random seed | `42` |

---

## Choosing a Method

| Method | Best for | Trade-off |
|--------|----------|-----------|
| **Shadow** | Tree-based models with native importances | Most rigorous, slowest |
| **Permutation** | Any model (no `feature_importances_` needed) | Model-agnostic, moderate speed |
| **Threshold** | Quick exploration | Fastest, least rigorous |

---

## Next Steps

- [Tuning](tuning.md) -- Optimization configuration
- [Reparameterization](reparameterization.md) -- Orthogonal parameter transforms
- [Search Spaces](search-spaces.md) -- Parameter definitions
