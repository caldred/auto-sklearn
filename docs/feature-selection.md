# Feature Selection

sklearn-meta provides automated feature selection using shadow features, permutation importance, and importance thresholds. The shadow feature method is a statistically robust approach that identifies genuinely important features while controlling for random chance.

---

## Quick Start

The recommended way to use feature selection is through the `GraphBuilder` fluent API:

```python
from sklearn_meta.api import GraphBuilder
from sklearn_meta.core.data.context import DataContext
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Prepare data
ctx = DataContext.from_Xy(X=pd.DataFrame(X), y=pd.Series(y))

# Build pipeline with feature selection
result = (
    GraphBuilder("my_pipeline")
    .add_model("rf", RandomForestClassifier)
    .with_search_space(
        n_estimators=(50, 500),
        max_depth=(3, 20),
    )
    .with_feature_selection(
        method="shadow",
        n_shadows=5,
        threshold_mult=1.414,
        retune_after_pruning=True,
    )
    .with_tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .with_cv(n_splits=5)
    .fit(ctx)
)

# Access selected features from FittedNode
fitted_node = result.fitted_nodes["rf"]
print(f"Selected features: {fitted_node.selected_features}")
```

---

## Selection Methods

### Shadow Features (default)

Shadow features are random permutations of real features. If a real feature is truly informative, its importance should exceed that of random noise.

```python
.with_feature_selection(
    method="shadow",
    n_shadows=5,            # Number of shadow copies per feature
    threshold_mult=1.414,   # Feature must beat threshold_mult * max shadow importance
)
```

```mermaid
graph TB
    subgraph "Original Features"
        F1[Feature 1<br/>Important]
        F2[Feature 2<br/>Important]
        F3[Feature 3<br/>Noise]
    end

    subgraph "Shadow Features (shuffled)"
        S1[Shadow 1<br/>Random]
        S2[Shadow 2<br/>Random]
        S3[Shadow 3<br/>Random]
    end

    subgraph "Comparison"
        F1 --> |importance > max shadow| K[Keep]
        F2 --> |importance > max shadow| K
        F3 --> |importance < max shadow| D[Drop]
    end
```

### Permutation Importance

Measures feature importance by shuffling each feature and observing the drop in model performance. Features whose permutation has little effect on score are dropped.

```python
.with_feature_selection(
    method="permutation",
    threshold_mult=1.414,
)
```

Features below `threshold_percentile` (default 10th percentile) of the permutation importance distribution are dropped.

### Importance Threshold

A simpler method that fits the model once and drops features below a percentile threshold of the model's native feature importances (e.g., `feature_importances_` for tree models).

```python
.with_feature_selection(
    method="threshold",
    threshold_mult=1.414,
)
```

---

## How Shadow Features Work

### Step 1: Create Shadow Features

For each original feature, create shadows by random permutation:

```
Original:  [1, 2, 3, 4, 5]
Shadow:    [3, 5, 1, 4, 2]  # Same values, random order
```

Shadow features have:
- Same distribution as the original
- Zero true predictive power
- Importance only from random chance

Multiple shadows (`n_shadows`) are created per feature for more stable estimates.

### Step 2: Fit Model and Get Importances

Train the model on original + shadow features together:

```python
X_combined = [original_features | shadow_features]
model.fit(X_combined, y)
importances = model.feature_importances_
```

### Step 3: Compare to Shadow Maximum

The maximum shadow importance represents the "noise floor":

```python
shadow_max = max(shadow_importances)
threshold = shadow_max * threshold_mult  # e.g., 1.414

keep = [f for f in features if importance[f] > threshold]
```

---

## Entropy Matching

### Why Entropy Matters

Shadow features should have the **same entropy** (information content) as original features. Otherwise, comparison is unfair.

```mermaid
graph LR
    subgraph "Low Entropy"
        A[0, 0, 0, 1, 0, 0]
    end

    subgraph "High Entropy"
        B[0.2, 0.8, 0.1, 0.9, 0.5, 0.3]
    end

    A --> |Different noise levels| C[Unfair comparison]
```

### Entropy Computation

```python
# Discretize continuous feature
bins = np.histogram_bin_edges(feature, bins='auto')
hist, _ = np.histogram(feature, bins=bins)
probs = hist / hist.sum()

# Shannon entropy
entropy = -sum(p * log(p) for p in probs if p > 0)
```

---

## Retune After Pruning

When `retune_after_pruning=True` (the default), sklearn-meta performs a two-phase optimization:

```mermaid
graph TB
    A[Phase 1: Initial Tuning] --> B[Find best params on all features]
    B --> C[Run feature selection with best params]
    C --> D[Drop unimportant features]
    D --> E[Phase 2: Narrow search around best]
    E --> F[Retune with selected features only]
    F --> G[Final best params]
```

### How the narrowed search space works

After feature selection, the search space is narrowed using `SearchSpace.narrow_around()`:

1. **Center on previous best**: The new search space is centered on the best parameters found in Phase 1.
2. **Reduce range by 50%**: Each parameter's range is narrowed by a factor of 0.5 around the best value.
3. **Bias toward less regularization**: Since removing features is itself a form of regularization, the narrowed space is biased (with `regularization_bias=0.25`) toward less regularization to compensate.
4. **Re-optimize**: A fresh round of optimization runs within this narrowed space using only the selected features.

This approach recognizes that the optimal hyperparameters may shift after features are pruned -- typically the model can afford to be slightly less regularized when noise features have been removed.

---

## FeatureSelectionConfig

For advanced usage, you can create a `FeatureSelectionConfig` directly:

```python
from sklearn_meta.selection.selector import FeatureSelectionConfig

config = FeatureSelectionConfig(
    enabled=True,
    method="shadow",           # "shadow", "permutation", or "threshold"
    n_shadows=5,               # Number of shadow copies (shadow method)
    threshold_mult=1.414,      # Multiplier for shadow max threshold
    threshold_percentile=10.0, # Percentile cutoff (permutation/threshold methods)
    retune_after_pruning=True, # Re-tune after feature pruning
    min_features=1,            # Minimum features to keep
    max_features=None,         # Maximum features to keep (None = no limit)
    random_state=42,
)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Whether feature selection is active | `True` |
| `method` | Selection method: `"shadow"`, `"permutation"`, `"threshold"` | `"shadow"` |
| `n_shadows` | Number of shadow features per real feature | `5` |
| `threshold_mult` | Multiplier for shadow max threshold | `1.414` |
| `threshold_percentile` | Percentile cutoff for permutation/threshold methods | `10.0` |
| `retune_after_pruning` | Whether to retune hyperparameters after selection | `True` |
| `min_features` | Minimum number of features to keep | `1` |
| `max_features` | Maximum number of features to keep (`None` = unlimited) | `None` |
| `random_state` | Random seed for reproducibility | `42` |

### Tuning the Threshold Multiplier

```mermaid
graph LR
    subgraph "threshold_mult"
        LOW[0.8<br/>Keep more] --> MED[1.0<br/>Balanced] --> HIGH[1.5+<br/>Keep fewer]
    end

    MED --> |Default: 1.414| R[Recommended]
```

---

## Accessing Results

After fitting, the `FittedNode` object contains the selected features:

```python
result = (
    GraphBuilder("pipeline")
    .add_model("xgb", XGBClassifier)
    .with_search_space(...)
    .with_feature_selection(method="shadow")
    .with_tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .with_cv(n_splits=5)
    .fit(ctx)
)

# Access selected features
fitted_node = result.fitted_nodes["xgb"]
selected = fitted_node.selected_features  # List[str] or None
print(f"Selected {len(selected)} features: {selected}")
```

---

## Advanced: Using FeatureSelector Directly

For programmatic control outside the `GraphBuilder`, you can use the `FeatureSelector` class directly:

```python
from sklearn_meta.selection.selector import FeatureSelector, FeatureSelectionConfig
from sklearn_meta.core.data.context import DataContext

config = FeatureSelectionConfig(
    method="shadow",
    n_shadows=5,
    threshold_mult=1.414,
)

selector = FeatureSelector(config)

# Select features for a model node
result = selector.select_for_node(node, ctx, best_params)

# result is a FeatureSelectionResult:
print(result.selected_features)   # List[str] - features to keep
print(result.dropped_features)    # List[str] - features removed
print(result.importances)         # Dict[str, float] - importance scores
print(result.method_used)         # str - which method was used
```

You can also call `select()` directly with raw data:

```python
from sklearn.ensemble import RandomForestClassifier

result = selector.select(
    model=RandomForestClassifier(n_estimators=100),
    X=X_train,
    y=y_train,
    feature_cols=list(X_train.columns),
)
```

---

## Complete Example

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from sklearn_meta.api import GraphBuilder
from sklearn_meta.core.data.context import DataContext

# Generate data with known informative/noise structure
X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=10,
    n_redundant=10,
    n_clusters_per_class=2,
    random_state=42,
)

# Add feature names
feature_names = (
    [f"informative_{i}" for i in range(10)]
    + [f"redundant_{i}" for i in range(10)]
    + [f"noise_{i}" for i in range(30)]
)
X = pd.DataFrame(X, columns=feature_names)
y = pd.Series(y)

# Create data context
ctx = DataContext.from_Xy(X=X, y=y)

# Build and fit with feature selection
result = (
    GraphBuilder("feature_selection_demo")
    .add_model("rf", RandomForestClassifier)
    .with_search_space(
        n_estimators=(50, 500),
        max_depth=(3, 20),
        min_samples_split=(2, 20),
    )
    .with_feature_selection(
        method="shadow",
        n_shadows=5,
        threshold_mult=1.414,
        retune_after_pruning=True,
        min_features=5,
    )
    .with_tuning(n_trials=30, metric="roc_auc", greater_is_better=True)
    .with_cv(n_splits=5)
    .fit(ctx)
)

# Inspect results
fitted = result.fitted_nodes["rf"]
selected = fitted.selected_features

print(f"Selected {len(selected)} of {len(feature_names)} features:")

informative_kept = sum(1 for f in selected if "informative" in f)
redundant_kept = sum(1 for f in selected if "redundant" in f)
noise_kept = sum(1 for f in selected if "noise" in f)

print(f"  Informative features kept: {informative_kept}/10")
print(f"  Redundant features kept: {redundant_kept}/10")
print(f"  Noise features kept: {noise_kept}/30")
print(f"  Best params: {fitted.best_params}")
```

---

## Best Practices

### 1. Start with Shadow Features

The shadow method is the most statistically principled. It controls for chance importance automatically:

```python
.with_feature_selection(method="shadow", n_shadows=5)
```

### 2. Enable Retune After Pruning

Hyperparameters tuned on all features may not be optimal for the selected subset. Retuning compensates:

```python
.with_feature_selection(retune_after_pruning=True)  # default
```

### 3. Set Feature Constraints for Safety

Prevent over-pruning or under-pruning with min/max bounds:

```python
.with_feature_selection(
    min_features=5,       # Never drop below 5 features
    max_features=30,      # Cap at 30 even if more pass threshold
)
```

### 4. Match Method to Use Case

- **Shadow** (default): Best for tree-based models with native feature importances. Most statistically rigorous.
- **Permutation**: Model-agnostic. Works with any estimator. Better for models without `feature_importances_`.
- **Threshold**: Fastest. Good for quick exploration when you trust the model's native importances.

### 5. Validate Selection

Always compare model performance with and without selection to confirm the pruning helps:

```python
# Without selection
result_all = builder.fit(ctx)

# With selection
result_selected = builder.with_feature_selection(method="shadow").fit(ctx)

# Compare CV scores
```

---

## Next Steps

- [Model Graphs](model-graphs.md) -- Integrate selection into pipelines
- [Tuning](tuning.md) -- Optimization configuration
- [Reparameterization](reparameterization.md) -- Orthogonal hyperparameter search
