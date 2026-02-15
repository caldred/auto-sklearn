# sklearn-meta

A Python library for automated machine learning pipelines with hyperparameter optimization, model stacking, feature selection, and knowledge distillation. Define complex model graphs as directed acyclic graphs (DAGs) using a fluent builder API, and let sklearn-meta handle cross-validated tuning, stacking, and training.

## Installation

```bash
pip install -e .
```

### Optional Dependencies

```bash
pip install xgboost        # XGBoost support
brew install libomp         # OpenMP on macOS (required for XGBoost)
pip install lightgbm        # LightGBM support
```

## Table of Contents

1. [Quick Start](#quick-start)
2. [Defining Models](#defining-models)
3. [Search Spaces](#search-spaces)
4. [Cross-Validation](#cross-validation)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Model Stacking](#model-stacking)
7. [Feature Selection](#feature-selection)
8. [Reparameterization](#reparameterization)
9. [Knowledge Distillation](#knowledge-distillation)
10. [Estimator Scaling](#estimator-scaling)
11. [Working with Results](#working-with-results)
12. [Advanced Usage](#advanced-usage)
13. [Project Structure](#project-structure)
14. [Running Tests](#running-tests)

## Quick Start

The simplest way to use sklearn-meta is through the `GraphBuilder` fluent API. Define your models, configure tuning, and call `.fit()`:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn_meta import GraphBuilder

fitted = (
    GraphBuilder("my_pipeline")
    .add_model("rf", RandomForestClassifier)
    .with_search_space(
        n_estimators=(50, 500),
        max_depth=(3, 20),
    )
    .with_fixed_params(random_state=42)
    .with_cv(n_splits=5, strategy="stratified")
    .with_tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .fit(X_train, y_train)
)

predictions = fitted.predict(X_test)
```

This tunes a random forest over 50 Optuna trials using 5-fold stratified CV, then refits on the full training set with the best hyperparameters.

## Defining Models

Add models to the graph with `add_model(name, estimator_class)`. Each model returns a `NodeBuilder` that supports chained configuration:

```python
builder = (
    GraphBuilder("classifier")
    .add_model("xgb", XGBClassifier)
    .with_search_space(learning_rate=(0.01, 0.3, "log"), max_depth=(3, 10))
    .with_fixed_params(random_state=42, use_label_encoder=False)
    .with_fit_params(verbose=False)
    .with_output_type("proba")           # "prediction", "proba", or "transform"
    .with_description("Base XGBoost model")
    .with_plugins("xgboost")             # Model-specific plugin hooks
)
```

### Conditional Models

Models can be conditionally included based on the data:

```python
.add_model("binary_only", LogisticRegression)
.with_condition(lambda ctx: ctx.y.nunique() == 2)
```

### Feature Subsets

Restrict a model to specific input features:

```python
.add_model("text_model", SGDClassifier)
.with_features("tfidf_1", "tfidf_2", "tfidf_3")
```

## Search Spaces

Hyperparameter search spaces are defined with a shorthand syntax:

```python
.with_search_space(
    n_estimators=(50, 500),              # Integer range [50, 500]
    max_depth=(3, 15),                   # Integer range [3, 15]
    learning_rate=(0.001, 0.3, "log"),   # Float range, log-uniform
    subsample=(0.5, 1.0),               # Float range [0.5, 1.0]
    booster=["gbtree", "dart"],          # Categorical choice
)
```

The shorthand rules:
- `(low, high)` with integers creates an integer range
- `(low, high)` with floats creates a float range
- `(low, high, "log")` uses log-uniform sampling
- `[a, b, c]` creates a categorical parameter

For full control, build a `SearchSpace` manually:

```python
from sklearn_meta.search.space import SearchSpace

space = SearchSpace()
space.add_int("n_estimators", 50, 500)
space.add_float("learning_rate", 0.001, 0.3, log=True)
space.add_float("subsample", 0.5, 1.0, step=0.1)
space.add_categorical("booster", ["gbtree", "dart"])

# Conditional parameters (active only when parent has a specific value)
space.add_conditional("dart_rate", parent_name="booster", parent_value="dart",
                      parameter=SearchParameter("dart_rate", 0.01, 0.5))

.add_model("xgb", XGBClassifier).with_search_space(space)
```

## Cross-Validation

Configure cross-validation with `.with_cv()`:

```python
.with_cv(
    n_splits=5,
    n_repeats=1,
    strategy="stratified",
    shuffle=True,
    random_state=42,
)
```

### CV Strategies

| Strategy | Use Case |
|----------|----------|
| `"stratified"` | Classification with imbalanced classes |
| `"random"` | General-purpose random splitting |
| `"group"` | Grouped data (pass `groups=` to `.fit()`) |
| `"time_series"` | Temporal data (no shuffling, expanding window) |

### Nested CV

For unbiased performance estimates during hyperparameter tuning:

```python
.with_cv(n_splits=5, nested=True, inner_splits=3)
```

This creates an outer 5-fold loop for evaluation and an inner 3-fold loop for tuning within each outer fold.

### Group CV

When observations are grouped (e.g., multiple samples per patient):

```python
(
    GraphBuilder("grouped")
    .add_model("rf", RandomForestClassifier)
    .with_search_space(n_estimators=(50, 200))
    .with_cv(n_splits=5, strategy="group")
    .with_tuning(n_trials=30, metric="roc_auc", greater_is_better=True)
    .fit(X_train, y_train, groups=patient_ids)
)
```

## Hyperparameter Tuning

Configure tuning with `.with_tuning()`:

```python
.with_tuning(
    n_trials=100,                  # Number of Optuna trials
    timeout=3600,                  # Optional timeout in seconds
    strategy="layer_by_layer",     # Optimization strategy
    metric="neg_mean_squared_error",
    greater_is_better=False,
    early_stopping_rounds=20,      # Stop if no improvement for 20 trials
    n_parallel_trials=4,           # Parallel Optuna workers
    show_progress=True,            # Display Optuna progress bar
)
```

### Optimization Strategies

| Strategy | Description |
|----------|-------------|
| `"layer_by_layer"` | Tune each DAG layer sequentially; models within a layer can be parallelized. **(Default)** |
| `"greedy"` | Tune models one at a time in topological order. |
| `"none"` | Skip tuning; use fixed parameters only. |

### Metrics

Any scikit-learn scorer name works. Common choices:

- Classification: `"roc_auc"`, `"accuracy"`, `"f1"`, `"log_loss"`, `"neg_log_loss"`
- Regression: `"neg_mean_squared_error"`, `"neg_mean_absolute_error"`, `"r2"`

Set `greater_is_better=True` for metrics where higher is better (e.g., `roc_auc`) and `False` for loss-style metrics (e.g., `neg_mean_squared_error`).

## Model Stacking

Build multi-layer stacking ensembles by declaring dependencies between models. Upstream model predictions (or probabilities) become features for downstream models:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_meta import GraphBuilder

fitted = (
    GraphBuilder("stacking_ensemble")

    # Layer 0: base models
    .add_model("rf", RandomForestClassifier)
    .with_search_space(n_estimators=(50, 300), max_depth=(3, 15))

    .add_model("gb", GradientBoostingClassifier)
    .with_search_space(
        learning_rate=(0.01, 0.3, "log"),
        n_estimators=(50, 300),
        max_depth=(3, 8),
    )

    # Layer 1: meta-learner stacks base predictions
    .add_model("meta", LogisticRegression)
    .with_fixed_params(C=1.0)
    .stacks("rf", "gb")               # Use predictions as features

    .with_cv(n_splits=5, strategy="stratified")
    .with_tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .fit(X_train, y_train)
)
```

### Stacking Modes

- `.stacks("rf", "gb")` -- Use point predictions as features
- `.stacks_proba("rf", "gb")` -- Use class probabilities as features
- `.depends_on("rf", dep_type=DependencyType.TRANSFORM)` -- Use transformed features

With `layer_by_layer` strategy, layer 0 models (`rf`, `gb`) are tuned first. Their out-of-fold predictions are then used as training features for the layer 1 meta-learner.

### Deeper Stacking

You can stack any number of layers:

```python
(
    GraphBuilder("deep_stack")
    # Layer 0
    .add_model("rf", RandomForestClassifier)
    .with_search_space(n_estimators=(50, 300))

    .add_model("xgb", XGBClassifier)
    .with_search_space(learning_rate=(0.01, 0.3, "log"))

    .add_model("lgb", LGBMClassifier)
    .with_search_space(learning_rate=(0.01, 0.3, "log"))

    # Layer 1: intermediate meta-learners
    .add_model("meta_tree", GradientBoostingClassifier)
    .stacks("rf", "xgb", "lgb")

    .add_model("meta_linear", LogisticRegression)
    .stacks_proba("rf", "xgb", "lgb")

    # Layer 2: final blender
    .add_model("blender", LogisticRegression)
    .stacks("meta_tree", "meta_linear")

    .with_cv(n_splits=5)
    .with_tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .fit(X_train, y_train)
)
```

## Feature Selection

Automatically drop uninformative features before final training. Three methods are available:

### Shadow Feature Selection (Recommended)

Creates shuffled "shadow" copies of each feature and compares real feature importance against the shadow baseline. Features that don't beat their shadows are dropped.

```python
.with_feature_selection(
    method="shadow",
    n_shadows=5,                   # Number of shadow copies per feature
    threshold_mult=1.414,          # Multiplier on shadow importance threshold
    retune_after_pruning=True,     # Re-tune hyperparameters with selected features
    min_features=1,                # Never drop below this many features
    max_features=None,             # Optional upper bound
)
```

### Permutation Importance

Measures importance by shuffling each feature and observing the drop in validation performance:

```python
.with_feature_selection(method="permutation", retune_after_pruning=True)
```

### Threshold

Simple threshold on raw feature importances:

```python
.with_feature_selection(method="threshold", retune_after_pruning=True)
```

### Retune After Pruning

When `retune_after_pruning=True` (the default), after features are selected the search space is narrowed around the previous best parameters and re-tuned using only the selected features. The narrowed space biases toward less regularization, since feature removal itself acts as regularization.

## Reparameterization

Many hyperparameters are correlated (e.g., `learning_rate` and `n_estimators` have an inverse relationship). Reparameterization transforms them into orthogonal dimensions so the optimizer explores the space more efficiently.

```python
.with_reparameterization(use_prebaked=True)
```

With `use_prebaked=True`, sklearn-meta automatically applies known reparameterizations for common model/parameter pairs (e.g., the learning-rate/n-estimators product for boosting models).

### Custom Reparameterizations

```python
from sklearn_meta.meta.reparameterization import LogProductReparameterization

.with_reparameterization(
    reparameterizations=[
        LogProductReparameterization("budget", "learning_rate", "n_estimators"),
    ],
    use_prebaked=False,
)
```

Available transforms:
- **LogProductReparameterization** -- For inversely related parameters (e.g., `lr * n_estimators ~ constant`). Transforms to `log_product` and `log_ratio`.
- **RatioReparameterization** -- For parameters that sum to a constant (e.g., regularization weights). Transforms to `total` and `ratio`.
- **LinearReparameterization** -- For weighted linear combinations.

## Knowledge Distillation

Train a smaller student model using soft targets from a larger teacher model. The student optimizes a blended loss: `alpha * KL_soft + (1 - alpha) * CE_hard`.

```python
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn_meta import GraphBuilder

fitted = (
    GraphBuilder("distillation")

    # Teacher: large model
    .add_model("teacher", GradientBoostingClassifier)
    .with_search_space(
        n_estimators=(200, 1000),
        learning_rate=(0.01, 0.3, "log"),
        max_depth=(4, 12),
    )
    .with_output_type("proba")

    # Student: smaller model learns from teacher's soft targets
    .add_model("student", XGBClassifier)
    .with_search_space(
        n_estimators=(10, 100),
        learning_rate=(0.01, 0.3, "log"),
        max_depth=(2, 6),
    )
    .distills("teacher", temperature=3.0, alpha=0.5)

    .with_cv(n_splits=5)
    .with_tuning(n_trials=50, metric="neg_log_loss", greater_is_better=False)
    .fit(X_train, y_train)
)
```

Parameters:
- **temperature** -- Softens the teacher's probability distribution. Higher values produce smoother targets.
- **alpha** -- Blend weight between the soft KL-divergence loss and the hard cross-entropy loss. `alpha=1.0` uses only soft targets; `alpha=0.0` ignores the teacher entirely.

The teacher must be fitted before the student in the graph's topological order. Only one teacher per student is supported.

## Estimator Scaling

For boosting models, you can tune with a small number of estimators for speed, then scale up for the final model:

### Fixed Scaling

```python
.with_tuning(
    n_trials=100,
    metric="neg_log_loss",
    greater_is_better=False,
    tuning_n_estimators=100,     # Use 100 trees during tuning (fast)
    final_n_estimators=500,      # Use 500 trees for the final model
)
```

The learning rate is automatically scaled by `tuning_n_estimators / final_n_estimators` to maintain equivalent training.

### Automatic Scaling Search

Search for the best scaling factor automatically:

```python
.with_tuning(
    n_trials=100,
    metric="neg_log_loss",
    greater_is_better=False,
    estimator_scaling_search=True,              # Search optimal multiplier
    estimator_scaling_factors=[2, 5, 10, 20],   # Factors to test (default)
)
```

This tunes hyperparameters first, then tests each scaling factor with early stopping if performance degrades.

## Working with Results

`.fit()` returns a `FittedGraph` containing all trained models and metadata.

### Predictions

```python
fitted = builder.fit(X_train, y_train)

# Predict using the final (leaf) node
predictions = fitted.predict(X_test)

# Predict from a specific node
rf_predictions = fitted.predict(X_test, node_name="rf")
```

For stacking graphs, `.predict()` automatically chains predictions through the graph: base model predictions are computed first and fed as features to downstream models.

### Inspecting Results

```python
# Access a fitted node
node = fitted.get_node("rf")

# Best hyperparameters found
print(node.best_params)

# Out-of-fold predictions (useful for analysis and diagnostics)
oof = fitted.get_oof_predictions("rf")

# CV performance
print(node.cv_result.mean_score)
print(node.cv_result.std_score)

# Features kept after selection
print(node.selected_features)

# Total training time
print(f"Completed in {fitted.total_time:.1f}s")
```

### Extracting Feature Importance

```python
from sklearn_meta.selection.importance import importance_registry

model = fitted.get_node("rf").cv_result  # or access individual fold models
importance = importance_registry.extract_importance(model, feature_names)
```

### Saving and Loading

```python
import joblib

# Save
joblib.dump(fitted, "my_pipeline.joblib")

# Load
fitted = joblib.load("my_pipeline.joblib")
predictions = fitted.predict(X_test)
```

## Advanced Usage

### Low-Level API

For full control, you can bypass the fluent builder and work with the core components directly:

```python
from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.node import ModelNode
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.tuning.orchestrator import TuningConfig, TuningOrchestrator
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.search.backends.optuna import OptunaBackend

# Build search space
space = SearchSpace()
space.add_int("n_estimators", 50, 200)
space.add_int("max_depth", 3, 10)

# Create model node
node = ModelNode(
    name="rf",
    estimator_class=RandomForestClassifier,
    search_space=space,
    fixed_params={"random_state": 42},
)

# Build graph
graph = ModelGraph()
graph.add_node(node)

# Configure
cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
tuning_config = TuningConfig(
    strategy=OptimizationStrategy.LAYER_BY_LAYER,
    n_trials=20,
    cv_config=cv_config,
    metric="accuracy",
    greater_is_better=True,
)

# Create data context and run
ctx = DataContext.from_Xy(X=X_train, y=y_train)
data_manager = DataManager(cv_config)
orchestrator = TuningOrchestrator(
    graph=graph,
    data_manager=data_manager,
    search_backend=OptunaBackend(),
    tuning_config=tuning_config,
)
fitted = orchestrator.fit(ctx)
```

### Custom Search Backend

The search backend is pluggable. The default is `OptunaBackend`, but you can configure it:

```python
from sklearn_meta.search.backends.optuna import OptunaBackend
import optuna

backend = OptunaBackend(
    direction="maximize",
    random_state=42,
    sampler=optuna.samplers.CmaEsSampler(seed=42),
    pruner=optuna.pruners.MedianPruner(),
    n_jobs=4,
    show_progress_bar=True,
    verbosity=optuna.logging.INFO,
)

fitted = builder.create_orchestrator(search_backend=backend).fit(ctx)
```

### Fit Caching

Cache fitted models to avoid redundant computation during optimization:

```python
from sklearn_meta.persistence.cache import FitCache

cache = FitCache(cache_dir="./cache", max_size_mb=1000.0)
orchestrator = TuningOrchestrator(..., fit_cache=cache)
```

### Plugins

Plugins hook into the model lifecycle (pre-fit, post-fit, search space modification, etc.). The built-in XGBoost plugin is applied with:

```python
.add_model("xgb", XGBClassifier).with_plugins("xgboost")
```

## Project Structure

```
sklearn_meta/
├── api.py                 # GraphBuilder fluent API
├── core/
│   ├── data/              # DataContext, CVConfig, DataManager
│   ├── model/             # ModelNode, ModelGraph, Dependencies, Distillation
│   └── tuning/            # TuningOrchestrator, Strategies
├── search/                # SearchSpace, SearchParameter, Backends
├── meta/                  # Reparameterization transforms
├── selection/             # Feature selection (shadow, permutation, threshold)
├── plugins/               # Model-specific plugins (XGBoost, etc.)
├── execution/             # Parallel execution backends
├── persistence/           # Fit caching
└── audit/                 # Logging
```

## Running Tests

```bash
pip install pytest

# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=sklearn_meta --cov-report=html
```

## License

MIT License
