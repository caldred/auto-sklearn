# sklearn-meta

A Python library for automated machine learning pipelines with hyperparameter optimization, model stacking, feature selection, and knowledge distillation. Define complex model graphs as directed acyclic graphs (DAGs) using a fluent builder API, and let sklearn-meta handle cross-validated tuning, stacking, and training.

## Installation

```bash
python3 -m pip install -e .[dev]
```

### Optional Dependencies

```bash
pip install xgboost        # XGBoost support
brew install libomp         # OpenMP on macOS (required for XGBoost)
pip install lightgbm        # LightGBM support
```

## Breaking Changes In 0.2.0

`0.2.0` is intentionally breaking. The old `sklearn_meta.api` module and the
`sklearn_meta.core.*` package tree were removed.

Use the top-level `sklearn_meta` exports and the new `spec`, `data`, `runtime`,
`engine`, and `artifacts` packages instead. See
[docs/upgrading-to-0.2.md](docs/upgrading-to-0.2.md) for migration guidance.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Management](#data-management)
3. [Defining Models](#defining-models)
4. [Search Spaces](#search-spaces)
5. [Cross-Validation](#cross-validation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Model Stacking](#model-stacking)
8. [Feature Selection](#feature-selection)
9. [Reparameterization](#reparameterization)
10. [Knowledge Distillation](#knowledge-distillation)
11. [Joint Quantile Regression](#joint-quantile-regression)
12. [Estimator Scaling](#estimator-scaling)
13. [Working with Results](#working-with-results)
14. [Advanced Usage](#advanced-usage)
15. [Project Structure](#project-structure)
16. [Running Tests](#running-tests)

## Quick Start

sklearn-meta separates graph definition from execution. First, define a model graph with `GraphBuilder`, then configure a `RunConfig`, and finally execute with `GraphRunner` (or the convenience `sklearn_meta.fit()` function):

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn_meta import GraphBuilder, RunConfigBuilder, DataView, fit

# 1. Define the model graph
graph = (
    GraphBuilder("my_pipeline")
    .add_model("rf", RandomForestClassifier)
    .param("n_estimators", 50, 500)
    .param("max_depth", 3, 20)
    .fixed_params(random_state=42)
    .compile()
)

# 2. Configure the run
config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .build()
)

# 3. Create data and fit
data = DataView.from_Xy(X_train, y_train)
training_run = fit(graph, data, config)

# 4. Compile for inference and predict
inference = training_run.compile_inference()
predictions = inference.predict(X_test)
```

This tunes a random forest over 50 Optuna trials using 5-fold stratified CV. At inference time, predictions from all fold models are averaged to produce the final output.

## Data Management

sklearn-meta uses `DataView` as a lazy, immutable view over datasets. It wraps a `DatasetRecord` and declares column roles (features, target, groups) rather than copying data. No data is materialized until `.materialize()` is called.

### Creating a DataView

```python
from sklearn_meta import DataView

# From separate X, y, groups
data = DataView.from_Xy(
    X=X_train,                         # pd.DataFrame of features
    y=y_train,                         # pd.Series or array-like target
    groups=group_labels,               # Optional array-like for group CV
    base_margin=margin_array,          # Optional aux channel for XGBoost base margins
)
```

Or construct from a `DatasetRecord` with column roles declared explicitly:

```python
from sklearn_meta.data.record import DatasetRecord

record = DatasetRecord.from_frame(full_dataframe)
data = DataView(
    dataset=record,
    feature_cols=("feat_1", "feat_2", "feat_3"),
).bind_target(full_dataframe["target"].values).bind_groups(full_dataframe["patient_id"].values)
```

### Accessing Data

DataView provides properties for inspecting the view without materializing:

```python
data.n_rows         # Number of rows (respecting any row selection)
data.n_features     # Number of feature columns + overlays
data.feature_cols   # Tuple of feature column names
data.target         # Default target channel reference (or None)
data.groups         # Group labels reference (or None)
```

To get concrete arrays, call `.materialize()`:

```python
batch = data.materialize()
batch.X              # Feature DataFrame (features + overlay columns)
batch.targets        # Dict of resolved target arrays
batch.row_ids        # Row identifiers
batch.feature_names  # List of feature column names
```

### Immutable Transformations

DataView is frozen -- every transformation returns a new instance, leaving the original unchanged. This prevents subtle bugs from shared mutable state during cross-validation and stacking.

```python
# Restrict to specific feature columns
data2 = data.select_features(["feat_1", "feat_3"])

# Subset to specific row indices
data3 = data.select_rows(train_indices)

# Add an overlay column (e.g., upstream OOF predictions for stacking)
data4 = data.with_overlay("pred_rf", rf_oof_predictions)

# Add multiple overlays at once
data5 = data.with_overlays({"pred_rf": rf_preds, "pred_xgb": xgb_preds})

# Bind a target channel
data6 = data.bind_target(new_target_array)

# Bind groups for CV splitting
data7 = data.bind_groups(group_labels)

# Add an auxiliary channel (e.g., soft targets for distillation)
data8 = data.with_aux("soft_targets", teacher_probabilities)
```

### CVEngine

`CVEngine` handles all cross-validation splitting logic. It is created automatically by `GraphRunner`, but can be used directly:

```python
from sklearn_meta import CVConfig, CVStrategy
from sklearn_meta.engine.cv import CVEngine

cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
engine = CVEngine(cv_config)

# Create CV folds
folds = engine.create_folds(data)
for fold in folds:
    print(f"Fold {fold.fold_idx}: {fold.n_train} train, {fold.n_val} val")

# Create nested folds (requires inner_cv in config)
nested_config = cv_config.with_inner_cv(n_splits=3)
nested_engine = CVEngine(nested_config)
nested_folds = nested_engine.create_nested_folds(data)
for nf in nested_folds:
    print(f"Outer fold {nf.fold_idx}, {nf.n_inner_folds} inner folds")
```

## Defining Models

Add models to the graph with `add_model(name, estimator_class)`. Each model returns a `NodeBuilder` that supports chained configuration:

```python
builder = (
    GraphBuilder("classifier")
    .add_model("xgb", XGBClassifier)
    .param("learning_rate", 0.01, 0.3, log=True)
    .int_param("max_depth", 3, 10)
    .fixed_params(random_state=42, use_label_encoder=False)
    .fit_params(verbose=False)
    .output_type("proba")                  # "prediction", "proba", or "transform"
    .description("Base XGBoost model")
    .plugins("xgboost")                    # Model-specific plugin hooks
)
```

### Conditional Models

Models can be conditionally included based on the data:

```python
.add_model("binary_only", LogisticRegression)
.condition(lambda data: len(set(data.targets.get("__default__", []))) == 2)
```

### Feature Subsets

Restrict a model to specific input features:

```python
.add_model("text_model", SGDClassifier)
.feature_cols(["tfidf_1", "tfidf_2", "tfidf_3"])
```

## Search Spaces

Hyperparameter search spaces are defined using typed builder methods on `NodeBuilder`:

```python
.add_model("xgb", XGBClassifier)
.int_param("n_estimators", 50, 500)              # Integer range [50, 500]
.int_param("max_depth", 3, 15)                   # Integer range [3, 15]
.param("learning_rate", 0.001, 0.3, log=True)    # Float range, log-uniform
.param("subsample", 0.5, 1.0)                    # Float range [0.5, 1.0]
.cat_param("booster", ["gbtree", "dart"])         # Categorical choice
```

The builder methods:
- `.param(name, low, high)` creates a float range; add `log=True` for log-uniform sampling, `step=` for discrete steps
- `.int_param(name, low, high)` creates an integer range; add `step=` for step size
- `.cat_param(name, choices)` creates a categorical parameter

For full control, build a `SearchSpace` manually and pass it with `.search_space()`:

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

.add_model("xgb", XGBClassifier).search_space(space)
```

## Cross-Validation

Configure cross-validation with `RunConfigBuilder.cv()`:

```python
RunConfigBuilder()
.cv(
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
| `"group"` | Grouped data (bind `groups` on `DataView`) |
| `"time_series"` | Temporal data (no shuffling, expanding window) |

### Nested CV

For unbiased performance estimates during hyperparameter tuning, use `CVConfig.with_inner_cv()`:

```python
from sklearn_meta import CVConfig, CVStrategy

cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED)
nested_config = cv_config.with_inner_cv(n_splits=3)

config = RunConfig(cv=nested_config, tuning=TuningConfig(...))
```

This creates an outer 5-fold loop for evaluation and an inner 3-fold loop for tuning within each outer fold.

### Group CV

When observations are grouped (e.g., multiple samples per patient):

```python
from sklearn_meta import GraphBuilder, RunConfigBuilder, DataView, fit

graph = (
    GraphBuilder("grouped")
    .add_model("rf", RandomForestClassifier)
    .param("n_estimators", 50, 200)
    .compile()
)

config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="group")
    .tuning(n_trials=30, metric="roc_auc", greater_is_better=True)
    .build()
)

data = DataView.from_Xy(X_train, y_train, groups=patient_ids)
training_run = fit(graph, data, config)
```

## Hyperparameter Tuning

Configure tuning with `RunConfigBuilder.tuning()`:

```python
RunConfigBuilder()
.tuning(
    n_trials=100,                  # Number of Optuna trials
    timeout=3600,                  # Optional timeout in seconds
    strategy="layer_by_layer",     # Optimization strategy
    metric="neg_mean_squared_error",
    greater_is_better=False,
    early_stopping_rounds=20,      # Stop if no improvement for 20 trials
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
from sklearn_meta import GraphBuilder, RunConfigBuilder, DataView, fit

# 1. Define the stacking graph
graph = (
    GraphBuilder("stacking_ensemble")

    # Layer 0: base models
    .add_model("rf", RandomForestClassifier)
    .param("n_estimators", 50, 300)
    .param("max_depth", 3, 15)

    .add_model("gb", GradientBoostingClassifier)
    .param("learning_rate", 0.01, 0.3, log=True)
    .param("n_estimators", 50, 300)
    .int_param("max_depth", 3, 8)

    # Layer 1: meta-learner stacks base predictions
    .add_model("meta", LogisticRegression)
    .fixed_params(C=1.0)
    .stacks("rf", "gb")               # Use predictions as features

    .compile()
)

# 2. Configure and run
config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .build()
)

data = DataView.from_Xy(X_train, y_train)
training_run = fit(graph, data, config)

inference = training_run.compile_inference()
predictions = inference.predict(X_test)
```

### Stacking Modes

- `.stacks("rf", "gb")` -- Use point predictions as features
- `.stacks_proba("rf", "gb")` -- Use class probabilities as features
- `.depends_on("rf", dep_type=DependencyType.TRANSFORM)` -- Use transformed features

With `layer_by_layer` strategy, layer 0 models (`rf`, `gb`) are tuned first. Their out-of-fold predictions are then used as training features for the layer 1 meta-learner.

### Deeper Stacking

You can stack any number of layers:

```python
graph = (
    GraphBuilder("deep_stack")
    # Layer 0
    .add_model("rf", RandomForestClassifier)
    .param("n_estimators", 50, 300)

    .add_model("xgb", XGBClassifier)
    .param("learning_rate", 0.01, 0.3, log=True)

    .add_model("lgb", LGBMClassifier)
    .param("learning_rate", 0.01, 0.3, log=True)

    # Layer 1: intermediate meta-learners
    .add_model("meta_tree", GradientBoostingClassifier)
    .stacks("rf", "xgb", "lgb")

    .add_model("meta_linear", LogisticRegression)
    .stacks_proba("rf", "xgb", "lgb")

    # Layer 2: final blender
    .add_model("blender", LogisticRegression)
    .stacks("meta_tree", "meta_linear")

    .compile()
)

config = (
    RunConfigBuilder()
    .cv(n_splits=5)
    .tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .build()
)

data = DataView.from_Xy(X_train, y_train)
training_run = fit(graph, data, config)
```

## Feature Selection

Automatically drop uninformative features before final training. Three methods are available. Configure feature selection on the `RunConfigBuilder`:

### Shadow Feature Selection (Recommended)

Uses round-based paired shadows. Across `n_shadows` rounds, each real feature
(or explicit feature group) is paired with synthetic noise and compared against
its paired shadow importance. Features that cannot beat their calibrated
shadow baseline are dropped.

```python
config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .feature_selection(
        method="shadow",
        n_shadows=5,                   # Number of shadow rounds
        threshold_mult=1.414,          # Multiplier on shadow importance threshold
        retune_after_pruning=True,     # Re-tune hyperparameters with selected features
        min_features=1,                # Never drop below this many features
        max_features=None,             # Optional upper bound
        feature_groups={               # Optional: select/drop grouped features together
            "gender_ohe": ["gender_f", "gender_m"],
            "state_ohe": ["state_ca", "state_ny", "state_tx"],
        },
    )
    .build()
)
```

Grouped features are scored by averaging member importances. Any group that passes or fails does so as a unit, so all group members are kept or dropped together.

### Permutation Importance

Measures importance by shuffling each feature and observing the drop in validation performance:

```python
.feature_selection(method="permutation", retune_after_pruning=True)
```

### Threshold

Simple threshold on raw feature importances:

```python
.feature_selection(method="threshold", retune_after_pruning=True)
```

### Retune After Pruning

When `retune_after_pruning=True` (the default), after features are selected the search space is narrowed around the previous best parameters and re-tuned using only the selected features. The narrowed space biases toward less regularization, since feature removal itself acts as regularization.

## Reparameterization

Many hyperparameters are correlated (e.g., `learning_rate` and `n_estimators` have an inverse relationship). Reparameterization transforms them into orthogonal dimensions so the optimizer explores the space more efficiently.

```python
config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=100, metric="roc_auc", greater_is_better=True)
    .reparameterization(use_prebaked=True)
    .build()
)
```

With `use_prebaked=True`, sklearn-meta automatically applies known reparameterizations for common model/parameter pairs (e.g., the learning-rate/n-estimators product for boosting models).

### Custom Reparameterizations

```python
from sklearn_meta import LogProductReparameterization

config = (
    RunConfigBuilder()
    .reparameterization(
        custom_reparameterizations=[
            LogProductReparameterization("budget", "learning_rate", "n_estimators"),
        ],
        use_prebaked=False,
    )
    .build()
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
from sklearn_meta import GraphBuilder, RunConfigBuilder, DataView, fit

# Define the distillation graph
graph = (
    GraphBuilder("distillation")

    # Teacher: large model
    .add_model("teacher", GradientBoostingClassifier)
    .param("n_estimators", 200, 1000)
    .param("learning_rate", 0.01, 0.3, log=True)
    .int_param("max_depth", 4, 12)
    .output_type("proba")

    # Student: smaller model learns from teacher's soft targets
    .add_model("student", XGBClassifier)
    .param("n_estimators", 10, 100)
    .param("learning_rate", 0.01, 0.3, log=True)
    .int_param("max_depth", 2, 6)
    .distill_from("teacher", temperature=3.0, alpha=0.5)

    .compile()
)

config = (
    RunConfigBuilder()
    .cv(n_splits=5)
    .tuning(n_trials=50, metric="neg_log_loss", greater_is_better=False)
    .build()
)

data = DataView.from_Xy(X_train, y_train)
training_run = fit(graph, data, config)
```

Parameters:
- **temperature** -- Softens the teacher's probability distribution. Higher values produce smoother targets.
- **alpha** -- Blend weight between the soft KL-divergence loss and the hard cross-entropy loss. `alpha=1.0` uses only soft targets; `alpha=0.0` ignores the teacher entirely.

The teacher must be fitted before the student in the graph's topological order. Only one teacher per student is supported.

## Joint Quantile Regression

Model multiple correlated targets with uncertainty quantification using sequential quantile regression via chain rule decomposition. Each conditional distribution is modeled with quantile regression (e.g., 10-20 quantile levels) using XGBoost:

```
P(Y1, Y2, ..., Yn | X) = P(Y1|X) x P(Y2|X,Y1) x P(Y3|X,Y1,Y2) x ...
```

```python
from sklearn_meta.spec.quantile import JointQuantileGraphSpec, JointQuantileConfig
from sklearn_meta.artifacts.inference import JointQuantileInferenceGraph
from xgboost import XGBRegressor

config = JointQuantileConfig(
    property_names=["price", "volume", "volatility"],
    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
    estimator_class=XGBRegressor,
    n_inference_samples=1000,
)

# ... build graph, runner, and fit ...

# Sample from the joint distribution
samples = jq_inference.sample_joint(X_test, n_samples=1000)

# Point predictions and prediction intervals
medians = jq_inference.predict_median(X_test)
q10 = jq_inference.predict_quantile(X_test, q=0.1)
q90 = jq_inference.predict_quantile(X_test, q=0.9)
```

For full details including order search, quantile scaling, sampling strategies, and save/load, see [Joint Quantile Regression](docs/joint-quantile-regression.md).

## Estimator Scaling

For boosting models, you can tune with a small number of estimators for speed, then scale up for the final cross-validated models used at inference:

### Fixed Scaling

```python
config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=100, metric="neg_log_loss", greater_is_better=False)
    .estimator_scaling(
        tuning_n_estimators=100,     # Use 100 trees during tuning (fast)
        final_n_estimators=500,      # Use 500 trees in the final CV pass
    )
    .build()
)
```

The learning rate is automatically scaled by `tuning_n_estimators / final_n_estimators` to maintain equivalent training during the final CV pass.

### Automatic Scaling Search

Search for the best scaling factor automatically:

```python
config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=100, metric="neg_log_loss", greater_is_better=False)
    .estimator_scaling(
        scaling_search=True,                        # Search optimal multiplier
        scaling_factors=[2, 5, 10, 20],             # Factors to test (default)
    )
    .build()
)
```

This tunes hyperparameters first, then tests each scaling factor with early stopping if performance degrades.

## Working with Results

`GraphRunner.fit()` returns a `TrainingRun` containing all trained models and metadata. To make predictions, compile an `InferenceGraph` from the training run.

### Predictions

```python
training_run = fit(graph, data, config)

# Compile to an InferenceGraph for prediction
inference = training_run.compile_inference()

# Predict using the final (leaf) node
predictions = inference.predict(X_test)

# For classifier leaves, get class probabilities directly
probabilities = inference.predict_proba(X_test)

# Predict from a specific node
rf_predictions = inference.predict(X_test, node_name="rf")
rf_probabilities = inference.predict_proba(X_test, node_name="rf")
```

For stacking graphs, `.predict()` automatically chains predictions through the graph: base model predictions are computed first and fed as features to downstream models.

### Probability Predictions

For stacking graphs where a base model's output type is set to `"proba"`, the probabilities are automatically passed as features to downstream meta-learners during training and inference. Set this on the base model:

```python
.add_model("rf", RandomForestClassifier)
.output_type("proba")
```

To get probability outputs from an `InferenceGraph`, call `predict_proba()` on the compiled inference graph:

```python
probas = inference.predict_proba(X_test)
rf_probas = inference.predict_proba(X_test, node_name="rf")
```

Manual fold-level averaging is still available for advanced inspection:

```python
import numpy as np

models = training_run.node_results["rf"].models
probas = np.mean(
    [model.predict_proba(X_test) for model in models],
    axis=0,
)
```

### Inspecting Results

```python
# Access a node's run result
result = training_run.node_results["rf"]

# Best hyperparameters found
print(result.best_params)

# Out-of-fold predictions (useful for analysis and diagnostics)
oof = result.oof_predictions

# CV performance
print(result.cv_result.mean_score)
print(result.cv_result.std_score)

# Features kept after selection
print(result.selected_features)

# Total training time
print(f"Completed in {training_run.total_time:.1f}s")
```

### Extracting Feature Importance

```python
from sklearn_meta.selection.importance import importance_registry

models = training_run.node_results["rf"].models
importance = importance_registry.extract_importance(models[0], feature_names)
```

### Saving and Loading

#### TrainingRun (full training artifacts)

```python
# Save -- creates a directory with fold models + JSON manifest
training_run.save("./models/my_pipeline/")

# Load -- restores the full TrainingRun with all fold results
loaded_run = TrainingRun.load("./models/my_pipeline/")
inference = loaded_run.compile_inference()
predictions = inference.predict(X_test)
```

#### InferenceGraph (lightweight, prediction-only)

```python
# Save -- only the models and graph structure needed for prediction
inference = training_run.compile_inference()
inference.save("./models/my_pipeline_inference/")

# Load -- ready for inference
loaded_inference = InferenceGraph.load("./models/my_pipeline_inference/")
predictions = loaded_inference.predict(X_test)
```

For joint quantile models, each property's model is saved as an independent artifact with a JSON manifest capturing the graph structure. See [Joint Quantile Regression -- Saving and Loading Models](docs/joint-quantile-regression.md#saving-and-loading-models):

```python
from sklearn_meta.artifacts.inference import JointQuantileInferenceGraph

# Save -- creates one .joblib per quantile/fold + manifest.json
jq_inference.save("./models/joint_quantile/")

# Load -- ready for inference
loaded = JointQuantileInferenceGraph.load("./models/joint_quantile/")
```

## Advanced Usage

### Low-Level API

For full control, you can bypass the fluent builders and work with the spec/runtime/engine components directly:

```python
from sklearn_meta import (
    DataView, GraphRunner, RuntimeServices,
    RunConfig, TuningConfig, CVConfig, CVStrategy,
    SearchSpace, OptunaBackend,
)
from sklearn_meta.spec.node import NodeSpec
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.engine.strategy import OptimizationStrategy

# Build search space
space = SearchSpace()
space.add_int("n_estimators", 50, 200)
space.add_int("max_depth", 3, 10)

# Create model node
node = NodeSpec(
    name="rf",
    estimator_class=RandomForestClassifier,
    search_space=space,
    fixed_params={"random_state": 42},
)

# Build graph
graph = GraphSpec()
graph.add_node(node)
graph.validate()

# Configure
cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
tuning_config = TuningConfig(
    n_trials=20,
    metric="accuracy",
    greater_is_better=True,
    strategy=OptimizationStrategy.LAYER_BY_LAYER,
)
run_config = RunConfig(cv=cv_config, tuning=tuning_config)

# Create data and run
data = DataView.from_Xy(X=X_train, y=y_train)
services = RuntimeServices.default()
runner = GraphRunner(services)
training_run = runner.fit(graph, data, run_config)
```

### Custom Search Backend

The search backend is pluggable. The default is `OptunaBackend`, but you can configure it:

```python
from sklearn_meta import OptunaBackend, RuntimeServices
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

services = RuntimeServices(search_backend=backend)
runner = GraphRunner(services)
training_run = runner.fit(graph, data, config)
```

### Fit Caching

Cache fitted models to avoid redundant computation during optimization:

```python
from sklearn_meta import FitCache, RuntimeServices

cache = FitCache(cache_dir="./cache", max_size_mb=1000.0)
services = RuntimeServices(search_backend=OptunaBackend(), fit_cache=cache)
runner = GraphRunner(services)
```

### Plugins

Plugins hook into the model lifecycle (pre-fit, post-fit, search space modification, etc.). The built-in XGBoost plugin is applied with:

```python
.add_model("xgb", XGBClassifier).plugins("xgboost")
```

## Project Structure

```
sklearn_meta/
├── __init__.py            # Public API re-exports and fit() convenience function
├── spec/                  # Graph & node specifications (pure data, no runtime)
│   ├── builder.py         # GraphBuilder & NodeBuilder fluent API
│   ├── graph.py           # GraphSpec (DAG of NodeSpec + DependencyEdge)
│   ├── node.py            # NodeSpec, OutputType
│   ├── dependency.py      # DependencyEdge, DependencyType
│   ├── distillation.py    # DistillationConfig
│   └── quantile.py        # JointQuantileGraphSpec, JointQuantileConfig
├── data/                  # DatasetRecord, DataView
│   ├── record.py          # DatasetRecord (immutable storage)
│   ├── view.py            # DataView (lazy, declarative view)
│   └── batch.py           # MaterializedBatch (concrete arrays)
├── runtime/               # RunConfig, RuntimeServices
│   ├── config.py          # RunConfig, TuningConfig, CVConfig, FeatureSelectionConfig
│   └── services.py        # RuntimeServices (search backend, cache, logger)
├── engine/                # Execution: GraphRunner, CVEngine, trainers
│   ├── runner.py          # GraphRunner (orchestrates training)
│   ├── cv.py              # CVEngine (fold splitting)
│   ├── trainer.py         # StandardNodeTrainer
│   ├── quantile_trainer.py # QuantileNodeTrainer
│   ├── search.py          # SearchService
│   ├── selection.py       # FeatureSelectionService
│   ├── strategy.py        # OptimizationStrategy enum
│   └── estimator_scaling.py # EstimatorScalingConfig, EstimatorScaler
├── artifacts/             # Training & inference outputs
│   ├── training.py        # TrainingRun, NodeRunResult, RunMetadata
│   ├── inference.py       # InferenceGraph, JointQuantileInferenceGraph
│   └── compiler.py        # InferenceCompiler (TrainingRun -> InferenceGraph)
├── search/                # SearchSpace, SearchParameter, Backends
├── meta/                  # Reparameterization transforms, CorrelationAnalyzer
├── selection/             # Feature selection (shadow, permutation, threshold)
│   └── importance.py      # Feature importance extraction registry
├── plugins/               # Model-specific plugins
│   ├── xgboost/           # XGBoost multiplier & importance plugins
│   └── joint_quantile/    # OrderSearchPlugin for property ordering
├── execution/             # Parallel execution backends
├── persistence/           # Fit caching, manifest I/O
└── audit/                 # Logging
```

## Running Tests

```bash
python3 -m pip install -e .[dev]

# Lint
python3 -m ruff check .

# Run all tests
python3 -m pytest

# Build wheel and sdist
python3 -m build --sdist --wheel
```

## License

MIT License
