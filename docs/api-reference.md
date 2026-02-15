# API Reference

Complete API documentation for sklearn-meta.

---

## GraphBuilder (Fluent API)

```python
from sklearn_meta import GraphBuilder
```

A fluent API for building model graphs with minimal boilerplate. Provides a chainable interface for defining models, search spaces, dependencies, and tuning configuration.

```python
GraphBuilder(name: str = "pipeline")
```

**Methods:**
```python
# Add a model to the graph
add_model(name: str, estimator_class: type) -> NodeBuilder

# Configure cross-validation
with_cv(
    n_splits: int = 5,
    n_repeats: int = 1,
    strategy: str | CVStrategy = "stratified",
    shuffle: bool = True,
    random_state: int = 42,
    nested: bool = False,
    inner_splits: int = 3,
) -> GraphBuilder

# Configure hyperparameter tuning
with_tuning(
    n_trials: int = 100,
    timeout: float | None = None,
    strategy: str | OptimizationStrategy = "layer_by_layer",
    metric: str = "neg_mean_squared_error",
    greater_is_better: bool = False,
    early_stopping_rounds: int | None = None,
    n_parallel_trials: int = 1,
    tuning_n_estimators: int | None = None,
    final_n_estimators: int | None = None,
    estimator_scaling_search: bool = False,
    estimator_scaling_factors: list[int] | None = None,
    show_progress: bool = False,
) -> GraphBuilder

# Configure feature selection
with_feature_selection(
    method: str = "shadow",
    n_shadows: int = 5,
    threshold_mult: float = 1.414,
    retune_after_pruning: bool = True,
    min_features: int = 1,
    max_features: int | None = None,
) -> GraphBuilder

# Configure reparameterization
with_reparameterization(
    reparameterizations: list[Reparameterization] | None = None,
    use_prebaked: bool = True,
) -> GraphBuilder

# Build the ModelGraph
build() -> ModelGraph

# Create TuningOrchestrator
create_orchestrator(
    search_backend: SearchBackend | None = None,
    executor: Executor | None = None,
) -> TuningOrchestrator

# Build and fit in one step
fit(
    X, y,
    groups=None,
    search_backend=None,
    executor=None,
) -> FittedGraph
```

### NodeBuilder

Returned by `add_model()` for configuring individual nodes.

```python
# Set search space (shorthand syntax)
with_search_space(
    space: SearchSpace | None = None,
    **kwargs,  # e.g., n_estimators=(50, 500), max_depth=(3, 20)
) -> NodeBuilder

# Set output type
with_output_type(output_type: str) -> NodeBuilder  # "prediction", "proba", "transform"

# Set execution condition
with_condition(condition: Callable[[DataContext], bool]) -> NodeBuilder

# Add plugins (by string name)
with_plugins(*plugins: str) -> NodeBuilder

# Set fixed (non-tuned) parameters
with_fixed_params(**params) -> NodeBuilder

# Set fit parameters
with_fit_params(**params) -> NodeBuilder

# Specify features to use
with_features(*feature_cols: str) -> NodeBuilder

# Add description
with_description(description: str) -> NodeBuilder

# Add dependencies
depends_on(*sources: str, dep_type: DependencyType = DependencyType.PREDICTION) -> NodeBuilder
stacks(*sources: str) -> NodeBuilder           # Shortcut for prediction dependencies
stacks_proba(*sources: str) -> NodeBuilder     # Shortcut for probability dependencies

# Knowledge distillation
distills(teacher: str, temperature: float = 3.0, alpha: float = 0.5) -> NodeBuilder
```

NodeBuilder also forwards the following methods back to GraphBuilder, allowing continued chaining:
- `add_model()`, `build()`, `with_cv()`, `with_tuning()`, `with_feature_selection()`, `with_reparameterization()`, `create_orchestrator()`, `fit()`

**Example:**
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_meta import GraphBuilder

# Build a stacking pipeline with fluent API
fitted = (
    GraphBuilder("stacking_pipeline")
    .add_model("rf", RandomForestClassifier)
    .with_search_space(
        n_estimators=(50, 300),
        max_depth=(3, 15),
    )
    .with_fixed_params(random_state=42, n_jobs=-1)
    .add_model("gbm", GradientBoostingClassifier)
    .with_search_space(
        n_estimators=(50, 200),
        learning_rate=(0.01, 0.3, "log"),
        max_depth=(3, 8),
    )
    .add_model("meta", LogisticRegression)
    .stacks("rf", "gbm")
    .with_cv(n_splits=5, strategy="stratified")
    .with_tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .fit(X_train, y_train)
)

predictions = fitted.predict(X_test)
```

**Knowledge Distillation Example:**
```python
fitted = (
    GraphBuilder("distillation")
    .add_model("teacher", XGBClassifier)
    .with_search_space(n_estimators=(100, 500), max_depth=(3, 10))
    .add_model("student", LogisticRegression)
    .distills("teacher", temperature=3.0, alpha=0.5)
    .with_cv(n_splits=5)
    .with_tuning(n_trials=50)
    .fit(X_train, y_train)
)
```

---

## Core Module

### DataContext

```python
from sklearn_meta.core.data.context import DataContext
```

Immutable container for training data and metadata.

**Constructor:**
```python
DataContext(
    df: pd.DataFrame,
    feature_cols: tuple,
    target_col: str | None,
    group_col: str | None = None,
    base_margin: np.ndarray | None = None,
    soft_targets: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    metadata: dict = {},
)
```

**Factory Method:**
```python
DataContext.from_Xy(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    base_margin: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    metadata: dict | None = None,
) -> DataContext
```

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `X` | DataFrame | Feature matrix |
| `y` | Series | Target values |
| `groups` | Series | Group labels |
| `n_samples` | int | Number of samples |
| `n_features` | int | Number of features |
| `feature_names` | list | Feature column names |

**Methods:**
```python
# Return new DataContext with different feature columns
ctx.with_feature_cols(feature_cols) -> DataContext

# Return new DataContext with a different target column
ctx.with_target_col(target_col) -> DataContext

# Add columns to the underlying DataFrame
ctx.with_columns(as_features=False, **cols) -> DataContext

# Return new DataContext with specific row indices
ctx.with_indices(indices) -> DataContext

# Return new DataContext with base margin
ctx.with_base_margin(base_margin) -> DataContext

# Return new DataContext with soft targets (for distillation)
ctx.with_soft_targets(soft_targets) -> DataContext

# Return new DataContext with updated metadata (key-value pair)
ctx.with_metadata(key: str, value: Any) -> DataContext

# Return new DataContext with replacement target
ctx.with_y(y) -> DataContext

# Augment context with upstream model predictions as new feature columns
ctx.augment_with_predictions(predictions: dict, prefix: str = "pred_") -> DataContext

# Deep copy
ctx.copy() -> DataContext
```

DataContext is immutable -- all `with_*` methods return a new instance rather than modifying in place.

---

### CVConfig

```python
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
```

Cross-validation configuration.

```python
CVConfig(
    n_splits: int = 5,
    n_repeats: int = 1,
    strategy: CVStrategy = CVStrategy.GROUP,
    shuffle: bool = True,
    random_state: int = 42,
    inner_cv: CVConfig | None = None,
)
```

**CVStrategy Enum:**
| Value | Description |
|-------|-------------|
| `GROUP` | Keep groups together |
| `STRATIFIED` | Preserve class ratios |
| `RANDOM` | Simple random splits |
| `TIME_SERIES` | Temporal ordering |

Note: There is no `KFOLD` strategy. Use `RANDOM` for simple random splits.

**Methods:**
```python
# Enable nested CV by adding an inner CV configuration
cv_config.with_inner_cv(n_splits: int = 3, strategy: CVStrategy | None = None) -> CVConfig
```

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `is_nested` | bool | Whether nested CV is configured |
| `total_folds` | int | Total number of folds (n_splits * n_repeats) |

---

### DataManager

```python
from sklearn_meta.core.data.manager import DataManager
```

Manages CV fold creation and data routing.

```python
DataManager(cv_config: CVConfig)
```

**Methods:**
```python
# Create CV folds
create_folds(ctx: DataContext) -> list[CVFold]

# Create nested CV folds (for nested cross-validation)
create_nested_folds(ctx: DataContext) -> list[NestedCVFold]

# Split context into train/val for a given fold
align_to_fold(ctx: DataContext, fold: CVFold) -> tuple[DataContext, DataContext]

# Combine out-of-fold predictions into full array
route_oof_predictions(ctx: DataContext, fold_results: list) -> np.ndarray

# Aggregate fold-level results into a single CVResult
aggregate_cv_result(
    node_name: str,
    fold_results: list,
    ctx: DataContext,
) -> CVResult
```

---

### ModelNode

```python
from sklearn_meta.core.model.node import ModelNode
```

Represents a single model in the graph. Defined as a dataclass.

```python
@dataclass
class ModelNode:
    name: str
    estimator_class: type
    search_space: SearchSpace | None = None
    output_type: str = "prediction"          # "prediction", "proba", "transform"
    condition: Callable | None = None
    plugins: list[str] = field(default_factory=list)  # Plugin names, not instances
    fixed_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)
    feature_cols: list[str] | None = None
    description: str = ""
    distillation_config: DistillationConfig | None = None
```

Note: The `plugins` field is a `list[str]` of plugin names (e.g., `["xgboost"]`), not a list of `ModelPlugin` instances. Plugins are resolved from the registry by name at runtime.

---

### ModelGraph

```python
from sklearn_meta.core.model.graph import ModelGraph
```

Directed acyclic graph of model nodes.

```python
ModelGraph()
```

**Methods:**
```python
# Add node to graph
add_node(node: ModelNode)

# Add edge between nodes
add_edge(edge: DependencyEdge)

# Get node by name
get_node(name: str) -> ModelNode

# Get topologically ordered node names
topological_order() -> list[str]

# Get nodes grouped by execution layer
get_layers() -> list[list[str]]

# Get root nodes (no incoming edges)
get_root_nodes() -> list[str]

# Get leaf nodes (no outgoing edges)
get_leaf_nodes() -> list[str]

# Get edges pointing into a node
get_upstream(name: str) -> list[DependencyEdge]

# Get edges pointing out from a node
get_downstream(name: str) -> list[DependencyEdge]

# Validate graph structure, returns list of warnings
validate() -> list[str]

# Extract a subgraph containing only specified nodes
subgraph(node_names: set[str]) -> ModelGraph

# Get all ancestor node names
ancestors(name: str) -> set[str]

# Get all descendant node names
descendants(name: str) -> set[str]
```

---

### DependencyEdge and DependencyType

```python
from sklearn_meta.core.model.dependency import DependencyEdge, DependencyType
```

```python
@dataclass
class DependencyEdge:
    source: str
    target: str
    dep_type: DependencyType = DependencyType.PREDICTION
    column_name: str | None = None
    conditional_config: ConditionalSampleConfig | None = None
```

**DependencyType Enum:**
| Value | Description |
|-------|-------------|
| `PREDICTION` | Pass class predictions as features |
| `PROBA` | Pass probability predictions as features |
| `TRANSFORM` | Pass transformed features |
| `FEATURE` | Pass raw features |
| `BASE_MARGIN` | Pass predictions as base margin (boosting init) |
| `CONDITIONAL_SAMPLE` | Conditional sample routing |
| `DISTILL` | Knowledge distillation (soft targets from teacher) |

---

### DistillationConfig

```python
from sklearn_meta.core.model.distillation import DistillationConfig
```

```python
@dataclass
class DistillationConfig:
    temperature: float = 3.0   # Softens probability distributions before KL computation
    alpha: float = 0.5         # Blending weight between soft and hard losses
```

---

### TuningConfig

```python
from sklearn_meta.core.tuning.orchestrator import TuningConfig
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
```

```python
TuningConfig(
    strategy: OptimizationStrategy = OptimizationStrategy.LAYER_BY_LAYER,
    n_trials: int = 100,
    timeout: float | None = None,
    early_stopping_rounds: int | None = None,
    cv_config: CVConfig | None = None,
    metric: str = "neg_mean_squared_error",
    greater_is_better: bool = False,
    feature_selection: FeatureSelectionConfig | None = None,
    use_reparameterization: bool = False,
    custom_reparameterizations: list | None = None,
    verbose: int = 1,
    tuning_n_estimators: int | None = None,
    final_n_estimators: int | None = None,
    estimator_scaling_search: bool = False,
    estimator_scaling_factors: list[int] | None = None,
    show_progress: bool = False,
)
```

**OptimizationStrategy Enum:**
| Value | Description |
|-------|-------------|
| `LAYER_BY_LAYER` | Tune each graph layer sequentially |
| `GREEDY` | Greedily tune one node at a time |
| `NONE` | Skip tuning, use fixed params only |

---

### TuningOrchestrator

```python
from sklearn_meta.core.tuning.orchestrator import TuningOrchestrator
```

```python
TuningOrchestrator(
    graph: ModelGraph,
    data_manager: DataManager,
    search_backend: SearchBackend,
    tuning_config: TuningConfig,
    executor: Executor | None = None,
)
```

**Methods:**
```python
# Run optimization and return fitted graph
fit(ctx: DataContext) -> FittedGraph
```

---

### FittedGraph

```python
from sklearn_meta.core.tuning.orchestrator import FittedGraph
```

Result of fitting and tuning a model graph.

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `graph` | ModelGraph | The original model graph |
| `fitted_nodes` | dict | Map of node name to FittedNode |
| `tuning_config` | TuningConfig | Configuration used for tuning |
| `total_time` | float | Total fit time in seconds |

**Methods:**
```python
# Generate predictions (uses leaf node by default)
predict(X: pd.DataFrame, node_name: str | None = None) -> np.ndarray

# Get a specific fitted node
get_node(name: str) -> FittedNode

# Get out-of-fold predictions for a node
get_oof_predictions(name: str) -> np.ndarray
```

Note: There is no `predict_proba()` method or `best_params` dict on FittedGraph. Access per-node best params via `fitted_graph.get_node("name").best_params`.

---

### FittedNode

```python
from sklearn_meta.core.tuning.orchestrator import FittedNode
```

Result for a single fitted node.

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `node` | ModelNode | The original model node |
| `cv_result` | CVResult | Cross-validation results |
| `best_params` | dict | Best hyperparameters found |
| `optimization_result` | OptimizationResult | Full optimization history |
| `selected_features` | list | Features selected (if feature selection enabled) |

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `oof_predictions` | ndarray | Out-of-fold predictions |
| `models` | list | List of fitted model instances (one per fold) |
| `mean_score` | float | Mean CV score |

---

## Search Module

### SearchSpace

```python
from sklearn_meta.search.space import SearchSpace
```

```python
SearchSpace()
```

**Methods:**
```python
# Add parameters (all return self for chaining)
add_float(name, low, high, log=False, step=None) -> SearchSpace
add_int(name, low, high, log=False, step=1) -> SearchSpace
add_categorical(name, choices) -> SearchSpace

# Conditional parameters
add_conditional(
    name: str,
    parent_name: str,
    parent_value: Any,
    parameter: Parameter,
) -> SearchSpace

# Shorthand notation
add_from_shorthand(**kwargs) -> SearchSpace

# Narrow search space around a center point
narrow_around(
    center: dict,
    factor: float,
    regularization_bias: float,
) -> SearchSpace

# Sampling
sample_optuna(trial) -> dict

# Operations
copy() -> SearchSpace
merge(other: SearchSpace) -> SearchSpace
remove_parameter(name: str) -> SearchSpace
```

---

### Parameter Classes

```python
from sklearn_meta.search.parameter import (
    FloatParameter,
    IntParameter,
    CategoricalParameter,
    ConditionalParameter,
)
```

```python
FloatParameter(name, low, high, log=False, step=None)
IntParameter(name, low, high, log=False, step=None)
CategoricalParameter(name, choices)
ConditionalParameter(name, parent_name, parent_value, parameter)
```

---

### OptunaBackend

```python
from sklearn_meta.search.backends.optuna import OptunaBackend
```

```python
OptunaBackend(
    direction: str = "minimize",
    random_state: int | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    n_jobs: int = 1,
    show_progress_bar: bool = False,
    verbosity: int | None = None,
)
```

---

## Meta Module

### CorrelationAnalyzer

```python
from sklearn_meta.meta.correlation import CorrelationAnalyzer, HyperparameterCorrelation, CorrelationType
```

Analyzes optimization history to discover hyperparameter correlations. This helps identify parameters that provide similar effects, have tradeoff relationships, or should be tuned together.

```python
CorrelationAnalyzer(
    min_trials: int = 20,
    significance_threshold: float = 0.1,
    correlation_threshold: float = 0.3,
)
```

**Methods:**
```python
# Analyze optimization results for correlations
analyze(
    optimization_result: OptimizationResult,
    param_names: list[str] | None = None,
) -> list[HyperparameterCorrelation]

# Suggest reparameterizations based on discovered correlations
suggest_reparameterization(
    correlations: list[HyperparameterCorrelation],
) -> dict  # {"substitutes": [...], "tradeoffs": [...], "correlation_details": [...]}
```

**CorrelationType Enum:**
| Value | Description |
|-------|-------------|
| `SUBSTITUTE` | Parameters providing similar effects (e.g., L1 and L2 regularization) |
| `COMPLEMENT` | Parameters that work together and move in the same direction |
| `TRADEOFF` | Parameters with inverse relationship (e.g., learning_rate x n_estimators) |
| `CONDITIONAL` | One parameter's optimal value depends on another |

**HyperparameterCorrelation:**
```python
@dataclass
class HyperparameterCorrelation:
    params: list[str]
    correlation_type: CorrelationType
    strength: float
    functional_form: str
    transform: Callable | None
    inverse_transform: Callable | None
    confidence: float

    # Methods
    effective_value(param_values: dict[str, float]) -> float
    decompose(effective: float, ratio: float = 0.5) -> dict[str, float]
```

**Example:**
```python
from sklearn_meta.meta.correlation import CorrelationAnalyzer

# After running optimization
analyzer = CorrelationAnalyzer(min_trials=30)
correlations = analyzer.analyze(optimization_result)

for corr in correlations:
    print(f"{corr.params}: {corr.correlation_type.value} (strength={corr.strength:.2f})")
    print(f"  Relationship: {corr.functional_form}")

# Get suggested reparameterizations
suggestions = analyzer.suggest_reparameterization(correlations)
print(f"Tradeoffs found: {suggestions['tradeoffs']}")
```

---

### Reparameterizations

```python
from sklearn_meta.meta.reparameterization import (
    Reparameterization,
    LogProductReparameterization,
    RatioReparameterization,
    LinearReparameterization,
)
```

**Reparameterization (abstract base):**
```python
class Reparameterization(ABC):
    def __init__(self, name: str, original_params: list[str]): ...
    def forward(self, params: dict) -> dict: ...   # Original -> transformed
    def inverse(self, params: dict) -> dict: ...   # Transformed -> original
```

**Concrete implementations:**
```python
LogProductReparameterization(
    name: str,
    param1: str,
    param2: str,
)

RatioReparameterization(
    name: str,
    param1: str,
    param2: str,
)

LinearReparameterization(
    name: str,
    params: list[str],
    weights: list[float] | None = None,
)
```

---

### Prebaked Configs

```python
from sklearn_meta.meta.prebaked import get_prebaked_reparameterization

# Get recommended reparameterizations for a model and its search space params
reparams = get_prebaked_reparameterization(
    model_class=XGBClassifier,
    param_names=["learning_rate", "n_estimators", "max_depth"],
)
```

---

## Selection Module

### FeatureSelectionConfig

```python
from sklearn_meta.selection.selector import FeatureSelectionConfig
```

```python
@dataclass
class FeatureSelectionConfig:
    enabled: bool = True
    method: str = "shadow"                # "shadow", "permutation", "threshold"
    n_shadows: int = 5
    threshold_mult: float = 1.414
    threshold_percentile: float = 10.0    # For threshold method
    retune_after_pruning: bool = True
    min_features: int = 1
    max_features: int | None = None
    random_state: int = 42
```

---

### ShadowFeatureSelector

```python
from sklearn_meta.selection.shadow import ShadowFeatureSelector
```

```python
ShadowFeatureSelector(
    importance_extractor: ImportanceExtractor | None = None,
    n_shadows: int = 5,
    n_clusters: int = 5,
    threshold_mult: float = 1.414,
    random_state: int = 42,
)
```

**Methods:**
```python
# Fit model with shadow features and return detailed selection result
fit_select(model, X, y, feature_cols=None, importance_type="gain") -> ShadowResult

# Convenience: return just the list of features to keep
select_features(model, X, y, feature_cols=None) -> list[str]
```

**ShadowResult:**
```python
@dataclass
class ShadowResult:
    features_to_keep: list[str]
    features_to_drop: list[str]
    feature_importances: dict[str, float]
    shadow_importances: dict[str, float]
    feature_to_shadow: dict[str, str]
    threshold_used: float
```

---

## Plugins Module

### ModelPlugin

```python
from sklearn_meta.plugins.base import ModelPlugin
```

Abstract base class for plugins. Plugins hook into the model lifecycle at specific points.

```python
class ModelPlugin(ABC):
    @property
    def name(self) -> str: ...                    # Default: class name

    @abstractmethod
    def applies_to(self, estimator_class) -> bool: ...

    def modify_search_space(self, space, node) -> SearchSpace: ...
    def modify_params(self, params, node) -> dict: ...
    def modify_fit_params(self, params, ctx) -> dict: ...
    def pre_fit(self, model, node, ctx) -> Any: ...
    def post_fit(self, model, node, ctx) -> Any: ...
    def post_tune(self, best_params, node, ctx) -> dict: ...
    def on_fold_start(self, fold_idx, node, ctx) -> None: ...
    def on_fold_end(self, fold_idx, model, score, node) -> None: ...
```

Plugins are referenced by name (string) via the GraphBuilder API:
```python
.with_plugins("xgboost")
```

The `plugins` field on `ModelNode` is `list[str]`, not `list[ModelPlugin]`.

---

### CompositePlugin

```python
from sklearn_meta.plugins.base import CompositePlugin
```

```python
CompositePlugin(plugins: list[ModelPlugin])
```

---

### PluginRegistry

```python
from sklearn_meta.plugins.registry import PluginRegistry, get_default_registry
```

```python
registry = PluginRegistry()
registry.register(plugin, priority=0)
registry.unregister(plugin_name)
registry.get_plugins_for(estimator_class) -> list[ModelPlugin]

# Global registry
registry = get_default_registry()
```

---

### XGBoost Plugins

```python
from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin
from sklearn_meta.plugins.xgboost.importance import XGBImportancePlugin
```

```python
XGBMultiplierPlugin(
    multipliers: list[float] = [0.5, 1.0, 2.0],
    cv_folds: int = 3,
    enable_post_tune: bool = True,
)

XGBImportancePlugin(
    importance_type: str = "gain",  # "gain", "weight", "cover"
)
```

---

## Persistence Module

### FitCache

```python
from sklearn_meta.persistence.cache import FitCache
```

```python
FitCache(
    cache_dir: str | None = None,
    max_size_mb: float = 1000.0,
    enabled: bool = True,
)
```

---

### ArtifactStore

```python
from sklearn_meta.persistence.store import ArtifactStore
```

Abstract interface for storing models, parameters, and CV ensembles. Subclasses implement a concrete backend.

**Abstract Methods:**
```python
# Save a fitted model
save_model(
    model: Any,
    node_name: str,
    fold_idx: int = 0,
    params: dict | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
) -> str  # Returns artifact_id

# Load a saved model
load_model(artifact_id: str) -> Any

# Save all models from a fitted node
save_fitted_node(
    fitted_node: FittedNode,
    tags: dict[str, str] | None = None,
) -> list[str]  # Returns list of artifact IDs

# Save an entire fitted graph
save_fitted_graph(
    fitted_graph: FittedGraph,
    name: str,
    tags: dict[str, str] | None = None,
) -> str  # Returns graph artifact ID

# List stored artifacts
list_artifacts(
    artifact_type: str | None = None,  # "model", "params", "graph"
    node_name: str | None = None,
) -> list[dict]

# Delete an artifact
delete_artifact(artifact_id: str) -> bool
```

---

## Audit Module

### AuditLogger

```python
from sklearn_meta.audit.logger import AuditLogger
```

```python
AuditLogger(log_dir: str = "./logs")
```

**Methods:**
```python
log_trial(node_name: str, trial: TrialLog) -> None
log_fold(node_name: str, fold: FoldLog) -> None
get_trial_history(node_name: str) -> list[TrialLog]
get_best_params(node_name: str) -> dict
```

---

### Log Dataclasses

```python
from sklearn_meta.audit.logger import TrialLog, FoldLog

@dataclass
class TrialLog:
    trial_number: int
    params: dict
    score: float
    duration_seconds: float
    timestamp: datetime

@dataclass
class FoldLog:
    fold_index: int
    train_score: float
    val_score: float
    n_train: int
    n_val: int
```

---

## Execution Module

### Executor

```python
from sklearn_meta.execution.base import Executor
from sklearn_meta.execution.local import LocalExecutor, SequentialExecutor
```

Abstract base class for execution backends. Executors handle parallel or distributed execution of tasks.

**Base Executor Interface:**
```python
class Executor(ABC):
    map(fn: Callable[[T], R], items: list[T]) -> list[R]
    submit(fn: Callable[..., R], *args, **kwargs) -> Future[R]
    shutdown(wait: bool = True) -> None

    n_workers: int
```

**LocalExecutor:**
```python
LocalExecutor(n_jobs: int = -1)  # -1 means use all CPU cores
```

**SequentialExecutor:**
```python
SequentialExecutor()
```

**Context Manager Support:**
```python
with LocalExecutor(n_jobs=4) as executor:
    results = executor.map(process_item, items)
```

---

## Quick Import Reference

```python
# Fluent API (recommended for most use cases)
from sklearn_meta import GraphBuilder

# Core
from sklearn_meta import (
    DataContext,
    CVConfig,
    DataManager,
    ModelNode,
    ModelGraph,
    DependencyType,
    DependencyEdge,
    TuningOrchestrator,
    TuningConfig,
    OptimizationStrategy,
    DistillationConfig,
)

# CV Strategy enum (not re-exported from sklearn_meta)
from sklearn_meta.core.data.cv import CVStrategy

# Search
from sklearn_meta import SearchSpace, OptunaBackend
from sklearn_meta.search.parameter import FloatParameter, IntParameter, CategoricalParameter

# Meta-learning
from sklearn_meta import (
    CorrelationAnalyzer,
    HyperparameterCorrelation,
    Reparameterization,
    LogProductReparameterization,
    RatioReparameterization,
    LinearReparameterization,
    ReparameterizedSpace,
    get_prebaked_reparameterization,
)

# Selection
from sklearn_meta import FeatureSelectionConfig
from sklearn_meta.selection.shadow import ShadowFeatureSelector

# Plugins
from sklearn_meta.plugins.base import ModelPlugin, CompositePlugin
from sklearn_meta.plugins.registry import PluginRegistry, get_default_registry
from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin
from sklearn_meta.plugins.xgboost.importance import XGBImportancePlugin

# Persistence
from sklearn_meta import FitCache, AuditLogger
from sklearn_meta.persistence.store import ArtifactStore

# Execution
from sklearn_meta.execution.local import LocalExecutor, SequentialExecutor
```
