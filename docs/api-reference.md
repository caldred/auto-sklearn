# API Reference

Complete API documentation for sklearn-meta v2.

---

## 1. spec/ --- Graph and Node Specifications

Pure data definitions for the model graph. No training logic; training is handled by the engine.

---

### GraphSpec

```python
from sklearn_meta.spec.graph import GraphSpec
```

Directed acyclic graph of model nodes with topological ordering and layer extraction.

```python
GraphSpec()
```

**Methods:**

```python
# Add a model node to the graph
add_node(node: NodeSpec) -> None

# Add a dependency edge (raises CycleError if it would create a cycle)
add_edge(edge: DependencyEdge) -> None

# Get a node by name (raises KeyError if not found)
get_node(name: str) -> NodeSpec

# Get topologically ordered node names (dependencies before dependents)
topological_order() -> list[str]

# Get nodes grouped by execution layer
# Layer 0 = no dependencies; Layer N = all dependencies in layers < N
get_layers() -> list[list[str]]

# Get root nodes (no incoming edges)
get_root_nodes() -> list[str]

# Get leaf nodes (no outgoing edges)
get_leaf_nodes() -> list[str]

# Get edges pointing into a node (its dependencies)
get_upstream(node_name: str) -> list[DependencyEdge]

# Get edges pointing out from a node (nodes that depend on it)
get_downstream(node_name: str) -> list[DependencyEdge]

# Validate graph structure; returns list of warnings, raises CycleError/ValueError on critical issues
validate() -> list[str]

# Extract a subgraph containing only specified nodes and their mutual edges
subgraph(node_names: set[str]) -> GraphSpec

# Get all transitive ancestor node names
ancestors(name: str) -> set[str]

# Get all transitive descendant node names
descendants(name: str) -> set[str]
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `nodes` | `dict[str, NodeSpec]` | Dictionary of all nodes (copy) |
| `edges` | `list[DependencyEdge]` | List of all edges (copy) |

**Dunder methods:** `__len__`, `__contains__`, `__iter__` (topological order), `__repr__`.

---

### NodeSpec

```python
from sklearn_meta.spec.node import NodeSpec
```

Definition of a single model in the graph. Contains no training logic.

```python
@dataclass
class NodeSpec:
    name: str
    estimator_class: Type
    search_space: SearchSpace | None = None
    output_type: OutputType = OutputType.PREDICTION
    condition: Callable[..., bool] | None = None
    plugins: list[str] = field(default_factory=list)       # Plugin names, not instances
    fixed_params: dict[str, Any] = field(default_factory=dict)
    fit_params: dict[str, Any] = field(default_factory=dict)
    feature_cols: list[str] | None = None
    description: str = ""
    distillation_config: DistillationConfig | None = None
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_distilled` | `bool` | Whether this node uses knowledge distillation |
| `has_search_space` | `bool` | Whether this node has hyperparameters to tune |
| `is_conditional` | `bool` | Whether this node has a condition for execution |

**Methods:**

```python
# Check if this node should run given the current data
should_run(data) -> bool

# Create an estimator instance with fixed_params merged with optional overrides
create_estimator(params: dict | None = None) -> Any

# Get the output from a fitted model based on output_type
get_output(model: Any, X) -> Any

# Serialize / deserialize
to_dict() -> dict[str, Any]

@classmethod
from_dict(data: dict[str, Any]) -> NodeSpec
```

**Validation (in `__post_init__`):**
- `name` must not be empty.
- `estimator_class` must have a `fit` method.
- If `output_type` is `PREDICTION`, estimator must have `predict`.
- If `output_type` is `PROBA`, estimator must have `predict_proba`.
- If `output_type` is `TRANSFORM`, estimator must have `transform`.
- If `distillation_config` is set, estimator must accept an `objective` parameter.

---

### OutputType

```python
from sklearn_meta.spec.node import OutputType
```

```python
class OutputType(str, Enum):
    PREDICTION = "prediction"
    PROBA = "proba"
    TRANSFORM = "transform"
    QUANTILES = "quantiles"
```

---

### DependencyType

```python
from sklearn_meta.spec.dependency import DependencyType
```

```python
class DependencyType(Enum):
    PREDICTION = "prediction"        # Stacking: predictions become features
    TRANSFORM = "transform"          # Pipeline: transformed X flows to target
    FEATURE = "feature"              # Single feature engineering output
    PROBA = "proba"                  # Stacking with class probabilities
    BASE_MARGIN = "base_margin"      # XGBoost-style base margin init
    CONDITIONAL_SAMPLE = "conditional_sample"  # Joint quantile conditioning
    DISTILL = "distill"              # Knowledge distillation soft targets
```

---

### DependencyEdge

```python
from sklearn_meta.spec.dependency import DependencyEdge
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

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `feature_name` | `str` | Auto-generated feature name (e.g. `pred_rf`, `proba_xgb`), or `column_name` if set |

**Methods:** `to_dict()`, `from_dict(data)`.

**Validation:** Source and target must be non-empty, cannot be equal (no self-loops). `conditional_config` is required when `dep_type` is `CONDITIONAL_SAMPLE`.

**ConditionalSampleConfig:**

```python
from sklearn_meta.spec.dependency import ConditionalSampleConfig

@dataclass
class ConditionalSampleConfig:
    property_name: str
    use_actual_during_training: bool = True
```

---

### DistillationConfig

```python
from sklearn_meta.spec.distillation import DistillationConfig
```

```python
@dataclass(frozen=True)
class DistillationConfig:
    temperature: float = 3.0   # Must be > 0. Softens distributions before KL computation.
    alpha: float = 0.5         # Must be in [0, 1]. Loss = alpha * KL_soft + (1-alpha) * CE_hard.
```

---

### QuantileNodeSpec

```python
from sklearn_meta.spec.quantile import QuantileNodeSpec
```

Extends `NodeSpec` for quantile regression. Output type is automatically set to `QUANTILES`.

```python
@dataclass
class QuantileNodeSpec(NodeSpec):
    property_name: str = ""                     # Required; name of the target property
    quantile_levels: list[float] = DEFAULT_QUANTILE_LEVELS  # 19 levels, 0.05-0.95
    quantile_scaling: QuantileScalingConfig | None = None
    xgboost_objective: str = "reg:quantileerror"
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `median_quantile` | `float` | Quantile level closest to 0.5 |
| `n_quantiles` | `int` | Number of quantile levels |

**Methods:**

```python
# Create an estimator configured for a specific quantile level
create_estimator_for_quantile(tau: float, params: dict | None = None) -> Any

# Get complete parameter dict for a specific quantile level
get_params_for_quantile(tau: float, tuned_params: dict | None = None) -> dict
```

**QuantileScalingConfig:**

```python
from sklearn_meta.spec.quantile import QuantileScalingConfig

@dataclass
class QuantileScalingConfig:
    base_params: dict[str, Any] = field(default_factory=dict)
    scaling_rules: dict[str, dict[str, float]] = field(default_factory=dict)

    # Get scaled parameters for a specific quantile level
    get_params_for_quantile(tau: float) -> dict[str, Any]
```

**JointQuantileGraphSpec:**

```python
from sklearn_meta.spec.quantile import JointQuantileGraphSpec, JointQuantileConfig

@dataclass
class JointQuantileConfig:
    property_names: list[str]
    quantile_levels: list[float] = DEFAULT_QUANTILE_LEVELS
    estimator_class: Type | None = None
    search_space: SearchSpace | None = None
    quantile_scaling: QuantileScalingConfig | None = None
    order_constraints: OrderConstraint | None = None
    sampling_strategy: SamplingStrategy = SamplingStrategy.LINEAR_INTERPOLATION
    n_inference_samples: int = 1000
    random_state: int | None = None
    fixed_params: dict[str, Any] = field(default_factory=dict)

class JointQuantileGraphSpec(GraphSpec):
    def __init__(self, config: JointQuantileConfig) -> None: ...

    set_order(new_order: list[str]) -> None
    swap_adjacent(position: int) -> None
    get_valid_swaps() -> list[tuple[int, int]]
    get_quantile_node(property_name: str) -> QuantileNodeSpec
    get_conditioning_properties(property_name: str) -> list[str]
    create_quantile_sampler() -> QuantileSampler

    @property
    property_order -> list[str]
    @property
    n_properties -> int
    @property
    quantile_levels -> list[float]
```

---

### GraphBuilder and NodeBuilder

```python
from sklearn_meta.spec.builder import GraphBuilder, NodeBuilder
```

Fluent API for building `GraphSpec` objects. The builder produces a pure `GraphSpec` with no runtime concerns; CV, tuning, and fitting are handled separately by `RunConfig` and `GraphRunner`.

```python
GraphBuilder(name: str | None = None)
```

**GraphBuilder Methods:**

```python
# Add a model node, returns NodeBuilder for fluent configuration
add_model(name: str, estimator_class: Type) -> NodeBuilder

# Build and validate the GraphSpec
compile() -> GraphSpec
```

**NodeBuilder Methods:**

```python
# --- Search space ---
search_space(space: SearchSpace) -> NodeBuilder              # Set a pre-built SearchSpace
param(name, low_or_choices, high=None, log=False, step=None) -> NodeBuilder  # Add inferred int/float or categorical shorthand
int_param(name, low, high, step=1) -> NodeBuilder            # Add integer hyperparameter
cat_param(name, choices) -> NodeBuilder                      # Add categorical hyperparameter

# --- Node configuration ---
output_type(t: str | OutputType) -> NodeBuilder              # "prediction", "proba", "transform"
condition(fn: Callable[..., bool]) -> NodeBuilder            # Set execution condition
plugins(*names: str) -> NodeBuilder                          # Add plugin names
fixed_params(**kwargs) -> NodeBuilder                        # Set non-tuned parameters
fit_params(**kwargs) -> NodeBuilder                          # Set fit() parameters
feature_cols(cols: list[str]) -> NodeBuilder                 # Restrict feature columns
description(desc: str) -> NodeBuilder                        # Add description

# --- Dependencies ---
depends_on(source, dep_type=DependencyType.PREDICTION, column_name=None) -> NodeBuilder
stacks(*sources: str) -> NodeBuilder                         # Shortcut for PREDICTION dependencies
stacks_proba(*sources: str) -> NodeBuilder                   # Shortcut for PROBA dependencies
distill_from(teacher_name, alpha=0.5, temperature=3.0) -> NodeBuilder
```

NodeBuilder delegates unknown attributes to the parent `GraphBuilder` via `__getattr__`, enabling seamless chaining between node-level and graph-level methods (e.g., calling `.add_model()` or `.compile()` on a `NodeBuilder`).

**Example:**

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_meta.spec.builder import GraphBuilder

graph = (
    GraphBuilder("stacking")
    .add_model("rf", RandomForestClassifier)
        .param("n_estimators", 50, 300)
        .param("max_depth", 3, 15)
        .fixed_params(random_state=42, n_jobs=-1)
    .add_model("gbm", GradientBoostingClassifier)
        .param("n_estimators", 50, 200)
        .param("learning_rate", 0.01, 0.3, log=True)
        .param("max_depth", 3, 8)
    .add_model("meta", LogisticRegression)
        .stacks("rf", "gbm")
    .compile()
)
```

**Knowledge Distillation Example:**

```python
graph = (
    GraphBuilder("distillation")
    .add_model("teacher", XGBClassifier)
        .param("n_estimators", 100, 500)
        .param("max_depth", 3, 10)
    .add_model("student", LogisticRegression)
        .distill_from("teacher", temperature=3.0, alpha=0.5)
    .compile()
)
```

---

## 2. data/ --- Data Layer

Immutable, lazy data structures that defer copying until materialization.

---

### DatasetRecord

```python
from sklearn_meta.data.record import DatasetRecord
```

Immutable base table holding the full dataset. The frame is never copied; views and slices are deferred to `DataView.materialize()`.

```python
@dataclass(frozen=True)
class DatasetRecord:
    frame: pd.DataFrame
    row_ids: pd.Index
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

**Factory Method:**

```python
@classmethod
DatasetRecord.from_frame(df: pd.DataFrame, metadata: Mapping[str, Any] | None = None) -> DatasetRecord
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `n_rows` | `int` | Number of rows in the frame |

---

### DataView

```python
from sklearn_meta.data.view import DataView
```

Lazy, declarative view over a `DatasetRecord`. All mutating operations return new `DataView` instances. No data is copied until `materialize()` is called.

```python
@dataclass(frozen=True)
class DataView:
    dataset: DatasetRecord
    row_sel: RowSelector | None = None            # Integer index array
    feature_cols: tuple[str, ...] = ()
    targets: Mapping[str, ChannelRef] = {}        # Named target channels
    groups: ChannelRef | None = None              # Group labels for CV splitting
    aux: Mapping[str, ChannelRef] = {}            # Auxiliary channels (e.g., sample_weight)
    overlays: Mapping[str, np.ndarray] = {}       # Full-length arrays added as features
```

`ChannelRef` is `str | np.ndarray` -- either a column name in the dataset frame or an explicit array.

**Lazy Operations (return new DataView, no copy):**

```python
select_rows(indices: np.ndarray) -> DataView        # Composes with existing row_sel
select_features(cols: Sequence[str]) -> DataView     # Restrict feature columns
with_overlay(name: str, values: np.ndarray) -> DataView   # Add full-length overlay
with_overlays(predictions: dict[str, np.ndarray]) -> DataView  # Add multiple overlays
bind_target(target: ChannelRef, name: str = "__default__") -> DataView
bind_groups(groups: ChannelRef) -> DataView
with_aux(key: str, value: ChannelRef) -> DataView
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `target` | `ChannelRef \| None` | Default target (shortcut for `targets["__default__"]`) |
| `effective_row_ids` | `np.ndarray` | Row IDs for current selection |
| `n_rows` | `int` | Number of selected rows |
| `n_features` | `int` | Number of feature columns plus overlays |

**Materialization:**

```python
materialize() -> MaterializedBatch
```

Resolves all lazy references into concrete arrays.

**Factory Methods:**

```python
@classmethod
DataView.from_Xy(X: pd.DataFrame, y=None, groups=None, **aux) -> DataView

@classmethod
DataView.from_X(X: pd.DataFrame) -> DataView
```

---

### MaterializedBatch

```python
from sklearn_meta.data.batch import MaterializedBatch
```

Concrete data ready for model fitting. Produced by `DataView.materialize()`.

```python
@dataclass
class MaterializedBatch:
    X: pd.DataFrame
    row_ids: np.ndarray
    targets: dict[str, np.ndarray]
    aux: dict[str, np.ndarray]
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `y` | `np.ndarray \| None` | Shortcut for `targets["__default__"]` |
| `n_samples` | `int` | Number of samples |
| `feature_names` | `list[str]` | List of feature column names |

---

## 3. runtime/ --- Configuration and Services

Runtime configuration types that are separate from graph structure.

---

### RunConfig

```python
from sklearn_meta.runtime.config import RunConfig
```

Unified configuration for a training run. All fields are frozen (immutable).

```python
@dataclass(frozen=True)
class RunConfig:
    cv: CVConfig = field(default_factory=CVConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    feature_selection: FeatureSelectionConfig | None = None
    reparameterization: ReparameterizationConfig | None = None
    estimator_scaling: EstimatorScalingConfig | None = None
    verbosity: int = 1
```

---

### CVConfig

```python
from sklearn_meta.runtime.config import CVConfig
```

```python
@dataclass
class CVConfig:
    n_splits: int = 5
    n_repeats: int = 1
    strategy: CVStrategy = CVStrategy.GROUP
    shuffle: bool = True
    random_state: int = 42
    inner_cv: CVConfig | None = None
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_nested` | `bool` | Whether nested CV is configured |
| `total_folds` | `int` | `n_splits * n_repeats` |

**Methods:**

```python
# Enable nested CV by adding an inner CV configuration
with_inner_cv(n_splits: int = 3, strategy: CVStrategy | None = None) -> CVConfig
```

---

### CVStrategy

```python
from sklearn_meta.runtime.config import CVStrategy
```

```python
class CVStrategy(Enum):
    GROUP = "group"              # Keep groups together
    STRATIFIED = "stratified"    # Preserve class ratios
    RANDOM = "random"            # Simple random splits
    TIME_SERIES = "time_series"  # Temporal ordering
```

Note: There is no `KFOLD` strategy. Use `RANDOM` for simple random splits.

---

### TuningConfig

```python
from sklearn_meta.runtime.config import TuningConfig
```

```python
@dataclass(frozen=True)
class TuningConfig:
    n_trials: int = 100
    timeout: float | None = None
    early_stopping_rounds: int | None = None
    metric: str = "neg_mean_squared_error"
    greater_is_better: bool | None = None  # auto-inferred from metric name
    strategy: OptimizationStrategy = OptimizationStrategy.LAYER_BY_LAYER
    show_progress: bool = False
```

---

### FeatureSelectionConfig and FeatureSelectionMethod

```python
from sklearn_meta.runtime.config import FeatureSelectionConfig, FeatureSelectionMethod
```

```python
class FeatureSelectionMethod(str, Enum):
    SHADOW = "shadow"
    PERMUTATION = "permutation"
    THRESHOLD = "threshold"

@dataclass
class FeatureSelectionConfig:
    enabled: bool = True
    method: FeatureSelectionMethod = FeatureSelectionMethod.SHADOW
    n_shadows: int = 5
    threshold_mult: float = 1.414
    threshold_percentile: float = 10.0
    retune_after_pruning: bool = True
    min_features: int = 1
    max_features: int | None = None
    random_state: int = 42
    feature_groups: dict[str, list[str]] | None = None
```

---

### ReparameterizationConfig

```python
from sklearn_meta.runtime.config import ReparameterizationConfig
```

```python
@dataclass(frozen=True)
class ReparameterizationConfig:
    enabled: bool = True
    use_prebaked: bool = True
    custom_reparameterizations: tuple = ()
```

---

### RunConfigBuilder

```python
from sklearn_meta.runtime.config import RunConfigBuilder
```

Fluent builder for `RunConfig`.

```python
RunConfigBuilder()
```

**Methods:**

```python
cv(
    n_splits=5, n_repeats=1,
    strategy: CVStrategy | str = CVStrategy.GROUP,
    shuffle=True, random_state=42,
) -> RunConfigBuilder

tuning(
    n_trials=100, timeout=None,
    early_stopping_rounds=None,
    metric="neg_mean_squared_error",
    greater_is_better=False,
    strategy: OptimizationStrategy | str = OptimizationStrategy.LAYER_BY_LAYER,
    show_progress=False,
) -> RunConfigBuilder

feature_selection(
    method="shadow", n_shadows=5,
    threshold_mult=1.414,
    retune_after_pruning=True,
    min_features=1, max_features=None,
    feature_groups=None,
) -> RunConfigBuilder

reparameterization(
    enabled=True, use_prebaked=True,
    custom_reparameterizations=None,
) -> RunConfigBuilder

estimator_scaling(
    tuning_n_estimators=None,
    final_n_estimators=None,
    scaling_search=False,
    scaling_factors=None,
) -> RunConfigBuilder

verbosity(level: int) -> RunConfigBuilder

build() -> RunConfig
```

**Example:**

```python
from sklearn_meta.runtime.config import RunConfigBuilder

config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=50, metric="roc_auc")
    .feature_selection(method="shadow")
    .build()
)
```

---

### RuntimeServices

```python
from sklearn_meta.runtime.services import RuntimeServices
```

Service wiring for training runs.

```python
@dataclass
class RuntimeServices:
    search_backend: SearchBackend
    executor: Executor | None = None
    plugin_registry: PluginRegistry | None = None
    audit_logger: AuditLogger | None = None
    fit_cache: FitCache | None = None

    @classmethod
    def default() -> RuntimeServices
```

`RuntimeServices.default()` creates a service object with only an `OptunaBackend`; no plugins, executor, cache, or logger.

---

## 4. engine/ --- Training Engine

Orchestrates the training process over a `GraphSpec`.

---

### GraphRunner

```python
from sklearn_meta.engine.runner import GraphRunner
```

Executes a `GraphSpec` with a `RunConfig` and `RuntimeServices`.

```python
GraphRunner(services: RuntimeServices)
```

**Methods:**

```python
fit(
    graph: GraphSpec,
    data: DataView,
    config: RunConfig,
) -> TrainingRun
```

The runner orchestrates the full training pipeline:

1. Validate graph
2. Create `CVEngine`, `SearchService`, `FeatureSelectionService`
3. Get layers from graph based on `config.tuning.strategy`
4. For each layer:
   - Add OOF overlays from upstream nodes
   - For each node: choose trainer, check conditional, inject distillation soft targets, call `trainer.fit_node()` to produce `NodeRunResult`
   - Cache OOF predictions
5. Build `RunMetadata`
6. Assemble `TrainingRun`

---

### CVEngine

```python
from sklearn_meta.engine.cv import CVEngine
```

Cross-validation fold creation, lazy splitting, and OOF routing.

```python
CVEngine(cv_config: CVConfig)
```

**Methods:**

```python
# Create CV folds from a DataView
create_folds(data: DataView) -> list[CVFold]

# Return lazy train/val DataView pair for a fold
split_for_fold(data: DataView, fold: CVFold) -> tuple[DataView, DataView]

# Combine per-fold predictions into OOF array; averages across repeats
route_oof_predictions(data: DataView, fold_results: list[FoldResult]) -> np.ndarray

# Aggregate fold results into a CVResult
aggregate_cv_result(node_name: str, fold_results: list[FoldResult], data: DataView) -> CVResult
```

**Supporting dataclasses (all in `sklearn_meta.runtime.config`):**

```python
@dataclass
class CVFold:
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    repeat_idx: int = 0

    @property n_train -> int
    @property n_val -> int

@dataclass
class NestedCVFold:
    outer_fold: CVFold
    inner_folds: list[CVFold]

@dataclass
class FoldResult:
    fold: CVFold
    model: object
    val_predictions: np.ndarray
    val_score: float
    train_score: float | None = None
    fit_time: float = 0.0
    predict_time: float = 0.0
    params: dict = field(default_factory=dict)

@dataclass
class CVResult:
    fold_results: list[FoldResult]
    oof_predictions: np.ndarray
    node_name: str
    repeat_oof: np.ndarray | None = None

    @property n_folds -> int
    @property val_scores -> np.ndarray
    @property mean_score -> float
    @property std_score -> float
    @property total_fit_time -> float
    @property models -> list[object]
```

---

### OptimizationStrategy

```python
from sklearn_meta.engine.strategy import OptimizationStrategy
```

```python
class OptimizationStrategy(Enum):
    LAYER_BY_LAYER = "layer_by_layer"   # Optimize each graph layer sequentially
    GREEDY = "greedy"                    # Optimize one node at a time in topological order
    NONE = "none"                        # Skip tuning, use fixed params only
```

---

## 5. artifacts/ --- Training and Inference Artifacts

Results from training runs and lightweight inference graphs.

---

### TrainingRun

```python
from sklearn_meta.artifacts.training import TrainingRun
```

Complete result of fitting a model graph.

```python
@dataclass
class TrainingRun:
    graph: GraphSpec
    config: RunConfig
    node_results: dict[str, NodeRunResult]   # Values may be QuantileNodeRunResult
    metadata: RunMetadata
    total_time: float = 0.0
```

**Methods:**

```python
# Compile into a lightweight inference graph
compile_inference() -> InferenceGraph

# Save to disk (joblib for models, JSON manifest for metadata)
save(path, include_training_artifacts=True) -> None

# Load from disk
@classmethod
load(path) -> TrainingRun
```

---

### NodeRunResult

```python
from sklearn_meta.artifacts.training import NodeRunResult
```

Result for a single fitted node.

```python
@dataclass
class NodeRunResult:
    node_name: str
    cv_result: CVResult
    best_params: dict[str, Any]
    selected_features: list[str] | None = None
    optimization_result: OptimizationResult | None = None
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `oof_predictions` | `np.ndarray` | Out-of-fold predictions |
| `models` | `list[Any]` | Fitted model instances (one per fold) |
| `mean_score` | `float` | Mean CV score |

---

### QuantileNodeRunResult

```python
from sklearn_meta.artifacts.training import QuantileNodeRunResult
```

Extends `NodeRunResult` for quantile regression nodes.

```python
@dataclass
class QuantileNodeRunResult(NodeRunResult):
    quantile_models: dict[float, list[Any]] = field(default_factory=dict)
    oof_quantile_predictions: np.ndarray | None = None  # (n_samples, n_quantiles)
```

---

### RunMetadata

```python
from sklearn_meta.artifacts.training import RunMetadata
```

```python
@dataclass
class RunMetadata:
    timestamp: str                      # ISO 8601
    sklearn_meta_version: str
    data_shape: tuple[int, int]
    feature_names: list[str]
    cv_config: dict[str, Any] | None
    tuning_config_summary: dict[str, Any]
    total_trials: int
    data_hash: str | None
    random_state: int | None
```

---

### InferenceGraph

```python
from sklearn_meta.artifacts.inference import InferenceGraph
```

Lightweight graph for inference (no training artifacts).

```python
@dataclass
class InferenceGraph:
    graph: GraphSpec
    node_models: dict[str, list[Any]]                # node_name -> fold models
    selected_features: dict[str, list[str] | None]
    node_params: dict[str, dict[str, Any]]
```

**Methods:**

```python
# Generate predictions (uses leaf node by default)
predict(X: pd.DataFrame, node_name: str | None = None) -> np.ndarray

# Save / load
save(path) -> None

@classmethod
load(path) -> InferenceGraph
```

Note: There is no `predict_proba()` method on `InferenceGraph`. Probability outputs are handled by the node's `output_type`.

---

### InferenceCompiler

```python
from sklearn_meta.artifacts.compiler import InferenceCompiler
```

```python
class InferenceCompiler:
    @staticmethod
    compile(run: TrainingRun) -> InferenceGraph

    @staticmethod
    compile_quantile(run: TrainingRun) -> JointQuantileInferenceGraph
```

---

## 6. search/ --- Search Space (unchanged from v1)

---

### SearchSpace

```python
from sklearn_meta.search.space import SearchSpace
```

Backend-agnostic hyperparameter search space.

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
    parameter: SearchParameter,
) -> SearchSpace

# Add a pre-constructed parameter
add_parameter(param: SearchParameter) -> SearchSpace

# Shorthand notation
add_from_shorthand(**kwargs) -> SearchSpace

# Create from dictionary (shorthand or explicit format)
@classmethod
from_dict(config: dict) -> SearchSpace

# Narrow search space around a center point
narrow_around(
    center: dict,
    factor: float = 0.5,
    regularization_bias: float = 0.25,
    regularization_params: list[str] | None = None,
) -> SearchSpace

# Sampling
sample_optuna(trial) -> dict

# Lookup
get_parameter(name: str) -> SearchParameter | None

# Operations
copy() -> SearchSpace
merge(other: SearchSpace) -> SearchSpace
remove_parameter(name: str) -> SearchSpace
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `parameter_names` | `list[str]` | List of all parameter names |

---

### SearchParameter Classes

```python
from sklearn_meta.search.parameter import (
    SearchParameter,
    FloatParameter,
    IntParameter,
    CategoricalParameter,
    ConditionalParameter,
    parse_shorthand,
)
```

```python
class SearchParameter(ABC):
    name: str
    sample_optuna(trial) -> Any

@dataclass
class FloatParameter(SearchParameter):
    name: str
    low: float
    high: float
    log: bool = False
    step: float | None = None

@dataclass
class IntParameter(SearchParameter):
    name: str
    low: int
    high: int
    log: bool = False
    step: int = 1

@dataclass
class CategoricalParameter(SearchParameter):
    name: str
    choices: list[Any]

@dataclass
class ConditionalParameter(SearchParameter):
    name: str
    parent_name: str
    parent_value: Any
    parameter: SearchParameter
```

**Shorthand Notation:**

```python
parse_shorthand(name, value) -> SearchParameter
```

Formats:
- `(low, high)` -- Float or Int range (inferred from types)
- `(low, high, "log")` -- Float/Int with log scale
- `[a, b, c]` -- Categorical choices

---

## 7. plugins/ --- Model Plugins

Plugins provide lifecycle hooks for customizing model behavior.

---

### ModelPlugin

```python
from sklearn_meta.plugins.base import ModelPlugin
```

Abstract base class for model-specific plugins.

```python
class ModelPlugin(ABC):
    @property
    def name(self) -> str: ...                        # Default: class name

    @abstractmethod
    def applies_to(self, estimator_class: Type) -> bool: ...

    def modify_search_space(self, space: SearchSpace, node: NodeSpec) -> SearchSpace: ...
    def modify_params(self, params: dict, node: NodeSpec) -> dict: ...
    def modify_fit_params(self, params: dict, batch: MaterializedBatch) -> dict: ...
    def pre_fit(self, model, node: NodeSpec, batch: MaterializedBatch) -> Any: ...
    def post_fit(self, model, node: NodeSpec, batch: MaterializedBatch) -> Any: ...
    def post_tune(self, best_params: dict, node: NodeSpec, data: DataView) -> dict: ...
    def on_fold_start(self, fold_idx: int, node: NodeSpec, data: DataView) -> None: ...
    def on_fold_end(self, fold_idx: int, model, score: float, node: NodeSpec) -> None: ...
```

**Hook Data Parameters:**
- `modify_fit_params`, `pre_fit`, and `post_fit` receive a `MaterializedBatch` (concrete arrays).
- `post_tune` and `on_fold_start` receive a `DataView` (lazy, pre-materialization).

Plugins are referenced by name (string) in `NodeSpec.plugins` and resolved from the plugin registry at runtime:

```python
.plugins("xgboost")
```

---

### CompositePlugin

```python
from sklearn_meta.plugins.base import CompositePlugin
```

Combines multiple plugins, applying all in order.

```python
CompositePlugin(plugins: list[ModelPlugin])
```

All lifecycle methods chain through applicable sub-plugins.

---

## Top-Level Convenience Function

```python
import sklearn_meta

sklearn_meta.fit(
    graph: GraphSpec,
    data_or_X: DataView | pd.DataFrame,
    y_or_config: RunConfig | Any = None,
    config: RunConfig | RuntimeServices | None = None,
    *,
    groups: Any = None,
    services: RuntimeServices | None = None,
) -> TrainingRun
```

Shortcut that creates a `GraphRunner` and calls `fit()`. Supports both `fit(graph, data, config, services)` and `fit(graph, X, y, config, groups=..., services=...)`. Uses `RuntimeServices.default()` if no services are provided.

---

## Quick Import Reference

```python
# Spec
from sklearn_meta import (
    GraphSpec,
    NodeSpec,
    OutputType,
    DependencyType,
    DependencyEdge,
    DistillationConfig,
    GraphBuilder,
)
from sklearn_meta.spec.builder import NodeBuilder
from sklearn_meta.spec.dependency import ConditionalSampleConfig
from sklearn_meta.spec.quantile import (
    QuantileNodeSpec,
    QuantileScalingConfig,
    JointQuantileGraphSpec,
    JointQuantileConfig,
    OrderConstraint,
)

# Data
from sklearn_meta import DatasetRecord, DataView
from sklearn_meta.data.batch import MaterializedBatch

# Runtime
from sklearn_meta import (
    RunConfig,
    TuningConfig,
    CVConfig,
    CVStrategy,
    FeatureSelectionConfig,
    RunConfigBuilder,
    RuntimeServices,
)
from sklearn_meta.runtime.config import (
    FeatureSelectionMethod,
    ReparameterizationConfig,
    CVFold,
    NestedCVFold,
    FoldResult,
    CVResult,
)
from sklearn_meta.engine.estimator_scaling import EstimatorScalingConfig

# Engine
from sklearn_meta import GraphRunner
from sklearn_meta.engine.cv import CVEngine
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.engine.trainer import StandardNodeTrainer

# Artifacts
from sklearn_meta import TrainingRun, NodeRunResult, InferenceGraph
from sklearn_meta.artifacts.training import QuantileNodeRunResult, RunMetadata
from sklearn_meta.artifacts.compiler import InferenceCompiler
from sklearn_meta.artifacts.inference import (
    JointQuantileInferenceGraph,
    QuantileFittedNode,
)

# Search (unchanged from v1)
from sklearn_meta import SearchSpace
from sklearn_meta.search.parameter import (
    SearchParameter,
    FloatParameter,
    IntParameter,
    CategoricalParameter,
    ConditionalParameter,
    parse_shorthand,
)
from sklearn_meta.search.backends.optuna import OptunaBackend

# Plugins
from sklearn_meta.plugins.base import ModelPlugin, CompositePlugin
from sklearn_meta.plugins.registry import PluginRegistry, get_default_registry

# Meta-learning (unchanged from v1)
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

# Persistence / Audit (unchanged from v1)
from sklearn_meta import FitCache, AuditLogger

# Convenience
from sklearn_meta import fit
```

---

## Full End-to-End Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

from sklearn_meta.spec.builder import GraphBuilder
from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import RunConfigBuilder
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.engine.runner import GraphRunner

# 1. Build the graph spec
graph = (
    GraphBuilder("stacking")
    .add_model("rf", RandomForestRegressor)
        .param("n_estimators", 50, 500)
        .param("max_depth", 3, 20)
        .fixed_params(random_state=42, n_jobs=-1)
    .add_model("xgb", XGBRegressor)
        .param("learning_rate", 0.01, 0.3, log=True)
        .param("max_depth", 3, 10)
        .param("n_estimators", 50, 500)
        .fixed_params(random_state=42)
    .add_model("meta", Ridge)
        .stacks("rf", "xgb")
        .param("alpha", 0.01, 100.0, log=True)
    .compile()
)

# 2. Create the data view
data = DataView.from_Xy(X_train, y_train)

# 3. Configure the run
config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="random")
    .tuning(n_trials=50, metric="neg_mean_squared_error")
    .feature_selection(method="shadow")
    .build()
)

# 4. Run training
services = RuntimeServices.default()
runner = GraphRunner(services)
training_run = runner.fit(graph, data, config)

# 5. Compile for inference and predict
inference = training_run.compile_inference()
predictions = inference.predict(X_test)

# 6. Inspect results
for name, result in training_run.node_results.items():
    print(f"{name}: mean_score={result.mean_score:.4f}, params={result.best_params}")

# 7. Save / load
training_run.save("./my_model")
loaded = TrainingRun.load("./my_model")
```

Or use the convenience function:

```python
import sklearn_meta

training_run = sklearn_meta.fit(graph, data, config)
```
