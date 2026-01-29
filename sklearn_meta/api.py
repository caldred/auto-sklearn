"""GraphBuilder: Fluent API for building model graphs."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.dependency import DependencyEdge, DependencyType
from sklearn_meta.core.model.distillation import DistillationConfig
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.model.node import ModelNode, OutputType
from sklearn_meta.core.tuning.orchestrator import (
    FittedGraph,
    TuningConfig,
    TuningOrchestrator,
)
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.execution.base import Executor
from sklearn_meta.search.backends.base import SearchBackend
from sklearn_meta.search.backends.optuna import OptunaBackend
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.selection.selector import FeatureSelectionConfig
from sklearn_meta.meta.reparameterization import Reparameterization, ReparameterizedSpace


class NodeBuilder:
    """Builder for a single model node."""

    def __init__(
        self,
        graph_builder: GraphBuilder,
        name: str,
        estimator_class: Type,
    ) -> None:
        """
        Initialize the node builder.

        Args:
            graph_builder: Parent graph builder.
            name: Node name.
            estimator_class: Estimator class.
        """
        self._graph_builder = graph_builder
        self._name = name
        self._estimator_class = estimator_class
        self._search_space: Optional[SearchSpace] = None
        self._output_type = OutputType.PREDICTION
        self._condition: Optional[Callable] = None
        self._plugins: List[str] = []
        self._fixed_params: Dict[str, Any] = {}
        self._fit_params: Dict[str, Any] = {}
        self._feature_cols: Optional[List[str]] = None
        self._description = ""
        self._distillation_config: Optional[DistillationConfig] = None

    def with_search_space(
        self,
        space: Optional[SearchSpace] = None,
        **kwargs: Union[Tuple, List],
    ) -> NodeBuilder:
        """
        Set the search space for this node.

        Args:
            space: Pre-built SearchSpace, or None to build from kwargs.
            **kwargs: Shorthand parameter definitions.
                     - (low, high): Int/Float range
                     - (low, high, "log"): Log-scale range
                     - [a, b, c]: Categorical choices

        Returns:
            Self for chaining.
        """
        if space is not None:
            self._search_space = space
        else:
            self._search_space = SearchSpace()
            self._search_space.add_from_shorthand(**kwargs)
        return self

    def with_output_type(self, output_type: str) -> NodeBuilder:
        """
        Set the output type.

        Args:
            output_type: One of "prediction", "proba", "transform".

        Returns:
            Self for chaining.
        """
        self._output_type = output_type
        return self

    def with_condition(
        self, condition: Callable[[DataContext], bool]
    ) -> NodeBuilder:
        """
        Set a condition for node execution.

        Args:
            condition: Function that returns True if node should run.

        Returns:
            Self for chaining.
        """
        self._condition = condition
        return self

    def with_plugins(self, *plugins: str) -> NodeBuilder:
        """
        Add plugins to this node.

        Args:
            *plugins: Plugin names to add.

        Returns:
            Self for chaining.
        """
        self._plugins.extend(plugins)
        return self

    def with_fixed_params(self, **params: Any) -> NodeBuilder:
        """
        Set fixed (non-tuned) parameters.

        Args:
            **params: Fixed parameter values.

        Returns:
            Self for chaining.
        """
        self._fixed_params.update(params)
        return self

    def with_fit_params(self, **params: Any) -> NodeBuilder:
        """
        Set parameters for fit().

        Args:
            **params: Fit parameter values.

        Returns:
            Self for chaining.
        """
        self._fit_params.update(params)
        return self

    def with_features(self, *feature_cols: str) -> NodeBuilder:
        """
        Specify which features to use.

        Args:
            *feature_cols: Feature column names.

        Returns:
            Self for chaining.
        """
        self._feature_cols = list(feature_cols)
        return self

    def with_description(self, description: str) -> NodeBuilder:
        """
        Add a description.

        Args:
            description: Human-readable description.

        Returns:
            Self for chaining.
        """
        self._description = description
        return self

    def depends_on(
        self,
        *sources: str,
        dep_type: DependencyType = DependencyType.PREDICTION,
    ) -> NodeBuilder:
        """
        Add dependencies to this node.

        Args:
            *sources: Names of source nodes.
            dep_type: Type of dependency.

        Returns:
            Self for chaining.
        """
        for source in sources:
            self._graph_builder._pending_edges.append(
                DependencyEdge(source=source, target=self._name, dep_type=dep_type)
            )
        return self

    def stacks(self, *sources: str) -> NodeBuilder:
        """
        Add stacking dependencies (predictions as features).

        Args:
            *sources: Names of source nodes to stack.

        Returns:
            Self for chaining.
        """
        return self.depends_on(*sources, dep_type=DependencyType.PREDICTION)

    def stacks_proba(self, *sources: str) -> NodeBuilder:
        """
        Add probability stacking dependencies.

        Args:
            *sources: Names of source nodes to stack.

        Returns:
            Self for chaining.
        """
        return self.depends_on(*sources, dep_type=DependencyType.PROBA)

    def distills(
        self,
        teacher: str,
        temperature: float = 3.0,
        alpha: float = 0.5,
    ) -> NodeBuilder:
        """
        Set up knowledge distillation from a teacher node.

        The student trains with a blended KL-divergence + cross-entropy loss
        using the teacher's OOF probabilities as soft targets. Only one
        teacher per student is supported.

        Args:
            teacher: Name of the teacher node.
            temperature: Softens distributions before KL computation.
            alpha: Blend weight: alpha * KL_soft + (1-alpha) * CE_hard.

        Returns:
            Self for chaining.
        """
        if self._distillation_config is not None:
            raise ValueError(
                f"Node '{self._name}' already has a distillation teacher. "
                f"Only one teacher per student is supported."
            )
        self._distillation_config = DistillationConfig(
            temperature=temperature, alpha=alpha
        )
        self._graph_builder._pending_edges.append(
            DependencyEdge(
                source=teacher,
                target=self._name,
                dep_type=DependencyType.DISTILL,
            )
        )
        return self

    def _build(self) -> ModelNode:
        """Build the ModelNode."""
        return ModelNode(
            name=self._name,
            estimator_class=self._estimator_class,
            search_space=self._search_space,
            output_type=self._output_type,
            condition=self._condition,
            plugins=self._plugins,
            fixed_params=self._fixed_params,
            fit_params=self._fit_params,
            feature_cols=self._feature_cols,
            description=self._description,
            distillation_config=self._distillation_config,
        )

    def add_model(
        self,
        name: str,
        estimator_class: Type,
    ) -> NodeBuilder:
        """
        Add another model to the graph (shortcut).

        Args:
            name: New node name.
            estimator_class: Estimator class.

        Returns:
            New NodeBuilder for the added node.
        """
        return self._graph_builder.add_model(name, estimator_class)

    def build(self) -> ModelGraph:
        """Build and return the ModelGraph."""
        return self._graph_builder.build()

    # Forward graph-level methods to GraphBuilder for fluent API
    def with_cv(self, *args, **kwargs) -> GraphBuilder:
        """Configure cross-validation (forwards to GraphBuilder)."""
        return self._graph_builder.with_cv(*args, **kwargs)

    def with_tuning(self, *args, **kwargs) -> GraphBuilder:
        """Configure hyperparameter tuning (forwards to GraphBuilder)."""
        return self._graph_builder.with_tuning(*args, **kwargs)

    def with_feature_selection(self, *args, **kwargs) -> GraphBuilder:
        """Configure feature selection (forwards to GraphBuilder)."""
        return self._graph_builder.with_feature_selection(*args, **kwargs)

    def with_reparameterization(self, *args, **kwargs) -> GraphBuilder:
        """Configure reparameterization (forwards to GraphBuilder)."""
        return self._graph_builder.with_reparameterization(*args, **kwargs)

    def create_orchestrator(self, *args, **kwargs):
        """Create TuningOrchestrator (forwards to GraphBuilder)."""
        return self._graph_builder.create_orchestrator(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """Fit the graph (forwards to GraphBuilder)."""
        return self._graph_builder.fit(*args, **kwargs)


class GraphBuilder:
    """
    Fluent API for building model graphs.

    Example:
        graph = (
            GraphBuilder("my_pipeline")
            .add_model("rf", RandomForestClassifier)
            .with_search_space(
                n_estimators=(50, 500),
                max_depth=(3, 20),
            )
            .add_model("xgb", XGBClassifier)
            .with_search_space(
                learning_rate=(0.01, 0.3, "log"),
                max_depth=(3, 10),
            )
            .add_model("meta", LogisticRegression)
            .stacks("rf", "xgb")
            .build()
        )
    """

    def __init__(self, name: str = "pipeline") -> None:
        """
        Initialize the graph builder.

        Args:
            name: Name for the pipeline.
        """
        self.name = name
        self._node_builders: Dict[str, NodeBuilder] = {}
        self._pending_edges: List[DependencyEdge] = []
        self._current_builder: Optional[NodeBuilder] = None

        # Tuning configuration
        self._cv_config: Optional[CVConfig] = None
        self._tuning_config: Optional[TuningConfig] = None
        self._feature_selection_config: Optional[FeatureSelectionConfig] = None
        self._n_parallel_trials: int = 1

        # Reparameterization configuration
        self._custom_reparameterizations: List[Reparameterization] = []
        self._use_prebaked_reparameterizations: bool = False

    def add_model(
        self,
        name: str,
        estimator_class: Type,
    ) -> NodeBuilder:
        """
        Add a model to the graph.

        Args:
            name: Unique name for this model.
            estimator_class: sklearn-compatible estimator class.

        Returns:
            NodeBuilder for configuring the model.
        """
        if name in self._node_builders:
            raise ValueError(f"Model '{name}' already exists")

        builder = NodeBuilder(self, name, estimator_class)
        self._node_builders[name] = builder
        self._current_builder = builder
        return builder

    def with_cv(
        self,
        n_splits: int = 5,
        n_repeats: int = 1,
        strategy: Union[str, CVStrategy] = CVStrategy.STRATIFIED,
        shuffle: bool = True,
        random_state: int = 42,
        nested: bool = False,
        inner_splits: int = 3,
    ) -> GraphBuilder:
        """
        Configure cross-validation.

        Args:
            n_splits: Number of CV folds.
            n_repeats: Number of CV repeats.
            strategy: CV strategy.
            shuffle: Whether to shuffle.
            random_state: Random seed.
            nested: Whether to use nested CV.
            inner_splits: Number of inner CV splits (for nested).

        Returns:
            Self for chaining.
        """
        if isinstance(strategy, str):
            strategy = CVStrategy(strategy)

        self._cv_config = CVConfig(
            n_splits=n_splits,
            n_repeats=n_repeats,
            strategy=strategy,
            shuffle=shuffle,
            random_state=random_state,
        )

        if nested:
            self._cv_config = self._cv_config.with_inner_cv(inner_splits)

        return self

    def with_tuning(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        strategy: Union[str, OptimizationStrategy] = OptimizationStrategy.LAYER_BY_LAYER,
        metric: str = "neg_mean_squared_error",
        greater_is_better: bool = False,
        early_stopping_rounds: Optional[int] = None,
        n_parallel_trials: int = 1,
        tuning_n_estimators: Optional[int] = None,
        final_n_estimators: Optional[int] = None,
        estimator_scaling_search: bool = False,
        estimator_scaling_factors: Optional[List[int]] = None,
    ) -> GraphBuilder:
        """
        Configure hyperparameter tuning.

        Args:
            n_trials: Number of optimization trials.
            timeout: Optional timeout in seconds.
            strategy: Optimization strategy.
            metric: Evaluation metric.
            greater_is_better: Whether higher metric is better.
            early_stopping_rounds: Stop after N trials without improvement.
            n_parallel_trials: Number of parallel Optuna trials.
            tuning_n_estimators: n_estimators to use during tuning for faster trials.
                If set along with final_n_estimators, learning_rate is scaled
                proportionally for the final model.
            final_n_estimators: n_estimators to use for the final model. If
                tuning_n_estimators is set, learning_rate is scaled by
                (tuning_n_estimators / final_n_estimators) to maintain equivalent
                training.
            estimator_scaling_search: If True, search for optimal n_estimators
                scaling after tuning. Tests multipliers [2, 5, 10, 20] with early
                stopping if performance degrades.
            estimator_scaling_factors: Custom scaling factors to search
                (default: [2, 5, 10, 20]).

        Returns:
            Self for chaining.
        """
        if isinstance(strategy, str):
            strategy = OptimizationStrategy(strategy)

        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")

        self._tuning_config = TuningConfig(
            strategy=strategy,
            n_trials=n_trials,
            timeout=timeout,
            early_stopping_rounds=early_stopping_rounds,
            metric=metric,
            greater_is_better=greater_is_better,
            tuning_n_estimators=tuning_n_estimators,
            final_n_estimators=final_n_estimators,
            estimator_scaling_search=estimator_scaling_search,
            estimator_scaling_factors=estimator_scaling_factors,
        )
        self._n_parallel_trials = n_parallel_trials

        return self

    def with_feature_selection(
        self,
        method: str = "shadow",
        n_shadows: int = 5,
        threshold_mult: float = 1.414,
        retune_after_pruning: bool = True,
        min_features: int = 1,
        max_features: Optional[int] = None,
    ) -> GraphBuilder:
        """
        Configure feature selection.

        Args:
            method: Selection method ("shadow", "permutation", "threshold").
            n_shadows: Number of shadow features (for shadow method).
            threshold_mult: Multiplier for shadow threshold.
            retune_after_pruning: Whether to retune after feature selection.
            min_features: Minimum features to keep.
            max_features: Maximum features to keep.

        Returns:
            Self for chaining.
        """
        self._feature_selection_config = FeatureSelectionConfig(
            enabled=True,
            method=method,
            n_shadows=n_shadows,
            threshold_mult=threshold_mult,
            retune_after_pruning=retune_after_pruning,
            min_features=min_features,
            max_features=max_features,
        )

        return self

    def with_reparameterization(
        self,
        reparameterizations: Optional[List[Reparameterization]] = None,
        use_prebaked: bool = True,
    ) -> GraphBuilder:
        """
        Configure hyperparameter reparameterization for orthogonal search.

        Reparameterization transforms correlated hyperparameters into more
        orthogonal dimensions for more efficient optimization.

        Args:
            reparameterizations: Custom reparameterizations to apply.
            use_prebaked: Whether to automatically apply pre-baked
                         reparameterizations for known model/param combinations.

        Returns:
            Self for chaining.
        """
        self._custom_reparameterizations = reparameterizations or []
        self._use_prebaked_reparameterizations = use_prebaked
        return self

    def build(self) -> ModelGraph:
        """
        Build and return the ModelGraph.

        Returns:
            Configured ModelGraph.
        """
        graph = ModelGraph()

        # Add all nodes
        for name, builder in self._node_builders.items():
            node = builder._build()
            graph.add_node(node)

        # Add all edges
        for edge in self._pending_edges:
            graph.add_edge(edge)

        # Validate
        graph.validate()

        return graph

    def create_orchestrator(
        self,
        search_backend: Optional[SearchBackend] = None,
        executor: Optional[Executor] = None,
    ) -> TuningOrchestrator:
        """
        Build graph and create a TuningOrchestrator.

        Args:
            search_backend: Search backend to use (default: OptunaBackend).
            executor: Optional executor for parallel execution.

        Returns:
            Configured TuningOrchestrator.
        """
        graph = self.build()

        # Use defaults if not configured
        cv_config = self._cv_config or CVConfig()
        tuning_config = self._tuning_config or TuningConfig()

        if self._cv_config:
            tuning_config.cv_config = cv_config

        if self._feature_selection_config:
            tuning_config.feature_selection = self._feature_selection_config

        # Wire reparameterization configuration
        if self._custom_reparameterizations or self._use_prebaked_reparameterizations:
            tuning_config.use_reparameterization = True
            tuning_config.custom_reparameterizations = self._custom_reparameterizations

        data_manager = DataManager(cv_config)
        backend = search_backend or OptunaBackend(n_jobs=self._n_parallel_trials)

        return TuningOrchestrator(
            graph=graph,
            data_manager=data_manager,
            search_backend=backend,
            tuning_config=tuning_config,
            executor=executor,
        )

    def fit(
        self,
        X,
        y,
        groups=None,
        search_backend: Optional[SearchBackend] = None,
        executor: Optional[Executor] = None,
    ) -> FittedGraph:
        """
        Build graph and fit to data.

        Args:
            X: Features DataFrame.
            y: Target Series.
            groups: Optional groups for group CV.
            search_backend: Search backend to use.
            executor: Optional executor for parallel execution.

        Returns:
            FittedGraph with trained models.
        """
        import pandas as pd

        # Convert to pandas if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if groups is not None and not isinstance(groups, pd.Series):
            groups = pd.Series(groups)

        # Validate inputs
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature")

        ctx = DataContext.from_Xy(X=X, y=y, groups=groups)
        orchestrator = self.create_orchestrator(search_backend, executor)

        return orchestrator.fit(ctx)

    def __repr__(self) -> str:
        return f"GraphBuilder(name={self.name}, models={list(self._node_builders.keys())})"
