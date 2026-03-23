"""GraphBuilder and NodeBuilder: Fluent API for building GraphSpec objects.

Extracted from the original api.py with runtime concerns removed.
GraphBuilder produces a pure GraphSpec (no tuning, CV, or fitting logic).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, Union, TYPE_CHECKING

from sklearn_meta.spec.dependency import DependencyEdge, DependencyType, ConditionalSampleConfig
from sklearn_meta.spec.distillation import DistillationConfig
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.search.parameter import FloatParameter, IntParameter, CategoricalParameter

if TYPE_CHECKING:
    from sklearn_meta.search.space import SearchSpace


class NodeBuilder:
    """Builder for configuring a single model node.

    Supports fluent chaining and delegates unknown attributes to the parent
    GraphBuilder, allowing seamless transitions between node-level and
    graph-level configuration.

    Example::

        builder = GraphBuilder()
        builder.add_model("xgb", XGBRegressor)
            .param("learning_rate", 0.01, 0.3, log=True)
            .param("max_depth", 3, 10)
            .fixed_params(n_jobs=-1)
    """

    _MISSING = object()

    def __init__(
        self,
        name: str,
        estimator_class: Type,
        graph_builder: GraphBuilder,
    ) -> None:
        """
        Initialize the node builder.

        Args:
            name: Unique node name.
            estimator_class: sklearn-compatible estimator class.
            graph_builder: Parent GraphBuilder instance.
        """
        self._name = name
        self._estimator_class = estimator_class
        self._graph_builder = graph_builder
        self._search_space: Optional[SearchSpace] = None
        self._output_type = OutputType.PREDICTION
        self._condition: Optional[Callable[..., bool]] = None
        self._plugins: List[str] = []
        self._fixed_params: Dict[str, Any] = {}
        self._fit_params: Dict[str, Any] = {}
        self._feature_cols: Optional[List[str]] = None
        self._description = ""
        self._distillation_config: Optional[DistillationConfig] = None

    # ------------------------------------------------------------------
    # Delegation to GraphBuilder for fluent chaining
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to GraphBuilder for fluent chaining.

        This allows fluent transitions from node-level configuration to
        graph-level configuration (e.g. ``.add_model()``, ``.compile()``)
        without maintaining explicit forwarding methods.
        """
        # Only delegate public methods to avoid masking internal errors
        if not name.startswith("_"):
            graph_builder = object.__getattribute__(self, "_graph_builder")
            attr = getattr(graph_builder, name, NodeBuilder._MISSING)
            if attr is not NodeBuilder._MISSING:
                return attr
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ------------------------------------------------------------------
    # Search space configuration
    # ------------------------------------------------------------------

    def search_space(self, space: SearchSpace) -> NodeBuilder:
        """
        Set a pre-built SearchSpace for this node.

        Args:
            space: A fully configured SearchSpace instance.

        Returns:
            Self for chaining.
        """
        self._search_space = space
        return self

    def param(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
        step: Optional[float] = None,
    ) -> NodeBuilder:
        """
        Add a float hyperparameter to this node's search space.

        Args:
            name: Parameter name.
            low: Lower bound.
            high: Upper bound.
            log: Whether to sample in log space.
            step: Optional step size for discrete sampling.

        Returns:
            Self for chaining.
        """
        self._ensure_search_space()
        self._search_space.add_float(name, low, high, log=log, step=step)
        return self

    def int_param(
        self,
        name: str,
        low: int,
        high: int,
        step: int = 1,
    ) -> NodeBuilder:
        """
        Add an integer hyperparameter to this node's search space.

        Args:
            name: Parameter name.
            low: Lower bound (inclusive).
            high: Upper bound (inclusive).
            step: Step size.

        Returns:
            Self for chaining.
        """
        self._ensure_search_space()
        self._search_space.add_int(name, low, high, step=step)
        return self

    def cat_param(self, name: str, choices: List[Any]) -> NodeBuilder:
        """
        Add a categorical hyperparameter to this node's search space.

        Args:
            name: Parameter name.
            choices: List of possible values.

        Returns:
            Self for chaining.
        """
        self._ensure_search_space()
        self._search_space.add_categorical(name, choices)
        return self

    # ------------------------------------------------------------------
    # Node configuration
    # ------------------------------------------------------------------

    def output_type(self, t: Union[str, OutputType]) -> NodeBuilder:
        """
        Set the output type for this node.

        Args:
            t: One of "prediction", "proba", "transform", or an OutputType enum.

        Returns:
            Self for chaining.
        """
        if isinstance(t, str):
            t = OutputType(t)
        self._output_type = t
        return self

    def condition(self, fn: Callable[..., bool]) -> NodeBuilder:
        """
        Set a condition for node execution.

        Args:
            fn: Callable that returns True if node should run.

        Returns:
            Self for chaining.
        """
        self._condition = fn
        return self

    def plugins(self, *names: str) -> NodeBuilder:
        """
        Add plugins to this node.

        Args:
            *names: Plugin names to register.

        Returns:
            Self for chaining.
        """
        self._plugins.extend(names)
        return self

    def fixed_params(self, **kwargs: Any) -> NodeBuilder:
        """
        Set fixed (non-tuned) parameters for the estimator.

        Args:
            **kwargs: Parameter name-value pairs.

        Returns:
            Self for chaining.
        """
        self._fixed_params.update(kwargs)
        return self

    def fit_params(self, **kwargs: Any) -> NodeBuilder:
        """
        Set parameters passed to the estimator's fit() method.

        Args:
            **kwargs: Fit parameter name-value pairs.

        Returns:
            Self for chaining.
        """
        self._fit_params.update(kwargs)
        return self

    def feature_cols(self, cols: List[str]) -> NodeBuilder:
        """
        Specify which feature columns this node should use.

        Args:
            cols: List of feature column names.

        Returns:
            Self for chaining.
        """
        self._feature_cols = list(cols)
        return self

    def description(self, desc: str) -> NodeBuilder:
        """
        Add a human-readable description for this node.

        Args:
            desc: Description text.

        Returns:
            Self for chaining.
        """
        self._description = desc
        return self

    # ------------------------------------------------------------------
    # Dependency configuration
    # ------------------------------------------------------------------

    def depends_on(
        self,
        source: str,
        dep_type: DependencyType = DependencyType.PREDICTION,
        column_name: Optional[str] = None,
    ) -> NodeBuilder:
        """
        Declare a dependency on another node.

        Args:
            source: Name of the source node.
            dep_type: Type of dependency (prediction, proba, transform, etc.).
            column_name: Optional custom feature name in target's input.

        Returns:
            Self for chaining.
        """
        self._graph_builder._edges.append(
            DependencyEdge(
                source=source,
                target=self._name,
                dep_type=dep_type,
                column_name=column_name,
            )
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
        for source in sources:
            self.depends_on(source, dep_type=DependencyType.PREDICTION)
        return self

    def stacks_proba(self, *sources: str) -> NodeBuilder:
        """
        Add probability stacking dependencies.

        Args:
            *sources: Names of source nodes to stack.

        Returns:
            Self for chaining.
        """
        for source in sources:
            self.depends_on(source, dep_type=DependencyType.PROBA)
        return self

    def distill_from(
        self,
        teacher_name: str,
        alpha: float = 0.5,
        temperature: float = 3.0,
    ) -> NodeBuilder:
        """
        Set up knowledge distillation from a teacher node.

        The student trains with a blended KL-divergence + cross-entropy loss
        using the teacher's OOF probabilities as soft targets. Only one
        teacher per student is supported.

        Args:
            teacher_name: Name of the teacher node.
            alpha: Blend weight: alpha * KL_soft + (1-alpha) * CE_hard.
            temperature: Softens distributions before KL computation.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If this node already has a distillation teacher.
        """
        if self._distillation_config is not None:
            raise ValueError(
                f"Node '{self._name}' already has a distillation teacher. "
                f"Only one teacher per student is supported."
            )
        self._distillation_config = DistillationConfig(
            temperature=temperature, alpha=alpha
        )
        self._graph_builder._edges.append(
            DependencyEdge(
                source=teacher_name,
                target=self._name,
                dep_type=DependencyType.DISTILL,
            )
        )
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> NodeSpec:
        """Build the NodeSpec from accumulated configuration."""
        return NodeSpec(
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_search_space(self) -> None:
        """Lazily create the SearchSpace if it doesn't exist yet."""
        if self._search_space is None:
            from sklearn_meta.search.space import SearchSpace
            self._search_space = SearchSpace()

    def __repr__(self) -> str:
        class_name = self._estimator_class.__name__
        return f"NodeBuilder(name={self._name!r}, estimator={class_name})"


class GraphBuilder:
    """Fluent API for building a GraphSpec.

    Produces a pure ``GraphSpec`` containing node definitions and edges.
    Runtime concerns (CV, tuning, fitting) are handled separately by
    RunConfig and GraphRunner.

    Example::

        graph = (
            GraphBuilder()
            .add_model("rf", RandomForestClassifier)
                .param("n_estimators", 50, 500)
                .param("max_depth", 3, 20)
            .add_model("xgb", XGBClassifier)
                .param("learning_rate", 0.01, 0.3, log=True)
                .param("max_depth", 3, 10)
            .add_model("meta", LogisticRegression)
                .stacks("rf", "xgb")
            .compile()
        )
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize an empty graph builder.

        Args:
            name: Optional graph name (unused, accepted for backward compatibility).
        """
        self._name = name
        self._nodes: Dict[str, NodeBuilder] = {}
        self._edges: List[DependencyEdge] = []

    def add_model(self, name: str, estimator_class: Type) -> NodeBuilder:
        """
        Add a model node to the graph.

        Args:
            name: Unique name for this model.
            estimator_class: sklearn-compatible estimator class.

        Returns:
            NodeBuilder for fluent configuration of the model.

        Raises:
            ValueError: If a model with the same name already exists.
        """
        if name in self._nodes:
            raise ValueError(f"Model '{name}' already exists in the graph")

        builder = NodeBuilder(name, estimator_class, self)
        self._nodes[name] = builder
        return builder

    def compile(self) -> GraphSpec:
        """
        Build the GraphSpec from accumulated nodes and edges.

        All nodes are built and added, then all edges are added.
        The resulting graph is validated before being returned.

        Returns:
            A fully constructed and validated GraphSpec.

        Raises:
            CycleError: If edges form a cycle.
            ValueError: If edges reference unknown nodes.
        """
        graph = GraphSpec()

        # Add all nodes
        for node_builder in self._nodes.values():
            graph.add_node(node_builder._build())

        # Add all edges
        for edge in self._edges:
            graph.add_edge(edge)

        # Validate the graph structure
        graph.validate()

        return graph

    def __repr__(self) -> str:
        return f"GraphBuilder(models={list(self._nodes.keys())})"
