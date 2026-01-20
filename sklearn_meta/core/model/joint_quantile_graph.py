"""Joint Quantile Graph: DAG for joint quantile regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TYPE_CHECKING

from sklearn_meta.core.model.dependency import (
    ConditionalSampleConfig,
    DependencyEdge,
    DependencyType,
)
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.model.quantile_node import (
    DEFAULT_QUANTILE_LEVELS,
    QuantileModelNode,
    QuantileScalingConfig,
)
from sklearn_meta.core.model.quantile_sampler import QuantileSampler, SamplingStrategy

if TYPE_CHECKING:
    from sklearn_meta.search.space import SearchSpace


@dataclass
class OrderConstraint:
    """
    Constraints on property ordering in joint quantile regression.

    The order of properties in the chain affects model quality since
    later properties condition on earlier ones. This class defines
    constraints on valid orderings for use in order search.

    Attributes:
        fixed_positions: Properties that must be at specific positions.
                        Format: {"property_name": position_index}
        must_precede: Pairs where first must come before second.
                     Format: [("prop_A", "prop_B")] means A before B.
        no_swap: Adjacent pairs that should never be swapped during search.
                Format: [("prop_A", "prop_B")]

    Example:
        constraint = OrderConstraint(
            fixed_positions={"price": 0},  # price must be first
            must_precede=[("volume", "volatility")],  # volume before volatility
            no_swap=[("price", "volume")],  # don't try swapping these
        )
    """

    fixed_positions: Dict[str, int] = field(default_factory=dict)
    must_precede: List[Tuple[str, str]] = field(default_factory=list)
    no_swap: List[Tuple[str, str]] = field(default_factory=list)

    def validate_order(self, order: List[str]) -> bool:
        """
        Check if an ordering satisfies all constraints.

        Args:
            order: Proposed property ordering.

        Returns:
            True if order is valid, False otherwise.
        """
        # Check fixed positions
        for prop_name, position in self.fixed_positions.items():
            if prop_name not in order:
                continue
            if order.index(prop_name) != position:
                return False

        # Check precedence constraints
        for first, second in self.must_precede:
            if first not in order or second not in order:
                continue
            if order.index(first) >= order.index(second):
                return False

        return True

    def get_valid_swaps(self, order: List[str]) -> List[Tuple[int, int]]:
        """
        Get all valid adjacent swap positions for the current order.

        Returns pairs (i, i+1) where swapping positions i and i+1
        would result in a valid ordering.

        Args:
            order: Current property ordering.

        Returns:
            List of (position, position+1) tuples for valid swaps.
        """
        valid_swaps = []

        for i in range(len(order) - 1):
            # Check if this swap is in the no_swap list
            prop_a, prop_b = order[i], order[i + 1]
            if (prop_a, prop_b) in self.no_swap or (prop_b, prop_a) in self.no_swap:
                continue

            # Create swapped order and check validity
            swapped = list(order)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]

            if self.validate_order(swapped):
                valid_swaps.append((i, i + 1))

        return valid_swaps

    def get_default_order(self, property_names: List[str]) -> List[str]:
        """
        Get a default ordering that satisfies fixed position constraints.

        Args:
            property_names: List of all property names.

        Returns:
            Ordered list satisfying fixed position constraints.
        """
        # Start with properties not in fixed positions
        remaining = [p for p in property_names if p not in self.fixed_positions]
        result = [None] * len(property_names)

        # Place fixed position properties
        for prop_name, position in self.fixed_positions.items():
            if prop_name in property_names:
                if position >= len(result):
                    raise ValueError(
                        f"Fixed position {position} for '{prop_name}' "
                        f"exceeds property count {len(property_names)}"
                    )
                result[position] = prop_name

        # Fill remaining positions
        remaining_idx = 0
        for i in range(len(result)):
            if result[i] is None:
                result[i] = remaining[remaining_idx]
                remaining_idx += 1

        # Validate must_precede constraints
        if not self.validate_order(result):
            # Try to fix by reordering remaining properties
            result = self._fix_precedence_order(result)

        return result

    def _fix_precedence_order(self, order: List[str]) -> List[str]:
        """Attempt to fix ordering to satisfy must_precede constraints."""
        order = list(order)

        # Simple bubble sort respecting precedence
        changed = True
        max_iterations = len(order) ** 2
        iterations = 0

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for first, second in self.must_precede:
                if first not in order or second not in order:
                    continue

                idx_first = order.index(first)
                idx_second = order.index(second)

                if idx_first > idx_second:
                    # Check if we can swap
                    if first not in self.fixed_positions and second not in self.fixed_positions:
                        order.remove(first)
                        new_idx = order.index(second)
                        order.insert(new_idx, first)
                        changed = True

        return order


@dataclass
class JointQuantileConfig:
    """
    Full configuration for joint quantile regression.

    Attributes:
        property_names: Names of properties to model jointly.
        quantile_levels: Quantile levels to predict (default: 19 levels).
        estimator_class: XGBoost-compatible estimator class.
        search_space: Hyperparameter search space.
        quantile_scaling: Parameter scaling by quantile level.
        order_constraints: Constraints on property ordering.
        sampling_strategy: How to sample from quantile distributions.
        n_inference_samples: Number of samples for joint sampling.
        random_state: Random seed for reproducibility.
        fixed_params: Fixed estimator parameters.
    """

    property_names: List[str]
    quantile_levels: List[float] = field(default_factory=lambda: list(DEFAULT_QUANTILE_LEVELS))
    estimator_class: Optional[Type] = None
    search_space: Optional[SearchSpace] = None
    quantile_scaling: Optional[QuantileScalingConfig] = None
    order_constraints: Optional[OrderConstraint] = None
    sampling_strategy: SamplingStrategy = SamplingStrategy.LINEAR_INTERPOLATION
    n_inference_samples: int = 1000
    random_state: Optional[int] = None
    fixed_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.property_names:
            raise ValueError("property_names cannot be empty")
        if len(self.property_names) != len(set(self.property_names)):
            raise ValueError("property_names must be unique")
        if not self.quantile_levels:
            raise ValueError("quantile_levels cannot be empty")
        for tau in self.quantile_levels:
            if not 0 < tau < 1:
                raise ValueError(f"Quantile level must be in (0, 1), got {tau}")

    @property
    def n_properties(self) -> int:
        """Number of properties to model."""
        return len(self.property_names)


class JointQuantileGraph(ModelGraph):
    """
    Model graph for joint quantile regression.

    Constructs a chain of QuantileModelNodes connected by
    CONDITIONAL_SAMPLE dependencies. The order of properties
    determines the conditioning structure:

    P(Y₁, Y₂, ..., Yₙ | X) = P(Y₁|X) × P(Y₂|X,Y₁) × P(Y₃|X,Y₁,Y₂) × ...

    Attributes:
        config: JointQuantileConfig defining the model structure.

    Example:
        config = JointQuantileConfig(
            property_names=["price", "volume", "volatility"],
            quantile_levels=[0.1, 0.5, 0.9],
        )
        graph = JointQuantileGraph(config)

        # Change ordering
        graph.set_order(["volume", "price", "volatility"])

        # Get valid swaps for order search
        swaps = graph.get_valid_swaps()
    """

    def __init__(self, config: JointQuantileConfig) -> None:
        """
        Initialize the joint quantile graph.

        Args:
            config: Configuration for the joint quantile model.
        """
        super().__init__()
        self.config = config
        self._property_order: List[str] = []

        # Get initial order from constraints or use provided order
        if config.order_constraints:
            self._property_order = config.order_constraints.get_default_order(
                config.property_names
            )
        else:
            self._property_order = list(config.property_names)

        self._build_graph()

    def _build_graph(self) -> None:
        """Build the graph structure based on current property order."""
        # Clear existing graph
        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()

        # Create nodes for each property
        for i, prop_name in enumerate(self._property_order):
            node = QuantileModelNode(
                name=f"quantile_{prop_name}",
                property_name=prop_name,
                estimator_class=self.config.estimator_class,
                search_space=self.config.search_space,
                quantile_levels=list(self.config.quantile_levels),
                quantile_scaling=self.config.quantile_scaling,
                fixed_params=dict(self.config.fixed_params),
            )
            self._nodes[node.name] = node

        # Create edges for conditional dependencies
        for i in range(1, len(self._property_order)):
            target_prop = self._property_order[i]
            target_node_name = f"quantile_{target_prop}"

            # Add dependency on all previous properties
            for j in range(i):
                source_prop = self._property_order[j]
                source_node_name = f"quantile_{source_prop}"

                edge = DependencyEdge(
                    source=source_node_name,
                    target=target_node_name,
                    dep_type=DependencyType.CONDITIONAL_SAMPLE,
                    conditional_config=ConditionalSampleConfig(
                        property_name=source_prop,
                        use_actual_during_training=True,
                    ),
                )
                self._edges.append(edge)
                self._adjacency[edge.source].append(edge)
                self._reverse_adjacency[edge.target].append(edge)

    def set_order(self, new_order: List[str]) -> None:
        """
        Change the property ordering and rebuild the graph.

        Args:
            new_order: New ordering of property names.

        Raises:
            ValueError: If order is invalid or violates constraints.
        """
        # Validate new order contains all properties
        if set(new_order) != set(self.config.property_names):
            raise ValueError(
                f"New order must contain all properties: {self.config.property_names}"
            )

        # Validate against constraints
        if self.config.order_constraints:
            if not self.config.order_constraints.validate_order(new_order):
                raise ValueError(f"Order {new_order} violates constraints")

        self._property_order = list(new_order)
        self._build_graph()

    def swap_adjacent(self, position: int) -> None:
        """
        Swap adjacent properties at the given position.

        Args:
            position: Index of first property to swap (swaps with position+1).

        Raises:
            ValueError: If swap is invalid or violates constraints.
        """
        if position < 0 or position >= len(self._property_order) - 1:
            raise ValueError(
                f"Position must be in [0, {len(self._property_order) - 2}]"
            )

        # Create swapped order
        new_order = list(self._property_order)
        new_order[position], new_order[position + 1] = (
            new_order[position + 1],
            new_order[position],
        )

        self.set_order(new_order)

    def get_valid_swaps(self) -> List[Tuple[int, int]]:
        """
        Get all valid adjacent swap positions.

        Returns:
            List of (i, i+1) tuples for valid swaps.
        """
        if self.config.order_constraints:
            return self.config.order_constraints.get_valid_swaps(self._property_order)

        # No constraints: all adjacent swaps are valid
        return [(i, i + 1) for i in range(len(self._property_order) - 1)]

    def get_quantile_node(self, property_name: str) -> QuantileModelNode:
        """
        Get the quantile node for a specific property.

        Args:
            property_name: Name of the property.

        Returns:
            QuantileModelNode for the property.
        """
        node_name = f"quantile_{property_name}"
        node = self.get_node(node_name)
        if not isinstance(node, QuantileModelNode):
            raise TypeError(f"Node {node_name} is not a QuantileModelNode")
        return node

    def get_conditioning_properties(self, property_name: str) -> List[str]:
        """
        Get properties that a given property conditions on.

        Args:
            property_name: Name of the property.

        Returns:
            List of upstream property names (in order).
        """
        try:
            idx = self._property_order.index(property_name)
        except ValueError:
            raise ValueError(f"Property '{property_name}' not in graph")

        return self._property_order[:idx]

    def create_quantile_sampler(self) -> QuantileSampler:
        """
        Create a QuantileSampler configured for this graph.

        Returns:
            Configured QuantileSampler instance.
        """
        return QuantileSampler(
            strategy=self.config.sampling_strategy,
            n_samples=self.config.n_inference_samples,
            random_state=self.config.random_state,
        )

    @property
    def property_order(self) -> List[str]:
        """Current property ordering."""
        return list(self._property_order)

    @property
    def n_properties(self) -> int:
        """Number of properties."""
        return len(self._property_order)

    @property
    def quantile_levels(self) -> List[float]:
        """Quantile levels being modeled."""
        return list(self.config.quantile_levels)

    def __repr__(self) -> str:
        return (
            f"JointQuantileGraph(properties={self._property_order}, "
            f"n_quantiles={len(self.config.quantile_levels)})"
        )
