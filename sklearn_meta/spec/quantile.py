"""Quantile regression specs: QuantileNodeSpec and JointQuantileGraphSpec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

from sklearn_meta.spec.dependency import (
    ConditionalSampleConfig,
    DependencyEdge,
    DependencyType,
)
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.spec.quantile_sampler import QuantileSampler, SamplingStrategy

if TYPE_CHECKING:
    from sklearn_meta.search.space import SearchSpace


# Default quantile levels: 19 levels from 0.05 to 0.95
DEFAULT_QUANTILE_LEVELS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]


@dataclass
class QuantileScalingConfig:
    """
    Configuration for scaling parameters by quantile level.

    Some hyperparameters (like regularization) may need different values
    at extreme quantiles compared to the median. This config defines
    how to scale parameters based on distance from the median (tau=0.5).

    Attributes:
        base_params: Base hyperparameters used for all quantiles.
        scaling_rules: Rules for scaling parameters at tail quantiles.
                      Format: {"param_name": {"base": value, "tail_multiplier": mult}}
                      The tail_multiplier is applied proportionally to |tau - 0.5|.

    Example:
        config = QuantileScalingConfig(
            base_params={"n_estimators": 100, "max_depth": 6},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
                "reg_alpha": {"base": 0.1, "tail_multiplier": 1.5},
            }
        )
        # At tau=0.1 (tail distance = 0.4, max distance = 0.45):
        # reg_lambda = 1.0 * (1 + (0.4/0.45) * 2.0) = 2.78
    """

    base_params: Dict[str, Any] = field(default_factory=dict)
    scaling_rules: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get_params_for_quantile(self, tau: float) -> Dict[str, Any]:
        """
        Get scaled parameters for a specific quantile level.

        Args:
            tau: Quantile level (0 < tau < 1).

        Returns:
            Dictionary of scaled parameters.
        """
        params = dict(self.base_params)

        # Distance from median, normalized by max distance (0.45 for 0.05-0.95 range)
        tail_distance = abs(tau - 0.5)
        max_distance = 0.45  # Assuming standard 0.05-0.95 range
        normalized_distance = tail_distance / max_distance

        for param_name, rule in self.scaling_rules.items():
            base_value = rule.get("base", 1.0)
            tail_multiplier = rule.get("tail_multiplier", 1.0)

            # Scale factor: 1 at median, up to (1 + tail_multiplier) at extremes
            scale_factor = 1.0 + normalized_distance * (tail_multiplier - 1.0)
            params[param_name] = base_value * scale_factor

        return params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_params": dict(self.base_params),
            "scaling_rules": dict(self.scaling_rules),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantileScalingConfig":
        return cls(
            base_params=dict(data.get("base_params", {})),
            scaling_rules=dict(data.get("scaling_rules", {})),
        )


@dataclass
class QuantileNodeSpec(NodeSpec):
    """
    Node spec specialized for quantile regression.

    Extends NodeSpec to handle multiple quantile levels, each requiring
    a separate model with quantile-specific objective function.

    Attributes:
        property_name: Name of the target property this node predicts.
        quantile_levels: List of quantile levels to model (default: 19 levels).
        quantile_scaling: Optional config for scaling params by quantile.
        xgboost_objective: XGBoost objective name for quantile regression.

    Note:
        The estimator_class should support quantile regression, typically via
        XGBoost with objective='reg:quantileerror' and quantile_alpha parameter.
    """

    property_name: str = ""
    quantile_levels: List[float] = field(default_factory=lambda: list(DEFAULT_QUANTILE_LEVELS))
    quantile_scaling: Optional[QuantileScalingConfig] = None
    xgboost_objective: str = "reg:quantileerror"

    def __post_init__(self) -> None:
        """Validate node configuration."""
        # Set output type to QUANTILES
        self.output_type = OutputType.QUANTILES

        # Validate property_name
        if not self.property_name:
            raise ValueError("property_name is required for QuantileNodeSpec")

        # Validate quantile levels
        if not self.quantile_levels:
            raise ValueError("quantile_levels cannot be empty")

        for tau in self.quantile_levels:
            if not 0 < tau < 1:
                raise ValueError(f"Quantile level must be in (0, 1), got {tau}")

        # Sort quantile levels
        self.quantile_levels = sorted(self.quantile_levels)

        # Set default name if not provided
        if not self.name:
            self.name = f"quantile_{self.property_name}"

        # Validate estimator has required methods
        if self.estimator_class:
            if not hasattr(self.estimator_class, "fit"):
                raise ValueError(
                    f"Estimator {self.estimator_class} must have a 'fit' method"
                )
            if not hasattr(self.estimator_class, "predict"):
                raise ValueError(
                    f"Estimator {self.estimator_class} must have a 'predict' method "
                    "for quantile regression"
                )

    def create_estimator_for_quantile(
        self,
        tau: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create an estimator configured for a specific quantile level."""
        all_params = self.get_params_for_quantile(tau, params)
        return self.estimator_class(**all_params)

    def get_params_for_quantile(
        self,
        tau: float,
        tuned_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get complete parameter dict for a specific quantile level.

        Args:
            tau: Quantile level.
            tuned_params: Tuned hyperparameters (from optimization at median).

        Returns:
            Complete parameter dictionary.
        """
        # Start with fixed params
        all_params = dict(self.fixed_params)

        # Apply tuned params
        if tuned_params:
            all_params.update(tuned_params)

        # Apply quantile scaling
        if self.quantile_scaling:
            scaled_params = self.quantile_scaling.get_params_for_quantile(tau)
            all_params.update(scaled_params)

        # Set quantile-specific parameters
        all_params["objective"] = self.xgboost_objective
        all_params["quantile_alpha"] = tau

        return all_params

    @property
    def median_quantile(self) -> float:
        """Get the median quantile level (or closest to 0.5)."""
        return min(self.quantile_levels, key=lambda x: abs(x - 0.5))

    @property
    def n_quantiles(self) -> int:
        """Number of quantile levels."""
        return len(self.quantile_levels)

    def __repr__(self) -> str:
        class_name = self.estimator_class.__name__ if self.estimator_class else "None"
        return (
            f"QuantileNodeSpec(name={self.name!r}, property={self.property_name!r}, "
            f"estimator={class_name}, n_quantiles={self.n_quantiles})"
        )

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "property_name": self.property_name,
            "quantile_levels": list(self.quantile_levels),
            "xgboost_objective": self.xgboost_objective,
        })
        if self.quantile_scaling is not None:
            result["quantile_scaling"] = self.quantile_scaling.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantileNodeSpec":
        fields = cls._base_fields_from_dict(data)

        quantile_scaling = None
        if data.get("quantile_scaling") is not None:
            quantile_scaling = QuantileScalingConfig.from_dict(
                data["quantile_scaling"]
            )

        return cls(
            **fields,
            property_name=data["property_name"],
            quantile_levels=list(data.get("quantile_levels", DEFAULT_QUANTILE_LEVELS)),
            quantile_scaling=quantile_scaling,
            xgboost_objective=data.get("xgboost_objective", "reg:quantileerror"),
        )


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
            result = self.fix_precedence_order(result)

        return result

    def fix_precedence_order(self, order: List[str]) -> List[str]:
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


class JointQuantileGraphSpec(GraphSpec):
    """
    Graph spec for joint quantile regression.

    Constructs a chain of QuantileNodeSpecs connected by
    CONDITIONAL_SAMPLE dependencies. The order of properties
    determines the conditioning structure:

    P(Y1, Y2, ..., Yn | X) = P(Y1|X) * P(Y2|X,Y1) * P(Y3|X,Y1,Y2) * ...

    Attributes:
        config: JointQuantileConfig defining the model structure.

    Example:
        config = JointQuantileConfig(
            property_names=["price", "volume", "volatility"],
            quantile_levels=[0.1, 0.5, 0.9],
        )
        graph = JointQuantileGraphSpec(config)

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
            node = QuantileNodeSpec(
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

    def get_quantile_node(self, property_name: str) -> QuantileNodeSpec:
        """
        Get the quantile node for a specific property.

        Args:
            property_name: Name of the property.

        Returns:
            QuantileNodeSpec for the property.
        """
        node_name = f"quantile_{property_name}"
        node = self.get_node(node_name)
        if not isinstance(node, QuantileNodeSpec):
            raise TypeError(f"Node {node_name} is not a QuantileNodeSpec")
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
            f"JointQuantileGraphSpec(properties={self._property_order}, "
            f"n_quantiles={len(self.config.quantile_levels)})"
        )
