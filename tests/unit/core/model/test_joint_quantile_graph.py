"""Tests for JointQuantileGraph and related classes."""

import pytest
import numpy as np

from sklearn_meta.core.model.dependency import DependencyType
from sklearn_meta.core.model.joint_quantile_graph import (
    JointQuantileConfig,
    JointQuantileGraph,
    OrderConstraint,
)
from sklearn_meta.core.model.quantile_sampler import SamplingStrategy


# =============================================================================
# Mock estimator for testing
# =============================================================================


class MockQuantileRegressor:
    """Mock estimator for testing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


# =============================================================================
# OrderConstraint Tests
# =============================================================================


class TestOrderConstraintCreation:
    """Tests for OrderConstraint creation."""

    def test_default_creation(self):
        """Verify default constraint creation."""
        constraint = OrderConstraint()

        assert constraint.fixed_positions == {}
        assert constraint.must_precede == []
        assert constraint.no_swap == []

    def test_creation_with_fixed_positions(self):
        """Verify constraint with fixed positions."""
        constraint = OrderConstraint(
            fixed_positions={"price": 0, "volume": 1},
        )

        assert constraint.fixed_positions["price"] == 0
        assert constraint.fixed_positions["volume"] == 1

    def test_creation_with_must_precede(self):
        """Verify constraint with precedence rules."""
        constraint = OrderConstraint(
            must_precede=[("A", "B"), ("B", "C")],
        )

        assert ("A", "B") in constraint.must_precede


class TestOrderConstraintValidateOrder:
    """Tests for validate_order method."""

    def test_validate_order_no_constraints(self):
        """Verify any order is valid without constraints."""
        constraint = OrderConstraint()

        assert constraint.validate_order(["A", "B", "C"]) is True
        assert constraint.validate_order(["C", "B", "A"]) is True

    def test_validate_order_fixed_positions(self):
        """Verify fixed position constraints are checked."""
        constraint = OrderConstraint(
            fixed_positions={"A": 0},
        )

        assert constraint.validate_order(["A", "B", "C"]) is True
        assert constraint.validate_order(["B", "A", "C"]) is False

    def test_validate_order_must_precede(self):
        """Verify precedence constraints are checked."""
        constraint = OrderConstraint(
            must_precede=[("A", "B")],
        )

        assert constraint.validate_order(["A", "B", "C"]) is True
        assert constraint.validate_order(["B", "A", "C"]) is False
        assert constraint.validate_order(["A", "C", "B"]) is True  # A before B, C anywhere

    def test_validate_order_multiple_constraints(self):
        """Verify multiple constraints are all checked."""
        constraint = OrderConstraint(
            fixed_positions={"A": 0},
            must_precede=[("B", "C")],
        )

        assert constraint.validate_order(["A", "B", "C"]) is True
        assert constraint.validate_order(["A", "C", "B"]) is False  # B must precede C
        assert constraint.validate_order(["B", "A", "C"]) is False  # A must be at 0


class TestOrderConstraintGetValidSwaps:
    """Tests for get_valid_swaps method."""

    def test_valid_swaps_no_constraints(self):
        """Verify all swaps valid without constraints."""
        constraint = OrderConstraint()

        swaps = constraint.get_valid_swaps(["A", "B", "C"])

        assert (0, 1) in swaps
        assert (1, 2) in swaps

    def test_valid_swaps_with_no_swap(self):
        """Verify no_swap list excludes pairs."""
        constraint = OrderConstraint(
            no_swap=[("A", "B")],
        )

        swaps = constraint.get_valid_swaps(["A", "B", "C"])

        assert (0, 1) not in swaps  # A-B swap blocked
        assert (1, 2) in swaps

    def test_valid_swaps_respects_fixed_positions(self):
        """Verify swaps don't violate fixed positions."""
        constraint = OrderConstraint(
            fixed_positions={"A": 0},
        )

        swaps = constraint.get_valid_swaps(["A", "B", "C"])

        # Swapping A with B would move A from position 0
        assert (0, 1) not in swaps
        assert (1, 2) in swaps


class TestOrderConstraintGetDefaultOrder:
    """Tests for get_default_order method."""

    def test_default_order_no_constraints(self):
        """Verify default order without constraints."""
        constraint = OrderConstraint()

        order = constraint.get_default_order(["A", "B", "C"])

        assert set(order) == {"A", "B", "C"}

    def test_default_order_with_fixed_positions(self):
        """Verify default order respects fixed positions."""
        constraint = OrderConstraint(
            fixed_positions={"C": 0, "A": 2},
        )

        order = constraint.get_default_order(["A", "B", "C"])

        assert order[0] == "C"
        assert order[2] == "A"


# =============================================================================
# JointQuantileConfig Tests
# =============================================================================


class TestJointQuantileConfigCreation:
    """Tests for JointQuantileConfig creation."""

    def test_basic_creation(self):
        """Verify basic config creation."""
        config = JointQuantileConfig(
            property_names=["price", "volume"],
        )

        assert config.property_names == ["price", "volume"]
        assert config.n_properties == 2

    def test_creation_with_quantile_levels(self):
        """Verify config with custom quantile levels."""
        config = JointQuantileConfig(
            property_names=["price", "volume"],
            quantile_levels=[0.1, 0.5, 0.9],
        )

        assert config.quantile_levels == [0.1, 0.5, 0.9]

    def test_empty_property_names_raises(self):
        """Verify empty property names raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            JointQuantileConfig(property_names=[])

    def test_duplicate_property_names_raises(self):
        """Verify duplicate property names raises error."""
        with pytest.raises(ValueError, match="must be unique"):
            JointQuantileConfig(property_names=["price", "price"])

    def test_invalid_quantile_level_raises(self):
        """Verify invalid quantile level raises error."""
        with pytest.raises(ValueError, match="must be in"):
            JointQuantileConfig(
                property_names=["price"],
                quantile_levels=[0.0, 0.5],
            )


# =============================================================================
# JointQuantileGraph Tests
# =============================================================================


class TestJointQuantileGraphCreation:
    """Tests for JointQuantileGraph creation."""

    def test_basic_creation(self):
        """Verify basic graph creation."""
        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)

        assert len(graph) == 2
        assert "quantile_price" in graph
        assert "quantile_volume" in graph

    def test_creates_conditional_edges(self):
        """Verify conditional edges are created."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)

        # B depends on A
        b_upstream = graph.get_upstream("quantile_B")
        assert len(b_upstream) == 1
        assert b_upstream[0].source == "quantile_A"

        # C depends on A and B
        c_upstream = graph.get_upstream("quantile_C")
        assert len(c_upstream) == 2

    def test_edge_type_is_conditional_sample(self):
        """Verify edges have CONDITIONAL_SAMPLE type."""
        config = JointQuantileConfig(
            property_names=["A", "B"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)
        edges = graph.get_upstream("quantile_B")

        assert all(e.dep_type == DependencyType.CONDITIONAL_SAMPLE for e in edges)


class TestJointQuantileGraphOrdering:
    """Tests for graph ordering operations."""

    def test_property_order(self):
        """Verify property_order returns correct order."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)

        assert graph.property_order == ["A", "B", "C"]

    def test_set_order(self):
        """Verify set_order changes property order."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)
        graph.set_order(["C", "B", "A"])

        assert graph.property_order == ["C", "B", "A"]

    def test_set_order_rebuilds_edges(self):
        """Verify set_order rebuilds graph edges."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)
        graph.set_order(["C", "B", "A"])

        # Now B depends on C, and A depends on C and B
        a_upstream = graph.get_upstream("quantile_A")
        assert len(a_upstream) == 2
        sources = {e.source for e in a_upstream}
        assert sources == {"quantile_C", "quantile_B"}

    def test_set_order_invalid_properties_raises(self):
        """Verify set_order with wrong properties raises error."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)

        with pytest.raises(ValueError, match="must contain all"):
            graph.set_order(["A", "B", "D"])

    def test_swap_adjacent(self):
        """Verify swap_adjacent swaps properties."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)
        graph.swap_adjacent(0)  # Swap A and B

        assert graph.property_order == ["B", "A", "C"]

    def test_swap_adjacent_invalid_position_raises(self):
        """Verify swap_adjacent with invalid position raises."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)

        with pytest.raises(ValueError):
            graph.swap_adjacent(5)


class TestJointQuantileGraphValidSwaps:
    """Tests for get_valid_swaps method."""

    def test_valid_swaps_no_constraints(self):
        """Verify all swaps valid without constraints."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)
        swaps = graph.get_valid_swaps()

        assert (0, 1) in swaps
        assert (1, 2) in swaps

    def test_valid_swaps_with_constraints(self):
        """Verify swaps respect order constraints."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
            order_constraints=OrderConstraint(
                fixed_positions={"A": 0},
            ),
        )

        graph = JointQuantileGraph(config)
        swaps = graph.get_valid_swaps()

        # Can't swap A (position 0)
        assert (0, 1) not in swaps
        assert (1, 2) in swaps


class TestJointQuantileGraphQuantileNode:
    """Tests for get_quantile_node method."""

    def test_get_quantile_node(self):
        """Verify get_quantile_node returns correct node."""
        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        graph = JointQuantileGraph(config)
        node = graph.get_quantile_node("price")

        assert node.property_name == "price"
        assert node.quantile_levels == [0.1, 0.5, 0.9]


class TestJointQuantileGraphConditioningProperties:
    """Tests for get_conditioning_properties method."""

    def test_conditioning_properties_first(self):
        """Verify first property has no conditioning."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)

        assert graph.get_conditioning_properties("A") == []

    def test_conditioning_properties_second(self):
        """Verify second property conditions on first."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)

        assert graph.get_conditioning_properties("B") == ["A"]

    def test_conditioning_properties_third(self):
        """Verify third property conditions on first two."""
        config = JointQuantileConfig(
            property_names=["A", "B", "C"],
            estimator_class=MockQuantileRegressor,
        )

        graph = JointQuantileGraph(config)

        assert graph.get_conditioning_properties("C") == ["A", "B"]


class TestJointQuantileGraphSampler:
    """Tests for create_quantile_sampler method."""

    def test_create_sampler(self):
        """Verify sampler creation with graph config."""
        config = JointQuantileConfig(
            property_names=["A", "B"],
            estimator_class=MockQuantileRegressor,
            sampling_strategy=SamplingStrategy.NORMAL,
            n_inference_samples=500,
            random_state=42,
        )

        graph = JointQuantileGraph(config)
        sampler = graph.create_quantile_sampler()

        assert sampler.strategy == SamplingStrategy.NORMAL
        assert sampler.n_samples == 500


class TestJointQuantileGraphRepr:
    """Tests for graph representation."""

    def test_repr(self):
        """Verify repr is informative."""
        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        graph = JointQuantileGraph(config)
        repr_str = repr(graph)

        assert "price" in repr_str
        assert "volume" in repr_str
        assert "n_quantiles=3" in repr_str
