"""Tests for QuantileModelNode and QuantileScalingConfig."""

import pytest
import numpy as np
from sklearn.linear_model import Ridge

from auto_sklearn.core.model.quantile_node import (
    DEFAULT_QUANTILE_LEVELS,
    QuantileModelNode,
    QuantileScalingConfig,
)
from auto_sklearn.core.model.node import OutputType


# =============================================================================
# Mock XGBoost-like estimator for testing
# =============================================================================


class MockQuantileRegressor:
    """Mock estimator that supports quantile regression parameters."""

    def __init__(
        self,
        objective="reg:squarederror",
        quantile_alpha=0.5,
        n_estimators=100,
        max_depth=6,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=None,
    ):
        self.objective = objective
        self.quantile_alpha = quantile_alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True
        return self

    def predict(self, X):
        return np.zeros(len(X))


# =============================================================================
# QuantileScalingConfig Tests
# =============================================================================


class TestQuantileScalingConfigCreation:
    """Tests for QuantileScalingConfig creation."""

    def test_basic_creation(self):
        """Verify basic config creation."""
        config = QuantileScalingConfig(
            base_params={"n_estimators": 100},
            scaling_rules={},
        )

        assert config.base_params == {"n_estimators": 100}
        assert config.scaling_rules == {}

    def test_creation_with_scaling_rules(self):
        """Verify config creation with scaling rules."""
        config = QuantileScalingConfig(
            base_params={"n_estimators": 100},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        assert "reg_lambda" in config.scaling_rules

    def test_default_creation(self):
        """Verify default config creation."""
        config = QuantileScalingConfig()

        assert config.base_params == {}
        assert config.scaling_rules == {}


class TestQuantileScalingConfigParams:
    """Tests for get_params_for_quantile method."""

    def test_params_at_median(self):
        """Verify params at median (tau=0.5) use base values."""
        config = QuantileScalingConfig(
            base_params={"n_estimators": 100},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        params = config.get_params_for_quantile(0.5)

        assert params["n_estimators"] == 100
        assert params["reg_lambda"] == pytest.approx(1.0, rel=0.01)

    def test_params_at_lower_tail(self):
        """Verify params at lower tail have scaling applied."""
        config = QuantileScalingConfig(
            base_params={"n_estimators": 100},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        # tau=0.1 is 0.4 from median, normalized = 0.4/0.45 = 0.889
        params = config.get_params_for_quantile(0.1)

        assert params["n_estimators"] == 100
        # Scale factor = 1 + 0.889 * (2.0 - 1.0) = 1.889
        assert params["reg_lambda"] == pytest.approx(1.889, rel=0.05)

    def test_params_at_upper_tail(self):
        """Verify params at upper tail have scaling applied."""
        config = QuantileScalingConfig(
            base_params={"n_estimators": 100},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        # tau=0.9 is also 0.4 from median
        params = config.get_params_for_quantile(0.9)

        assert params["reg_lambda"] == pytest.approx(1.889, rel=0.05)

    def test_params_at_extreme_tail(self):
        """Verify params at extreme tail (tau=0.05)."""
        config = QuantileScalingConfig(
            base_params={},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        # tau=0.05 is 0.45 from median, normalized = 1.0
        params = config.get_params_for_quantile(0.05)

        # Scale factor = 1 + 1.0 * (2.0 - 1.0) = 2.0
        assert params["reg_lambda"] == pytest.approx(2.0, rel=0.01)

    def test_multiple_scaling_rules(self):
        """Verify multiple scaling rules are applied."""
        config = QuantileScalingConfig(
            base_params={"n_estimators": 100},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
                "reg_alpha": {"base": 0.1, "tail_multiplier": 1.5},
            },
        )

        params = config.get_params_for_quantile(0.1)

        assert "reg_lambda" in params
        assert "reg_alpha" in params


# =============================================================================
# QuantileModelNode Tests
# =============================================================================


class TestQuantileModelNodeCreation:
    """Tests for QuantileModelNode creation."""

    def test_basic_creation(self):
        """Verify basic node creation."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
        )

        assert node.name == "test"
        assert node.property_name == "price"
        assert node.output_type == OutputType.QUANTILES

    def test_creation_with_quantile_levels(self):
        """Verify node creation with custom quantile levels."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        assert node.quantile_levels == [0.1, 0.5, 0.9]
        assert node.n_quantiles == 3

    def test_default_quantile_levels(self):
        """Verify default quantile levels are used."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
        )

        assert node.quantile_levels == sorted(DEFAULT_QUANTILE_LEVELS)
        assert len(node.quantile_levels) == 19

    def test_quantile_levels_sorted(self):
        """Verify quantile levels are sorted."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.9, 0.1, 0.5],
        )

        assert node.quantile_levels == [0.1, 0.5, 0.9]

    def test_auto_name_from_property(self):
        """Verify automatic name generation from property."""
        node = QuantileModelNode(
            name="",  # Will be auto-generated
            property_name="price",
            estimator_class=MockQuantileRegressor,
        )

        assert node.name == "quantile_price"

    def test_missing_property_name_raises(self):
        """Verify missing property_name raises error."""
        with pytest.raises(ValueError, match="property_name"):
            QuantileModelNode(
                name="test",
                property_name="",
                estimator_class=MockQuantileRegressor,
            )

    def test_invalid_quantile_level_raises(self):
        """Verify invalid quantile level raises error."""
        with pytest.raises(ValueError, match="must be in"):
            QuantileModelNode(
                name="test",
                property_name="price",
                estimator_class=MockQuantileRegressor,
                quantile_levels=[0.0, 0.5, 1.0],
            )

    def test_empty_quantile_levels_raises(self):
        """Verify empty quantile levels raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            QuantileModelNode(
                name="test",
                property_name="price",
                estimator_class=MockQuantileRegressor,
                quantile_levels=[],
            )


class TestQuantileModelNodeProperties:
    """Tests for QuantileModelNode properties."""

    def test_median_quantile_exact(self):
        """Verify median_quantile when 0.5 is in levels."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        assert node.median_quantile == 0.5

    def test_median_quantile_closest(self):
        """Verify median_quantile finds closest level."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.4, 0.6, 0.9],
        )

        # Both 0.4 and 0.6 are equidistant; min() picks 0.4
        assert node.median_quantile in [0.4, 0.6]

    def test_n_quantiles(self):
        """Verify n_quantiles property."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        )

        assert node.n_quantiles == 5


class TestQuantileModelNodeCreateEstimator:
    """Tests for create_estimator_for_quantile method."""

    def test_create_estimator_sets_quantile_params(self):
        """Verify estimator is created with quantile parameters."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        model = node.create_estimator_for_quantile(0.1)

        assert model.objective == "reg:quantileerror"
        assert model.quantile_alpha == 0.1

    def test_create_estimator_with_fixed_params(self):
        """Verify fixed params are applied."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            fixed_params={"n_estimators": 50, "max_depth": 4},
        )

        model = node.create_estimator_for_quantile(0.5)

        assert model.n_estimators == 50
        assert model.max_depth == 4

    def test_create_estimator_with_scaling(self):
        """Verify quantile scaling is applied."""
        scaling = QuantileScalingConfig(
            base_params={"reg_lambda": 1.0},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            quantile_scaling=scaling,
        )

        model_median = node.create_estimator_for_quantile(0.5)
        model_tail = node.create_estimator_for_quantile(0.1)

        assert model_tail.reg_lambda > model_median.reg_lambda

    def test_create_estimator_params_override(self):
        """Verify provided params override others."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            fixed_params={"n_estimators": 50},
        )

        model = node.create_estimator_for_quantile(0.5, params={"n_estimators": 200})

        assert model.n_estimators == 200


class TestQuantileModelNodeGetParams:
    """Tests for get_params_for_quantile method."""

    def test_get_params_includes_quantile_settings(self):
        """Verify params include quantile-specific settings."""
        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        params = node.get_params_for_quantile(0.1)

        assert params["objective"] == "reg:quantileerror"
        assert params["quantile_alpha"] == 0.1

    def test_get_params_merges_all_sources(self):
        """Verify params merge fixed, tuned, and scaled values."""
        scaling = QuantileScalingConfig(
            base_params={"reg_lambda": 1.0},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        node = QuantileModelNode(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            fixed_params={"n_estimators": 50},
            quantile_scaling=scaling,
        )

        params = node.get_params_for_quantile(0.1, tuned_params={"max_depth": 8})

        assert params["n_estimators"] == 50
        assert params["max_depth"] == 8
        assert "reg_lambda" in params


class TestQuantileModelNodeRepr:
    """Tests for node representation."""

    def test_repr(self):
        """Verify repr is informative."""
        node = QuantileModelNode(
            name="quantile_price",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        repr_str = repr(node)

        assert "quantile_price" in repr_str
        assert "price" in repr_str
        assert "n_quantiles=3" in repr_str
