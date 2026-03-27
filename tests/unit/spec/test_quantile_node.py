"""Tests for QuantileNodeSpec and QuantileScalingConfig."""

import pytest
import numpy as np

from sklearn_meta.spec.quantile import (
    DEFAULT_QUANTILE_LEVELS,
    QuantileNodeSpec,
    QuantileScalingConfig,
)
from sklearn_meta.spec.node import OutputType


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

# =============================================================================
# QuantileNodeSpec Tests
# =============================================================================


class TestQuantileNodeSpecCreation:
    """Tests for QuantileNodeSpec creation and validation."""

    def test_missing_property_name_raises(self):
        """Verify missing property_name raises error."""
        with pytest.raises(ValueError, match="property_name"):
            QuantileNodeSpec(
                name="test",
                property_name="",
                estimator_class=MockQuantileRegressor,
            )

    def test_invalid_quantile_level_raises(self):
        """Verify invalid quantile level raises error."""
        with pytest.raises(ValueError, match="must be in"):
            QuantileNodeSpec(
                name="test",
                property_name="price",
                estimator_class=MockQuantileRegressor,
                quantile_levels=[0.0, 0.5, 1.0],
            )

    def test_empty_quantile_levels_raises(self):
        """Verify empty quantile levels raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            QuantileNodeSpec(
                name="test",
                property_name="price",
                estimator_class=MockQuantileRegressor,
                quantile_levels=[],
            )


class TestQuantileNodeSpecProperties:
    """Tests for QuantileNodeSpec properties."""

    def test_median_quantile_closest(self):
        """Verify median_quantile finds closest level."""
        node = QuantileNodeSpec(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.4, 0.6, 0.9],
        )

        # Both 0.4 and 0.6 are equidistant; min() picks 0.4
        assert node.median_quantile in [0.4, 0.6]


class TestQuantileNodeSpecCreateEstimator:
    """Tests for create_estimator_for_quantile method."""

    def test_create_estimator_with_scaling(self):
        """Verify quantile scaling is applied."""
        scaling = QuantileScalingConfig(
            base_params={"reg_lambda": 1.0},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        node = QuantileNodeSpec(
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
        node = QuantileNodeSpec(
            name="test",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            fixed_params={"n_estimators": 50},
        )

        model = node.create_estimator_for_quantile(0.5, params={"n_estimators": 200})

        assert model.n_estimators == 200


class TestQuantileNodeSpecGetParams:
    """Tests for get_params_for_quantile method."""

    def test_get_params_merges_all_sources(self):
        """Verify params merge fixed, tuned, and scaled values."""
        scaling = QuantileScalingConfig(
            base_params={"reg_lambda": 1.0},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        node = QuantileNodeSpec(
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


class TestQuantileNodeSpecSerialization:
    """Tests for quantile node serialization."""

    def test_round_trips_quantile_fields(self):
        scaling = QuantileScalingConfig(
            base_params={"reg_lambda": 1.0},
            scaling_rules={"reg_lambda": {"base": 1.0, "tail_multiplier": 2.0}},
        )
        node = QuantileNodeSpec(
            name="quantile_price",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            quantile_scaling=scaling,
        )

        restored = QuantileNodeSpec.from_dict(node.to_dict())

        assert restored.property_name == "price"
        assert restored.quantile_levels == [0.1, 0.5, 0.9]
        assert restored.quantile_scaling is not None
        assert restored.quantile_scaling.scaling_rules["reg_lambda"]["tail_multiplier"] == 2.0
