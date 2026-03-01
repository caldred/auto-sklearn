"""Tests for the EstimatorScaler component extracted from TuningOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from sklearn_meta.core.tuning.estimator_scaling import (
    EstimatorScaler,
    EstimatorScalingConfig,
    supports_param,
)


# ---------------------------------------------------------------------------
# supports_param tests
# ---------------------------------------------------------------------------

class TestSupportsParam:
    """Test supports_param with various estimator classes."""

    def test_present_param(self):
        assert supports_param(GradientBoostingRegressor, "n_estimators") is True

    def test_absent_param(self):
        assert supports_param(Ridge, "n_estimators") is False

    def test_present_alpha_on_ridge(self):
        assert supports_param(Ridge, "alpha") is True

    def test_var_keyword_catches_all(self):
        """A class with **kwargs should accept any param."""
        class _ModelWithKwargs:
            def __init__(self, **kwargs):
                pass

        assert supports_param(_ModelWithKwargs, "n_estimators") is True
        assert supports_param(_ModelWithKwargs, "anything") is True

    def test_no_var_keyword(self):
        """A class without **kwargs should only accept declared params."""
        class _StrictModel:
            def __init__(self, alpha=1.0):
                pass

        assert supports_param(_StrictModel, "alpha") is True
        assert supports_param(_StrictModel, "n_estimators") is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(estimator_class, name="test_node"):
    """Create a mock node with an estimator_class and name."""
    node = MagicMock()
    node.estimator_class = estimator_class
    node.name = name
    return node


@dataclass
class _FakeCVResult:
    """Minimal stand-in for CVResult with a mean_score attribute."""
    mean_score: float


# ---------------------------------------------------------------------------
# apply_fixed_scaling tests
# ---------------------------------------------------------------------------

class TestApplyFixedScaling:
    """Test EstimatorScaler.apply_fixed_scaling."""

    def test_scales_n_estimators_and_learning_rate(self):
        config = EstimatorScalingConfig(
            tuning_n_estimators=100,
            final_n_estimators=500,
        )
        scaler = EstimatorScaler(config)
        node = _make_node(GradientBoostingRegressor)

        params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
        result = scaler.apply_fixed_scaling(node, params)

        assert result["n_estimators"] == 500
        # scale_factor = 100/500 = 0.2, so lr = 0.1 * 0.2 = 0.02
        assert result["learning_rate"] == pytest.approx(0.02)
        # Other params untouched
        assert result["max_depth"] == 3

    def test_does_not_mutate_original(self):
        config = EstimatorScalingConfig(
            tuning_n_estimators=100,
            final_n_estimators=500,
        )
        scaler = EstimatorScaler(config)
        node = _make_node(GradientBoostingRegressor)

        params = {"n_estimators": 100, "learning_rate": 0.1}
        result = scaler.apply_fixed_scaling(node, params)

        assert params["n_estimators"] == 100  # original unchanged
        assert result["n_estimators"] == 500

    def test_no_learning_rate_only_scales_n_estimators(self):
        config = EstimatorScalingConfig(
            tuning_n_estimators=100,
            final_n_estimators=500,
        )
        scaler = EstimatorScaler(config)
        node = _make_node(GradientBoostingRegressor)

        params = {"n_estimators": 100, "max_depth": 3}
        result = scaler.apply_fixed_scaling(node, params)

        assert result["n_estimators"] == 500
        assert "learning_rate" not in result

    def test_skips_when_estimator_lacks_n_estimators(self):
        config = EstimatorScalingConfig(
            tuning_n_estimators=100,
            final_n_estimators=500,
        )
        scaler = EstimatorScaler(config)
        node = _make_node(Ridge, name="ridge_node")

        params = {"alpha": 1.0}
        result = scaler.apply_fixed_scaling(node, params)

        # Should return params unchanged
        assert result == params


# ---------------------------------------------------------------------------
# search_scaling tests
# ---------------------------------------------------------------------------

class TestSearchScaling:
    """Test EstimatorScaler.search_scaling."""

    def test_finds_best_scaling_lower_is_better(self):
        """When lower scores are better, picks the factor with lowest score."""
        config = EstimatorScalingConfig(
            scaling_search=True,
            scaling_factors=[2, 5],
        )
        scaler = EstimatorScaler(config, greater_is_better=False)
        node = _make_node(GradientBoostingRegressor)
        ctx = MagicMock()

        # Scores: 1x -> 0.5, 2x -> 0.3, 5x -> 0.4 (2x is best)
        scores = {
            (100, 0.1): 0.5,     # 1x
            (200, 0.05): 0.3,    # 2x
            (500, 0.02): 0.4,    # 5x (worse than 2x, triggers early stop)
        }

        def mock_cv(params):
            key = (params["n_estimators"], params["learning_rate"])
            return _FakeCVResult(mean_score=scores[key])

        params = {"n_estimators": 100, "learning_rate": 0.1}
        result_params, cv_result = scaler.search_scaling(node, ctx, params, mock_cv)

        assert result_params["n_estimators"] == 200
        assert result_params["learning_rate"] == pytest.approx(0.05)
        assert cv_result.mean_score == 0.3

    def test_finds_best_scaling_greater_is_better(self):
        """When higher scores are better, picks the factor with highest score."""
        config = EstimatorScalingConfig(
            scaling_search=True,
            scaling_factors=[2, 5],
        )
        scaler = EstimatorScaler(config, greater_is_better=True)
        node = _make_node(GradientBoostingRegressor)
        ctx = MagicMock()

        scores = {
            (100, 0.1): 0.5,
            (200, 0.05): 0.7,    # 2x best
            (500, 0.02): 0.6,    # worse
        }

        def mock_cv(params):
            key = (params["n_estimators"], params["learning_rate"])
            return _FakeCVResult(mean_score=scores[key])

        params = {"n_estimators": 100, "learning_rate": 0.1}
        result_params, cv_result = scaler.search_scaling(node, ctx, params, mock_cv)

        assert result_params["n_estimators"] == 200
        assert cv_result.mean_score == 0.7

    def test_early_stopping(self):
        """Search stops after first degradation beyond baseline."""
        config = EstimatorScalingConfig(
            scaling_search=True,
            scaling_factors=[2, 5, 10],
        )
        scaler = EstimatorScaler(config, greater_is_better=False)
        node = _make_node(GradientBoostingRegressor)
        ctx = MagicMock()

        call_count = 0

        def mock_cv(params):
            nonlocal call_count
            call_count += 1
            # 1x -> 0.5, 2x -> 0.6 (worse), should stop
            if params["n_estimators"] == 100:
                return _FakeCVResult(mean_score=0.5)
            return _FakeCVResult(mean_score=0.6)

        params = {"n_estimators": 100, "learning_rate": 0.1}
        result_params, cv_result = scaler.search_scaling(node, ctx, params, mock_cv)

        # Should have only called CV twice (1x baseline + 2x which was worse)
        assert call_count == 2
        # Should return baseline params
        assert result_params["n_estimators"] == 100

    def test_skips_when_no_n_estimators_support(self):
        config = EstimatorScalingConfig(scaling_search=True)
        scaler = EstimatorScaler(config, greater_is_better=False)
        node = _make_node(Ridge, name="ridge_node")
        ctx = MagicMock()

        def mock_cv(params):
            raise AssertionError("CV should not be called in early exit")

        params = {"alpha": 1.0}
        result_params, cv_result = scaler.search_scaling(node, ctx, params, mock_cv)

        assert result_params == params
        assert cv_result is None  # Caller handles fallback CV

    def test_skips_when_no_learning_rate(self):
        config = EstimatorScalingConfig(scaling_search=True)
        scaler = EstimatorScaler(config, greater_is_better=False)
        node = _make_node(RandomForestRegressor)
        ctx = MagicMock()

        def mock_cv(params):
            raise AssertionError("CV should not be called in early exit")

        params = {"n_estimators": 100}
        result_params, cv_result = scaler.search_scaling(node, ctx, params, mock_cv)

        assert result_params == params
        assert cv_result is None  # Caller handles fallback CV

    def test_custom_scaling_factors(self):
        """Custom factors should be used instead of defaults."""
        config = EstimatorScalingConfig(
            scaling_search=True,
            scaling_factors=[3, 7],
        )
        scaler = EstimatorScaler(config, greater_is_better=False)
        node = _make_node(GradientBoostingRegressor)
        ctx = MagicMock()

        tested_n_estimators = []
        # Each factor improves so all are tested
        score_map = {100: 0.5, 300: 0.4, 700: 0.3}

        def mock_cv(params):
            tested_n_estimators.append(params["n_estimators"])
            return _FakeCVResult(mean_score=score_map[params["n_estimators"]])

        params = {"n_estimators": 100, "learning_rate": 0.1}
        scaler.search_scaling(node, ctx, params, mock_cv)

        # Should test 1x, 3x, 7x (all improving, no early stop)
        assert tested_n_estimators == [100, 300, 700]
