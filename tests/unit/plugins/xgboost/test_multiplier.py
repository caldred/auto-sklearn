"""Tests for XGBMultiplierPlugin."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin


def _has_xgboost():
    """Check if XGBoost is available and loadable."""
    try:
        import xgboost
        # Try to actually use it to verify it's loadable
        xgboost.XGBClassifier()
        return True
    except (ImportError, Exception):
        # Catches ImportError and XGBoostError (e.g., missing libomp)
        return False


class MockXGBClassifier:
    """Mock XGBoost classifier for testing."""

    __name__ = "XGBClassifier"

    def __init__(self, **params):
        self.params = params
        self._booster = MagicMock()

    def get_booster(self):
        return self._booster

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def score(self, X, y):
        return 0.85


class MockXGBRegressor:
    """Mock XGBoost regressor for testing."""

    __name__ = "XGBRegressor"

    def get_booster(self):
        return MagicMock()


class MockXGBRanker:
    """Mock XGBoost ranker for testing."""

    __name__ = "XGBRanker"

    def get_booster(self):
        return MagicMock()


class NonXGBModel:
    """Non-XGBoost model for testing."""

    __name__ = "NonXGBModel"


class TestXGBMultiplierPluginInit:
    """Tests for XGBMultiplierPlugin initialization."""

    def test_default_multipliers(self):
        """Verify default multipliers are set."""
        plugin = XGBMultiplierPlugin()

        assert 0.5 in plugin.multipliers
        assert 1.0 in plugin.multipliers
        assert 2.0 in plugin.multipliers

    def test_custom_multipliers(self):
        """Verify custom multipliers can be set."""
        custom = [0.1, 0.5, 1.0]
        plugin = XGBMultiplierPlugin(multipliers=custom)

        assert plugin.multipliers == custom

    def test_default_cv_folds(self):
        """Verify default CV folds."""
        plugin = XGBMultiplierPlugin()

        assert plugin.cv_folds == 3

    def test_custom_cv_folds(self):
        """Verify custom CV folds."""
        plugin = XGBMultiplierPlugin(cv_folds=5)

        assert plugin.cv_folds == 5

    def test_name_property(self):
        """Verify name property."""
        plugin = XGBMultiplierPlugin()

        assert plugin.name == "xgb_multiplier"

    def test_repr(self):
        """Verify repr includes multipliers."""
        plugin = XGBMultiplierPlugin()

        repr_str = repr(plugin)

        assert "XGBMultiplierPlugin" in repr_str
        assert "multipliers" in repr_str


class TestXGBMultiplierPluginAppliesTo:
    """Tests for applies_to method."""

    def test_applies_to_xgb_classifier(self):
        """Verify applies to XGBClassifier."""
        plugin = XGBMultiplierPlugin()

        assert plugin.applies_to(MockXGBClassifier) is True

    def test_applies_to_xgb_regressor(self):
        """Verify applies to XGBRegressor."""
        plugin = XGBMultiplierPlugin()

        assert plugin.applies_to(MockXGBRegressor) is True

    def test_applies_to_xgb_ranker(self):
        """Verify applies to XGBRanker."""
        plugin = XGBMultiplierPlugin()

        assert plugin.applies_to(MockXGBRanker) is True

    def test_not_applies_to_non_xgb(self):
        """Verify doesn't apply to non-XGBoost models."""
        plugin = XGBMultiplierPlugin()

        assert plugin.applies_to(NonXGBModel) is False

    def test_applies_to_class_with_get_booster(self):
        """Verify applies to any class with get_booster."""

        class CustomBooster:
            def get_booster(self):
                pass

        plugin = XGBMultiplierPlugin()

        assert plugin.applies_to(CustomBooster) is True


class TestXGBMultiplierPluginModifyFitParams:
    """Tests for modify_fit_params method."""

    def test_adds_verbose_false(self, data_context):
        """Verify verbose=False is added by default."""
        plugin = XGBMultiplierPlugin()
        params = {}

        result = plugin.modify_fit_params(params, data_context)

        assert result["verbose"] is False

    def test_preserves_existing_early_stopping(self, data_context):
        """Verify existing early_stopping_rounds is preserved."""
        plugin = XGBMultiplierPlugin()
        params = {"early_stopping_rounds": 50}

        result = plugin.modify_fit_params(params, data_context)

        # Should not modify if early_stopping_rounds already set
        assert result == params

    def test_does_not_mutate_original(self, data_context):
        """Verify original params are not mutated."""
        plugin = XGBMultiplierPlugin()
        original = {"a": 1}

        result = plugin.modify_fit_params(original, data_context)

        assert "verbose" not in original
        assert "verbose" in result


class TestXGBMultiplierPluginPostTune:
    """Tests for post_tune method."""

    def test_disabled_returns_unchanged(self, data_context):
        """Verify disabled plugin returns params unchanged."""
        plugin = XGBMultiplierPlugin(enable_post_tune=False)
        node = MagicMock()
        params = {"learning_rate": 0.1, "n_estimators": 100}

        result = plugin.post_tune(params, node, data_context)

        assert result == params

    def test_missing_learning_rate_uses_default(self, data_context):
        """Verify missing learning_rate uses default value 0.1."""
        plugin = XGBMultiplierPlugin(multipliers=[1.0], enable_post_tune=True)
        node = MagicMock()
        params = {"n_estimators": 100}

        with patch.object(plugin, "_evaluate_params", return_value=0.5):
            result = plugin.post_tune(params, node, data_context)

        # Should use default learning_rate=0.1 and proceed with tuning
        assert "learning_rate" in result
        assert result["learning_rate"] == pytest.approx(0.1)

    def test_missing_n_estimators_uses_default(self, data_context):
        """Verify missing n_estimators uses default value 100."""
        plugin = XGBMultiplierPlugin(multipliers=[1.0], enable_post_tune=True)
        node = MagicMock()
        params = {"learning_rate": 0.1}

        with patch.object(plugin, "_evaluate_params", return_value=0.5):
            result = plugin.post_tune(params, node, data_context)

        # Should use default n_estimators=100 and proceed with tuning
        assert "n_estimators" in result
        assert result["n_estimators"] == 100

    def test_none_learning_rate_returns_unchanged(self, data_context):
        """Verify None learning_rate returns unchanged."""
        plugin = XGBMultiplierPlugin()
        node = MagicMock()
        params = {"learning_rate": None, "n_estimators": 100}

        result = plugin.post_tune(params, node, data_context)

        assert result == params

    def test_adjusts_learning_rate_and_n_estimators(self, data_context):
        """Verify multiplier adjusts both params inversely."""
        plugin = XGBMultiplierPlugin(multipliers=[2.0], enable_post_tune=True)

        node = MagicMock()
        node.create_estimator = MagicMock(return_value=MockXGBClassifier())

        params = {"learning_rate": 0.1, "n_estimators": 100}

        with patch.object(plugin, "_evaluate_params", return_value=0.5):
            result = plugin.post_tune(params, node, data_context)

        # With multiplier 2.0: lr = 0.1 * 2.0 = 0.2, n_estimators = 100 / 2.0 = 50
        assert result["learning_rate"] == 0.2
        assert result["n_estimators"] == 50

    def test_n_estimators_minimum(self, data_context):
        """Verify n_estimators has minimum of 10."""
        plugin = XGBMultiplierPlugin(multipliers=[100.0], enable_post_tune=True)

        node = MagicMock()
        node.create_estimator = MagicMock(return_value=MockXGBClassifier())

        params = {"learning_rate": 0.1, "n_estimators": 100}

        with patch.object(plugin, "_evaluate_params", return_value=0.5):
            result = plugin.post_tune(params, node, data_context)

        # With multiplier 100: n_estimators = 100 / 100 = 1 -> clamped to 10
        assert result["n_estimators"] >= 10

    def test_selects_best_multiplier(self, data_context):
        """Verify selects multiplier with best score."""
        plugin = XGBMultiplierPlugin(multipliers=[0.5, 1.0, 2.0], enable_post_tune=True)

        node = MagicMock()
        node.create_estimator = MagicMock(return_value=MockXGBClassifier())

        params = {"learning_rate": 0.1, "n_estimators": 100}

        # Return different scores for different multipliers
        scores = {0.5: 0.3, 1.0: 0.2, 2.0: 0.5}  # 1.0 is best (lower is better)

        def mock_evaluate(node, ctx, test_params):
            lr = test_params["learning_rate"]
            if lr == pytest.approx(0.05):  # 0.1 * 0.5
                return scores[0.5]
            elif lr == pytest.approx(0.1):  # 1.0
                return scores[1.0]
            else:  # 2.0
                return scores[2.0]

        with patch.object(plugin, "_evaluate_params", side_effect=mock_evaluate):
            result = plugin.post_tune(params, node, data_context)

        # Should select multiplier 1.0 (best score)
        assert result["learning_rate"] == pytest.approx(0.1)
        assert result["n_estimators"] == 100


class TestXGBMultiplierPluginEvaluateParams:
    """Tests for _evaluate_params method."""

    def test_returns_inf_on_exception(self, data_context):
        """Verify returns inf on evaluation failure."""
        plugin = XGBMultiplierPlugin()
        node = MagicMock()
        node.create_estimator = MagicMock(side_effect=Exception("Test error"))

        result = plugin._evaluate_params(node, data_context, {})

        assert result == float("inf")

    @pytest.mark.skipif(
        not _has_xgboost(),
        reason="XGBoost not installed"
    )
    def test_returns_score_on_success(self, classification_data):
        """Verify returns actual score on success."""
        import pandas as pd
        from sklearn_meta.core.data.context import DataContext

        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        plugin = XGBMultiplierPlugin(cv_folds=2)

        node = MagicMock()

        # This test requires actual xgboost
        import xgboost as xgb
        node.create_estimator = MagicMock(
            return_value=xgb.XGBClassifier(
                n_estimators=10,
                max_depth=3,
                random_state=42,
                eval_metric="logloss",
            )
        )

        result = plugin._evaluate_params(node, ctx, {})

        assert result != float("inf")
        assert result > 0
