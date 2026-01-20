"""Tests for XGBImportanceExtractor and XGBImportancePlugin."""

import pytest
import numpy as np
import warnings
from unittest.mock import MagicMock, patch

from sklearn_meta.plugins.xgboost.importance import (
    XGBImportanceExtractor,
    XGBImportancePlugin,
)


class MockBooster:
    """Mock XGBoost booster for testing."""

    def __init__(self, scores=None, feature_names=None):
        self._scores = scores or {"f0": 10.0, "f1": 20.0, "f2": 30.0}
        self.feature_names = feature_names

    def get_score(self, importance_type=None):
        return self._scores


class MockXGBModel:
    """Mock XGBoost model for testing."""

    def __init__(self, scores=None, feature_names=None):
        self._booster = MockBooster(scores, feature_names)
        self._sklearn_meta_meta = {}

    def get_booster(self):
        return self._booster


class NonXGBModel:
    """Non-XGBoost model for testing."""
    pass


class TestXGBImportanceExtractor:
    """Tests for XGBImportanceExtractor."""

    def test_default_importance_type(self):
        """Verify default importance type is total_gain."""
        extractor = XGBImportanceExtractor()

        assert extractor.importance_type == "total_gain"

    def test_custom_importance_type(self):
        """Verify custom importance type can be set."""
        extractor = XGBImportanceExtractor(importance_type="weight")

        assert extractor.importance_type == "weight"

    def test_applies_to_xgb_model(self):
        """Verify applies to models with get_booster."""
        extractor = XGBImportanceExtractor()
        model = MockXGBModel()

        assert extractor.applies_to(model) is True

    def test_not_applies_to_non_xgb(self):
        """Verify doesn't apply to non-XGBoost models."""
        extractor = XGBImportanceExtractor()
        model = NonXGBModel()

        assert extractor.applies_to(model) is False

    def test_extract_returns_dict(self):
        """Verify extract returns dictionary."""
        extractor = XGBImportanceExtractor()
        model = MockXGBModel(scores={"f0": 1.0, "f1": 2.0})
        feature_names = ["feat_a", "feat_b"]

        result = extractor.extract(model, feature_names)

        assert isinstance(result, dict)
        assert len(result) == 2

    def test_extract_maps_feature_names(self):
        """Verify extract maps XGBoost feature names to actual names."""
        extractor = XGBImportanceExtractor()
        model = MockXGBModel(scores={"f0": 10.0, "f1": 20.0, "f2": 30.0})
        feature_names = ["alpha", "beta", "gamma"]

        result = extractor.extract(model, feature_names)

        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result
        assert result["alpha"] == 10.0
        assert result["beta"] == 20.0
        assert result["gamma"] == 30.0

    def test_extract_missing_feature_is_zero(self):
        """Verify missing features get zero importance."""
        extractor = XGBImportanceExtractor()
        model = MockXGBModel(scores={"f0": 10.0})  # Only f0 has score
        feature_names = ["alpha", "beta", "gamma"]

        result = extractor.extract(model, feature_names)

        assert result["alpha"] == 10.0
        assert result["beta"] == 0.0  # Missing -> 0
        assert result["gamma"] == 0.0

    def test_extract_override_importance_type(self):
        """Verify importance_type can be overridden per call."""
        extractor = XGBImportanceExtractor(importance_type="total_gain")
        model = MockXGBModel()
        feature_names = ["a", "b", "c"]

        # Extract with override
        result = extractor.extract(model, feature_names, importance_type="weight")

        # Should not raise and return valid dict
        assert isinstance(result, dict)

    def test_extract_handles_get_score_exception(self):
        """Verify extract handles get_score exception gracefully."""
        extractor = XGBImportanceExtractor()
        model = MockXGBModel()

        # Make get_score raise on specific importance_type but succeed on default
        model._booster.get_score = MagicMock(
            side_effect=[ValueError("bad type"), {"f0": 1.0}]
        )

        feature_names = ["a"]
        result = extractor.extract(model, feature_names)

        # Should fall back to default get_score
        assert isinstance(result, dict)


class TestXGBImportancePlugin:
    """Tests for XGBImportancePlugin."""

    def test_default_importance_type(self):
        """Verify default importance type."""
        plugin = XGBImportancePlugin()

        assert plugin.importance_type == "total_gain"

    def test_custom_importance_type(self):
        """Verify custom importance type."""
        plugin = XGBImportancePlugin(importance_type="weight")

        assert plugin.importance_type == "weight"

    def test_name_property(self):
        """Verify name property."""
        plugin = XGBImportancePlugin()

        assert plugin.name == "xgb_importance"

    def test_repr(self):
        """Verify repr includes importance_type."""
        plugin = XGBImportancePlugin(importance_type="gain")

        repr_str = repr(plugin)

        assert "XGBImportancePlugin" in repr_str
        assert "gain" in repr_str

    def test_applies_to_xgb_classifier(self):
        """Verify applies to XGBClassifier."""

        class XGBClassifier:
            pass

        plugin = XGBImportancePlugin()

        assert plugin.applies_to(XGBClassifier) is True

    def test_applies_to_xgb_regressor(self):
        """Verify applies to XGBRegressor."""

        class XGBRegressor:
            pass

        plugin = XGBImportancePlugin()

        assert plugin.applies_to(XGBRegressor) is True

    def test_applies_to_xgb_ranker(self):
        """Verify applies to XGBRanker."""

        class XGBRanker:
            pass

        plugin = XGBImportancePlugin()

        assert plugin.applies_to(XGBRanker) is True

    def test_not_applies_to_other(self):
        """Verify doesn't apply to non-XGBoost."""
        plugin = XGBImportancePlugin()

        assert plugin.applies_to(NonXGBModel) is False


class TestXGBImportancePluginPostFit:
    """Tests for XGBImportancePlugin.post_fit."""

    def test_extracts_and_caches_importance(self, data_context):
        """Verify post_fit extracts and caches importance."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel(scores={"f0": 10.0, "f1": 20.0})
        node = MagicMock()

        result = plugin.post_fit(model, node, data_context)

        assert "_sklearn_meta_meta" in dir(result)
        assert "feature_importance" in result._sklearn_meta_meta
        assert "importance_type" in result._sklearn_meta_meta

    def test_returns_same_model(self, data_context):
        """Verify post_fit returns the same model object."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel()
        node = MagicMock()

        result = plugin.post_fit(model, node, data_context)

        assert result is model

    def test_prune_zero_importance_warns(self, data_context):
        """Verify warning when prune_zero_importance is enabled."""
        plugin = XGBImportancePlugin(prune_zero_importance=True)
        model = MockXGBModel(scores={"f0": 10.0})  # f1, f2 will be zero
        node = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plugin.post_fit(model, node, data_context)

            # Should have a warning about zero importance features
            zero_warnings = [warning for warning in w if "zero importance" in str(warning.message)]
            assert len(zero_warnings) > 0

    def test_no_warning_without_prune(self, data_context):
        """Verify no warning when prune_zero_importance is disabled."""
        plugin = XGBImportancePlugin(prune_zero_importance=False)
        model = MockXGBModel(scores={"f0": 10.0})  # f1, f2 will be zero
        node = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plugin.post_fit(model, node, data_context)

            zero_warnings = [warning for warning in w if "zero importance" in str(warning.message)]
            assert len(zero_warnings) == 0


class TestXGBImportancePluginGetImportance:
    """Tests for XGBImportancePlugin.get_importance."""

    def test_returns_cached_importance(self):
        """Verify returns cached importance if available."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel()
        model._sklearn_meta_meta["feature_importance"] = {"a": 1.0, "b": 2.0}

        result = plugin.get_importance(model)

        assert result == {"a": 1.0, "b": 2.0}

    def test_extracts_fresh_with_feature_names(self):
        """Verify extracts fresh importance with provided feature names."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel(scores={"f0": 10.0, "f1": 20.0})
        model._sklearn_meta_meta = {}  # No cached importance

        result = plugin.get_importance(model, feature_names=["x", "y"])

        assert "x" in result
        assert "y" in result

    def test_raises_without_feature_names(self):
        """Verify raises when no feature names and no cache."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel()
        model._sklearn_meta_meta = {}
        model._booster.feature_names = None

        # Either ValueError from explicit check or TypeError from iterating None
        with pytest.raises((ValueError, TypeError)):
            plugin.get_importance(model)

    def test_uses_booster_feature_names(self):
        """Verify uses booster's feature names as fallback."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel(scores={"f0": 10.0}, feature_names=["alpha"])
        model._sklearn_meta_meta = {}

        result = plugin.get_importance(model)

        assert "alpha" in result


class TestXGBImportancePluginGetTopFeatures:
    """Tests for XGBImportancePlugin.get_top_features."""

    def test_returns_top_n(self):
        """Verify returns top N features."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel()
        model._sklearn_meta_meta["feature_importance"] = {
            "a": 10.0,
            "b": 30.0,
            "c": 20.0,
            "d": 5.0,
        }

        result = plugin.get_top_features(model, n=2)

        assert len(result) == 2
        assert result[0][0] == "b"  # Highest
        assert result[1][0] == "c"  # Second highest

    def test_returns_tuples(self):
        """Verify returns list of tuples."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel()
        model._sklearn_meta_meta["feature_importance"] = {"a": 10.0, "b": 20.0}

        result = plugin.get_top_features(model, n=2)

        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    def test_sorted_descending(self):
        """Verify features are sorted by importance descending."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel()
        model._sklearn_meta_meta["feature_importance"] = {
            "low": 1.0,
            "medium": 5.0,
            "high": 10.0,
        }

        result = plugin.get_top_features(model, n=3)

        importances = [imp for _, imp in result]
        assert importances == sorted(importances, reverse=True)

    def test_default_n_is_10(self):
        """Verify default n is 10."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel()
        model._sklearn_meta_meta["feature_importance"] = {
            f"f{i}": float(i) for i in range(20)
        }

        result = plugin.get_top_features(model)

        assert len(result) == 10

    def test_returns_all_if_less_than_n(self):
        """Verify returns all features if fewer than n."""
        plugin = XGBImportancePlugin()
        model = MockXGBModel()
        model._sklearn_meta_meta["feature_importance"] = {"a": 1.0, "b": 2.0}

        result = plugin.get_top_features(model, n=10)

        assert len(result) == 2
