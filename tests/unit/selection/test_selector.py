"""Tests for FeatureSelector."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn_meta.api import GraphBuilder
from sklearn_meta.selection.selector import FeatureSelectionConfig, FeatureSelector


class _DummyModel:
    def fit(self, X, y):
        return self


class _DummyExtractor:
    def __init__(self, importances):
        self._importances = importances

    def extract(self, model, feature_names, **kwargs):
        return {name: self._importances[name] for name in feature_names}


class TestFeatureSelectorGroups:
    """Tests for grouped feature selection behavior."""

    def test_threshold_uses_group_average_and_keeps_group_together(self):
        """Grouped features should share averaged importance and selection decision."""
        X = pd.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [1, 0, 1, 0],
                "c": [0, 0, 1, 1],
            }
        )
        y = pd.Series([0, 1, 0, 1])

        config = FeatureSelectionConfig(
            method="threshold",
            threshold_percentile=50.0,
            min_features=1,
            feature_groups={"ab": ["a", "b"]},
        )
        selector = FeatureSelector(config)
        selector._importance_registry.get_extractor = lambda _model: _DummyExtractor(
            {"a": 0.9, "b": 0.1, "c": 0.4}
        )

        result = selector.select(_DummyModel(), X, y)

        assert set(result.selected_features) == {"a", "b"}
        assert set(result.dropped_features) == {"c"}
        assert result.importances["a"] == result.importances["b"] == 0.5
        assert result.importances["c"] == 0.4
        assert result.details["group_importances"]["ab"] == 0.5

    def test_threshold_drops_entire_group_when_group_average_is_low(self):
        """If a group's average importance is low, all its members are dropped."""
        X = pd.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [1, 0, 1, 0],
                "c": [0, 0, 1, 1],
            }
        )
        y = pd.Series([0, 1, 0, 1])

        config = FeatureSelectionConfig(
            method="threshold",
            threshold_percentile=50.0,
            min_features=1,
            feature_groups={"ab": ["a", "b"]},
        )
        selector = FeatureSelector(config)
        selector._importance_registry.get_extractor = lambda _model: _DummyExtractor(
            {"a": 0.2, "b": 0.0, "c": 0.4}
        )

        result = selector.select(_DummyModel(), X, y)

        assert result.selected_features == ["c"]
        assert set(result.dropped_features) == {"a", "b"}

    def test_min_features_never_splits_group(self):
        """Min-feature expansion should add complete groups, not partial groups."""
        X = pd.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [1, 0, 1, 0],
                "c": [0, 0, 1, 1],
            }
        )
        y = pd.Series([0, 1, 0, 1])

        config = FeatureSelectionConfig(
            method="threshold",
            threshold_percentile=50.0,
            min_features=2,
            feature_groups={"ab": ["a", "b"]},
        )
        selector = FeatureSelector(config)
        selector._importance_registry.get_extractor = lambda _model: _DummyExtractor(
            {"a": 0.1, "b": 0.0, "c": 0.4}
        )

        result = selector.select(_DummyModel(), X, y)

        assert set(result.selected_features) == {"a", "b", "c"}

    def test_shadow_mode_uses_single_shadow_representative_per_group(self):
        """Grouped shadow selection should map all group members to one shadow."""
        rng = np.random.default_rng(42)
        n = 200
        a = rng.normal(size=n)
        b = a + rng.normal(scale=0.1, size=n)
        c = rng.normal(size=n)
        y = (a + 0.5 * c > 0.0).astype(int)

        X = pd.DataFrame({"a": a, "b": b, "c": c})
        y = pd.Series(y)

        config = FeatureSelectionConfig(
            method="shadow",
            n_shadows=3,
            threshold_mult=1.0,
            min_features=1,
            feature_groups={"ab": ["a", "b"]},
        )
        selector = FeatureSelector(config)
        model = RandomForestClassifier(n_estimators=40, random_state=42)

        result = selector.select(model, X, y)

        feature_to_shadow = result.details["feature_to_shadow"]
        shadow_importances = result.details["shadow_importances"]

        assert feature_to_shadow["a"] == feature_to_shadow["b"]
        assert feature_to_shadow["a"].startswith("__shadow_group_avg__")
        assert feature_to_shadow["a"] in shadow_importances


class TestGraphBuilderFeatureSelection:
    """Tests for GraphBuilder.with_feature_selection."""

    def test_with_feature_selection_passes_feature_groups(self):
        """GraphBuilder should store feature_groups in FeatureSelectionConfig."""
        builder = GraphBuilder("test").with_feature_selection(
            feature_groups={"cat_one_hot": ["cat_a", "cat_b", "cat_c"]}
        )

        assert builder._feature_selection_config is not None
        assert builder._feature_selection_config.feature_groups == {
            "cat_one_hot": ["cat_a", "cat_b", "cat_c"]
        }
