"""Tests for ImportanceExtractor classes."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from sklearn_meta.selection.importance import (
    ImportanceExtractor,
    ImportanceRegistry,
    LinearImportanceExtractor,
    PermutationImportanceExtractor,
    TreeImportanceExtractor,
)


class TestTreeImportanceExtractor:
    """Tests for TreeImportanceExtractor."""

    def test_applies_to_rf(self, fitted_rf):
        """Verify applies to RandomForest."""
        extractor = TreeImportanceExtractor()

        assert extractor.applies_to(fitted_rf) is True

    def test_applies_to_non_tree_false(self, fitted_lr):
        """Verify doesn't apply to non-tree models."""
        extractor = TreeImportanceExtractor()

        # LR has coef_, not feature_importances_
        # But TreeImportanceExtractor checks for feature_importances_ first
        assert extractor.applies_to(fitted_lr) is False

    def test_extract_rf_importance(self, fitted_rf, classification_data):
        """Verify extraction from RandomForest."""
        X, y = classification_data
        extractor = TreeImportanceExtractor()

        feature_names = list(X.columns)
        importance = extractor.extract(fitted_rf, feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        assert all(f in importance for f in feature_names)
        assert all(v >= 0 for v in importance.values())

    def test_extract_sums_to_reasonable_value(self, fitted_rf, classification_data):
        """Verify importances sum to approximately 1."""
        X, y = classification_data
        extractor = TreeImportanceExtractor()

        feature_names = list(X.columns)
        importance = extractor.extract(fitted_rf, feature_names)

        total = sum(importance.values())
        # RF feature_importances_ sum to 1
        assert 0.99 <= total <= 1.01


class TestLinearImportanceExtractor:
    """Tests for LinearImportanceExtractor."""

    def test_applies_to_lr(self, fitted_lr):
        """Verify applies to LogisticRegression."""
        extractor = LinearImportanceExtractor()

        assert extractor.applies_to(fitted_lr) is True

    def test_applies_to_non_linear_false(self, fitted_rf):
        """Verify doesn't apply to non-linear models."""
        extractor = LinearImportanceExtractor()

        # RF has feature_importances_, not coef_
        assert extractor.applies_to(fitted_rf) is False

    def test_extract_lr_importance(self, fitted_lr, classification_data):
        """Verify extraction from LogisticRegression."""
        X, y = classification_data
        extractor = LinearImportanceExtractor()

        feature_names = list(X.columns)
        importance = extractor.extract(fitted_lr, feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        assert all(v >= 0 for v in importance.values())  # Absolute values


class TestPermutationImportanceExtractor:
    """Tests for PermutationImportanceExtractor."""

    def test_applies_to_any_model(self, fitted_rf, fitted_lr):
        """Verify applies to any model with predict."""
        extractor = PermutationImportanceExtractor()

        assert extractor.applies_to(fitted_rf) is True
        assert extractor.applies_to(fitted_lr) is True

    def test_extract_requires_validation_data(self, fitted_rf, classification_data):
        """Verify extraction requires X_val and y_val."""
        X, y = classification_data
        extractor = PermutationImportanceExtractor(n_repeats=2)

        feature_names = list(X.columns)

        with pytest.raises(ValueError, match="requires X_val and y_val"):
            extractor.extract(fitted_rf, feature_names)

    def test_extract_with_validation_data(self, fitted_rf, classification_data):
        """Verify extraction works with validation data."""
        X, y = classification_data
        extractor = PermutationImportanceExtractor(n_repeats=2, random_state=42)

        feature_names = list(X.columns)
        importance = extractor.extract(
            fitted_rf, feature_names,
            X_val=X.iloc[:100],
            y_val=y.iloc[:100],
        )

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)


class TestImportanceRegistry:
    """Tests for ImportanceRegistry."""

    def test_default_extractors(self):
        """Verify default extractors are registered."""
        registry = ImportanceRegistry()

        # Should have tree and linear extractors
        # Plus a fallback permutation extractor
        assert len(registry._extractors) >= 2

    def test_get_extractor_tree(self, fitted_rf):
        """Verify returns TreeImportanceExtractor for RF."""
        registry = ImportanceRegistry()

        extractor = registry.get_extractor(fitted_rf)

        assert isinstance(extractor, TreeImportanceExtractor)

    def test_get_extractor_linear(self, fitted_lr):
        """Verify returns LinearImportanceExtractor for LR."""
        registry = ImportanceRegistry()

        extractor = registry.get_extractor(fitted_lr)

        assert isinstance(extractor, LinearImportanceExtractor)

    def test_get_extractor_fallback(self):
        """Verify returns fallback for unknown model."""
        registry = ImportanceRegistry()

        class UnknownModel:
            def predict(self, X):
                return np.zeros(len(X))

        extractor = registry.get_extractor(UnknownModel())

        assert isinstance(extractor, PermutationImportanceExtractor)

    def test_register_custom_extractor(self):
        """Verify custom extractor can be registered."""
        registry = ImportanceRegistry()

        class CustomExtractor(ImportanceExtractor):
            def applies_to(self, model):
                return hasattr(model, "custom_importance")

            def extract(self, model, feature_names, **kwargs):
                return {f: 1.0 for f in feature_names}

        initial_count = len(registry._extractors)
        registry.register(CustomExtractor())

        assert len(registry._extractors) == initial_count + 1

    def test_extract_importance(self, fitted_rf, classification_data):
        """Verify extract_importance uses correct extractor."""
        X, y = classification_data
        registry = ImportanceRegistry()

        feature_names = list(X.columns)
        importance = registry.extract_importance(fitted_rf, feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)


class TestImportanceNormalization:
    """Tests for importance normalization."""

    def test_normalize_sums_to_one(self):
        """Verify normalize makes importances sum to 1."""
        extractor = TreeImportanceExtractor()

        importance = {"a": 10, "b": 20, "c": 70}
        normalized = extractor.normalize(importance)

        assert sum(normalized.values()) == pytest.approx(1.0)
        assert normalized["a"] == pytest.approx(0.1)
        assert normalized["b"] == pytest.approx(0.2)
        assert normalized["c"] == pytest.approx(0.7)

    def test_normalize_zero_total(self):
        """Verify normalize handles zero total."""
        extractor = TreeImportanceExtractor()

        importance = {"a": 0, "b": 0, "c": 0}
        normalized = extractor.normalize(importance)

        assert all(v == 0 for v in normalized.values())

    def test_normalize_preserves_order(self):
        """Verify normalize preserves relative ordering."""
        extractor = TreeImportanceExtractor()

        importance = {"a": 5, "b": 10, "c": 15}
        normalized = extractor.normalize(importance)

        assert normalized["a"] < normalized["b"] < normalized["c"]
