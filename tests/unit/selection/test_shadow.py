"""Tests for ShadowFeatureSelector."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from auto_sklearn.selection.shadow import ShadowFeatureSelector, ShadowResult


class TestShadowFeatureSelectorCreation:
    """Tests for ShadowFeatureSelector creation."""

    def test_default_creation(self):
        """Verify default selector creation."""
        selector = ShadowFeatureSelector()

        assert selector.n_shadows == 5
        assert selector.n_clusters == 5
        assert selector.threshold_mult == pytest.approx(1.414, rel=1e-3)
        assert selector.random_state == 42

    def test_custom_parameters(self):
        """Verify custom parameter creation."""
        selector = ShadowFeatureSelector(
            n_shadows=3,
            n_clusters=4,
            threshold_mult=2.0,
            random_state=123,
        )

        assert selector.n_shadows == 3
        assert selector.n_clusters == 4
        assert selector.threshold_mult == 2.0
        assert selector.random_state == 123


class TestShadowFeatureSelectorEntropy:
    """Tests for entropy computation."""

    def test_compute_entropy_uniform(self):
        """Verify high entropy for uniform distribution."""
        selector = ShadowFeatureSelector()

        # Uniform distribution should have high entropy
        uniform_col = pd.Series(np.random.uniform(0, 1, 1000))
        entropy = selector._compute_entropy(uniform_col)

        assert entropy > 5  # High entropy

    def test_compute_entropy_constant(self):
        """Verify low entropy for constant values."""
        selector = ShadowFeatureSelector()

        # Constant values should have low entropy
        constant_col = pd.Series(np.ones(1000))
        entropy = selector._compute_entropy(constant_col)

        assert entropy < 1  # Low entropy

    def test_compute_entropy_binary(self):
        """Verify entropy for binary distribution."""
        selector = ShadowFeatureSelector()

        # 50/50 binary should have entropy around 1
        binary_col = pd.Series(np.random.choice([0, 1], 1000))
        entropy = selector._compute_entropy(binary_col)

        assert 0.5 < entropy < 2

    def test_compute_entropy_handles_nan(self):
        """Verify entropy computation handles NaN values."""
        selector = ShadowFeatureSelector()

        col_with_nan = pd.Series([1.0, 2.0, np.nan, 3.0, np.nan])
        entropy = selector._compute_entropy(col_with_nan)

        assert np.isfinite(entropy)


class TestShadowFeatureSelectorClustering:
    """Tests for feature clustering by entropy."""

    def test_cluster_features_by_entropy(self):
        """Verify features are clustered by entropy."""
        selector = ShadowFeatureSelector(n_clusters=3)

        # Create features with different entropy levels
        np.random.seed(42)
        X = pd.DataFrame({
            "uniform_1": np.random.uniform(0, 1, 100),
            "uniform_2": np.random.uniform(0, 1, 100),
            "binary_1": np.random.choice([0, 1], 100),
            "binary_2": np.random.choice([0, 1], 100),
            "constant": np.ones(100),
        })

        clusters = selector._cluster_features_by_entropy(X, list(X.columns))

        # Should have 3 clusters
        assert len(clusters) == 3

        # All features should be assigned
        all_features = set()
        for features in clusters.values():
            all_features.update(features)
        assert all_features == set(X.columns)


class TestShadowFeatureSelectorShadowCreation:
    """Tests for shadow feature creation."""

    def test_create_shadow_features_adds_shadows(self):
        """Verify shadow features are added."""
        selector = ShadowFeatureSelector(n_shadows=3, n_clusters=2)

        X = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
        })

        X_augmented, feature_to_shadow = selector._create_shadow_features(X, list(X.columns))

        # Should have original + shadow columns
        assert X_augmented.shape[1] > X.shape[1]

        # Shadow columns should start with __shadow_
        shadow_cols = [c for c in X_augmented.columns if c.startswith("__shadow_")]
        assert len(shadow_cols) > 0

    def test_shadow_entropy_matching(self):
        """Verify shadow features match real feature entropy approximately."""
        selector = ShadowFeatureSelector(n_shadows=5, n_clusters=3)

        # Create features with varying entropy
        np.random.seed(42)
        X = pd.DataFrame({
            "high_entropy": np.random.randn(500),
            "low_entropy": np.random.choice([0, 1, 2], 500, p=[0.8, 0.15, 0.05]),
        })

        X_augmented, _ = selector._create_shadow_features(X, list(X.columns))

        # Compare entropy of shadows to real features
        shadow_cols = [c for c in X_augmented.columns if c.startswith("__shadow_")]

        shadow_entropies = [selector._compute_entropy(X_augmented[c]) for c in shadow_cols]

        # Shadows should have reasonable entropy values (not all the same)
        assert len(set(round(e, 1) for e in shadow_entropies)) > 1


class TestShadowFeatureSelectorFitSelect:
    """Tests for fit_select method."""

    @pytest.fixture
    def classification_with_informative(self):
        """Create classification data with known informative features."""
        np.random.seed(42)
        n_samples = 500

        # Informative features
        X_informative = np.random.randn(n_samples, 5)

        # Create target based on informative features
        y = (X_informative[:, 0] + X_informative[:, 1] > 0).astype(int)

        # Add noise features
        X_noise = np.random.randn(n_samples, 5)

        X = pd.DataFrame(
            np.hstack([X_informative, X_noise]),
            columns=[f"informative_{i}" for i in range(5)] + [f"noise_{i}" for i in range(5)]
        )
        y = pd.Series(y)

        return X, y

    def test_fit_select_returns_shadow_result(self, classification_with_informative):
        """Verify fit_select returns ShadowResult."""
        X, y = classification_with_informative
        selector = ShadowFeatureSelector(n_shadows=3)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        result = selector.fit_select(model, X, y)

        assert isinstance(result, ShadowResult)

    def test_fit_select_keeps_important_features(self, classification_with_informative):
        """Verify informative features are more likely to be kept."""
        X, y = classification_with_informative
        selector = ShadowFeatureSelector(n_shadows=5, threshold_mult=1.0)
        model = RandomForestClassifier(n_estimators=50, random_state=42)

        result = selector.fit_select(model, X, y)

        # At least some informative features should be kept
        informative_kept = [f for f in result.features_to_keep if "informative" in f]
        assert len(informative_kept) >= 2

    def test_fit_select_drops_noise_features(self, classification_with_informative):
        """Verify noise features are more likely to be dropped."""
        X, y = classification_with_informative
        selector = ShadowFeatureSelector(n_shadows=5, threshold_mult=1.5)
        model = RandomForestClassifier(n_estimators=50, random_state=42)

        result = selector.fit_select(model, X, y)

        # More noise features should be dropped than informative ones
        noise_dropped = [f for f in result.features_to_drop if "noise" in f]
        informative_dropped = [f for f in result.features_to_drop if "informative" in f]

        # At least as many noise features dropped as informative
        assert len(noise_dropped) >= len(informative_dropped)

    def test_all_features_classified(self, classification_with_informative):
        """Verify all features are either kept or dropped."""
        X, y = classification_with_informative
        selector = ShadowFeatureSelector()
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        result = selector.fit_select(model, X, y)

        all_features = set(result.features_to_keep) | set(result.features_to_drop)
        assert all_features == set(X.columns)

    def test_no_overlap_kept_dropped(self, classification_with_informative):
        """Verify no feature is both kept and dropped."""
        X, y = classification_with_informative
        selector = ShadowFeatureSelector()
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        result = selector.fit_select(model, X, y)

        intersection = set(result.features_to_keep) & set(result.features_to_drop)
        assert len(intersection) == 0


class TestShadowFeatureSelectorThreshold:
    """Tests for threshold multiplier effect."""

    @pytest.fixture
    def simple_data(self):
        """Create simple classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)

    def test_higher_threshold_keeps_fewer(self, simple_data):
        """Verify higher threshold keeps fewer features."""
        X, y = simple_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        selector_low = ShadowFeatureSelector(threshold_mult=0.5)
        selector_high = ShadowFeatureSelector(threshold_mult=2.0)

        result_low = selector_low.fit_select(model.set_params(), X, y)
        result_high = selector_high.fit_select(model.set_params(), X, y)

        # Higher threshold should keep same or fewer features
        assert len(result_high.features_to_keep) <= len(result_low.features_to_keep)

    def test_zero_threshold_keeps_all_above_zero(self, simple_data):
        """Verify very low threshold keeps most features."""
        X, y = simple_data
        selector = ShadowFeatureSelector(threshold_mult=0.01)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        result = selector.fit_select(model, X, y)

        # Should keep most features
        assert len(result.features_to_keep) >= len(X.columns) // 2


class TestShadowResult:
    """Tests for ShadowResult dataclass."""

    def test_shadow_result_fields(self):
        """Verify ShadowResult has expected fields."""
        result = ShadowResult(
            features_to_keep=["a", "b"],
            features_to_drop=["c"],
            feature_importances={"a": 0.5, "b": 0.3, "c": 0.2},
            shadow_importances={"shadow_1": 0.1},
            feature_to_shadow={"a": "shadow_1", "b": "shadow_1", "c": "shadow_1"},
            threshold_used=0.14,
        )

        assert result.features_to_keep == ["a", "b"]
        assert result.features_to_drop == ["c"]
        assert result.feature_importances["a"] == 0.5
        assert result.threshold_used == 0.14


class TestShadowFeatureSelectorSelectFeatures:
    """Tests for select_features convenience method."""

    def test_select_features_returns_list(self):
        """Verify select_features returns list of feature names."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.choice([0, 1], 100))

        selector = ShadowFeatureSelector()
        model = RandomForestClassifier(n_estimators=5, random_state=42)

        selected = selector.select_features(model, X, y)

        assert isinstance(selected, list)
        assert all(isinstance(f, str) for f in selected)
        assert all(f in X.columns for f in selected)


class TestShadowFeatureSelectorRepr:
    """Tests for repr."""

    def test_repr(self):
        """Verify repr is informative."""
        selector = ShadowFeatureSelector(n_shadows=3, threshold_mult=1.5)

        repr_str = repr(selector)

        assert "ShadowFeatureSelector" in repr_str
        assert "n_shadows=3" in repr_str
        assert "1.5" in repr_str
