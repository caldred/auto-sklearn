"""Tests for DataContext."""

import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError

from sklearn_meta.core.data.context import DataContext


class TestDataContextImmutability:
    """Tests for DataContext immutability."""

    def test_datacontext_is_frozen(self, classification_data):
        """Verify DataContext is frozen and cannot be mutated."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        with pytest.raises(FrozenInstanceError):
            ctx.X = pd.DataFrame()

    def test_datacontext_with_methods_return_new_instance(self, classification_data):
        """Verify with_* methods return new instances."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        new_ctx = ctx.with_metadata("key", "value")

        assert new_ctx is not ctx
        assert ctx.metadata == {}
        assert new_ctx.metadata == {"key": "value"}


class TestDataContextWithSubset:
    """Tests for DataContext.with_indices()."""

    def test_with_indices_returns_correct_subset(self, classification_data):
        """Verify with_indices returns the correct data subset."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        indices = np.array([0, 5, 10, 15, 20])
        subset_ctx = ctx.with_indices(indices)

        assert subset_ctx.n_samples == len(indices)
        assert subset_ctx.n_features == ctx.n_features
        np.testing.assert_array_equal(subset_ctx.indices, indices)

    def test_with_indices_preserves_feature_values(self, classification_data):
        """Verify subset X values match original at indices."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        indices = np.array([0, 5, 10])
        subset_ctx = ctx.with_indices(indices)

        for i, idx in enumerate(indices):
            np.testing.assert_array_almost_equal(
                subset_ctx.X.iloc[i].values,
                X.iloc[idx].values,
            )

    def test_with_indices_preserves_target_values(self, classification_data):
        """Verify subset y values match original at indices."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        indices = np.array([0, 5, 10])
        subset_ctx = ctx.with_indices(indices)

        for i, idx in enumerate(indices):
            assert subset_ctx.y.iloc[i] == y.iloc[idx]

    def test_with_indices_subsets_groups(self, grouped_data):
        """Verify groups are also subset correctly."""
        X, y, groups = grouped_data
        ctx = DataContext(X=X, y=y, groups=groups)

        indices = np.array([0, 10, 20])  # Different groups
        subset_ctx = ctx.with_indices(indices)

        for i, idx in enumerate(indices):
            assert subset_ctx.groups.iloc[i] == groups.iloc[idx]

    def test_with_indices_subsets_upstream_outputs(self, classification_data):
        """Verify upstream outputs are subset correctly."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        # Add upstream output
        upstream = np.random.randn(len(X))
        ctx = ctx.with_upstream_output("model_1", upstream)

        indices = np.array([0, 5, 10])
        subset_ctx = ctx.with_indices(indices)

        np.testing.assert_array_almost_equal(
            subset_ctx.upstream_outputs["model_1"],
            upstream[indices],
        )

    def test_with_indices_subsets_base_margin(self, classification_data):
        """Verify base margin is subset correctly."""
        X, y = classification_data
        base_margin = np.random.randn(len(X))
        ctx = DataContext(X=X, y=y, base_margin=base_margin)

        indices = np.array([0, 5, 10])
        subset_ctx = ctx.with_indices(indices)

        np.testing.assert_array_almost_equal(
            subset_ctx.base_margin,
            base_margin[indices],
        )


class TestDataContextWithUpstream:
    """Tests for DataContext.with_upstream_output()."""

    def test_with_upstream_output_adds_output(self, classification_data):
        """Verify with_upstream_output adds the output."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        upstream = np.random.randn(len(X))
        new_ctx = ctx.with_upstream_output("model_1", upstream)

        assert "model_1" in new_ctx.upstream_outputs
        np.testing.assert_array_almost_equal(
            new_ctx.upstream_outputs["model_1"],
            upstream,
        )

    def test_with_upstream_output_preserves_original(self, classification_data):
        """Verify original context is unchanged."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        upstream = np.random.randn(len(X))
        new_ctx = ctx.with_upstream_output("model_1", upstream)

        assert "model_1" not in ctx.upstream_outputs
        assert len(ctx.upstream_outputs) == 0

    def test_multiple_upstream_outputs(self, classification_data):
        """Verify multiple upstream outputs can be added."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        upstream1 = np.random.randn(len(X))
        upstream2 = np.random.randn(len(X))

        ctx = ctx.with_upstream_output("model_1", upstream1)
        ctx = ctx.with_upstream_output("model_2", upstream2)

        assert len(ctx.upstream_outputs) == 2
        assert "model_1" in ctx.upstream_outputs
        assert "model_2" in ctx.upstream_outputs


class TestDataContextBaseMargin:
    """Tests for DataContext base margin handling."""

    def test_with_base_margin_sets_margin(self, classification_data):
        """Verify with_base_margin sets the base margin."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        margin = np.random.randn(len(X))
        new_ctx = ctx.with_base_margin(margin)

        np.testing.assert_array_almost_equal(new_ctx.base_margin, margin)

    def test_base_margin_shape_matches(self, classification_data):
        """Verify base margin must match X length."""
        X, y = classification_data
        wrong_margin = np.random.randn(len(X) + 10)

        with pytest.raises(ValueError, match="same length"):
            DataContext(X=X, y=y, base_margin=wrong_margin)

    def test_base_margin_preserved_in_copy(self, classification_data):
        """Verify base margin is preserved in copy."""
        X, y = classification_data
        margin = np.random.randn(len(X))
        ctx = DataContext(X=X, y=y, base_margin=margin)

        copy_ctx = ctx.copy()

        np.testing.assert_array_almost_equal(copy_ctx.base_margin, margin)


class TestDataContextProperties:
    """Tests for DataContext properties."""

    def test_feature_columns_match_dataframe(self, classification_data):
        """Verify feature_names matches DataFrame columns."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        assert ctx.feature_names == list(X.columns)

    def test_n_samples_correct(self, classification_data):
        """Verify n_samples property is correct."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        assert ctx.n_samples == len(X)

    def test_n_features_correct(self, classification_data):
        """Verify n_features property is correct."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        assert ctx.n_features == X.shape[1]


class TestDataContextValidation:
    """Tests for DataContext validation."""

    def test_x_y_length_mismatch_raises(self, classification_data):
        """Verify X and y length mismatch raises error."""
        X, y = classification_data
        y_short = y.iloc[:100]

        with pytest.raises(ValueError, match="same length"):
            DataContext(X=X, y=y_short)

    def test_x_groups_length_mismatch_raises(self, classification_data):
        """Verify X and groups length mismatch raises error."""
        X, y = classification_data
        groups = pd.Series(range(100))

        with pytest.raises(ValueError, match="same length"):
            DataContext(X=X, y=y, groups=groups)


class TestDataContextAugmentWithPredictions:
    """Tests for DataContext.augment_with_predictions()."""

    def test_augment_adds_prediction_columns(self, classification_data):
        """Verify predictions are added as columns."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        predictions = {"model_1": np.random.randn(len(X))}
        augmented = ctx.augment_with_predictions(predictions)

        assert "pred_model_1" in augmented.X.columns
        assert augmented.n_features == ctx.n_features + 1

    def test_augment_with_multiclass_probabilities(self, classification_data):
        """Verify multi-class probabilities are expanded."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        # 3-class probabilities
        proba = np.random.rand(len(X), 3)
        predictions = {"model_1": proba}
        augmented = ctx.augment_with_predictions(predictions)

        assert "pred_model_1_0" in augmented.X.columns
        assert "pred_model_1_1" in augmented.X.columns
        assert "pred_model_1_2" in augmented.X.columns
        assert augmented.n_features == ctx.n_features + 3

    def test_augment_preserves_original_features(self, classification_data):
        """Verify original features are preserved."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        predictions = {"model_1": np.random.randn(len(X))}
        augmented = ctx.augment_with_predictions(predictions)

        for col in X.columns:
            assert col in augmented.X.columns
            np.testing.assert_array_almost_equal(
                augmented.X[col].values,
                X[col].values,
            )

    def test_augment_with_custom_prefix(self, classification_data):
        """Verify custom prefix is applied."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        predictions = {"model_1": np.random.randn(len(X))}
        augmented = ctx.augment_with_predictions(predictions, prefix="oof_")

        assert "oof_model_1" in augmented.X.columns
        assert "pred_model_1" not in augmented.X.columns


class TestDataContextCopy:
    """Tests for DataContext.copy()."""

    def test_copy_creates_new_dataframe(self, classification_data):
        """Verify copy creates a new DataFrame."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        copy_ctx = ctx.copy()

        assert copy_ctx.X is not ctx.X
        pd.testing.assert_frame_equal(copy_ctx.X, ctx.X)

    def test_copy_creates_new_series(self, classification_data):
        """Verify copy creates a new Series for y."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        copy_ctx = ctx.copy()

        assert copy_ctx.y is not ctx.y
        pd.testing.assert_series_equal(copy_ctx.y, ctx.y)

    def test_copy_creates_new_metadata_dict(self, classification_data):
        """Verify copy creates a new metadata dict."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y, metadata={"key": "value"})

        copy_ctx = ctx.copy()

        assert copy_ctx.metadata is not ctx.metadata
        assert copy_ctx.metadata == ctx.metadata


class TestDataContextWithX:
    """Tests for DataContext.with_X()."""

    def test_with_x_replaces_features(self, classification_data):
        """Verify with_X replaces the feature DataFrame."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        new_X = pd.DataFrame(np.random.randn(len(X), 5))
        new_ctx = ctx.with_X(new_X)

        assert new_ctx.n_features == 5
        pd.testing.assert_frame_equal(new_ctx.X, new_X)

    def test_with_x_preserves_y(self, classification_data):
        """Verify with_X preserves the target."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        new_X = pd.DataFrame(np.random.randn(len(X), 5))
        new_ctx = ctx.with_X(new_X)

        pd.testing.assert_series_equal(new_ctx.y, y)


class TestDataContextWithY:
    """Tests for DataContext.with_y()."""

    def test_with_y_replaces_target(self, classification_data):
        """Verify with_y replaces the target Series."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        new_y = pd.Series(np.random.randn(len(X)))
        new_ctx = ctx.with_y(new_y)

        pd.testing.assert_series_equal(new_ctx.y, new_y)

    def test_with_y_preserves_x(self, classification_data):
        """Verify with_y preserves features."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        new_y = pd.Series(np.random.randn(len(X)))
        new_ctx = ctx.with_y(new_y)

        pd.testing.assert_frame_equal(new_ctx.X, X)


class TestDataContextWithMetadata:
    """Tests for DataContext.with_metadata()."""

    def test_with_metadata_adds_key(self, classification_data):
        """Verify with_metadata adds a key-value pair."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        new_ctx = ctx.with_metadata("key", "value")

        assert new_ctx.metadata["key"] == "value"

    def test_with_metadata_preserves_existing(self, classification_data):
        """Verify with_metadata preserves existing metadata."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y, metadata={"existing": "data"})

        new_ctx = ctx.with_metadata("key", "value")

        assert new_ctx.metadata["existing"] == "data"
        assert new_ctx.metadata["key"] == "value"

    def test_with_metadata_does_not_modify_original(self, classification_data):
        """Verify original metadata is unchanged."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        new_ctx = ctx.with_metadata("key", "value")

        assert "key" not in ctx.metadata
        assert "key" in new_ctx.metadata
