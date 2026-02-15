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
        ctx = DataContext.from_Xy(X, y)

        with pytest.raises(FrozenInstanceError):
            ctx.df = pd.DataFrame()

    def test_datacontext_with_methods_return_new_instance(self, classification_data):
        """Verify with_* methods return new instances."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        new_ctx = ctx.with_metadata("key", "value")

        assert new_ctx is not ctx
        assert ctx.metadata == {}
        assert new_ctx.metadata == {"key": "value"}


class TestDataContextFromXy:
    """Tests for DataContext.from_Xy() factory."""

    def test_from_xy_sets_feature_cols(self, classification_data):
        """Verify from_Xy correctly sets feature_cols from X columns."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        assert ctx.feature_cols == tuple(X.columns)

    def test_from_xy_sets_target(self, classification_data):
        """Verify from_Xy correctly sets the target."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        assert ctx.target_col is not None
        pd.testing.assert_series_equal(ctx.y, y, check_names=False)

    def test_from_xy_sets_groups(self, grouped_data):
        """Verify from_Xy correctly sets groups."""
        X, y, groups = grouped_data
        ctx = DataContext.from_Xy(X, y, groups=groups)

        assert ctx.group_col is not None
        pd.testing.assert_series_equal(ctx.groups, groups, check_names=False)

    def test_from_xy_df_contains_all_columns(self, grouped_data):
        """Verify df contains features, target, and groups."""
        X, y, groups = grouped_data
        ctx = DataContext.from_Xy(X, y, groups=groups)

        # df should have feature cols + target + groups
        assert len(ctx.df.columns) == len(X.columns) + 2

    def test_from_xy_no_target(self, classification_data):
        """Verify from_Xy works without y."""
        X, _ = classification_data
        ctx = DataContext.from_Xy(X)

        assert ctx.target_col is None
        assert ctx.y is None


class TestDataContextWithSubset:
    """Tests for DataContext.with_indices()."""

    def test_with_indices_returns_correct_subset(self, classification_data):
        """Verify with_indices returns the correct data subset."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        indices = np.array([0, 5, 10, 15, 20])
        subset_ctx = ctx.with_indices(indices)

        assert subset_ctx.n_samples == len(indices)
        assert subset_ctx.n_features == ctx.n_features
        np.testing.assert_array_equal(subset_ctx.indices, indices)

    def test_with_indices_preserves_feature_values(self, classification_data):
        """Verify subset X values match original at indices."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

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
        ctx = DataContext.from_Xy(X, y)

        indices = np.array([0, 5, 10])
        subset_ctx = ctx.with_indices(indices)

        for i, idx in enumerate(indices):
            assert subset_ctx.y.iloc[i] == y.iloc[idx]

    def test_with_indices_subsets_groups(self, grouped_data):
        """Verify groups are also subset correctly."""
        X, y, groups = grouped_data
        ctx = DataContext.from_Xy(X, y, groups=groups)

        indices = np.array([0, 10, 20])  # Different groups
        subset_ctx = ctx.with_indices(indices)

        for i, idx in enumerate(indices):
            assert subset_ctx.groups.iloc[i] == groups.iloc[idx]

    def test_with_indices_subsets_base_margin(self, classification_data):
        """Verify base margin is subset correctly."""
        X, y = classification_data
        base_margin = np.random.randn(len(X))
        ctx = DataContext.from_Xy(X, y, base_margin=base_margin)

        indices = np.array([0, 5, 10])
        subset_ctx = ctx.with_indices(indices)

        np.testing.assert_array_almost_equal(
            subset_ctx.base_margin,
            base_margin[indices],
        )


class TestDataContextWithColumns:
    """Tests for DataContext.with_columns()."""

    def test_with_columns_adds_non_feature_column(self, classification_data):
        """Verify with_columns adds a column without extending features."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        new_col = np.random.randn(len(X))
        new_ctx = ctx.with_columns(extra=new_col)

        assert "extra" in new_ctx.df.columns
        assert "extra" not in new_ctx.feature_cols
        assert new_ctx.n_features == ctx.n_features

    def test_with_columns_adds_feature_column(self, classification_data):
        """Verify with_columns(as_features=True) extends feature_cols."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        new_col = np.random.randn(len(X))
        new_ctx = ctx.with_columns(as_features=True, extra=new_col)

        assert "extra" in new_ctx.df.columns
        assert "extra" in new_ctx.feature_cols
        assert new_ctx.n_features == ctx.n_features + 1

    def test_with_columns_preserves_original(self, classification_data):
        """Verify original context is unchanged."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        new_ctx = ctx.with_columns(as_features=True, extra=np.zeros(len(X)))

        assert "extra" not in ctx.df.columns
        assert ctx.n_features == len(X.columns)


class TestDataContextWithFeatureCols:
    """Tests for DataContext.with_feature_cols()."""

    def test_with_feature_cols_narrows_features(self, classification_data):
        """Verify with_feature_cols narrows the feature set."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)
        subset = list(X.columns[:5])

        new_ctx = ctx.with_feature_cols(subset)

        assert new_ctx.n_features == 5
        assert list(new_ctx.feature_cols) == subset
        # df still has all columns
        assert len(new_ctx.df.columns) == len(ctx.df.columns)


class TestDataContextBaseMargin:
    """Tests for DataContext base margin handling."""

    def test_with_base_margin_sets_margin(self, classification_data):
        """Verify with_base_margin sets the base margin."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        margin = np.random.randn(len(X))
        new_ctx = ctx.with_base_margin(margin)

        np.testing.assert_array_almost_equal(new_ctx.base_margin, margin)

    def test_base_margin_shape_matches(self, classification_data):
        """Verify base margin must match df length."""
        X, y = classification_data
        wrong_margin = np.random.randn(len(X) + 10)

        with pytest.raises(ValueError, match="same length"):
            DataContext.from_Xy(X, y, base_margin=wrong_margin)

    def test_base_margin_preserved_in_copy(self, classification_data):
        """Verify base margin is preserved in copy."""
        X, y = classification_data
        margin = np.random.randn(len(X))
        ctx = DataContext.from_Xy(X, y, base_margin=margin)

        copy_ctx = ctx.copy()

        np.testing.assert_array_almost_equal(copy_ctx.base_margin, margin)


class TestDataContextProperties:
    """Tests for DataContext properties."""

    def test_feature_columns_match_dataframe(self, classification_data):
        """Verify feature_names matches DataFrame columns."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        assert ctx.feature_names == list(X.columns)

    def test_n_samples_correct(self, classification_data):
        """Verify n_samples property is correct."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        assert ctx.n_samples == len(X)

    def test_n_features_correct(self, classification_data):
        """Verify n_features property is correct."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        assert ctx.n_features == X.shape[1]


class TestDataContextValidation:
    """Tests for DataContext validation."""

    def test_x_y_length_mismatch_raises(self, classification_data):
        """Verify X and y length mismatch raises error."""
        X, y = classification_data
        y_short = y.iloc[:100]

        with pytest.raises(ValueError, match="same length"):
            DataContext.from_Xy(X, y_short)

    def test_x_groups_length_mismatch_raises(self, classification_data):
        """Verify X and groups length mismatch raises error."""
        X, y = classification_data
        groups = pd.Series(range(100))

        with pytest.raises(ValueError, match="same length"):
            DataContext.from_Xy(X, y, groups=groups)

    def test_missing_feature_col_raises(self, classification_data):
        """Verify referencing missing feature columns raises error."""
        X, y = classification_data
        df = X.copy()
        df["__target__"] = y.values

        with pytest.raises(ValueError, match="feature_cols not found"):
            DataContext(
                df=df,
                feature_cols=("nonexistent_col",),
                target_col="__target__",
            )


class TestDataContextAugmentWithPredictions:
    """Tests for DataContext.augment_with_predictions()."""

    def test_augment_adds_prediction_columns(self, classification_data):
        """Verify predictions are added as columns."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        predictions = {"model_1": np.random.randn(len(X))}
        augmented = ctx.augment_with_predictions(predictions)

        assert "pred_model_1" in augmented.X.columns
        assert augmented.n_features == ctx.n_features + 1

    def test_augment_with_multiclass_probabilities(self, classification_data):
        """Verify multi-class probabilities are expanded."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

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
        ctx = DataContext.from_Xy(X, y)

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
        ctx = DataContext.from_Xy(X, y)

        predictions = {"model_1": np.random.randn(len(X))}
        augmented = ctx.augment_with_predictions(predictions, prefix="oof_")

        assert "oof_model_1" in augmented.X.columns
        assert "pred_model_1" not in augmented.X.columns


class TestDataContextCopy:
    """Tests for DataContext.copy()."""

    def test_copy_creates_new_dataframe(self, classification_data):
        """Verify copy creates a new DataFrame."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        copy_ctx = ctx.copy()

        assert copy_ctx.df is not ctx.df
        pd.testing.assert_frame_equal(copy_ctx.X, ctx.X)

    def test_copy_creates_new_target(self, classification_data):
        """Verify copy creates independent target data."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        copy_ctx = ctx.copy()

        pd.testing.assert_series_equal(copy_ctx.y, ctx.y)

    def test_copy_creates_new_metadata_dict(self, classification_data):
        """Verify copy creates a new metadata dict."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y, metadata={"key": "value"})

        copy_ctx = ctx.copy()

        assert copy_ctx.metadata is not ctx.metadata
        assert copy_ctx.metadata == ctx.metadata


class TestDataContextWithY:
    """Tests for DataContext.with_y()."""

    def test_with_y_replaces_target(self, classification_data):
        """Verify with_y replaces the target Series."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        new_y = pd.Series(np.random.randn(len(X)))
        new_ctx = ctx.with_y(new_y)

        pd.testing.assert_series_equal(new_ctx.y, new_y, check_names=False)

    def test_with_y_preserves_x(self, classification_data):
        """Verify with_y preserves features."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        new_y = pd.Series(np.random.randn(len(X)))
        new_ctx = ctx.with_y(new_y)

        pd.testing.assert_frame_equal(new_ctx.X, X)


class TestDataContextSoftTargets:
    """Tests for DataContext soft_targets handling."""

    def test_soft_targets_default_none(self, classification_data):
        """Verify soft_targets defaults to None."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        assert ctx.soft_targets is None

    def test_with_soft_targets(self, classification_data):
        """Verify with_soft_targets sets soft targets."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)
        st = np.random.rand(len(X))

        new_ctx = ctx.with_soft_targets(st)

        np.testing.assert_array_equal(new_ctx.soft_targets, st)
        assert ctx.soft_targets is None  # original unchanged

    def test_soft_targets_length_validation(self, classification_data):
        """Verify soft_targets length must match df length."""
        X, y = classification_data
        wrong_st = np.random.rand(len(X) + 10)

        with pytest.raises(ValueError, match="same length"):
            DataContext(
                df=X.copy().assign(__target__=y.values),
                feature_cols=tuple(X.columns),
                target_col="__target__",
                soft_targets=wrong_st,
            )

    def test_with_indices_slices_soft_targets(self, classification_data):
        """Verify with_indices correctly slices soft_targets."""
        X, y = classification_data
        st = np.random.rand(len(X))
        ctx = DataContext.from_Xy(X, y)
        ctx = ctx.with_soft_targets(st)

        indices = np.array([0, 5, 10])
        subset_ctx = ctx.with_indices(indices)

        np.testing.assert_array_almost_equal(
            subset_ctx.soft_targets,
            st[indices],
        )

    def test_with_indices_none_soft_targets(self, classification_data):
        """Verify with_indices works when soft_targets is None."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        indices = np.array([0, 5, 10])
        subset_ctx = ctx.with_indices(indices)

        assert subset_ctx.soft_targets is None

    def test_soft_targets_preserved_in_copy(self, classification_data):
        """Verify soft_targets is preserved in copy."""
        X, y = classification_data
        st = np.random.rand(len(X))
        ctx = DataContext.from_Xy(X, y)
        ctx = ctx.with_soft_targets(st)

        copy_ctx = ctx.copy()

        np.testing.assert_array_almost_equal(copy_ctx.soft_targets, st)

    def test_soft_targets_preserved_in_with_feature_cols(self, classification_data):
        """Verify soft_targets propagated through with_feature_cols."""
        X, y = classification_data
        st = np.random.rand(len(X))
        ctx = DataContext.from_Xy(X, y).with_soft_targets(st)

        new_ctx = ctx.with_feature_cols(list(X.columns[:5]))

        np.testing.assert_array_equal(new_ctx.soft_targets, st)

    def test_soft_targets_preserved_in_with_columns(self, classification_data):
        """Verify soft_targets propagated through with_columns."""
        X, y = classification_data
        st = np.random.rand(len(X))
        ctx = DataContext.from_Xy(X, y).with_soft_targets(st)

        new_ctx = ctx.with_columns(extra=np.zeros(len(X)))

        np.testing.assert_array_equal(new_ctx.soft_targets, st)

    def test_soft_targets_preserved_in_with_base_margin(self, classification_data):
        """Verify soft_targets propagated through with_base_margin."""
        X, y = classification_data
        st = np.random.rand(len(X))
        ctx = DataContext.from_Xy(X, y).with_soft_targets(st)

        new_ctx = ctx.with_base_margin(np.zeros(len(X)))

        np.testing.assert_array_equal(new_ctx.soft_targets, st)

    def test_soft_targets_preserved_in_with_metadata(self, classification_data):
        """Verify soft_targets propagated through with_metadata."""
        X, y = classification_data
        st = np.random.rand(len(X))
        ctx = DataContext.from_Xy(X, y).with_soft_targets(st)

        new_ctx = ctx.with_metadata("key", "value")

        np.testing.assert_array_equal(new_ctx.soft_targets, st)

    def test_soft_targets_preserved_in_augment_with_predictions(self, classification_data):
        """Verify soft_targets propagated through augment_with_predictions."""
        X, y = classification_data
        st = np.random.rand(len(X))
        ctx = DataContext.from_Xy(X, y).with_soft_targets(st)

        predictions = {"model_1": np.random.randn(len(X))}
        new_ctx = ctx.augment_with_predictions(predictions)

        np.testing.assert_array_equal(new_ctx.soft_targets, st)


    def test_soft_targets_preserved_in_with_y(self, classification_data):
        """Verify soft_targets propagated through with_y."""
        X, y = classification_data
        st = np.random.rand(len(X))
        ctx = DataContext.from_Xy(X, y).with_soft_targets(st)

        new_y = pd.Series(np.random.randn(len(X)))
        new_ctx = ctx.with_y(new_y)

        np.testing.assert_array_equal(new_ctx.soft_targets, st)

    def test_soft_targets_preserved_in_with_target_col(self, classification_data):
        """Verify soft_targets propagated through with_target_col."""
        X, y = classification_data
        st = np.random.rand(len(X))
        ctx = DataContext.from_Xy(X, y).with_soft_targets(st)
        ctx = ctx.with_columns(alt_target=np.zeros(len(X)))

        new_ctx = ctx.with_target_col("alt_target")

        np.testing.assert_array_equal(new_ctx.soft_targets, st)


class TestDataContextWithTargetCol:
    """Tests for DataContext.with_target_col()."""

    def test_with_target_col_switches_target(self, classification_data):
        """Verify with_target_col can switch to a different column."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)
        ctx = ctx.with_columns(alt_target=np.zeros(len(X)))

        new_ctx = ctx.with_target_col("alt_target")

        assert new_ctx.target_col == "alt_target"
        np.testing.assert_array_equal(new_ctx.y.values, np.zeros(len(X)))


class TestDataContextWithMetadata:
    """Tests for DataContext.with_metadata()."""

    def test_with_metadata_adds_key(self, classification_data):
        """Verify with_metadata adds a key-value pair."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        new_ctx = ctx.with_metadata("key", "value")

        assert new_ctx.metadata["key"] == "value"

    def test_with_metadata_preserves_existing(self, classification_data):
        """Verify with_metadata preserves existing metadata."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y, metadata={"existing": "data"})

        new_ctx = ctx.with_metadata("key", "value")

        assert new_ctx.metadata["existing"] == "data"
        assert new_ctx.metadata["key"] == "value"

    def test_with_metadata_does_not_modify_original(self, classification_data):
        """Verify original metadata is unchanged."""
        X, y = classification_data
        ctx = DataContext.from_Xy(X, y)

        new_ctx = ctx.with_metadata("key", "value")

        assert "key" not in ctx.metadata
        assert "key" in new_ctx.metadata
