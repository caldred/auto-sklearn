"""Tests for DataView (formerly DataContext)."""

import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError

from sklearn_meta.data.view import DataView


class TestDataViewImmutability:
    """Tests for DataView immutability."""

    def test_dataview_is_frozen(self, classification_data):
        """Verify DataView is frozen and cannot be mutated."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        with pytest.raises(FrozenInstanceError):
            view.dataset = None

    def test_dataview_select_methods_return_new_instance(self, classification_data):
        """Verify select/bind methods return new instances."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        new_view = view.with_aux("key", np.zeros(len(X)))

        assert new_view is not view
        assert "key" not in view.aux
        assert "key" in new_view.aux


class TestDataViewFromXy:
    """Tests for DataView.from_Xy() factory."""

    def test_from_xy_sets_feature_cols(self, classification_data):
        """Verify from_Xy correctly sets feature_cols from X columns."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        assert view.feature_cols == tuple(X.columns)

    def test_from_xy_sets_target(self, classification_data):
        """Verify from_Xy correctly sets the target."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        assert view.target is not None
        batch = view.materialize()
        np.testing.assert_array_equal(batch.y, y.values)

    def test_from_xy_sets_groups(self, grouped_data):
        """Verify from_Xy correctly sets groups."""
        X, y, groups = grouped_data
        view = DataView.from_Xy(X, y, groups=groups)

        assert view.groups is not None
        resolved_groups = view.resolve_channel(view.groups)
        np.testing.assert_array_equal(resolved_groups, groups.values)

    def test_from_xy_df_contains_all_feature_columns(self, grouped_data):
        """Verify feature columns are tracked correctly."""
        X, y, groups = grouped_data
        view = DataView.from_Xy(X, y, groups=groups)

        # feature_cols should match X columns
        assert len(view.feature_cols) == len(X.columns)

    def test_from_xy_no_target(self, classification_data):
        """Verify from_Xy works without y."""
        X, _ = classification_data
        view = DataView.from_Xy(X)

        assert view.target is None
        batch = view.materialize()
        assert batch.y is None


class TestDataViewSelectRows:
    """Tests for DataView.select_rows() (formerly DataContext.with_indices())."""

    def test_select_rows_returns_correct_subset(self, classification_data):
        """Verify select_rows returns the correct data subset."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        indices = np.array([0, 5, 10, 15, 20])
        subset_view = view.select_rows(indices)

        assert subset_view.n_rows == len(indices)
        assert subset_view.n_features == view.n_features
        np.testing.assert_array_equal(subset_view.row_sel, indices)

    def test_select_rows_preserves_feature_values(self, classification_data):
        """Verify subset X values match original at indices."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        indices = np.array([0, 5, 10])
        subset_view = view.select_rows(indices)

        subset_batch = subset_view.materialize()
        for i, idx in enumerate(indices):
            np.testing.assert_array_almost_equal(
                subset_batch.X.iloc[i].values,
                X.iloc[idx].values,
            )

    def test_select_rows_preserves_target_values(self, classification_data):
        """Verify subset y values match original at indices."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        indices = np.array([0, 5, 10])
        subset_view = view.select_rows(indices)

        subset_batch = subset_view.materialize()
        for i, idx in enumerate(indices):
            assert subset_batch.y[i] == y.iloc[idx]

    def test_select_rows_subsets_groups(self, grouped_data):
        """Verify groups are also subset correctly."""
        X, y, groups = grouped_data
        view = DataView.from_Xy(X, y, groups=groups)

        indices = np.array([0, 10, 20])  # Different groups
        subset_view = view.select_rows(indices)

        subset_groups = subset_view.resolve_channel(subset_view.groups)
        for i, idx in enumerate(indices):
            assert subset_groups[i] == groups.iloc[idx]

    def test_select_rows_subsets_aux(self, classification_data):
        """Verify aux channels are subset correctly."""
        X, y = classification_data
        base_margin = np.random.randn(len(X))
        view = DataView.from_Xy(X, y, base_margin=base_margin)

        indices = np.array([0, 5, 10])
        subset_view = view.select_rows(indices)

        subset_batch = subset_view.materialize()
        np.testing.assert_array_almost_equal(
            subset_batch.aux["base_margin"],
            base_margin[indices],
        )


class TestDataViewSelectFeatures:
    """Tests for DataView.select_features() (formerly DataContext.with_feature_cols())."""

    def test_select_features_narrows_features(self, classification_data):
        """Verify select_features narrows the feature set."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)
        subset = list(X.columns[:5])

        new_view = view.select_features(subset)

        assert new_view.n_features == 5
        assert list(new_view.feature_cols) == subset


class TestDataViewOverlays:
    """Tests for DataView overlay operations (formerly DataContext.with_columns())."""

    def test_with_overlay_adds_data(self, classification_data):
        """Verify with_overlay adds overlay data."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        new_col = np.random.randn(len(X))
        new_view = view.with_overlay("extra", new_col)

        assert "extra" in new_view.overlays
        # Overlay columns appear in materialized X
        batch = new_view.materialize()
        assert "extra" in batch.X.columns

    def test_with_overlay_preserves_original(self, classification_data):
        """Verify original view is unchanged."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        view.with_overlay("extra", np.zeros(len(X)))

        assert "extra" not in view.overlays


class TestDataViewAux:
    """Tests for DataView aux channel handling (formerly base_margin / soft_targets)."""

    def test_with_aux_sets_channel(self, classification_data):
        """Verify with_aux sets an auxiliary channel."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        margin = np.random.randn(len(X))
        new_view = view.with_aux("base_margin", margin)

        resolved = new_view.resolve_channel(new_view.aux["base_margin"])
        np.testing.assert_array_almost_equal(resolved, margin)

    def test_with_aux_soft_targets(self, classification_data):
        """Verify soft targets can be stored as aux."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)
        st = np.random.rand(len(X))

        new_view = view.with_aux("soft_targets", st)

        resolved = new_view.resolve_channel(new_view.aux["soft_targets"])
        np.testing.assert_array_equal(resolved, st)
        assert "soft_targets" not in view.aux  # original unchanged

    def test_select_rows_slices_aux(self, classification_data):
        """Verify select_rows correctly slices aux channels."""
        X, y = classification_data
        st = np.random.rand(len(X))
        view = DataView.from_Xy(X, y).with_aux("soft_targets", st)

        indices = np.array([0, 5, 10])
        subset_view = view.select_rows(indices)

        subset_batch = subset_view.materialize()
        np.testing.assert_array_almost_equal(
            subset_batch.aux["soft_targets"],
            st[indices],
        )

class TestDataViewBindTarget:
    """Tests for DataView.bind_target() (formerly DataContext.with_y())."""

    def test_bind_target_replaces_target(self, classification_data):
        """Verify bind_target replaces the target."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        new_y = np.random.randn(len(X))
        new_view = view.bind_target(new_y)

        batch = new_view.materialize()
        np.testing.assert_array_almost_equal(batch.y, new_y)

    def test_bind_target_preserves_x(self, classification_data):
        """Verify bind_target preserves features."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        new_y = np.random.randn(len(X))
        new_view = view.bind_target(new_y)

        batch = new_view.materialize()
        pd.testing.assert_frame_equal(batch.X, X)


class TestDataViewValidation:
    """Tests for DataView validation."""

    def test_overlay_length_mismatch_raises(self, classification_data):
        """Verify overlay length mismatch raises error."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)
        wrong_overlay = np.random.randn(len(X) + 10)

        with pytest.raises(ValueError, match="must match"):
            view.with_overlay("bad", wrong_overlay)


class TestDataViewWithOverlayPredictions:
    """Tests for DataView.with_overlays() (formerly DataContext.augment_with_predictions())."""

    def test_overlays_add_prediction_columns(self, classification_data):
        """Verify predictions are added as overlay columns."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        predictions = {"model_1": np.random.randn(len(X))}
        augmented = view.with_overlays(predictions)

        batch = augmented.materialize()
        assert "model_1" in batch.X.columns

    def test_overlays_with_multiclass_probabilities(self, classification_data):
        """Verify multi-class probabilities are expanded."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        # 3-class probabilities
        proba = np.random.rand(len(X), 3)
        predictions = {"model_1": proba}
        augmented = view.with_overlays(predictions)

        batch = augmented.materialize()
        assert "model_1_0" in batch.X.columns
        assert "model_1_1" in batch.X.columns
        assert "model_1_2" in batch.X.columns

    def test_overlays_preserve_original_features(self, classification_data):
        """Verify original features are preserved."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)

        predictions = {"model_1": np.random.randn(len(X))}
        augmented = view.with_overlays(predictions)

        batch = augmented.materialize()
        for col in X.columns:
            assert col in batch.X.columns
            np.testing.assert_array_almost_equal(
                batch.X[col].values,
                X[col].values,
            )


class TestDataViewBindGroups:
    """Tests for DataView.bind_groups()."""

    def test_bind_groups_sets_groups(self, classification_data):
        """Verify bind_groups sets groups."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)
        groups = np.repeat(np.arange(100), 10)

        new_view = view.bind_groups(groups)

        assert new_view.groups is not None
        resolved = new_view.resolve_channel(new_view.groups)
        np.testing.assert_array_equal(resolved, groups)

    def test_bind_groups_does_not_modify_original(self, classification_data):
        """Verify original view is unchanged."""
        X, y = classification_data
        view = DataView.from_Xy(X, y)
        groups = np.repeat(np.arange(100), 10)

        new_view = view.bind_groups(groups)

        assert view.groups is None
        assert new_view.groups is not None
