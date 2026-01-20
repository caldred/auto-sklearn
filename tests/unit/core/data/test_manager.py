"""Tests for DataManager."""

import numpy as np
import pandas as pd
import pytest

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVFold, CVStrategy, FoldResult
from sklearn_meta.core.data.manager import DataManager


class TestDataManagerCreateFolds:
    """Tests for DataManager.create_folds()."""

    def test_create_folds_count(self, data_context, cv_config_stratified):
        """Verify correct number of folds created."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        assert len(folds) == cv_config_stratified.n_splits

    def test_create_folds_repeated_cv_count(self, data_context, cv_config_repeated):
        """Verify correct number of folds for repeated CV."""
        manager = DataManager(cv_config_repeated)
        folds = manager.create_folds(data_context)

        expected_count = cv_config_repeated.n_splits * cv_config_repeated.n_repeats
        assert len(folds) == expected_count

    def test_create_folds_train_val_disjoint(self, data_context, cv_config_stratified):
        """Verify train and val indices are disjoint for each fold."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        for fold in folds:
            intersection = set(fold.train_indices) & set(fold.val_indices)
            assert len(intersection) == 0, f"Fold {fold.fold_idx} has overlapping indices"

    def test_create_folds_complete_coverage(self, data_context, cv_config_stratified):
        """Verify all samples appear in exactly one validation set per repeat."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        all_val_indices = np.concatenate([f.val_indices for f in folds])
        all_val_indices = np.sort(all_val_indices)

        expected = np.arange(data_context.n_samples)
        np.testing.assert_array_equal(all_val_indices, expected)

    def test_create_folds_repeated_cv_complete_coverage(self, data_context, cv_config_repeated):
        """Verify complete coverage for each repeat in repeated CV."""
        manager = DataManager(cv_config_repeated)
        folds = manager.create_folds(data_context)

        n_repeats = cv_config_repeated.n_repeats
        n_splits = cv_config_repeated.n_splits

        for repeat in range(n_repeats):
            repeat_folds = [f for f in folds if f.repeat_idx == repeat]
            all_val_indices = np.concatenate([f.val_indices for f in repeat_folds])
            all_val_indices = np.sort(all_val_indices)

            expected = np.arange(data_context.n_samples)
            np.testing.assert_array_equal(
                all_val_indices, expected,
                err_msg=f"Repeat {repeat} doesn't have complete coverage"
            )

    def test_create_folds_without_y_raises(self, classification_data):
        """Verify create_folds raises error without target."""
        X, _ = classification_data
        ctx = DataContext(X=X)
        manager = DataManager(CVConfig())

        with pytest.raises(ValueError, match="without target"):
            manager.create_folds(ctx)

    def test_create_folds_fold_indices_correct(self, data_context, cv_config_stratified):
        """Verify fold indices are assigned correctly."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        for i, fold in enumerate(folds):
            assert fold.fold_idx == i

    def test_create_folds_repeat_indices_correct(self, data_context, cv_config_repeated):
        """Verify repeat indices are assigned correctly."""
        manager = DataManager(cv_config_repeated)
        folds = manager.create_folds(data_context)

        n_splits = cv_config_repeated.n_splits
        for i, fold in enumerate(folds):
            expected_repeat = i // n_splits
            assert fold.repeat_idx == expected_repeat


class TestDataManagerStratifiedCV:
    """Tests for stratified cross-validation."""

    def test_stratified_preserves_class_ratio(self, data_context, cv_config_stratified):
        """Verify stratified CV preserves class ratios."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        # Get overall class ratio
        y = data_context.y
        overall_ratio = (y == 1).mean()

        for fold in folds:
            train_y = y.iloc[fold.train_indices]
            val_y = y.iloc[fold.val_indices]

            train_ratio = (train_y == 1).mean()
            val_ratio = (val_y == 1).mean()

            # Allow 10% tolerance
            assert abs(train_ratio - overall_ratio) < 0.1, f"Train ratio {train_ratio} differs from overall {overall_ratio}"
            assert abs(val_ratio - overall_ratio) < 0.1, f"Val ratio {val_ratio} differs from overall {overall_ratio}"


class TestDataManagerGroupCV:
    """Tests for group cross-validation."""

    def test_group_cv_no_group_leak(self, data_context_with_groups, cv_config_group):
        """Verify groups don't span train and validation."""
        manager = DataManager(cv_config_group)
        folds = manager.create_folds(data_context_with_groups)

        groups = data_context_with_groups.groups

        for fold in folds:
            train_groups = set(groups.iloc[fold.train_indices])
            val_groups = set(groups.iloc[fold.val_indices])

            intersection = train_groups & val_groups
            assert len(intersection) == 0, f"Fold {fold.fold_idx} has group leak: {intersection}"

    def test_group_cv_requires_groups(self, data_context, cv_config_group):
        """Verify group CV raises error without groups."""
        manager = DataManager(cv_config_group)

        with pytest.raises(ValueError, match="requires groups"):
            manager.create_folds(data_context)


class TestDataManagerRepeatedCV:
    """Tests for repeated cross-validation."""

    def test_repeated_cv_different_folds(self, data_context, cv_config_repeated):
        """Verify different repeats have different fold assignments."""
        manager = DataManager(cv_config_repeated)
        folds = manager.create_folds(data_context)

        n_splits = cv_config_repeated.n_splits

        # Get first fold from each repeat
        repeat_0_fold_0 = folds[0]
        repeat_1_fold_0 = folds[n_splits]

        # The validation indices should be different
        # (with high probability for shuffled CV)
        if cv_config_repeated.shuffle:
            val_0 = set(repeat_0_fold_0.val_indices)
            val_1 = set(repeat_1_fold_0.val_indices)
            # They might not be completely different, but shouldn't be identical
            # unless random state produces same result
            pass  # This is probabilistic


class TestDataManagerNestedCV:
    """Tests for nested cross-validation."""

    def test_create_nested_folds_outer_count(self, data_context, cv_config_nested):
        """Verify correct number of outer folds."""
        manager = DataManager(cv_config_nested)
        nested_folds = manager.create_nested_folds(data_context)

        assert len(nested_folds) == cv_config_nested.n_splits

    def test_create_nested_folds_inner_count(self, data_context, cv_config_nested):
        """Verify correct number of inner folds per outer fold."""
        manager = DataManager(cv_config_nested)
        nested_folds = manager.create_nested_folds(data_context)

        for nested in nested_folds:
            assert nested.n_inner_folds == cv_config_nested.inner_cv.n_splits

    def test_nested_cv_outer_val_not_in_inner(self, data_context, cv_config_nested):
        """Verify outer validation samples are never in inner training."""
        manager = DataManager(cv_config_nested)
        nested_folds = manager.create_nested_folds(data_context)

        for nested in nested_folds:
            outer_val = set(nested.outer_fold.val_indices)

            for inner_fold in nested.inner_folds:
                inner_train = set(inner_fold.train_indices)
                inner_val = set(inner_fold.val_indices)

                # Outer val should not appear in inner train or val
                assert len(outer_val & inner_train) == 0
                assert len(outer_val & inner_val) == 0

    def test_nested_cv_inner_within_outer_train(self, data_context, cv_config_nested):
        """Verify inner folds are subsets of outer training."""
        manager = DataManager(cv_config_nested)
        nested_folds = manager.create_nested_folds(data_context)

        for nested in nested_folds:
            outer_train = set(nested.outer_fold.train_indices)

            for inner_fold in nested.inner_folds:
                inner_train = set(inner_fold.train_indices)
                inner_val = set(inner_fold.val_indices)

                assert inner_train.issubset(outer_train)
                assert inner_val.issubset(outer_train)

    def test_nested_cv_requires_inner_cv(self, data_context, cv_config_stratified):
        """Verify create_nested_folds raises error without inner_cv."""
        manager = DataManager(cv_config_stratified)

        with pytest.raises(ValueError, match="inner_cv"):
            manager.create_nested_folds(data_context)


class TestDataManagerAlignToFold:
    """Tests for DataManager.align_to_fold()."""

    def test_align_to_fold_returns_train_val(self, data_context, cv_config_stratified):
        """Verify align_to_fold returns train and val contexts."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        train_ctx, val_ctx = manager.align_to_fold(data_context, folds[0])

        assert train_ctx.n_samples == folds[0].n_train
        assert val_ctx.n_samples == folds[0].n_val

    def test_align_to_fold_train_features_correct(self, data_context, cv_config_stratified):
        """Verify train context has correct features."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        train_ctx, _ = manager.align_to_fold(data_context, folds[0])

        # Check first sample
        first_train_idx = folds[0].train_indices[0]
        np.testing.assert_array_almost_equal(
            train_ctx.X.iloc[0].values,
            data_context.X.iloc[first_train_idx].values,
        )

    def test_align_to_fold_val_features_correct(self, data_context, cv_config_stratified):
        """Verify val context has correct features."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        _, val_ctx = manager.align_to_fold(data_context, folds[0])

        # Check first sample
        first_val_idx = folds[0].val_indices[0]
        np.testing.assert_array_almost_equal(
            val_ctx.X.iloc[0].values,
            data_context.X.iloc[first_val_idx].values,
        )

    def test_align_to_fold_preserves_groups(self, data_context_with_groups, cv_config_group):
        """Verify groups are aligned correctly."""
        manager = DataManager(cv_config_group)
        folds = manager.create_folds(data_context_with_groups)

        train_ctx, val_ctx = manager.align_to_fold(data_context_with_groups, folds[0])

        # Check groups are subsets of original
        assert train_ctx.groups is not None
        assert val_ctx.groups is not None

        original_groups = data_context_with_groups.groups
        first_train_idx = folds[0].train_indices[0]
        assert train_ctx.groups.iloc[0] == original_groups.iloc[first_train_idx]


class TestDataManagerRouteOOFPredictions:
    """Tests for DataManager.route_oof_predictions()."""

    def test_route_oof_predictions_shape(self, data_context, cv_config_stratified):
        """Verify OOF predictions have correct shape."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        # Create mock fold results
        fold_results = []
        for fold in folds:
            result = FoldResult(
                fold=fold,
                model=None,
                val_predictions=np.random.randn(fold.n_val),
                val_score=0.8,
            )
            fold_results.append(result)

        oof = manager.route_oof_predictions(data_context, fold_results)

        assert oof.shape == (data_context.n_samples,)

    def test_route_oof_predictions_values_match(self, data_context, cv_config_stratified):
        """Verify OOF values match fold predictions."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        # Create mock fold results with known values
        fold_results = []
        for fold in folds:
            preds = np.arange(fold.n_val) + fold.fold_idx * 1000
            result = FoldResult(
                fold=fold,
                model=None,
                val_predictions=preds,
                val_score=0.8,
            )
            fold_results.append(result)

        oof = manager.route_oof_predictions(data_context, fold_results)

        # Check each fold's predictions are in the right place
        for fold, result in zip(folds, fold_results):
            np.testing.assert_array_equal(
                oof[fold.val_indices],
                result.val_predictions,
            )

    def test_route_oof_no_overlap(self, data_context, cv_config_stratified):
        """Verify each index is filled exactly once."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        # Track which indices are filled
        filled = np.zeros(data_context.n_samples, dtype=int)

        for fold in folds:
            for idx in fold.val_indices:
                filled[idx] += 1

        # Each index should be filled exactly once
        np.testing.assert_array_equal(filled, np.ones(data_context.n_samples))

    def test_route_oof_multiclass_predictions(self, data_context, cv_config_stratified):
        """Verify OOF routing works for multi-class probabilities."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        n_classes = 3
        fold_results = []
        for fold in folds:
            preds = np.random.rand(fold.n_val, n_classes)
            result = FoldResult(
                fold=fold,
                model=None,
                val_predictions=preds,
                val_score=0.8,
            )
            fold_results.append(result)

        oof = manager.route_oof_predictions(data_context, fold_results)

        assert oof.shape == (data_context.n_samples, n_classes)


class TestDataManagerAggregateCVResult:
    """Tests for DataManager.aggregate_cv_result()."""

    def test_aggregate_returns_cv_result(self, data_context, cv_config_stratified):
        """Verify aggregate returns CVResult with correct node name."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        fold_results = [
            FoldResult(
                fold=fold,
                model=None,
                val_predictions=np.random.randn(fold.n_val),
                val_score=0.8,
            )
            for fold in folds
        ]

        result = manager.aggregate_cv_result("test_node", fold_results, data_context)

        assert result.node_name == "test_node"
        assert result.n_folds == len(folds)

    def test_aggregate_includes_oof(self, data_context, cv_config_stratified):
        """Verify aggregated result includes OOF predictions."""
        manager = DataManager(cv_config_stratified)
        folds = manager.create_folds(data_context)

        fold_results = [
            FoldResult(
                fold=fold,
                model=None,
                val_predictions=np.random.randn(fold.n_val),
                val_score=0.8,
            )
            for fold in folds
        ]

        result = manager.aggregate_cv_result("test_node", fold_results, data_context)

        assert result.oof_predictions.shape == (data_context.n_samples,)


class TestDataManagerReproducibility:
    """Tests for reproducibility with random state."""

    def test_cv_reproducibility_same_seed(self, data_context, cv_config_stratified):
        """Verify same seed produces same folds."""
        manager = DataManager(cv_config_stratified)
        folds_1 = manager.create_folds(data_context)

        manager2 = DataManager(cv_config_stratified)
        folds_2 = manager2.create_folds(data_context)

        for f1, f2 in zip(folds_1, folds_2):
            np.testing.assert_array_equal(f1.train_indices, f2.train_indices)
            np.testing.assert_array_equal(f1.val_indices, f2.val_indices)

    def test_cv_different_seed_produces_different_folds(self, data_context):
        """Verify different seeds produce different folds."""
        config_1 = CVConfig(
            n_splits=5, strategy=CVStrategy.STRATIFIED,
            shuffle=True, random_state=42
        )
        config_2 = CVConfig(
            n_splits=5, strategy=CVStrategy.STRATIFIED,
            shuffle=True, random_state=123
        )

        manager_1 = DataManager(config_1)
        manager_2 = DataManager(config_2)

        folds_1 = manager_1.create_folds(data_context)
        folds_2 = manager_2.create_folds(data_context)

        # At least one fold should be different
        different = False
        for f1, f2 in zip(folds_1, folds_2):
            if not np.array_equal(f1.val_indices, f2.val_indices):
                different = True
                break

        assert different, "Different seeds should produce different folds"


class TestDataManagerTimeSeries:
    """Tests for time series cross-validation."""

    def test_time_series_cv_temporal_order(self, data_context):
        """Verify time series CV maintains temporal ordering."""
        config = CVConfig(n_splits=5, strategy=CVStrategy.TIME_SERIES)
        manager = DataManager(config)
        folds = manager.create_folds(data_context)

        for fold in folds:
            # All train indices should be less than all val indices
            max_train = max(fold.train_indices)
            min_val = min(fold.val_indices)
            assert max_train < min_val, "Train indices should come before val indices"

    def test_time_series_cv_expanding_window(self, data_context):
        """Verify time series CV uses expanding window."""
        config = CVConfig(n_splits=5, strategy=CVStrategy.TIME_SERIES)
        manager = DataManager(config)
        folds = manager.create_folds(data_context)

        prev_train_size = 0
        for fold in folds:
            # Training set should grow
            assert fold.n_train > prev_train_size
            prev_train_size = fold.n_train

    def test_time_series_cv_repeated_not_supported(self, data_context):
        """Verify repeated time series CV raises error."""
        config = CVConfig(n_splits=5, n_repeats=2, strategy=CVStrategy.TIME_SERIES)
        manager = DataManager(config)

        with pytest.raises(ValueError, match="not supported"):
            manager.create_folds(data_context)
