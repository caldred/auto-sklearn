"""Tests for CVEngine (formerly DataManager)."""

import numpy as np
import pytest

from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import CVConfig, CVStrategy, FoldResult
from sklearn_meta.engine.cv import CVEngine


class TestCVEngineCreateFolds:
    """Tests for CVEngine.create_folds()."""

    def test_create_folds_count(self, data_context, cv_config_stratified):
        """Verify correct number of folds created."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        assert len(folds) == cv_config_stratified.n_splits

    def test_create_folds_repeated_cv_count(self, data_context, cv_config_repeated):
        """Verify correct number of folds for repeated CV."""
        cv_engine = CVEngine(cv_config_repeated)
        folds = cv_engine.create_folds(data_context)

        expected_count = cv_config_repeated.n_splits * cv_config_repeated.n_repeats
        assert len(folds) == expected_count

    def test_create_folds_train_val_disjoint(self, data_context, cv_config_stratified):
        """Verify train and val indices are disjoint for each fold."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        for fold in folds:
            intersection = set(fold.train_indices) & set(fold.val_indices)
            assert len(intersection) == 0, f"Fold {fold.fold_idx} has overlapping indices"

    def test_create_folds_complete_coverage(self, data_context, cv_config_stratified):
        """Verify all samples appear in exactly one validation set per repeat."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        all_val_indices = np.concatenate([f.val_indices for f in folds])
        all_val_indices = np.sort(all_val_indices)

        expected = np.arange(data_context.n_rows)
        np.testing.assert_array_equal(all_val_indices, expected)

    def test_create_folds_repeated_cv_complete_coverage(self, data_context, cv_config_repeated):
        """Verify complete coverage for each repeat in repeated CV."""
        cv_engine = CVEngine(cv_config_repeated)
        folds = cv_engine.create_folds(data_context)

        n_repeats = cv_config_repeated.n_repeats

        for repeat in range(n_repeats):
            repeat_folds = [f for f in folds if f.repeat_idx == repeat]
            all_val_indices = np.concatenate([f.val_indices for f in repeat_folds])
            all_val_indices = np.sort(all_val_indices)

            expected = np.arange(data_context.n_rows)
            np.testing.assert_array_equal(
                all_val_indices, expected,
                err_msg=f"Repeat {repeat} doesn't have complete coverage"
            )

    def test_create_folds_without_y_works_for_random(self, classification_data):
        """Non-stratified strategies work without a target."""
        X, _ = classification_data
        data_view = DataView.from_Xy(X)
        cv_engine = CVEngine(CVConfig(strategy=CVStrategy.RANDOM))
        folds = cv_engine.create_folds(data_view)
        assert len(folds) == 5  # default n_splits

    def test_create_folds_without_y_raises_for_stratified(self, classification_data):
        """Stratified strategy requires a target."""
        X, _ = classification_data
        data_view = DataView.from_Xy(X)
        cv_engine = CVEngine(CVConfig(strategy=CVStrategy.STRATIFIED))

        with pytest.raises(ValueError, match="without target"):
            cv_engine.create_folds(data_view)

    def test_create_folds_fold_indices_correct(self, data_context, cv_config_stratified):
        """Verify fold indices are assigned correctly."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        for i, fold in enumerate(folds):
            assert fold.fold_idx == i

    def test_create_folds_repeat_indices_correct(self, data_context, cv_config_repeated):
        """Verify repeat indices are assigned correctly."""
        cv_engine = CVEngine(cv_config_repeated)
        folds = cv_engine.create_folds(data_context)

        n_splits = cv_config_repeated.n_splits
        for i, fold in enumerate(folds):
            expected_repeat = i // n_splits
            assert fold.repeat_idx == expected_repeat


class TestCVEngineStratifiedCV:
    """Tests for stratified cross-validation."""

    def test_stratified_preserves_class_ratio(self, data_context, cv_config_stratified):
        """Verify stratified CV preserves class ratios."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        # Get overall class ratio
        y = data_context.materialize().y
        overall_ratio = (y == 1).mean()

        for fold in folds:
            train_y = y[fold.train_indices]
            val_y = y[fold.val_indices]

            train_ratio = (train_y == 1).mean()
            val_ratio = (val_y == 1).mean()

            # Allow 10% tolerance
            assert abs(train_ratio - overall_ratio) < 0.1, f"Train ratio {train_ratio} differs from overall {overall_ratio}"
            assert abs(val_ratio - overall_ratio) < 0.1, f"Val ratio {val_ratio} differs from overall {overall_ratio}"


class TestCVEngineGroupCV:
    """Tests for group cross-validation."""

    def test_group_cv_no_group_leak(self, data_context_with_groups, cv_config_group):
        """Verify groups don't span train and validation."""
        cv_engine = CVEngine(cv_config_group)
        folds = cv_engine.create_folds(data_context_with_groups)

        groups = data_context_with_groups.resolve_channel(data_context_with_groups.groups)

        for fold in folds:
            train_groups = set(groups[fold.train_indices])
            val_groups = set(groups[fold.val_indices])

            intersection = train_groups & val_groups
            assert len(intersection) == 0, f"Fold {fold.fold_idx} has group leak: {intersection}"

    def test_group_cv_falls_back_without_groups(self, data_context, cv_config_group, caplog):
        """Verify group CV falls back to KFOLD without groups."""
        import logging

        cv_engine = CVEngine(cv_config_group)

        with caplog.at_level(logging.WARNING):
            folds = cv_engine.create_folds(data_context)

        # Should have logged a warning about fallback
        assert "Falling back to RANDOM" in caplog.text
        # Should still create valid folds
        assert len(folds) == cv_config_group.n_splits

    def test_stratified_group_cv_no_group_leak(self, classification_data):
        """Verify stratified-group CV never leaks a group across splits."""
        X, y = classification_data
        groups = np.repeat(np.arange(len(X) // 5), 5)
        data_view = DataView.from_Xy(X, y=y, groups=groups)
        cv_engine = CVEngine(
            CVConfig(
                n_splits=3,
                strategy=CVStrategy.STRATIFIED_GROUP,
                shuffle=True,
                random_state=42,
            )
        )

        folds = cv_engine.create_folds(data_view)

        resolved_groups = data_view.resolve_channel(data_view.groups)
        for fold in folds:
            train_groups = set(resolved_groups[fold.train_indices])
            val_groups = set(resolved_groups[fold.val_indices])
            assert train_groups.isdisjoint(val_groups)

    def test_stratified_group_cv_requires_groups(self, data_context):
        """Verify stratified-group CV fails loudly without groups."""
        cv_engine = CVEngine(CVConfig(strategy=CVStrategy.STRATIFIED_GROUP))

        with pytest.raises(ValueError, match="STRATIFIED_GROUP"):
            cv_engine.create_folds(data_context)

    def test_stratified_group_cv_produces_valid_folds(self, classification_data):
        """Verify stratified-group CV produces the requested number of folds."""
        X, y = classification_data
        groups = np.repeat(np.arange(len(X) // 5), 5)
        data_view = DataView.from_Xy(X, y=y, groups=groups)
        cv_engine = CVEngine(
            CVConfig(
                n_splits=3,
                strategy=CVStrategy.STRATIFIED_GROUP,
                shuffle=True,
                random_state=42,
            )
        )

        folds = cv_engine.create_folds(data_view)

        assert len(folds) == 3
        for fold in folds:
            val_y = y[fold.val_indices]
            assert len(np.unique(val_y)) == 2


class TestCVEngineRepeatedCV:
    """Tests for repeated cross-validation."""

    def test_repeated_cv_different_folds(self, data_context, cv_config_repeated):
        """Verify different repeats have different fold assignments."""
        cv_engine = CVEngine(cv_config_repeated)
        folds = cv_engine.create_folds(data_context)

        n_splits = cv_config_repeated.n_splits

        # The validation indices should be different
        # (with high probability for shuffled CV)
        if cv_config_repeated.shuffle:
            repeat_0_fold_0 = folds[0]
            repeat_1_fold_0 = folds[n_splits]
            assert not np.array_equal(
                repeat_0_fold_0.val_indices,
                repeat_1_fold_0.val_indices,
            )


class TestCVEngineSplitForFold:
    """Tests for CVEngine.split_for_fold() (formerly DataManager.align_to_fold())."""

    def test_split_for_fold_returns_train_val(self, data_context, cv_config_stratified):
        """Verify split_for_fold returns train and val views."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        train_view, val_view = cv_engine.split_for_fold(data_context, folds[0])

        assert train_view.n_rows == folds[0].n_train
        assert val_view.n_rows == folds[0].n_val

    def test_split_for_fold_train_features_correct(self, data_context, cv_config_stratified):
        """Verify train view has correct features."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        train_view, _ = cv_engine.split_for_fold(data_context, folds[0])

        # Check first sample
        first_train_idx = folds[0].train_indices[0]
        np.testing.assert_array_almost_equal(
            train_view.materialize().X.iloc[0].values,
            data_context.materialize().X.iloc[first_train_idx].values,
        )

    def test_split_for_fold_val_features_correct(self, data_context, cv_config_stratified):
        """Verify val view has correct features."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        _, val_view = cv_engine.split_for_fold(data_context, folds[0])

        # Check first sample
        first_val_idx = folds[0].val_indices[0]
        np.testing.assert_array_almost_equal(
            val_view.materialize().X.iloc[0].values,
            data_context.materialize().X.iloc[first_val_idx].values,
        )

    def test_split_for_fold_preserves_groups(self, data_context_with_groups, cv_config_group):
        """Verify groups are aligned correctly."""
        cv_engine = CVEngine(cv_config_group)
        folds = cv_engine.create_folds(data_context_with_groups)

        train_view, val_view = cv_engine.split_for_fold(data_context_with_groups, folds[0])

        # Check groups are subsets of original
        assert train_view.groups is not None
        assert val_view.groups is not None

        original_groups = data_context_with_groups.resolve_channel(data_context_with_groups.groups)
        train_groups = train_view.resolve_channel(train_view.groups)
        first_train_idx = folds[0].train_indices[0]
        assert train_groups[0] == original_groups[first_train_idx]


class TestCVEngineRouteOOFPredictions:
    """Tests for CVEngine.route_oof_predictions()."""

    def test_route_oof_predictions_shape(self, data_context, cv_config_stratified):
        """Verify OOF predictions have correct shape."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

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

        oof = cv_engine.route_oof_predictions(data_context, fold_results)

        assert oof.shape == (data_context.n_rows,)

    def test_route_oof_predictions_values_match(self, data_context, cv_config_stratified):
        """Verify OOF values match fold predictions."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

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

        oof = cv_engine.route_oof_predictions(data_context, fold_results)

        # Check each fold's predictions are in the right place
        for fold, result in zip(folds, fold_results):
            np.testing.assert_array_equal(
                oof[fold.val_indices],
                result.val_predictions,
            )

    def test_route_oof_no_overlap(self, data_context, cv_config_stratified):
        """Verify each index is filled exactly once."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        # Track which indices are filled
        filled = np.zeros(data_context.n_rows, dtype=int)

        for fold in folds:
            for idx in fold.val_indices:
                filled[idx] += 1

        # Each index should be filled exactly once
        np.testing.assert_array_equal(filled, np.ones(data_context.n_rows))

    def test_route_oof_multiclass_predictions(self, data_context, cv_config_stratified):
        """Verify OOF routing works for multi-class probabilities."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

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

        oof = cv_engine.route_oof_predictions(data_context, fold_results)

        assert oof.shape == (data_context.n_rows, n_classes)


class TestCVEngineAggregateCVResult:
    """Tests for CVEngine.aggregate_cv_result()."""

    def test_aggregate_returns_cv_result(self, data_context, cv_config_stratified):
        """Verify aggregate returns CVResult with correct node name."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        fold_results = [
            FoldResult(
                fold=fold,
                model=None,
                val_predictions=np.random.randn(fold.n_val),
                val_score=0.8,
            )
            for fold in folds
        ]

        result = cv_engine.aggregate_cv_result("test_node", fold_results, data_context)

        assert result.node_name == "test_node"
        assert result.n_folds == len(folds)

    def test_aggregate_includes_oof(self, data_context, cv_config_stratified):
        """Verify aggregated result includes OOF predictions."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        fold_results = [
            FoldResult(
                fold=fold,
                model=None,
                val_predictions=np.random.randn(fold.n_val),
                val_score=0.8,
            )
            for fold in folds
        ]

        result = cv_engine.aggregate_cv_result("test_node", fold_results, data_context)

        assert result.oof_predictions.shape == (data_context.n_rows,)


class TestCVEngineReproducibility:
    """Tests for reproducibility with random state."""

    def test_cv_reproducibility_same_seed(self, data_context, cv_config_stratified):
        """Verify same seed produces same folds."""
        cv_engine = CVEngine(cv_config_stratified)
        folds_1 = cv_engine.create_folds(data_context)

        cv_engine2 = CVEngine(cv_config_stratified)
        folds_2 = cv_engine2.create_folds(data_context)

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

        cv_engine_1 = CVEngine(config_1)
        cv_engine_2 = CVEngine(config_2)

        folds_1 = cv_engine_1.create_folds(data_context)
        folds_2 = cv_engine_2.create_folds(data_context)

        # At least one fold should be different
        different = False
        for f1, f2 in zip(folds_1, folds_2):
            if not np.array_equal(f1.val_indices, f2.val_indices):
                different = True
                break

        assert different, "Different seeds should produce different folds"


class TestCVEngineTimeSeries:
    """Tests for time series cross-validation."""

    def test_time_series_cv_temporal_order(self, data_context):
        """Verify time series CV maintains temporal ordering."""
        config = CVConfig(n_splits=5, strategy=CVStrategy.TIME_SERIES)
        cv_engine = CVEngine(config)
        folds = cv_engine.create_folds(data_context)

        for fold in folds:
            # All train indices should be less than all val indices
            max_train = max(fold.train_indices)
            min_val = min(fold.val_indices)
            assert max_train < min_val, "Train indices should come before val indices"

    def test_time_series_cv_expanding_window(self, data_context):
        """Verify time series CV uses expanding window."""
        config = CVConfig(n_splits=5, strategy=CVStrategy.TIME_SERIES)
        cv_engine = CVEngine(config)
        folds = cv_engine.create_folds(data_context)

        prev_train_size = 0
        for fold in folds:
            # Training set should grow
            assert fold.n_train > prev_train_size
            prev_train_size = fold.n_train

    def test_time_series_cv_repeated_not_supported(self, data_context):
        """Verify repeated time series CV raises error."""
        config = CVConfig(n_splits=5, n_repeats=2, strategy=CVStrategy.TIME_SERIES)
        cv_engine = CVEngine(config)

        with pytest.raises(ValueError, match="not supported"):
            cv_engine.create_folds(data_context)
