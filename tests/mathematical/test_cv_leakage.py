"""Tests to verify no data leakage in cross-validation."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import CVConfig, CVStrategy, FoldResult
from sklearn_meta.engine.cv import CVEngine


class TestTrainValDisjoint:
    """Tests verifying train and validation sets are disjoint."""

    def test_train_val_indices_disjoint(self, data_context, cv_config_stratified):
        """Verify train and val indices have no overlap in any fold."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        for fold in folds:
            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)

            intersection = train_set & val_set
            assert len(intersection) == 0, (
                f"Fold {fold.fold_idx} has {len(intersection)} overlapping indices: {intersection}"
            )

    def test_train_val_union_covers_all(self, data_context, cv_config_stratified):
        """Verify train + val covers all samples in each fold."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        for fold in folds:
            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)
            all_indices = train_set | val_set

            expected = set(range(data_context.n_rows))
            assert all_indices == expected, (
                f"Fold {fold.fold_idx} doesn't cover all samples"
            )

    def test_each_sample_in_val_once_per_repeat(self, data_context, cv_config_stratified):
        """Verify each sample appears in validation exactly once per repeat."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        val_counts = np.zeros(data_context.n_rows)

        for fold in folds:
            for idx in fold.val_indices:
                val_counts[idx] += 1

        # Each sample should be in validation exactly once
        np.testing.assert_array_equal(
            val_counts,
            np.ones(data_context.n_rows),
            err_msg="Some samples appear in validation multiple times or never"
        )


class TestOOFFromValOnly:
    """Tests verifying OOF predictions come from validation, not training."""

    def test_oof_indices_match_val_indices(self, data_context, cv_config_stratified):
        """Verify OOF predictions are placed at validation indices."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        # Create mock fold results with unique predictions per fold
        fold_results = []
        for fold in folds:
            # Make predictions identifiable by fold
            preds = np.ones(fold.n_val) * (fold.fold_idx + 1)
            result = FoldResult(
                fold=fold,
                model=None,
                val_predictions=preds,
                val_score=0.8,
            )
            fold_results.append(result)

        oof = cv_engine.route_oof_predictions(data_context, fold_results)

        # Verify each fold's predictions are at the right indices
        for fold, result in zip(folds, fold_results):
            expected_value = fold.fold_idx + 1
            actual_values = oof[fold.val_indices]
            np.testing.assert_array_equal(
                actual_values,
                np.ones(fold.n_val) * expected_value,
                err_msg=f"Fold {fold.fold_idx} predictions not at correct indices"
            )

    def test_oof_not_in_train_set(self, data_context, cv_config_stratified):
        """Verify predictions at training indices don't come from that fold."""
        cv_engine = CVEngine(cv_config_stratified)
        folds = cv_engine.create_folds(data_context)

        # Create mock fold results with unique predictions per fold
        fold_results = []
        for fold in folds:
            preds = np.ones(fold.n_val) * (fold.fold_idx + 1) * 100
            result = FoldResult(
                fold=fold,
                model=None,
                val_predictions=preds,
                val_score=0.8,
            )
            fold_results.append(result)

        oof = cv_engine.route_oof_predictions(data_context, fold_results)

        # For each fold, verify its train indices don't have its predictions
        for fold in folds:
            fold_value = (fold.fold_idx + 1) * 100
            train_oof_values = oof[fold.train_indices]

            # Train indices should NOT have this fold's predictions
            # (they should have predictions from other folds)
            assert not np.all(train_oof_values == fold_value), (
                "Train indices have predictions from same fold (data leakage)"
            )


@pytest.mark.skip(reason="CVEngine.create_nested_folds() not yet implemented in new API")
class TestNestedCVLeakage:
    """Tests verifying no leakage in nested cross-validation."""

    def test_outer_val_never_in_inner(self, data_context, cv_config_nested):
        """Verify outer validation samples never appear in inner training or validation."""
        cv_engine = CVEngine(cv_config_nested)
        nested_folds = cv_engine.create_nested_folds(data_context)

        for nested in nested_folds:
            outer_val_set = set(nested.outer_fold.val_indices)

            for inner_fold in nested.inner_folds:
                inner_train_set = set(inner_fold.train_indices)
                inner_val_set = set(inner_fold.val_indices)

                # Outer validation should not appear in inner training
                train_leak = outer_val_set & inner_train_set
                assert len(train_leak) == 0, (
                    f"Outer val samples in inner train: {train_leak}"
                )

                # Outer validation should not appear in inner validation
                val_leak = outer_val_set & inner_val_set
                assert len(val_leak) == 0, (
                    f"Outer val samples in inner val: {val_leak}"
                )

    def test_inner_within_outer_train(self, data_context, cv_config_nested):
        """Verify inner CV is entirely within outer training set."""
        cv_engine = CVEngine(cv_config_nested)
        nested_folds = cv_engine.create_nested_folds(data_context)

        for nested in nested_folds:
            outer_train_set = set(nested.outer_fold.train_indices)

            for inner_fold in nested.inner_folds:
                inner_train_set = set(inner_fold.train_indices)
                inner_val_set = set(inner_fold.val_indices)

                # All inner samples should be in outer training
                assert inner_train_set.issubset(outer_train_set), (
                    "Inner train samples outside outer train"
                )
                assert inner_val_set.issubset(outer_train_set), (
                    "Inner val samples outside outer train"
                )

    def test_inner_folds_disjoint(self, data_context, cv_config_nested):
        """Verify inner folds have disjoint validation sets."""
        cv_engine = CVEngine(cv_config_nested)
        nested_folds = cv_engine.create_nested_folds(data_context)

        for nested in nested_folds:
            inner_val_sets = [set(f.val_indices) for f in nested.inner_folds]

            # Check pairwise disjoint
            for i in range(len(inner_val_sets)):
                for j in range(i + 1, len(inner_val_sets)):
                    intersection = inner_val_sets[i] & inner_val_sets[j]
                    assert len(intersection) == 0, (
                        f"Inner folds {i} and {j} have overlapping validation"
                    )


class TestGroupCVLeakage:
    """Tests verifying no group leakage in group cross-validation."""

    def test_groups_not_split(self, data_context_with_groups, cv_config_group):
        """Verify no group appears in both train and validation."""
        cv_engine = CVEngine(cv_config_group)
        folds = cv_engine.create_folds(data_context_with_groups)
        groups = data_context_with_groups.groups

        for fold in folds:
            train_groups = set(groups[fold.train_indices])
            val_groups = set(groups[fold.val_indices])

            intersection = train_groups & val_groups
            assert len(intersection) == 0, (
                f"Fold {fold.fold_idx} splits groups: {intersection}"
            )


class TestIntentionalLeakDetection:
    """Tests to verify we can detect when leakage occurs."""

    def test_detect_high_score_with_leak(self, classification_data):
        """Verify artificially high score detected as potential leak."""
        X, y = classification_data

        # Create a feature that directly encodes the target (obvious leak)
        X_leaked = X.copy()
        X_leaked["leak_feature"] = y.values

        ctx_leaked = DataView.from_Xy(X_leaked, y)
        ctx_clean = DataView.from_Xy(X, y)

        cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
        cv_engine = CVEngine(cv_config)
        folds = cv_engine.create_folds(ctx_leaked)

        # Train and predict with leaked data
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        leaked_scores = []
        clean_scores = []

        for fold in folds:
            # With leak
            train_view, val_view = cv_engine.split_for_fold(ctx_leaked, fold)
            train_data = train_view.materialize()
            val_data = val_view.materialize()
            model.fit(train_data.X, train_data.y)
            leaked_score = model.score(val_data.X, val_data.y)
            leaked_scores.append(leaked_score)

            # Without leak
            train_view, val_view = cv_engine.split_for_fold(ctx_clean, fold)
            train_data = train_view.materialize()
            val_data = val_view.materialize()
            model.fit(train_data.X, train_data.y)
            clean_score = model.score(val_data.X, val_data.y)
            clean_scores.append(clean_score)

        # Leaked should have near-perfect scores
        assert np.mean(leaked_scores) > 0.99, "Leak detection failed - leaked model should be near-perfect"

        # Clean should have reasonable but not perfect scores
        assert np.mean(clean_scores) < 0.95, "Clean model unexpectedly high - possible leak"

    def test_oof_score_matches_cv_average(self, classification_data):
        """Verify OOF score is consistent with CV fold scores."""
        X, y = classification_data
        ctx = DataView.from_Xy(X, y)

        cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
        cv_engine = CVEngine(cv_config)
        folds = cv_engine.create_folds(ctx)

        model = LogisticRegression(max_iter=1000, random_state=42)

        fold_results = []
        for fold in folds:
            train_view, val_view = cv_engine.split_for_fold(ctx, fold)
            train_data = train_view.materialize()
            val_data = val_view.materialize()
            model.fit(train_data.X, train_data.y)
            val_preds = model.predict(val_data.X)
            val_score = (val_preds == val_data.y).mean()

            result = FoldResult(
                fold=fold,
                model=model,
                val_predictions=val_preds,
                val_score=val_score,
            )
            fold_results.append(result)

        # Route OOF predictions
        oof = cv_engine.route_oof_predictions(ctx, fold_results)

        # OOF score should match average CV score (approximately)
        oof_score = (oof == y.values).mean()
        cv_score = np.mean([r.val_score for r in fold_results])

        assert abs(oof_score - cv_score) < 0.01, (
            f"OOF score {oof_score} differs from CV score {cv_score}"
        )


class TestTimeSeriesCVLeakage:
    """Tests for time series CV leakage prevention."""

    def test_time_series_no_future_leak(self, data_context):
        """Verify time series CV doesn't use future data in training."""
        cv_config = CVConfig(n_splits=5, strategy=CVStrategy.TIME_SERIES)
        cv_engine = CVEngine(cv_config)
        folds = cv_engine.create_folds(data_context)

        for fold in folds:
            max_train_idx = max(fold.train_indices)
            min_val_idx = min(fold.val_indices)

            # All training indices should be less than all validation indices
            assert max_train_idx < min_val_idx, (
                f"Fold {fold.fold_idx}: train indices overlap with val indices temporally"
            )

    def test_time_series_expanding_window(self, data_context):
        """Verify time series CV uses expanding training window."""
        cv_config = CVConfig(n_splits=5, strategy=CVStrategy.TIME_SERIES)
        cv_engine = CVEngine(cv_config)
        folds = cv_engine.create_folds(data_context)

        prev_train_size = 0
        for fold in folds:
            # Training window should expand
            assert fold.n_train > prev_train_size, (
                f"Fold {fold.fold_idx}: training window not expanding"
            )
            prev_train_size = fold.n_train
