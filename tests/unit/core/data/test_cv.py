"""Tests for cross-validation configuration and fold management."""

import numpy as np
import pandas as pd
import pytest

from auto_sklearn.core.data.cv import (
    CVConfig,
    CVFold,
    CVResult,
    CVStrategy,
    FoldResult,
    NestedCVFold,
)


class TestCVFold:
    """Tests for CVFold dataclass."""

    def test_cvfold_train_val_disjoint(self):
        """Verify train and validation indices have no overlap."""
        train_indices = np.array([0, 1, 2, 3, 4])
        val_indices = np.array([5, 6, 7, 8, 9])

        fold = CVFold(
            fold_idx=0,
            train_indices=train_indices,
            val_indices=val_indices,
        )

        intersection = set(train_indices) & set(val_indices)
        assert len(intersection) == 0

    def test_cvfold_n_train_property(self):
        """Verify n_train property returns correct count."""
        fold = CVFold(
            fold_idx=0,
            train_indices=np.array([0, 1, 2, 3, 4]),
            val_indices=np.array([5, 6, 7]),
        )

        assert fold.n_train == 5

    def test_cvfold_n_val_property(self):
        """Verify n_val property returns correct count."""
        fold = CVFold(
            fold_idx=0,
            train_indices=np.array([0, 1, 2, 3, 4]),
            val_indices=np.array([5, 6, 7]),
        )

        assert fold.n_val == 3

    def test_cvfold_repeat_idx_default(self):
        """Verify repeat_idx defaults to 0."""
        fold = CVFold(
            fold_idx=0,
            train_indices=np.array([0, 1, 2]),
            val_indices=np.array([3, 4]),
        )

        assert fold.repeat_idx == 0

    def test_cvfold_repr(self):
        """Verify CVFold repr is informative."""
        fold = CVFold(
            fold_idx=2,
            train_indices=np.array([0, 1, 2, 3]),
            val_indices=np.array([4, 5]),
            repeat_idx=1,
        )

        repr_str = repr(fold)
        assert "fold=2" in repr_str
        assert "repeat=1" in repr_str
        assert "n_train=4" in repr_str
        assert "n_val=2" in repr_str


class TestNestedCVFold:
    """Tests for NestedCVFold dataclass."""

    def test_nested_fold_contains_outer_fold(self):
        """Verify NestedCVFold contains outer fold."""
        outer = CVFold(
            fold_idx=0,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            val_indices=np.array([8, 9]),
        )
        inner_folds = [
            CVFold(fold_idx=0, train_indices=np.array([0, 1, 2, 3, 4]), val_indices=np.array([5, 6, 7])),
            CVFold(fold_idx=1, train_indices=np.array([0, 1, 2, 5, 6, 7]), val_indices=np.array([3, 4])),
        ]

        nested = NestedCVFold(outer_fold=outer, inner_folds=inner_folds)

        assert nested.outer_fold is outer
        assert len(nested.inner_folds) == 2

    def test_nested_fold_idx_property(self):
        """Verify fold_idx returns outer fold index."""
        outer = CVFold(fold_idx=3, train_indices=np.array([0, 1, 2]), val_indices=np.array([3, 4]))
        nested = NestedCVFold(outer_fold=outer, inner_folds=[])

        assert nested.fold_idx == 3

    def test_n_inner_folds_property(self):
        """Verify n_inner_folds returns correct count."""
        outer = CVFold(fold_idx=0, train_indices=np.array([0, 1, 2, 3]), val_indices=np.array([4, 5]))
        inner_folds = [
            CVFold(fold_idx=i, train_indices=np.array([0, 1]), val_indices=np.array([2, 3]))
            for i in range(5)
        ]

        nested = NestedCVFold(outer_fold=outer, inner_folds=inner_folds)

        assert nested.n_inner_folds == 5


class TestCVConfig:
    """Tests for CVConfig dataclass."""

    def test_cvconfig_default_values(self):
        """Verify default values are set correctly."""
        config = CVConfig()

        assert config.n_splits == 5
        assert config.n_repeats == 1
        assert config.strategy == CVStrategy.GROUP
        assert config.shuffle is True
        assert config.random_state == 42
        assert config.inner_cv is None

    def test_cvconfig_total_folds_no_repeat(self):
        """Verify total_folds with no repeats."""
        config = CVConfig(n_splits=5, n_repeats=1)

        assert config.total_folds == 5

    def test_cvconfig_total_folds_with_repeats(self):
        """Verify total_folds with repeats."""
        config = CVConfig(n_splits=5, n_repeats=3)

        assert config.total_folds == 15

    def test_cvconfig_is_nested_false(self):
        """Verify is_nested is False without inner_cv."""
        config = CVConfig()

        assert config.is_nested is False

    def test_cvconfig_with_inner_cv_creates_nested(self):
        """Verify with_inner_cv creates nested configuration."""
        config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED)
        nested = config.with_inner_cv(n_splits=3)

        assert nested.is_nested is True
        assert nested.inner_cv.n_splits == 3

    def test_cvconfig_with_inner_cv_inherits_strategy(self):
        """Verify inner CV inherits strategy by default."""
        config = CVConfig(strategy=CVStrategy.STRATIFIED)
        nested = config.with_inner_cv(n_splits=3)

        assert nested.inner_cv.strategy == CVStrategy.STRATIFIED

    def test_cvconfig_with_inner_cv_custom_strategy(self):
        """Verify inner CV can have custom strategy."""
        config = CVConfig(strategy=CVStrategy.STRATIFIED)
        nested = config.with_inner_cv(n_splits=3, strategy=CVStrategy.RANDOM)

        assert nested.inner_cv.strategy == CVStrategy.RANDOM

    def test_cvconfig_validation_n_splits_minimum(self):
        """Verify n_splits must be >= 2."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            CVConfig(n_splits=1)

    def test_cvconfig_validation_n_repeats_minimum(self):
        """Verify n_repeats must be >= 1."""
        with pytest.raises(ValueError, match="n_repeats must be >= 1"):
            CVConfig(n_repeats=0)

    def test_cvconfig_strategy_from_string(self):
        """Verify strategy can be set from string."""
        config = CVConfig(strategy="stratified")

        assert config.strategy == CVStrategy.STRATIFIED

    def test_cvconfig_inner_cv_has_different_random_state(self):
        """Verify inner CV has different random state."""
        config = CVConfig(random_state=42)
        nested = config.with_inner_cv(n_splits=3)

        assert nested.inner_cv.random_state == 43


class TestCVStrategy:
    """Tests for CVStrategy enum."""

    def test_all_strategies_defined(self):
        """Verify all expected strategies exist."""
        strategies = [CVStrategy.GROUP, CVStrategy.STRATIFIED, CVStrategy.RANDOM, CVStrategy.TIME_SERIES]

        assert len(CVStrategy) == 4
        for strategy in strategies:
            assert strategy in CVStrategy

    def test_strategy_values(self):
        """Verify strategy string values."""
        assert CVStrategy.GROUP.value == "group"
        assert CVStrategy.STRATIFIED.value == "stratified"
        assert CVStrategy.RANDOM.value == "random"
        assert CVStrategy.TIME_SERIES.value == "time_series"


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_fold_result_creation(self):
        """Verify FoldResult can be created with required fields."""
        fold = CVFold(fold_idx=0, train_indices=np.array([0, 1]), val_indices=np.array([2, 3]))

        class MockModel:
            pass

        result = FoldResult(
            fold=fold,
            model=MockModel(),
            val_predictions=np.array([0.5, 0.3]),
            val_score=0.85,
        )

        assert result.fold_idx == 0
        assert result.val_score == 0.85

    def test_fold_result_default_times(self):
        """Verify default times are 0."""
        fold = CVFold(fold_idx=0, train_indices=np.array([0]), val_indices=np.array([1]))

        result = FoldResult(
            fold=fold,
            model=None,
            val_predictions=np.array([0.5]),
            val_score=0.8,
        )

        assert result.fit_time == 0.0
        assert result.predict_time == 0.0

    def test_fold_result_default_params(self):
        """Verify default params is empty dict."""
        fold = CVFold(fold_idx=0, train_indices=np.array([0]), val_indices=np.array([1]))

        result = FoldResult(
            fold=fold,
            model=None,
            val_predictions=np.array([0.5]),
            val_score=0.8,
        )

        assert result.params == {}

    def test_fold_result_repr(self):
        """Verify FoldResult repr is informative."""
        fold = CVFold(fold_idx=2, train_indices=np.array([0, 1]), val_indices=np.array([2, 3]))

        result = FoldResult(
            fold=fold,
            model=None,
            val_predictions=np.array([0.5, 0.3]),
            val_score=0.8567,
            fit_time=1.23,
        )

        repr_str = repr(result)
        assert "fold=2" in repr_str
        assert "0.8567" in repr_str
        assert "1.23" in repr_str


class TestCVResult:
    """Tests for CVResult dataclass."""

    def test_cvresult_n_folds(self):
        """Verify n_folds property."""
        fold_results = [
            FoldResult(
                fold=CVFold(fold_idx=i, train_indices=np.array([0]), val_indices=np.array([1])),
                model=None,
                val_predictions=np.array([0.5]),
                val_score=0.8 + i * 0.01,
            )
            for i in range(5)
        ]

        result = CVResult(
            fold_results=fold_results,
            oof_predictions=np.array([0.5] * 2),
            node_name="test",
        )

        assert result.n_folds == 5

    def test_cvresult_val_scores(self):
        """Verify val_scores returns array of scores."""
        fold_results = [
            FoldResult(
                fold=CVFold(fold_idx=i, train_indices=np.array([0]), val_indices=np.array([1])),
                model=None,
                val_predictions=np.array([0.5]),
                val_score=0.8 + i * 0.02,
            )
            for i in range(3)
        ]

        result = CVResult(
            fold_results=fold_results,
            oof_predictions=np.array([0.5]),
            node_name="test",
        )

        expected = np.array([0.8, 0.82, 0.84])
        np.testing.assert_array_almost_equal(result.val_scores, expected)

    def test_cvresult_mean_score(self):
        """Verify mean_score is calculated correctly."""
        scores = [0.8, 0.82, 0.84]
        fold_results = [
            FoldResult(
                fold=CVFold(fold_idx=i, train_indices=np.array([0]), val_indices=np.array([1])),
                model=None,
                val_predictions=np.array([0.5]),
                val_score=score,
            )
            for i, score in enumerate(scores)
        ]

        result = CVResult(
            fold_results=fold_results,
            oof_predictions=np.array([0.5]),
            node_name="test",
        )

        assert result.mean_score == pytest.approx(np.mean(scores))

    def test_cvresult_std_score(self):
        """Verify std_score is calculated correctly."""
        scores = [0.8, 0.85, 0.9]
        fold_results = [
            FoldResult(
                fold=CVFold(fold_idx=i, train_indices=np.array([0]), val_indices=np.array([1])),
                model=None,
                val_predictions=np.array([0.5]),
                val_score=score,
            )
            for i, score in enumerate(scores)
        ]

        result = CVResult(
            fold_results=fold_results,
            oof_predictions=np.array([0.5]),
            node_name="test",
        )

        assert result.std_score == pytest.approx(np.std(scores))

    def test_cvresult_total_fit_time(self):
        """Verify total_fit_time sums all fold times."""
        fold_results = [
            FoldResult(
                fold=CVFold(fold_idx=i, train_indices=np.array([0]), val_indices=np.array([1])),
                model=None,
                val_predictions=np.array([0.5]),
                val_score=0.8,
                fit_time=1.0 + i,
            )
            for i in range(3)
        ]

        result = CVResult(
            fold_results=fold_results,
            oof_predictions=np.array([0.5]),
            node_name="test",
        )

        assert result.total_fit_time == 6.0  # 1 + 2 + 3

    def test_cvresult_models(self):
        """Verify models returns list of models from folds."""
        class MockModel:
            def __init__(self, idx):
                self.idx = idx

        models = [MockModel(i) for i in range(3)]
        fold_results = [
            FoldResult(
                fold=CVFold(fold_idx=i, train_indices=np.array([0]), val_indices=np.array([1])),
                model=models[i],
                val_predictions=np.array([0.5]),
                val_score=0.8,
            )
            for i in range(3)
        ]

        result = CVResult(
            fold_results=fold_results,
            oof_predictions=np.array([0.5]),
            node_name="test",
        )

        assert len(result.models) == 3
        for i, model in enumerate(result.models):
            assert model.idx == i

    def test_cvresult_repr(self):
        """Verify CVResult repr is informative."""
        fold_results = [
            FoldResult(
                fold=CVFold(fold_idx=i, train_indices=np.array([0]), val_indices=np.array([1])),
                model=None,
                val_predictions=np.array([0.5]),
                val_score=0.8,
            )
            for i in range(5)
        ]

        result = CVResult(
            fold_results=fold_results,
            oof_predictions=np.array([0.5]),
            node_name="test_node",
        )

        repr_str = repr(result)
        assert "test_node" in repr_str
        assert "n_folds=5" in repr_str
