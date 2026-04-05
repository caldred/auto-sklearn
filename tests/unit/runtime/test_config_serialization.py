"""Tests for runtime config serialization helpers."""

import json

from sklearn_meta.engine.estimator_scaling import EstimatorScalingConfig
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.meta.reparameterization import LogProductReparameterization
from sklearn_meta.runtime.config import (
    CVConfig,
    CVStrategy,
    FeatureSelectionConfig,
    FeatureSelectionMethod,
    ReparameterizationConfig,
    RunConfig,
    TuningConfig,
)


def test_run_config_round_trip():
    config = RunConfig(
        cv=CVConfig(
            n_splits=4,
            n_repeats=2,
            strategy=CVStrategy.STRATIFIED,
            shuffle=True,
            random_state=7,
        ),
        tuning=TuningConfig(
            n_trials=25,
            timeout=10.0,
            early_stopping_rounds=5,
            metric="accuracy",
            greater_is_better=True,
            strategy=OptimizationStrategy.GREEDY,
            show_progress=True,
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method=FeatureSelectionMethod.THRESHOLD,
            min_features=2,
            random_state=11,
        ),
        reparameterization=ReparameterizationConfig(
            enabled=True,
            use_prebaked=False,
        ),
        estimator_scaling=EstimatorScalingConfig(
            tuning_n_estimators=50,
            final_n_estimators=200,
            scaling_search=True,
            scaling_factors=[1.5, 2, 4],
            scaling_estimators=[300, 500],
        ),
        verbosity=2,
    )

    restored = RunConfig.from_dict(config.to_dict())

    assert restored.cv.n_splits == 4
    assert restored.cv.n_repeats == 2
    assert restored.cv.strategy == CVStrategy.STRATIFIED
    assert restored.tuning.n_trials == 25
    assert restored.tuning.strategy == OptimizationStrategy.GREEDY
    assert restored.feature_selection is not None
    assert restored.feature_selection.method == FeatureSelectionMethod.THRESHOLD
    assert restored.reparameterization is not None
    assert restored.reparameterization.use_prebaked is False
    assert restored.estimator_scaling is not None
    assert restored.estimator_scaling.final_n_estimators == 200
    assert restored.estimator_scaling.scaling_factors == [1.5, 2, 4]
    assert restored.estimator_scaling.scaling_estimators == [300, 500]
    assert restored.verbosity == 2


def test_run_config_round_trips_custom_reparameterizations():
    config = RunConfig(
        reparameterization=ReparameterizationConfig(
            enabled=True,
            use_prebaked=False,
            custom_reparameterizations=(
                LogProductReparameterization(
                    name="lr_budget",
                    param1="learning_rate",
                    param2="n_estimators",
                ),
            ),
        ),
    )

    payload = config.to_dict()
    json.dumps(payload)
    restored = RunConfig.from_dict(payload)

    assert restored.reparameterization is not None
    restored_reparameterization = (
        restored.reparameterization.custom_reparameterizations[0]
    )
    assert isinstance(restored_reparameterization, LogProductReparameterization)
    assert restored_reparameterization.param1 == "learning_rate"
    assert restored_reparameterization.param2 == "n_estimators"


# ------------------------------------------------------------------
# CVConfig validation
# ------------------------------------------------------------------

import warnings
import pytest


class TestCVConfigValidation:
    """Test boundary conditions in CVConfig validation."""

    def test_n_splits_one_raises(self):
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            CVConfig(n_splits=1)

    def test_n_splits_zero_raises(self):
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            CVConfig(n_splits=0)

    def test_n_repeats_zero_raises(self):
        with pytest.raises(ValueError, match="n_repeats must be >= 1"):
            CVConfig(n_repeats=0)

    def test_n_repeats_negative_raises(self):
        with pytest.raises(ValueError, match="n_repeats must be >= 1"):
            CVConfig(n_repeats=-1)

    def test_total_folds_property(self):
        config = CVConfig(n_splits=5, n_repeats=3)
        assert config.total_folds == 15


class TestCVConfigNestedRoundTrip:
    """Test nested CV survives serialization."""

    def test_nested_cv_round_trip(self):
        config = CVConfig(
            n_splits=5,
            n_repeats=2,
            strategy=CVStrategy.STRATIFIED,
            random_state=7,
        ).with_inner_cv(n_splits=3)

        assert config.is_nested
        assert config.inner_cv.n_splits == 3

        restored = CVConfig.from_dict(config.to_dict())

        assert restored.is_nested
        assert restored.inner_cv is not None
        assert restored.inner_cv.n_splits == 3
        assert restored.inner_cv.strategy == CVStrategy.STRATIFIED
        assert restored.n_splits == 5
        assert restored.n_repeats == 2

class TestTuningConfigInference:
    """Test greater_is_better inference from metric name."""

    def test_accuracy_inferred_true(self):
        config = TuningConfig(metric="accuracy")
        assert config.greater_is_better is True

    def test_roc_auc_inferred_true(self):
        config = TuningConfig(metric="roc_auc")
        assert config.greater_is_better is True

    def test_neg_mean_squared_error_inferred_true(self):
        # sklearn negates loss metrics, so all scorers are higher-is-better
        config = TuningConfig(metric="neg_mean_squared_error")
        assert config.greater_is_better is True

    def test_unknown_metric_defaults_false_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = TuningConfig(metric="totally_custom_metric")
            assert config.greater_is_better is False
            assert any("Unknown metric" in str(warning.message) for warning in w)

    def test_explicit_override_preserved(self):
        config = TuningConfig(metric="accuracy", greater_is_better=False)
        assert config.greater_is_better is False

    def test_tuning_config_frozen(self):
        config = TuningConfig(metric="accuracy")
        with pytest.raises(AttributeError):
            config.n_trials = 999


class TestRunConfigRoundTripCompleteness:
    """Ensure every RunConfig field survives a full round-trip."""

    def test_all_none_optionals_round_trip(self):
        config = RunConfig()
        restored = RunConfig.from_dict(config.to_dict())

        assert restored.feature_selection is None
        assert restored.reparameterization is None
        assert restored.estimator_scaling is None
        assert restored.verbosity == 1

    def test_feature_selection_all_fields_round_trip(self):
        fs = FeatureSelectionConfig(
            enabled=True,
            method=FeatureSelectionMethod.PERMUTATION,
            n_shadows=7,
            threshold_mult=2.5,
            threshold_percentile=15.0,
            retune_after_pruning=False,
            min_features=3,
            max_features=50,
            random_state=99,
            feature_groups={"group_a": ["f1", "f2"], "group_b": ["f3"]},
        )
        config = RunConfig(feature_selection=fs)
        restored = RunConfig.from_dict(config.to_dict())

        rfs = restored.feature_selection
        assert rfs is not None
        assert rfs.method == FeatureSelectionMethod.PERMUTATION
        assert rfs.n_shadows == 7
        assert rfs.threshold_mult == 2.5
        assert rfs.threshold_percentile == 15.0
        assert rfs.retune_after_pruning is False
        assert rfs.min_features == 3
        assert rfs.max_features == 50
        assert rfs.random_state == 99
        assert rfs.feature_groups == {"group_a": ["f1", "f2"], "group_b": ["f3"]}

