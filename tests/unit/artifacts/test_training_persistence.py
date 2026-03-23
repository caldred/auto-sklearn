"""Tests for TrainingRun save/load round-trip."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from sklearn_meta.artifacts.training import (
    TrainingRun,
    NodeRunResult,
    RunMetadata,
)
from sklearn_meta.runtime.config import (
    CVConfig,
    CVFold,
    CVResult,
    CVStrategy,
    FeatureSelectionConfig,
    FeatureSelectionMethod,
    FoldResult,
    ReparameterizationConfig,
    RunConfig,
    TuningConfig,
)
from sklearn_meta.engine.estimator_scaling import EstimatorScalingConfig
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec


def _make_training_run(config: RunConfig, repeat_oof: np.ndarray | None = None) -> TrainingRun:
    """Build a minimal TrainingRun for round-trip testing."""
    node = NodeSpec(name="ridge", estimator_class=Ridge)
    graph = GraphSpec()
    graph.add_node(node)

    fold = CVFold(
        fold_idx=0,
        train_indices=np.arange(80),
        val_indices=np.arange(80, 100),
    )
    model = Ridge().fit(np.random.randn(80, 3), np.random.randn(80))
    fold_result = FoldResult(
        fold=fold,
        model=model,
        val_predictions=np.random.randn(20),
        val_score=0.85,
        train_score=0.90,
        fit_time=0.1,
        predict_time=0.01,
        params={"alpha": 1.0},
    )
    cv_result = CVResult(
        fold_results=[fold_result],
        oof_predictions=np.random.randn(100),
        node_name="ridge",
        repeat_oof=repeat_oof,
    )
    node_result = NodeRunResult(
        node_name="ridge",
        cv_result=cv_result,
        best_params={"alpha": 1.0},
        selected_features=["f0", "f1", "f2"],
    )

    metadata = RunMetadata(
        timestamp="2026-01-01T00:00:00+00:00",
        sklearn_meta_version="0.1.0",
        data_shape=(100, 3),
        feature_names=["f0", "f1", "f2"],
        cv_config={"n_splits": 5},
        tuning_config_summary={"n_trials": 50},
        total_trials=50,
        data_hash="abc123",
        random_state=42,
    )

    return TrainingRun(
        graph=graph,
        config=config,
        node_results={"ridge": node_result},
        metadata=metadata,
        total_time=1.23,
    )


class TestTrainingRunPersistence:
    def test_round_trips_default_config(self, tmp_path):
        config = RunConfig()
        run = _make_training_run(config)
        run.save(tmp_path / "run")
        loaded = TrainingRun.load(tmp_path / "run")

        assert loaded.config.cv.n_splits == config.cv.n_splits
        assert loaded.config.cv.n_repeats == config.cv.n_repeats
        assert loaded.config.cv.strategy == config.cv.strategy
        assert loaded.config.tuning.n_trials == config.tuning.n_trials
        assert loaded.config.tuning.metric == config.tuning.metric
        assert loaded.config.verbosity == config.verbosity
        assert loaded.config.feature_selection is None
        assert loaded.config.reparameterization is None

    def test_round_trips_custom_config(self, tmp_path):
        config = RunConfig(
            cv=CVConfig(n_splits=10, n_repeats=3, strategy=CVStrategy.STRATIFIED, random_state=99),
            tuning=TuningConfig(
                n_trials=200,
                timeout=600.0,
                early_stopping_rounds=20,
                metric="neg_log_loss",
                greater_is_better=False,
                strategy=OptimizationStrategy.GREEDY,
                show_progress=True,
            ),
            feature_selection=FeatureSelectionConfig(
                enabled=True,
                method=FeatureSelectionMethod.PERMUTATION,
                n_shadows=10,
                threshold_mult=2.0,
                min_features=3,
                max_features=50,
                random_state=7,
            ),
            reparameterization=ReparameterizationConfig(enabled=True, use_prebaked=False),
            estimator_scaling=EstimatorScalingConfig(
                tuning_n_estimators=100,
                final_n_estimators=500,
                scaling_search=True,
                scaling_factors=[1, 2, 4],
            ),
            verbosity=2,
        )
        run = _make_training_run(config)
        run.save(tmp_path / "run")
        loaded = TrainingRun.load(tmp_path / "run")

        # CV
        assert loaded.config.cv.n_splits == 10
        assert loaded.config.cv.n_repeats == 3
        assert loaded.config.cv.strategy == CVStrategy.STRATIFIED
        assert loaded.config.cv.random_state == 99

        # Tuning
        assert loaded.config.tuning.n_trials == 200
        assert loaded.config.tuning.timeout == 600.0
        assert loaded.config.tuning.early_stopping_rounds == 20
        assert loaded.config.tuning.metric == "neg_log_loss"
        assert loaded.config.tuning.strategy == OptimizationStrategy.GREEDY
        assert loaded.config.tuning.show_progress is True

        # Feature selection
        assert loaded.config.feature_selection is not None
        assert loaded.config.feature_selection.method == FeatureSelectionMethod.PERMUTATION
        assert loaded.config.feature_selection.n_shadows == 10
        assert loaded.config.feature_selection.min_features == 3
        assert loaded.config.feature_selection.max_features == 50

        # Reparameterization
        assert loaded.config.reparameterization is not None
        assert loaded.config.reparameterization.use_prebaked is False

        # Estimator scaling
        assert loaded.config.estimator_scaling is not None
        assert loaded.config.estimator_scaling.tuning_n_estimators == 100
        assert loaded.config.estimator_scaling.final_n_estimators == 500
        assert loaded.config.estimator_scaling.scaling_search is True
        assert loaded.config.estimator_scaling.scaling_factors == [1, 2, 4]

        # Verbosity
        assert loaded.config.verbosity == 2

    def test_round_trips_node_results(self, tmp_path):
        run = _make_training_run(RunConfig())
        run.save(tmp_path / "run")
        loaded = TrainingRun.load(tmp_path / "run")

        assert "ridge" in loaded.node_results
        result = loaded.node_results["ridge"]
        assert result.best_params == {"alpha": 1.0}
        assert result.selected_features == ["f0", "f1", "f2"]
        np.testing.assert_allclose(result.mean_score, 0.85, atol=1e-6)

    def test_round_trips_total_time(self, tmp_path):
        run = _make_training_run(RunConfig())
        run.save(tmp_path / "run")
        loaded = TrainingRun.load(tmp_path / "run")
        assert loaded.total_time == pytest.approx(1.23)

    def test_round_trips_metadata(self, tmp_path):
        run = _make_training_run(RunConfig())
        run.save(tmp_path / "run")
        loaded = TrainingRun.load(tmp_path / "run")

        assert loaded.metadata.timestamp == "2026-01-01T00:00:00+00:00"
        assert loaded.metadata.sklearn_meta_version == "0.1.0"
        assert loaded.metadata.data_shape == (100, 3)
        assert loaded.metadata.feature_names == ["f0", "f1", "f2"]
        assert loaded.metadata.data_hash == "abc123"

    def test_round_trips_repeat_oof(self, tmp_path):
        repeat_oof = np.random.randn(2, 100)
        run = _make_training_run(
            RunConfig(cv=CVConfig(n_splits=5, n_repeats=2)),
            repeat_oof=repeat_oof,
        )
        run.save(tmp_path / "run")
        loaded = TrainingRun.load(tmp_path / "run")

        assert loaded.node_results["ridge"].cv_result.repeat_oof is not None
        np.testing.assert_allclose(
            loaded.node_results["ridge"].cv_result.repeat_oof,
            repeat_oof,
        )

    def test_loads_legacy_manifest_without_run_config(self, tmp_path):
        """If run_config key is missing (old manifest), load still works with defaults."""
        run = _make_training_run(RunConfig())
        run.save(tmp_path / "run")

        # Manually strip run_config from manifest
        import json
        manifest_path = tmp_path / "run" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("run_config", None)
        manifest_path.write_text(json.dumps(manifest))

        loaded = TrainingRun.load(tmp_path / "run")
        # Should get defaults without error
        assert loaded.config.cv.n_splits == 5
        assert loaded.config.tuning.n_trials == 100

    def test_loads_manifest_without_repeat_oof_or_estimator_scaling(self, tmp_path):
        import json

        config = RunConfig(
            estimator_scaling=EstimatorScalingConfig(
                tuning_n_estimators=100,
                final_n_estimators=500,
                scaling_search=True,
                scaling_factors=[1, 2, 4],
            ),
        )
        repeat_oof = np.random.randn(2, 100)
        run = _make_training_run(config, repeat_oof=repeat_oof)
        run.save(tmp_path / "run")

        manifest_path = tmp_path / "run" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["run_config"].pop("estimator_scaling", None)
        manifest_path.write_text(json.dumps(manifest))

        artifacts_path = tmp_path / "run" / "nodes" / "ridge" / "training_artifacts.joblib"
        import joblib
        artifacts = joblib.load(artifacts_path)
        artifacts.pop("repeat_oof", None)
        joblib.dump(artifacts, artifacts_path)

        loaded = TrainingRun.load(tmp_path / "run")
        assert loaded.config.estimator_scaling is None
        assert loaded.node_results["ridge"].cv_result.repeat_oof is None
