"""Tests for RunMetadata in FittedGraph."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVFold, CVResult, FoldResult
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.node import ModelNode, OutputType
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.tuning.orchestrator import (
    FittedGraph,
    FittedNode,
    RunMetadata,
    TuningConfig,
)
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.meta.reparameterization import LogProductReparameterization
from sklearn_meta.selection.selector import (
    FeatureSelectionConfig,
    FeatureSelectionMethod,
)
from sklearn_meta.search.backends.optuna import OptunaBackend


def _make_simple_ctx(n_samples=50, n_features=3, seed=42):
    """Create a simple DataContext for testing."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = X["f0"] * 2 + rng.randn(n_samples) * 0.1
    return DataContext.from_Xy(X, y)


def _fit_simple_graph(ctx=None, n_splits=2):
    """Fit a simple single-node graph and return the FittedGraph."""
    if ctx is None:
        ctx = _make_simple_ctx()

    node = ModelNode(
        name="tree",
        estimator_class=DecisionTreeRegressor,
        output_type=OutputType.PREDICTION,
        fixed_params={"max_depth": 3, "random_state": 42},
    )

    graph = ModelGraph()
    graph.add_node(node)

    cv_config = CVConfig(n_splits=n_splits, random_state=42)
    tuning_config = TuningConfig(
        strategy=OptimizationStrategy.NONE,
        n_trials=0,
        metric="neg_mean_squared_error",
        greater_is_better=False,
        cv_config=cv_config,
    )

    backend = OptunaBackend(direction="minimize", random_state=42)
    dm = DataManager(cv_config=cv_config)

    from sklearn_meta.core.tuning.orchestrator import TuningOrchestrator

    orchestrator = TuningOrchestrator(
        graph=graph,
        data_manager=dm,
        search_backend=backend,
        tuning_config=tuning_config,
    )

    return orchestrator.fit(ctx)


class TestRunMetadataPopulated:
    """Test that RunMetadata is populated after fit()."""

    def test_metadata_not_none(self):
        fg = _fit_simple_graph()
        assert fg.metadata is not None

    def test_metadata_is_run_metadata(self):
        fg = _fit_simple_graph()
        assert isinstance(fg.metadata, RunMetadata)

    def test_timestamp_is_iso_format(self):
        fg = _fit_simple_graph()
        # Should parse without error
        from datetime import datetime
        dt = datetime.fromisoformat(fg.metadata.timestamp)
        assert dt is not None

    def test_sklearn_meta_version(self):
        import sklearn_meta
        fg = _fit_simple_graph()
        assert fg.metadata.sklearn_meta_version == sklearn_meta.__version__

    def test_data_shape(self):
        ctx = _make_simple_ctx(n_samples=50, n_features=3)
        fg = _fit_simple_graph(ctx=ctx)
        assert fg.metadata.data_shape == (50, 3)

    def test_feature_names(self):
        ctx = _make_simple_ctx(n_samples=50, n_features=3)
        fg = _fit_simple_graph(ctx=ctx)
        assert fg.metadata.feature_names == ["f0", "f1", "f2"]

    def test_cv_config_populated(self):
        fg = _fit_simple_graph(n_splits=2)
        assert fg.metadata.cv_config is not None
        assert fg.metadata.cv_config["n_splits"] == 2
        assert fg.metadata.cv_config["random_state"] == 42

    def test_tuning_config_summary(self):
        fg = _fit_simple_graph()
        summary = fg.metadata.tuning_config_summary
        assert summary["strategy"] == "none"
        assert summary["metric"] == "neg_mean_squared_error"
        assert summary["greater_is_better"] is False

    def test_data_hash_is_hex_string(self):
        fg = _fit_simple_graph()
        assert fg.metadata.data_hash is not None
        assert len(fg.metadata.data_hash) == 64  # SHA256 hex length

    def test_random_state_from_cv(self):
        fg = _fit_simple_graph()
        assert fg.metadata.random_state == 42


class TestRunMetadataSaveLoad:
    """Test that metadata survives save/load round-trip."""

    def test_round_trip(self, tmp_path):
        fg = _fit_simple_graph()
        fg.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.metadata is not None
        assert loaded.metadata.timestamp == fg.metadata.timestamp
        assert loaded.metadata.sklearn_meta_version == fg.metadata.sklearn_meta_version
        assert loaded.metadata.data_shape == fg.metadata.data_shape
        assert loaded.metadata.feature_names == fg.metadata.feature_names
        assert loaded.metadata.cv_config == fg.metadata.cv_config
        assert loaded.metadata.tuning_config_summary == fg.metadata.tuning_config_summary
        assert loaded.metadata.total_trials == fg.metadata.total_trials
        assert loaded.metadata.data_hash == fg.metadata.data_hash
        assert loaded.metadata.random_state == fg.metadata.random_state

    def test_metadata_in_manifest(self, tmp_path):
        fg = _fit_simple_graph()
        fg.save(tmp_path / "model")

        with open(tmp_path / "model" / "manifest.json") as f:
            manifest = json.load(f)

        assert "metadata" in manifest
        assert manifest["metadata"]["data_hash"] == fg.metadata.data_hash


class TestTuningConfigSaveLoad:
    """Test that persisted tuning_config is materially complete."""

    def test_round_trip_preserves_nested_config(self, tmp_path):
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.GREEDY,
            n_trials=7,
            timeout=12.0,
            early_stopping_rounds=3,
            cv_config=CVConfig(
                n_splits=4,
                n_repeats=2,
                strategy="random",
                shuffle=False,
                random_state=17,
            ).with_inner_cv(n_splits=2, strategy="stratified"),
            metric="accuracy",
            greater_is_better=True,
            feature_selection=FeatureSelectionConfig(
                enabled=True,
                method=FeatureSelectionMethod.THRESHOLD,
                retune_after_pruning=False,
                min_features=2,
                threshold_percentile=25.0,
            ),
            use_reparameterization=True,
            custom_reparameterizations=[
                LogProductReparameterization(
                    name="lr_budget",
                    param1="learning_rate",
                    param2="n_estimators",
                )
            ],
            verbose=2,
            tuning_n_estimators=100,
            final_n_estimators=500,
            estimator_scaling_search=True,
            estimator_scaling_factors=[2, 4],
            show_progress=True,
        )

        node = ModelNode(
            name="tree",
            estimator_class=DecisionTreeRegressor,
            output_type=OutputType.PREDICTION,
        )
        graph = ModelGraph()
        graph.add_node(node)

        model = DecisionTreeRegressor().fit(np.array([[1], [2]]), np.array([1, 2]))
        fold = CVFold(
            fold_idx=0,
            train_indices=np.array([0]),
            val_indices=np.array([1]),
        )
        cv = CVResult(
            fold_results=[
                FoldResult(
                    fold=fold,
                    model=model,
                    val_predictions=np.array([2.0]),
                    val_score=-0.1,
                )
            ],
            oof_predictions=np.array([0.0, 2.0]),
            node_name="tree",
        )
        fg = FittedGraph(
            graph=graph,
            fitted_nodes={"tree": FittedNode(node=node, cv_result=cv, best_params={})},
            tuning_config=tuning_config,
            total_time=1.0,
        )

        fg.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.tuning_config.strategy == OptimizationStrategy.GREEDY
        assert loaded.tuning_config.n_trials == 7
        assert loaded.tuning_config.timeout == 12.0
        assert loaded.tuning_config.early_stopping_rounds == 3
        assert loaded.tuning_config.metric == "accuracy"
        assert loaded.tuning_config.greater_is_better is True
        assert loaded.tuning_config.cv_config is not None
        assert loaded.tuning_config.cv_config.n_splits == 4
        assert loaded.tuning_config.cv_config.n_repeats == 2
        assert loaded.tuning_config.cv_config.strategy.value == "random"
        assert loaded.tuning_config.cv_config.shuffle is False
        assert loaded.tuning_config.cv_config.random_state == 17
        assert loaded.tuning_config.cv_config.inner_cv is not None
        assert loaded.tuning_config.cv_config.inner_cv.n_splits == 2
        assert loaded.tuning_config.cv_config.inner_cv.strategy.value == "stratified"
        assert loaded.tuning_config.feature_selection is not None
        assert loaded.tuning_config.feature_selection.method == FeatureSelectionMethod.THRESHOLD
        assert loaded.tuning_config.feature_selection.min_features == 2
        assert loaded.tuning_config.use_reparameterization is True
        assert loaded.tuning_config.custom_reparameterizations is None
        assert loaded.tuning_config.tuning_n_estimators == 100
        assert loaded.tuning_config.final_n_estimators == 500
        assert loaded.tuning_config.estimator_scaling_search is True
        assert loaded.tuning_config.estimator_scaling_factors == [2, 4]
        assert loaded.tuning_config.show_progress is True


class TestRunMetadataManualFittedGraph:
    """Test that metadata is None when not populated."""

    def test_metadata_none_by_default(self):
        from sklearn_meta.core.data.cv import CVFold, CVResult, FoldResult

        node = ModelNode(
            name="tree",
            estimator_class=DecisionTreeRegressor,
            output_type=OutputType.PREDICTION,
        )
        graph = ModelGraph()
        graph.add_node(node)

        model = DecisionTreeRegressor()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        model.fit(X, y)

        fold = CVFold(fold_idx=0, train_indices=np.array([0]), val_indices=np.array([1]))
        fr = FoldResult(fold=fold, model=model, val_predictions=np.array([1.0]), val_score=-0.1)
        cv = CVResult(fold_results=[fr], oof_predictions=np.array([0.0, 1.0]), node_name="tree")

        fg = FittedGraph(
            graph=graph,
            fitted_nodes={"tree": FittedNode(node=node, cv_result=cv, best_params={})},
            tuning_config=TuningConfig(),
            total_time=1.0,
        )
        assert fg.metadata is None

    def test_save_load_without_metadata(self, tmp_path):
        from sklearn_meta.core.data.cv import CVFold, CVResult, FoldResult

        node = ModelNode(
            name="tree",
            estimator_class=DecisionTreeRegressor,
            output_type=OutputType.PREDICTION,
        )
        graph = ModelGraph()
        graph.add_node(node)

        model = DecisionTreeRegressor()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        model.fit(X, y)

        fold = CVFold(fold_idx=0, train_indices=np.array([0]), val_indices=np.array([1]))
        fr = FoldResult(fold=fold, model=model, val_predictions=np.array([1.0]), val_score=-0.1)
        cv = CVResult(fold_results=[fr], oof_predictions=np.array([0.0, 1.0]), node_name="tree")

        fg = FittedGraph(
            graph=graph,
            fitted_nodes={"tree": FittedNode(node=node, cv_result=cv, best_params={})},
            tuning_config=TuningConfig(),
            total_time=1.0,
        )

        fg.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")
        assert loaded.metadata is None


class TestDataHashDeterministic:
    """Test that data_hash is deterministic for same input."""

    def test_same_data_same_hash(self):
        ctx1 = _make_simple_ctx(seed=42)
        ctx2 = _make_simple_ctx(seed=42)
        fg1 = _fit_simple_graph(ctx=ctx1)
        fg2 = _fit_simple_graph(ctx=ctx2)
        assert fg1.metadata.data_hash == fg2.metadata.data_hash

    def test_different_data_different_hash(self):
        ctx1 = _make_simple_ctx(seed=42)
        ctx2 = _make_simple_ctx(seed=99)
        fg1 = _fit_simple_graph(ctx=ctx1)
        fg2 = _fit_simple_graph(ctx=ctx2)
        assert fg1.metadata.data_hash != fg2.metadata.data_hash


class TestTotalTrials:
    """Test total_trials counts correctly."""

    def test_no_tuning_zero_trials(self):
        fg = _fit_simple_graph()
        # With OptimizationStrategy.NONE, no optimization_result is set
        assert fg.metadata.total_trials == 0
