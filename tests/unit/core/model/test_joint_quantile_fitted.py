"""Tests for JointQuantileFittedGraph save/load serialization."""

import json

import numpy as np
import pandas as pd
import pytest

from sklearn_meta.core.model.joint_quantile_fitted import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    JointQuantileFittedGraph,
)
from sklearn_meta.core.model.joint_quantile_graph import JointQuantileConfig, JointQuantileGraph
from sklearn_meta.core.model.quantile_sampler import QuantileSampler, SamplingStrategy
from sklearn_meta.core.tuning.joint_quantile_orchestrator import FittedQuantileNode


# =============================================================================
# Mock classes
# =============================================================================


class MockQuantileRegressor:
    """Mock quantile regressor for testing."""

    def __init__(
        self,
        objective="reg:squarederror",
        quantile_alpha=0.5,
        n_estimators=100,
        max_depth=6,
        **kwargs,
    ):
        self.objective = objective
        self.quantile_alpha = quantile_alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._fitted = False

    def fit(self, X, y, **fit_params):
        self._fitted = True
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)
        return self

    def predict(self, X):
        offset = (self.quantile_alpha - 0.5) * self._y_std * 2
        return np.full(len(X), self._y_mean + offset)


# =============================================================================
# Fixtures
# =============================================================================


def _make_fitted_node(property_name, quantile_levels):
    """Create a FittedQuantileNode with fitted mock models."""
    config = JointQuantileConfig(
        property_names=[property_name],
        estimator_class=MockQuantileRegressor,
        quantile_levels=quantile_levels,
    )
    graph = JointQuantileGraph(config)
    node = graph.get_quantile_node(property_name)

    np.random.seed(hash(property_name) % 2**31)
    y_train = np.random.randn(20)

    models = {}
    for tau in quantile_levels:
        m = MockQuantileRegressor(quantile_alpha=tau)
        m.fit(np.random.randn(20, 5), y_train)
        models[tau] = [m]

    return FittedQuantileNode(
        node=node,
        quantile_models=models,
        oof_quantile_predictions=np.random.randn(20, len(quantile_levels)),
        best_params={"max_depth": 5},
    )


@pytest.fixture
def quantile_levels():
    return [0.1, 0.5, 0.9]


@pytest.fixture
def property_names():
    return ["price", "volume", "volatility"]


@pytest.fixture
def fitted_graph(property_names, quantile_levels):
    """Create a JointQuantileFittedGraph with 3 properties."""
    config = JointQuantileConfig(
        property_names=property_names,
        estimator_class=MockQuantileRegressor,
        quantile_levels=quantile_levels,
        n_inference_samples=100,
        random_state=42,
    )
    graph = JointQuantileGraph(config)

    fitted_nodes = {}
    for prop_name in property_names:
        fitted_nodes[prop_name] = _make_fitted_node(prop_name, quantile_levels)

    sampler = graph.create_quantile_sampler()
    return JointQuantileFittedGraph(
        graph=graph,
        fitted_nodes=fitted_nodes,
        quantile_sampler=sampler,
    )


@pytest.fixture
def test_X():
    np.random.seed(99)
    return pd.DataFrame(
        np.random.randn(5, 5),
        columns=[f"feature_{i}" for i in range(5)],
    )


# =============================================================================
# Tests
# =============================================================================


class TestFittedGraphSaveLoad:
    """Tests for JointQuantileFittedGraph save/load."""

    def test_fitted_graph_save_load_roundtrip(self, fitted_graph, test_X, tmp_path):
        """Full save/load cycle should produce matching predictions."""
        save_dir = tmp_path / "model"
        fitted_graph.save(save_dir)

        loaded = JointQuantileFittedGraph.load(save_dir)

        original_medians = fitted_graph.predict_median(test_X)
        loaded_medians = loaded.predict_median(test_X)
        np.testing.assert_array_almost_equal(original_medians, loaded_medians)

        assert loaded.property_order == fitted_graph.property_order
        assert loaded.quantile_levels == fitted_graph.quantile_levels

    def test_fitted_graph_save_creates_correct_files(
        self, fitted_graph, property_names, tmp_path
    ):
        """Save should create manifest + one .joblib per property."""
        save_dir = tmp_path / "model"
        fitted_graph.save(save_dir)

        assert (save_dir / MANIFEST_FILENAME).exists()
        for prop_name in property_names:
            assert (save_dir / f"{prop_name}.joblib").exists()

    def test_fitted_graph_load_manifest_structure(self, fitted_graph, tmp_path):
        """Manifest JSON should have the expected schema."""
        save_dir = tmp_path / "model"
        fitted_graph.save(save_dir)

        with open(save_dir / MANIFEST_FILENAME) as f:
            manifest = json.load(f)

        assert manifest["version"] == MANIFEST_VERSION
        assert manifest["property_order"] == ["price", "volume", "volatility"]
        assert manifest["quantile_levels"] == [0.1, 0.5, 0.9]
        assert manifest["sampling_strategy"] == "linear_interpolation"
        assert manifest["n_inference_samples"] == 100
        assert manifest["random_state"] == 42
        assert set(manifest["node_files"].keys()) == {
            "price",
            "volume",
            "volatility",
        }

    def test_fitted_graph_node_swap_workflow(self, fitted_graph, test_X, tmp_path):
        """Load, swap one node, and verify inference still works."""
        save_dir = tmp_path / "model"
        fitted_graph.save(save_dir)

        loaded = JointQuantileFittedGraph.load(save_dir)

        # Create a replacement node for "volume"
        new_volume_node = _make_fitted_node("volume", [0.1, 0.5, 0.9])
        loaded.fitted_nodes["volume"] = new_volume_node

        # Inference should still work (no crash)
        result = loaded.predict_median(test_X)
        assert result.shape == (5, 3)

    def test_from_fit_result_unchanged(self, property_names, quantile_levels):
        """Existing from_fit_result construction path should still work."""
        from sklearn_meta.core.tuning.joint_quantile_orchestrator import (
            JointQuantileFitResult,
        )
        from sklearn_meta.core.tuning.orchestrator import TuningConfig
        from sklearn_meta.core.tuning.strategy import OptimizationStrategy
        from sklearn_meta.core.data.cv import CVConfig, CVStrategy

        config = JointQuantileConfig(
            property_names=property_names,
            estimator_class=MockQuantileRegressor,
            quantile_levels=quantile_levels,
            n_inference_samples=100,
            random_state=42,
        )
        graph = JointQuantileGraph(config)

        fitted_nodes = {
            prop: _make_fitted_node(prop, quantile_levels)
            for prop in property_names
        }

        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="neg_mean_squared_error",
            greater_is_better=False,
            verbose=0,
            cv_config=CVConfig(
                n_splits=3,
                strategy=CVStrategy.RANDOM,
                shuffle=True,
                random_state=42,
            ),
        )

        fit_result = JointQuantileFitResult(
            graph=graph,
            fitted_nodes=fitted_nodes,
            tuning_config=tuning_config,
            total_time=1.0,
        )

        fitted_graph = JointQuantileFittedGraph.from_fit_result(fit_result)
        assert fitted_graph.property_order == property_names
        assert fitted_graph.quantile_levels == quantile_levels

        X = pd.DataFrame(
            np.random.randn(3, 5),
            columns=[f"feature_{i}" for i in range(5)],
        )
        result = fitted_graph.predict_median(X)
        assert result.shape == (3, 3)
