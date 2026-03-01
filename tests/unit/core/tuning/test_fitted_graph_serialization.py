"""Tests for FittedGraph save/load serialization."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sklearn_meta.core.data.cv import CVFold, CVResult, FoldResult
from sklearn_meta.core.model.dependency import DependencyEdge, DependencyType
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.model.node import ModelNode, OutputType
from sklearn_meta.core.tuning.orchestrator import (
    FittedGraph,
    FittedNode,
    TuningConfig,
)
from sklearn_meta.search.backends.base import OptimizationResult, TrialResult
from sklearn_meta.core.tuning.strategy import OptimizationStrategy


class _OuterEstimator:
    class InnerEstimator:
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return np.zeros(len(X))


def _make_single_node_fitted_graph(n_folds: int = 3) -> tuple:
    """Create a simple single-node FittedGraph with real sklearn models.

    Returns:
        Tuple of (FittedGraph, X_test DataFrame for predictions).
    """
    rng = np.random.RandomState(42)
    n_train = 100
    n_test = 20

    X_train = pd.DataFrame({
        "f1": rng.randn(n_train),
        "f2": rng.randn(n_train),
    })
    y_train = X_train["f1"] * 2 + X_train["f2"] + rng.randn(n_train) * 0.1
    X_test = pd.DataFrame({
        "f1": rng.randn(n_test),
        "f2": rng.randn(n_test),
    })

    node = ModelNode(
        name="tree",
        estimator_class=DecisionTreeRegressor,
        output_type=OutputType.PREDICTION,
        fixed_params={"max_depth": 5, "random_state": 42},
        description="A decision tree",
    )

    graph = ModelGraph()
    graph.add_node(node)

    # Train fold models
    fold_results = []
    fold_size = n_train // n_folds
    for i in range(n_folds):
        val_idx = np.arange(i * fold_size, (i + 1) * fold_size)
        train_idx = np.setdiff1d(np.arange(n_train), val_idx)

        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])

        val_preds = model.predict(X_train.iloc[val_idx])

        fold = CVFold(fold_idx=i, train_indices=train_idx, val_indices=val_idx)
        fold_results.append(FoldResult(
            fold=fold,
            model=model,
            val_predictions=val_preds,
            val_score=-0.1 * (i + 1),
        ))

    oof = np.zeros(n_train)
    for fr in fold_results:
        oof[fr.fold.val_indices] = fr.val_predictions

    cv_result = CVResult(
        fold_results=fold_results,
        oof_predictions=oof,
        node_name="tree",
    )

    fitted_node = FittedNode(
        node=node,
        cv_result=cv_result,
        best_params={"max_depth": 5, "random_state": 42},
        selected_features=None,
    )

    tuning_config = TuningConfig(
        strategy=OptimizationStrategy.LAYER_BY_LAYER,
        n_trials=50,
        metric="neg_mean_squared_error",
        greater_is_better=False,
    )

    fitted_graph = FittedGraph(
        graph=graph,
        fitted_nodes={"tree": fitted_node},
        tuning_config=tuning_config,
        total_time=12.5,
    )

    return fitted_graph, X_test


def _make_stacking_fitted_graph(n_folds: int = 3) -> tuple:
    """Create a 2-node stacking FittedGraph with real sklearn models.

    Returns:
        Tuple of (FittedGraph, X_test DataFrame for predictions).
    """
    rng = np.random.RandomState(42)
    n_train = 100
    n_test = 20

    X_train = pd.DataFrame({
        "f1": rng.randn(n_train),
        "f2": rng.randn(n_train),
    })
    y_train = X_train["f1"] * 2 + X_train["f2"] + rng.randn(n_train) * 0.1
    X_test = pd.DataFrame({
        "f1": rng.randn(n_test),
        "f2": rng.randn(n_test),
    })

    # Base node
    base_node = ModelNode(
        name="base",
        estimator_class=DecisionTreeRegressor,
        output_type=OutputType.PREDICTION,
        fixed_params={"max_depth": 3, "random_state": 0},
    )

    # Stacker node
    stacker_node = ModelNode(
        name="stacker",
        estimator_class=DecisionTreeRegressor,
        output_type=OutputType.PREDICTION,
        fixed_params={"max_depth": 2, "random_state": 1},
    )

    graph = ModelGraph()
    graph.add_node(base_node)
    graph.add_node(stacker_node)
    graph.add_edge(DependencyEdge(
        source="base",
        target="stacker",
        dep_type=DependencyType.PREDICTION,
    ))

    # Train base fold models
    fold_size = n_train // n_folds
    base_oof = np.zeros(n_train)
    base_fold_results = []
    for i in range(n_folds):
        val_idx = np.arange(i * fold_size, (i + 1) * fold_size)
        train_idx = np.setdiff1d(np.arange(n_train), val_idx)

        model = DecisionTreeRegressor(max_depth=3, random_state=0)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])

        val_preds = model.predict(X_train.iloc[val_idx])
        base_oof[val_idx] = val_preds

        fold = CVFold(fold_idx=i, train_indices=train_idx, val_indices=val_idx)
        base_fold_results.append(FoldResult(
            fold=fold, model=model, val_predictions=val_preds, val_score=-0.2,
        ))

    base_cv = CVResult(fold_results=base_fold_results, oof_predictions=base_oof, node_name="base")
    base_fitted = FittedNode(node=base_node, cv_result=base_cv, best_params={"max_depth": 3, "random_state": 0})

    # Train stacker fold models with base OOF as feature
    X_train_stacker = X_train.copy()
    X_train_stacker["pred_base"] = base_oof

    stacker_oof = np.zeros(n_train)
    stacker_fold_results = []
    for i in range(n_folds):
        val_idx = np.arange(i * fold_size, (i + 1) * fold_size)
        train_idx = np.setdiff1d(np.arange(n_train), val_idx)

        model = DecisionTreeRegressor(max_depth=2, random_state=1)
        model.fit(X_train_stacker.iloc[train_idx], y_train.iloc[train_idx])

        val_preds = model.predict(X_train_stacker.iloc[val_idx])
        stacker_oof[val_idx] = val_preds

        fold = CVFold(fold_idx=i, train_indices=train_idx, val_indices=val_idx)
        stacker_fold_results.append(FoldResult(
            fold=fold, model=model, val_predictions=val_preds, val_score=-0.15,
        ))

    stacker_cv = CVResult(fold_results=stacker_fold_results, oof_predictions=stacker_oof, node_name="stacker")
    stacker_fitted = FittedNode(
        node=stacker_node,
        cv_result=stacker_cv,
        best_params={"max_depth": 2, "random_state": 1},
        selected_features=["f1", "f2", "pred_base"],
    )

    tuning_config = TuningConfig(
        strategy=OptimizationStrategy.LAYER_BY_LAYER,
        n_trials=100,
        metric="neg_mean_squared_error",
        greater_is_better=False,
    )

    fitted_graph = FittedGraph(
        graph=graph,
        fitted_nodes={"base": base_fitted, "stacker": stacker_fitted},
        tuning_config=tuning_config,
        total_time=25.0,
    )

    return fitted_graph, X_test


def _make_conditional_inference_graph() -> FittedGraph:
    """Create a graph with one skipped conditional node omitted from fitted_nodes."""
    normal = ModelNode(
        name="normal",
        estimator_class=DecisionTreeRegressor,
        output_type=OutputType.PREDICTION,
        fixed_params={"max_depth": 2, "random_state": 0},
    )
    conditional = ModelNode(
        name="conditional",
        estimator_class=DecisionTreeRegressor,
        output_type=OutputType.PREDICTION,
        condition=lambda ctx: False,
    )

    graph = ModelGraph()
    graph.add_node(normal)
    graph.add_node(conditional)

    model = DecisionTreeRegressor(max_depth=2, random_state=0)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model.fit(X, y)

    fold = CVFold(
        fold_idx=0,
        train_indices=np.array([0, 1]),
        val_indices=np.array([2]),
    )
    cv_result = CVResult(
        fold_results=[
            FoldResult(
                fold=fold,
                model=model,
                val_predictions=np.array([3.0]),
                val_score=-0.1,
            )
        ],
        oof_predictions=np.array([0.0, 0.0, 3.0]),
        node_name="normal",
    )

    return FittedGraph(
        graph=graph,
        fitted_nodes={
            "normal": FittedNode(
                node=normal,
                cv_result=cv_result,
                best_params={"max_depth": 2, "random_state": 0},
            )
        },
        tuning_config=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            metric="neg_mean_squared_error",
            greater_is_better=False,
        ),
        total_time=1.0,
    )


class TestSingleNodeRoundTrip:
    """Test save/load round-trip with a single node graph."""

    def test_predictions_identical(self, tmp_path):
        fitted_graph, X_test = _make_single_node_fitted_graph()

        preds_before = fitted_graph.predict(X_test)
        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")
        preds_after = loaded.predict(X_test)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_graph_structure_matches(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert list(loaded.graph.nodes.keys()) == list(fitted_graph.graph.nodes.keys())
        assert len(loaded.graph.edges) == len(fitted_graph.graph.edges)

    def test_best_params_match(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        for name in fitted_graph.fitted_nodes:
            assert loaded.fitted_nodes[name].best_params == fitted_graph.fitted_nodes[name].best_params

    def test_selected_features_match(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.fitted_nodes["tree"].selected_features is None

    def test_training_artifacts_disabled_by_default(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.training_artifacts_available is False
        with pytest.raises(RuntimeError, match="include_training_artifacts=True"):
            loaded.get_oof_predictions("tree")

    def test_round_trip_with_training_artifacts(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()
        optimization_result = OptimizationResult(
            best_params={"max_depth": 5},
            best_value=-0.05,
            trials=[TrialResult(params={"max_depth": 5}, value=-0.05, trial_id=1)],
            n_trials=1,
            study_name="tree_tuning",
        )
        fitted_graph.fitted_nodes["tree"].optimization_result = optimization_result

        fitted_graph.save(
            tmp_path / "model",
            include_training_artifacts=True,
        )
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.training_artifacts_available is True
        np.testing.assert_array_equal(
            loaded.get_oof_predictions("tree"),
            fitted_graph.get_oof_predictions("tree"),
        )
        loaded_fold = loaded.get_node("tree").cv_result.fold_results[0]
        orig_fold = fitted_graph.get_node("tree").cv_result.fold_results[0]
        np.testing.assert_array_equal(
            loaded_fold.fold.train_indices,
            orig_fold.fold.train_indices,
        )
        np.testing.assert_array_equal(
            loaded_fold.fold.val_indices,
            orig_fold.fold.val_indices,
        )
        assert loaded.get_node("tree").optimization_result is not None
        assert loaded.get_node("tree").optimization_result.n_trials == 1

    def test_save_normalizes_numpy_scalar_params(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()
        fitted_graph.fitted_nodes["tree"].best_params = {"max_depth": np.int64(5)}

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.fitted_nodes["tree"].best_params == {"max_depth": 5}

    def test_total_time_preserved(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.total_time == pytest.approx(12.5)

    def test_tuning_config_preserved(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.tuning_config.strategy == OptimizationStrategy.LAYER_BY_LAYER
        assert loaded.tuning_config.n_trials == 50
        assert loaded.tuning_config.metric == "neg_mean_squared_error"
        assert loaded.tuning_config.greater_is_better is False

    def test_node_metadata_preserved(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        orig_node = fitted_graph.graph.get_node("tree")
        loaded_node = loaded.graph.get_node("tree")

        assert loaded_node.name == orig_node.name
        assert loaded_node.estimator_class == orig_node.estimator_class
        assert loaded_node.output_type == orig_node.output_type
        assert loaded_node.description == orig_node.description

    def test_directory_structure(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph(n_folds=3)

        save_path = tmp_path / "model"
        fitted_graph.save(save_path)

        assert (save_path / "manifest.json").exists()
        assert (save_path / "nodes" / "tree" / "fold_0.joblib").exists()
        assert (save_path / "nodes" / "tree" / "fold_1.joblib").exists()
        assert (save_path / "nodes" / "tree" / "fold_2.joblib").exists()

    def test_manifest_schema(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()

        save_path = tmp_path / "model"
        fitted_graph.save(save_path)

        with open(save_path / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["version"] == 2
        assert manifest["training_artifacts_included"] is False
        assert "graph" in manifest
        assert "nodes" in manifest["graph"]
        assert "edges" in manifest["graph"]
        assert "fitted_nodes" in manifest
        assert "tuning_config" in manifest
        assert "total_time" in manifest


class TestStackingGraphRoundTrip:
    """Test save/load round-trip with a 2-node stacking graph."""

    def test_predictions_identical(self, tmp_path):
        fitted_graph, X_test = _make_stacking_fitted_graph()

        preds_before = fitted_graph.predict(X_test)
        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")
        preds_after = loaded.predict(X_test)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_graph_structure_matches(self, tmp_path):
        fitted_graph, _ = _make_stacking_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert set(loaded.graph.nodes.keys()) == {"base", "stacker"}
        assert len(loaded.graph.edges) == 1
        edge = loaded.graph.edges[0]
        assert edge.source == "base"
        assert edge.target == "stacker"
        assert edge.dep_type == DependencyType.PREDICTION

    def test_selected_features_preserved(self, tmp_path):
        fitted_graph, _ = _make_stacking_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.fitted_nodes["base"].selected_features is None
        assert loaded.fitted_nodes["stacker"].selected_features == ["f1", "f2", "pred_base"]

    def test_best_params_match(self, tmp_path):
        fitted_graph, _ = _make_stacking_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.fitted_nodes["base"].best_params == {"max_depth": 3, "random_state": 0}
        assert loaded.fitted_nodes["stacker"].best_params == {"max_depth": 2, "random_state": 1}

    def test_mean_score_preserved(self, tmp_path):
        fitted_graph, _ = _make_stacking_fitted_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert loaded.fitted_nodes["base"].mean_score == pytest.approx(-0.2)
        assert loaded.fitted_nodes["stacker"].mean_score == pytest.approx(-0.15)


class TestErrorHandling:
    """Test error handling for save/load."""

    def test_missing_manifest(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="manifest.json not found"):
            FittedGraph.load(tmp_path / "nonexistent")

    def test_corrupt_manifest(self, tmp_path):
        save_path = tmp_path / "corrupt"
        save_path.mkdir()
        with open(save_path / "manifest.json", "w") as f:
            f.write("{invalid json")

        with pytest.raises(ValueError, match="Corrupt manifest.json"):
            FittedGraph.load(save_path)

    def test_unsupported_version(self, tmp_path):
        save_path = tmp_path / "bad_version"
        save_path.mkdir()
        manifest = {"version": 99, "graph": {}, "fitted_nodes": {}, "tuning_config": {}}
        with open(save_path / "manifest.json", "w") as f:
            json.dump(manifest, f)

        with pytest.raises(ValueError, match="Unsupported manifest version"):
            FittedGraph.load(save_path)

    def test_missing_fold_file(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph(n_folds=3)

        save_path = tmp_path / "model"
        fitted_graph.save(save_path)

        # Delete one fold file
        os.remove(save_path / "nodes" / "tree" / "fold_1.joblib")

        with pytest.raises(FileNotFoundError, match="Missing fold model file"):
            FittedGraph.load(save_path)

    def test_save_overwrites_existing(self, tmp_path):
        """Saving to an existing directory should work (overwrite)."""
        fitted_graph, X_test = _make_single_node_fitted_graph()

        save_path = tmp_path / "model"
        fitted_graph.save(save_path)
        fitted_graph.save(save_path)  # Save again

        loaded = FittedGraph.load(save_path)
        preds = loaded.predict(X_test)
        assert preds.shape == (20,)


class TestInferenceArtifactSemantics:
    """Tests for inference-first persistence behavior."""

    def test_skipped_conditional_nodes_are_omitted_from_loaded_graph(self, tmp_path):
        fitted_graph = _make_conditional_inference_graph()

        fitted_graph.save(tmp_path / "model")
        loaded = FittedGraph.load(tmp_path / "model")

        assert list(loaded.graph.nodes.keys()) == ["normal"]
        assert "conditional" not in loaded.graph.nodes

    def test_legacy_v1_manifest_loads_as_inference_only(self, tmp_path):
        fitted_graph, _ = _make_single_node_fitted_graph()
        save_path = tmp_path / "legacy_model"
        save_path.mkdir()
        node_dir = save_path / "nodes" / "tree"
        node_dir.mkdir(parents=True)

        for i, model in enumerate(fitted_graph.get_node("tree").models):
            import joblib
            joblib.dump(model, node_dir / f"fold_{i}.joblib")

        legacy_manifest = {
            "version": 1,
            "graph": {
                "nodes": [fitted_graph.graph.get_node("tree").to_dict()],
                "edges": [],
            },
            "fitted_nodes": {
                "tree": {
                    "best_params": {"max_depth": 5, "random_state": 42},
                    "selected_features": None,
                    "mean_score": fitted_graph.get_node("tree").mean_score,
                    "n_folds": len(fitted_graph.get_node("tree").models),
                }
            },
            "tuning_config": {
                "strategy": OptimizationStrategy.LAYER_BY_LAYER.value,
                "n_trials": 50,
                "metric": "neg_mean_squared_error",
                "greater_is_better": False,
            },
            "total_time": 12.5,
        }
        with open(save_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(legacy_manifest, f)

        loaded = FittedGraph.load(save_path)

        assert loaded.training_artifacts_available is False
        with pytest.raises(RuntimeError, match="include_training_artifacts=True"):
            loaded.get_oof_predictions("tree")


class TestModelNodeSerialization:
    """Test ModelNode to_dict/from_dict."""

    def test_round_trip(self):
        node = ModelNode(
            name="test",
            estimator_class=DecisionTreeRegressor,
            output_type=OutputType.PREDICTION,
            fixed_params={"max_depth": 5},
            description="test node",
            plugins=["plugin_a"],
        )

        data = node.to_dict()
        restored = ModelNode.from_dict(data)

        assert restored.name == node.name
        assert restored.estimator_class == node.estimator_class
        assert restored.output_type == node.output_type
        assert restored.fixed_params == node.fixed_params
        assert restored.description == node.description
        assert restored.plugins == node.plugins

    def test_estimator_class_string(self):
        node = ModelNode(
            name="test",
            estimator_class=DecisionTreeRegressor,
        )
        data = node.to_dict()
        assert data["estimator_class"] == "sklearn.tree._classes.DecisionTreeRegressor"

    def test_nested_estimator_class_round_trip(self):
        node = ModelNode(
            name="nested",
            estimator_class=_OuterEstimator.InnerEstimator,
        )

        data = node.to_dict()
        restored = ModelNode.from_dict(data)

        assert restored.estimator_class is _OuterEstimator.InnerEstimator


class TestPredictOutputModes:
    """Test output-type-specific prediction behavior."""

    def test_transform_output_uses_transform(self):
        X = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0],
                "f2": [10.0, 20.0, 30.0],
            }
        )
        model = StandardScaler().fit(X)
        node = ModelNode(
            name="scale",
            estimator_class=StandardScaler,
            output_type=OutputType.TRANSFORM,
        )
        graph = ModelGraph()
        graph.add_node(node)

        fold = CVFold(
            fold_idx=0,
            train_indices=np.array([0, 1]),
            val_indices=np.array([2]),
        )
        cv = CVResult(
            fold_results=[
                FoldResult(
                    fold=fold,
                    model=model,
                    val_predictions=np.array([[0.0, 0.0]]),
                    val_score=0.0,
                )
            ],
            oof_predictions=np.zeros((3, 2)),
            node_name="scale",
        )
        fitted_graph = FittedGraph(
            graph=graph,
            fitted_nodes={
                "scale": FittedNode(node=node, cv_result=cv, best_params={})
            },
            tuning_config=TuningConfig(),
        )

        transformed = fitted_graph.predict(X, node_name="scale")

        np.testing.assert_allclose(transformed, model.transform(X))


class TestDependencyEdgeSerialization:
    """Test DependencyEdge to_dict/from_dict."""

    def test_round_trip(self):
        edge = DependencyEdge(
            source="a",
            target="b",
            dep_type=DependencyType.PREDICTION,
            column_name="pred_a",
        )

        data = edge.to_dict()
        restored = DependencyEdge.from_dict(data)

        assert restored.source == edge.source
        assert restored.target == edge.target
        assert restored.dep_type == edge.dep_type
        assert restored.column_name == edge.column_name

    def test_round_trip_no_column_name(self):
        edge = DependencyEdge(source="x", target="y", dep_type=DependencyType.FEATURE)

        data = edge.to_dict()
        restored = DependencyEdge.from_dict(data)

        assert restored.column_name is None
        assert restored.feature_name == "feat_x"
