"""Tests for ArtifactStore."""

import pytest
import json
import pickle
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from auto_sklearn.persistence.store import ArtifactStore, ArtifactMetadata


class MockModel:
    """Mock model for testing."""

    def __init__(self, value=42):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


class TestArtifactMetadata:
    """Tests for ArtifactMetadata dataclass."""

    def test_create_metadata(self):
        """Verify metadata can be created."""
        metadata = ArtifactMetadata(
            artifact_id="test_123",
            artifact_type="model",
            created_at="2024-01-01T00:00:00",
            node_name="rf_classifier",
            params={"n_estimators": 100},
            metrics={"accuracy": 0.95},
            tags={"experiment": "exp1"},
        )

        assert metadata.artifact_id == "test_123"
        assert metadata.artifact_type == "model"
        assert metadata.node_name == "rf_classifier"

    def test_metadata_defaults(self):
        """Verify metadata has sensible defaults."""
        metadata = ArtifactMetadata(
            artifact_id="test",
            artifact_type="model",
            created_at="2024-01-01",
        )

        assert metadata.node_name is None
        assert metadata.params == {}
        assert metadata.metrics == {}
        assert metadata.tags == {}


class TestArtifactStoreInit:
    """Tests for ArtifactStore initialization."""

    def test_default_path(self, tmp_path, monkeypatch):
        """Verify default path is used."""
        monkeypatch.chdir(tmp_path)
        store = ArtifactStore()

        assert store.base_path == Path(".auto_sklearn_artifacts")

    def test_custom_path(self, tmp_path):
        """Verify custom path is used."""
        custom_path = tmp_path / "custom_store"
        store = ArtifactStore(base_path=str(custom_path))

        assert store.base_path == custom_path

    def test_creates_directories(self, tmp_path):
        """Verify required directories are created."""
        store_path = tmp_path / "test_store"
        store = ArtifactStore(base_path=str(store_path))

        assert (store_path / "models").exists()
        assert (store_path / "params").exists()
        assert (store_path / "graphs").exists()
        assert (store_path / "metadata").exists()

    def test_repr(self, tmp_path):
        """Verify repr includes base path."""
        store = ArtifactStore(base_path=str(tmp_path))

        assert "ArtifactStore" in repr(store)


class TestArtifactStoreSaveModel:
    """Tests for ArtifactStore.save_model."""

    def test_save_model_returns_id(self, tmp_path):
        """Verify save_model returns artifact ID."""
        store = ArtifactStore(base_path=str(tmp_path))
        model = MockModel()

        artifact_id = store.save_model(model, "test_node")

        assert artifact_id is not None
        assert isinstance(artifact_id, str)
        assert "test_node" in artifact_id

    def test_save_model_creates_file(self, tmp_path):
        """Verify model file is created."""
        store = ArtifactStore(base_path=str(tmp_path))
        model = MockModel()

        artifact_id = store.save_model(model, "test_node")

        model_path = tmp_path / "models" / f"{artifact_id}.pkl"
        assert model_path.exists()

    def test_save_model_with_params(self, tmp_path):
        """Verify params are saved in metadata."""
        store = ArtifactStore(base_path=str(tmp_path))
        model = MockModel()
        params = {"n_estimators": 100, "max_depth": 5}

        artifact_id = store.save_model(model, "test_node", params=params)

        meta_path = tmp_path / "metadata" / f"{artifact_id}.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["params"] == params

    def test_save_model_with_metrics(self, tmp_path):
        """Verify metrics are saved in metadata."""
        store = ArtifactStore(base_path=str(tmp_path))
        model = MockModel()
        metrics = {"accuracy": 0.95, "f1": 0.92}

        artifact_id = store.save_model(model, "test_node", metrics=metrics)

        meta_path = tmp_path / "metadata" / f"{artifact_id}.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["metrics"] == metrics

    def test_save_model_with_tags(self, tmp_path):
        """Verify tags are saved in metadata."""
        store = ArtifactStore(base_path=str(tmp_path))
        model = MockModel()
        tags = {"experiment": "exp1", "version": "1.0"}

        artifact_id = store.save_model(model, "test_node", tags=tags)

        meta_path = tmp_path / "metadata" / f"{artifact_id}.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["tags"] == tags

    def test_save_model_with_fold_idx(self, tmp_path):
        """Verify fold_idx is included in artifact ID."""
        store = ArtifactStore(base_path=str(tmp_path))
        model = MockModel()

        artifact_id = store.save_model(model, "test_node", fold_idx=3)

        assert "_3_" in artifact_id


class TestArtifactStoreLoadModel:
    """Tests for ArtifactStore.load_model."""

    def test_load_model_returns_model(self, tmp_path):
        """Verify load_model returns saved model."""
        store = ArtifactStore(base_path=str(tmp_path))
        original = MockModel(value=99)

        artifact_id = store.save_model(original, "test_node")
        loaded = store.load_model(artifact_id)

        assert isinstance(loaded, MockModel)
        assert loaded.value == 99

    def test_load_nonexistent_raises(self, tmp_path):
        """Verify loading nonexistent artifact raises."""
        store = ArtifactStore(base_path=str(tmp_path))

        with pytest.raises(FileNotFoundError, match="not found"):
            store.load_model("nonexistent_id")

    def test_load_model_preserves_state(self, tmp_path):
        """Verify loaded model has correct state."""
        store = ArtifactStore(base_path=str(tmp_path))

        # Use sklearn model if available
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        model = LogisticRegression(max_iter=1000, random_state=42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)

        artifact_id = store.save_model(model, "lr")
        loaded = store.load_model(artifact_id)

        # Predictions should match
        original_preds = model.predict(X)
        loaded_preds = loaded.predict(X)

        np.testing.assert_array_equal(original_preds, loaded_preds)


class TestArtifactStoreSaveParams:
    """Tests for ArtifactStore.save_params."""

    def test_save_params_returns_id(self, tmp_path):
        """Verify save_params returns artifact ID."""
        store = ArtifactStore(base_path=str(tmp_path))
        params = {"n_estimators": 100}

        artifact_id = store.save_params(params, "test_node")

        assert artifact_id is not None
        assert "test_node" in artifact_id

    def test_save_params_creates_file(self, tmp_path):
        """Verify params file is created."""
        store = ArtifactStore(base_path=str(tmp_path))
        params = {"n_estimators": 100}

        artifact_id = store.save_params(params, "test_node")

        params_path = tmp_path / "params" / f"{artifact_id}.json"
        assert params_path.exists()

    def test_save_params_with_description(self, tmp_path):
        """Verify description is saved."""
        store = ArtifactStore(base_path=str(tmp_path))
        params = {"n_estimators": 100}

        artifact_id = store.save_params(params, "test_node", description="Best params")

        params_path = tmp_path / "params" / f"{artifact_id}.json"
        with open(params_path) as f:
            data = json.load(f)

        assert data["description"] == "Best params"


class TestArtifactStoreLoadParams:
    """Tests for ArtifactStore.load_params."""

    def test_load_params_returns_dict(self, tmp_path):
        """Verify load_params returns saved params."""
        store = ArtifactStore(base_path=str(tmp_path))
        original = {"n_estimators": 100, "max_depth": 5}

        artifact_id = store.save_params(original, "test_node")
        loaded = store.load_params(artifact_id)

        assert loaded == original

    def test_load_nonexistent_raises(self, tmp_path):
        """Verify loading nonexistent params raises."""
        store = ArtifactStore(base_path=str(tmp_path))

        with pytest.raises(FileNotFoundError, match="not found"):
            store.load_params("nonexistent_id")


class TestArtifactStoreListArtifacts:
    """Tests for ArtifactStore.list_artifacts."""

    def test_list_empty(self, tmp_path):
        """Verify empty store returns empty list."""
        store = ArtifactStore(base_path=str(tmp_path))

        artifacts = store.list_artifacts()

        assert artifacts == []

    def test_list_all_artifacts(self, tmp_path):
        """Verify all artifacts are listed."""
        store = ArtifactStore(base_path=str(tmp_path))

        store.save_model(MockModel(), "node1")
        store.save_model(MockModel(), "node2")

        artifacts = store.list_artifacts()

        assert len(artifacts) == 2

    def test_list_filter_by_type(self, tmp_path):
        """Verify filtering by artifact type."""
        store = ArtifactStore(base_path=str(tmp_path))

        store.save_model(MockModel(), "node1")
        # save_params doesn't create metadata, so we need to check models only

        artifacts = store.list_artifacts(artifact_type="model")

        assert len(artifacts) == 1
        assert artifacts[0].artifact_type == "model"

    def test_list_filter_by_node_name(self, tmp_path):
        """Verify filtering by node name."""
        store = ArtifactStore(base_path=str(tmp_path))

        store.save_model(MockModel(), "node1")
        store.save_model(MockModel(), "node2")

        artifacts = store.list_artifacts(node_name="node1")

        assert len(artifacts) == 1
        assert artifacts[0].node_name == "node1"

    def test_list_returns_metadata(self, tmp_path):
        """Verify list returns ArtifactMetadata objects."""
        store = ArtifactStore(base_path=str(tmp_path))

        store.save_model(MockModel(), "node1", params={"a": 1})

        artifacts = store.list_artifacts()

        assert len(artifacts) == 1
        assert isinstance(artifacts[0], ArtifactMetadata)
        assert artifacts[0].params == {"a": 1}


class TestArtifactStoreDeleteArtifact:
    """Tests for ArtifactStore.delete_artifact."""

    def test_delete_existing(self, tmp_path):
        """Verify deleting existing artifact works."""
        store = ArtifactStore(base_path=str(tmp_path))
        artifact_id = store.save_model(MockModel(), "node1")

        result = store.delete_artifact(artifact_id)

        assert result is True
        assert not (tmp_path / "models" / f"{artifact_id}.pkl").exists()
        assert not (tmp_path / "metadata" / f"{artifact_id}.json").exists()

    def test_delete_nonexistent(self, tmp_path):
        """Verify deleting nonexistent returns False."""
        store = ArtifactStore(base_path=str(tmp_path))

        result = store.delete_artifact("nonexistent_id")

        assert result is False


class TestArtifactStoreSaveFittedNode:
    """Tests for ArtifactStore.save_fitted_node."""

    def test_save_fitted_node_returns_ids(self, tmp_path):
        """Verify save_fitted_node returns artifact IDs."""
        store = ArtifactStore(base_path=str(tmp_path))

        # Create mock fitted node
        from auto_sklearn.core.data.cv import CVFold, FoldResult, CVResult

        fold = CVFold(
            fold_idx=0,
            repeat_idx=0,
            train_indices=[0, 1, 2],
            val_indices=[3, 4],
        )

        fold_result = FoldResult(
            fold=fold,
            model=MockModel(),
            val_predictions=np.array([0, 1]),
            val_score=0.9,
        )

        cv_result = CVResult(
            fold_results=[fold_result],
            oof_predictions=np.array([0, 0, 0, 0, 1]),
            node_name="test_node",
        )

        fitted_node = MagicMock()
        fitted_node.node.name = "test_node"
        fitted_node.cv_result = cv_result
        fitted_node.models = [MockModel()]
        fitted_node.best_params = {"n_estimators": 100}

        artifact_ids = store.save_fitted_node(fitted_node)

        assert len(artifact_ids) == 1
        assert "test_node" in artifact_ids[0]


class TestArtifactStoreSaveFittedGraph:
    """Tests for ArtifactStore.save_fitted_graph."""

    def test_save_fitted_graph_returns_id(self, tmp_path):
        """Verify save_fitted_graph returns graph ID."""
        store = ArtifactStore(base_path=str(tmp_path))

        # Create mock fitted graph
        from auto_sklearn.core.data.cv import CVFold, FoldResult, CVResult
        from auto_sklearn.core.tuning.strategy import OptimizationStrategy

        fold = CVFold(
            fold_idx=0,
            repeat_idx=0,
            train_indices=[0, 1, 2],
            val_indices=[3, 4],
        )

        fold_result = FoldResult(
            fold=fold,
            model=MockModel(),
            val_predictions=np.array([0, 1]),
            val_score=0.9,
        )

        cv_result = CVResult(
            fold_results=[fold_result],
            oof_predictions=np.array([0, 0, 0, 0, 1]),
            node_name="test_node",
        )

        fitted_node = MagicMock()
        fitted_node.node.name = "test_node"
        fitted_node.cv_result = cv_result
        fitted_node.models = [MockModel()]
        fitted_node.best_params = {"n_estimators": 100}

        tuning_config = MagicMock()
        tuning_config.strategy = OptimizationStrategy.NONE
        tuning_config.n_trials = 10
        tuning_config.metric = "accuracy"

        fitted_graph = MagicMock()
        fitted_graph.fitted_nodes = {"test_node": fitted_node}
        fitted_graph.total_time = 10.5
        fitted_graph.tuning_config = tuning_config

        graph_id = store.save_fitted_graph(fitted_graph, "my_experiment")

        assert "graph_my_experiment" in graph_id
        assert (tmp_path / "graphs" / f"{graph_id}.json").exists()


class TestArtifactStoreGenerateId:
    """Tests for artifact ID generation."""

    def test_generate_id_unique(self, tmp_path):
        """Verify generated IDs are unique."""
        store = ArtifactStore(base_path=str(tmp_path))

        ids = set()
        for i in range(10):
            artifact_id = store._generate_id("node", i, {"a": i})
            ids.add(artifact_id)

        assert len(ids) == 10  # All unique

    def test_generate_id_includes_node_name(self, tmp_path):
        """Verify ID includes node name."""
        store = ArtifactStore(base_path=str(tmp_path))

        artifact_id = store._generate_id("my_node", 0, {})

        assert "my_node" in artifact_id

    def test_generate_id_includes_fold_idx(self, tmp_path):
        """Verify ID includes fold index."""
        store = ArtifactStore(base_path=str(tmp_path))

        artifact_id = store._generate_id("node", 5, {})

        assert "_5_" in artifact_id
