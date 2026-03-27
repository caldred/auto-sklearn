"""Tests for TrainingRun.predict() and predict_proba() convenience methods."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sklearn_meta.artifacts.training import TrainingRun, NodeRunResult, RunMetadata
from sklearn_meta.artifacts.inference import InferenceGraph
from sklearn_meta.runtime.config import RunConfig, TuningConfig, CVConfig, CVResult, FoldResult, CVFold
from sklearn_meta.spec.graph import GraphSpec


def _make_training_run(graph=None):
    """Create a minimal TrainingRun for testing."""
    if graph is None:
        graph = MagicMock(spec=GraphSpec)

    fold = CVFold(fold_idx=0, train_indices=np.array([0, 1]), val_indices=np.array([2, 3]))
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.1, 0.2])
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.8, 0.2]])

    fold_result = FoldResult(fold=fold, model=mock_model, val_predictions=np.array([0.1, 0.2]), val_score=0.9)
    cv_result = CVResult(fold_results=[fold_result], oof_predictions=np.array([0.1, 0.2, 0.3, 0.4]), node_name="m")

    node_result = NodeRunResult(node_name="m", cv_result=cv_result, best_params={"x": 1})

    metadata = RunMetadata(
        timestamp="2024-01-01T00:00:00",
        sklearn_meta_version="0.2.0",
        data_shape=(4, 2),
        feature_names=["a", "b"],
        cv_config=None,
        tuning_config_summary={},
        total_trials=1,
        data_hash=None,
        random_state=42,
    )

    config = RunConfig(
        cv=CVConfig(n_splits=2),
        tuning=TuningConfig(metric="accuracy", greater_is_better=True),
    )

    return TrainingRun(
        graph=graph,
        config=config,
        node_results={"m": node_result},
        metadata=metadata,
        total_time=1.0,
    )


class TestTrainingRunPredict:
    """TrainingRun.predict() compiles and delegates to InferenceGraph."""

    def test_predict_delegates_to_inference_graph(self):
        run = _make_training_run()
        mock_inference = MagicMock(spec=InferenceGraph)
        mock_inference.predict.return_value = np.array([0.1, 0.2])
        run._inference_graph = mock_inference

        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = run.predict(X)

        mock_inference.predict.assert_called_once()
        np.testing.assert_array_equal(result, [0.1, 0.2])

    def test_predict_proba_delegates(self):
        run = _make_training_run()
        mock_inference = MagicMock(spec=InferenceGraph)
        mock_inference.predict_proba.return_value = np.array([[0.9, 0.1]])
        run._inference_graph = mock_inference

        X = pd.DataFrame({"a": [1], "b": [2]})
        run.predict_proba(X)

        mock_inference.predict_proba.assert_called_once()

    def test_predict_caches_inference_graph(self):
        run = _make_training_run()
        mock_inference = MagicMock(spec=InferenceGraph)
        mock_inference.predict.return_value = np.array([0.1])

        with patch.object(run, "compile_inference", return_value=mock_inference) as mock_compile:
            X = pd.DataFrame({"a": [1], "b": [2]})
            run.predict(X)
            run.predict(X)

            mock_compile.assert_called_once()

    def test_predict_with_node_name(self):
        run = _make_training_run()
        mock_inference = MagicMock(spec=InferenceGraph)
        mock_inference.predict.return_value = np.array([0.5])
        run._inference_graph = mock_inference

        X = pd.DataFrame({"a": [1], "b": [2]})
        run.predict(X, node_name="m")

        mock_inference.predict.assert_called_once_with(X, node_name="m")


class TestTrainingRunPredictQuantileError:
    """Quantile graphs raise a helpful TypeError."""

    def test_predict_raises_for_quantile_graph(self):
        from sklearn_meta.spec.quantile import JointQuantileGraphSpec

        graph = MagicMock(spec=JointQuantileGraphSpec)
        run = _make_training_run(graph=graph)

        X = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(TypeError, match="quantile"):
            run.predict(X)

    def test_predict_proba_raises_for_quantile_graph(self):
        from sklearn_meta.spec.quantile import JointQuantileGraphSpec

        graph = MagicMock(spec=JointQuantileGraphSpec)
        run = _make_training_run(graph=graph)

        X = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(TypeError, match="quantile"):
            run.predict_proba(X)
