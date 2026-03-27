"""Tests for the fit() convenience function overloads."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from sklearn_meta import fit, GraphSpec, DataView, RunConfig
from sklearn_meta.runtime.config import TuningConfig, CVConfig


@pytest.fixture
def dummy_graph():
    """A minimal valid GraphSpec."""
    graph = MagicMock(spec=GraphSpec)
    return graph


@pytest.fixture
def dummy_config():
    return RunConfig(
        cv=CVConfig(n_splits=2),
        tuning=TuningConfig(metric="accuracy", greater_is_better=True),
    )


@pytest.fixture
def sample_xy():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y = np.array([0, 1, 0, 1])
    return X, y


class TestFitWithDataView:
    """Original DataView path is unchanged."""

    @patch("sklearn_meta.GraphRunner")
    def test_dataview_path(self, MockRunner, dummy_graph, dummy_config, sample_xy):
        X, y = sample_xy
        data = DataView.from_Xy(X, y)
        mock_runner_inst = MockRunner.return_value
        mock_runner_inst.fit.return_value = MagicMock()

        fit(dummy_graph, data, dummy_config)

        mock_runner_inst.fit.assert_called_once_with(dummy_graph, data, dummy_config)

    def test_dataview_rejects_groups_kwarg(self, dummy_graph, dummy_config, sample_xy):
        X, y = sample_xy
        data = DataView.from_Xy(X, y)
        with pytest.raises(TypeError, match="groups"):
            fit(dummy_graph, data, dummy_config, groups=np.array([0, 0, 1, 1]))

    def test_dataview_rejects_aux_kwarg(self, dummy_graph, dummy_config, sample_xy):
        X, y = sample_xy
        data = DataView.from_Xy(X, y)
        with pytest.raises(TypeError, match="groups"):
            fit(dummy_graph, data, dummy_config, base_margin=np.zeros(4))


class TestFitWithRawXy:
    """New DataFrame + y path."""

class TestFitTypeErrors:
    """Helpful error messages for common mistakes."""

    def test_string_raises_type_error(self, dummy_graph, dummy_config):
        with pytest.raises(TypeError, match="array-like features"):
            fit(dummy_graph, "not_data", None, dummy_config)
