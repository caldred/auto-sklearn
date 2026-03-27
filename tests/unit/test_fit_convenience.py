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

    @patch("sklearn_meta.GraphRunner")
    def test_dataview_positional_services(self, MockRunner, dummy_graph, dummy_config, sample_xy):
        X, y = sample_xy
        data = DataView.from_Xy(X, y)
        mock_services = MagicMock()
        mock_runner_inst = MockRunner.return_value
        mock_runner_inst.fit.return_value = MagicMock()

        fit(dummy_graph, data, dummy_config, mock_services)

        MockRunner.assert_called_once_with(mock_services)
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

    @patch("sklearn_meta.GraphRunner")
    def test_raw_xy_path(self, MockRunner, dummy_graph, dummy_config, sample_xy):
        X, y = sample_xy
        mock_runner_inst = MockRunner.return_value
        mock_runner_inst.fit.return_value = MagicMock()

        fit(dummy_graph, X, y, dummy_config)

        call_args = mock_runner_inst.fit.call_args
        passed_data = call_args[0][1]
        assert isinstance(passed_data, DataView)

    @patch("sklearn_meta.GraphRunner")
    def test_raw_xy_with_groups(self, MockRunner, dummy_graph, dummy_config, sample_xy):
        X, y = sample_xy
        groups = np.array([0, 0, 1, 1])
        mock_runner_inst = MockRunner.return_value
        mock_runner_inst.fit.return_value = MagicMock()

        fit(dummy_graph, X, y, dummy_config, groups=groups)

        call_args = mock_runner_inst.fit.call_args
        passed_data = call_args[0][1]
        assert isinstance(passed_data, DataView)
        np.testing.assert_array_equal(passed_data.groups, groups)


class TestFitTypeErrors:
    """Helpful error messages for common mistakes."""

    def test_ndarray_raises_type_error(self, dummy_graph, dummy_config):
        X = np.array([[1, 2], [3, 4]])
        with pytest.raises(TypeError, match="DataFrame"):
            fit(dummy_graph, X, np.array([0, 1]), dummy_config)

    def test_string_raises_type_error(self, dummy_graph, dummy_config):
        with pytest.raises(TypeError, match="DataFrame"):
            fit(dummy_graph, "not_data", None, dummy_config)
