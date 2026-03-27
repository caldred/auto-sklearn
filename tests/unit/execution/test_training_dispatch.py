"""Tests for node-level training dispatch helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from sklearn_meta.data.view import DataView
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.execution.training import (
    LocalTrainingDispatcher,
    NodeTrainingJob,
    NodeTrainingJobRunner,
    NodeTrainingJobBuilder,
    NodeTrainingResult,
    NodeTrainingResultReconstructor,
    validate_dispatchable,
)
from sklearn_meta.runtime.config import CVConfig, CVStrategy, RunConfig, TuningConfig
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.search.backends.base import (
    OptimizationResult,
    SearchBackend,
    TrialResult,
)
from sklearn_meta.search.backends.optuna import OptunaBackend
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec


def _make_regression_context():
    X = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x2": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})
    y = pd.Series([1.0, 2.2, 2.9, 4.1, 5.2, 6.0])
    return DataView.from_Xy(X, y)


def test_job_builder_and_reconstruction_round_trip():
    view = _make_regression_context()
    node = NodeSpec(name="linreg", estimator_class=LinearRegression)
    graph = GraphSpec()
    graph.add_node(node)
    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="r2",
            greater_is_better=True,
        ),
        verbosity=0,
    )
    services = RuntimeServices(search_backend=OptunaBackend())

    job = NodeTrainingJobBuilder.build_serialized(
        node=node,
        node_data=view,
        config=config,
    )

    dispatcher = LocalTrainingDispatcher()
    result = dispatcher.dispatch([job], services)[0]
    reconstructed = NodeTrainingResultReconstructor.reconstruct(job, result)

    assert reconstructed.node_name == "linreg"
    assert len(reconstructed.models) == 3
    assert reconstructed.oof_predictions.shape[0] == 6


def test_job_builder_can_skip_payload_for_live_dispatch():
    view = _make_regression_context()
    node = NodeSpec(name="linreg", estimator_class=LinearRegression)
    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="r2",
            greater_is_better=True,
        ),
        verbosity=0,
    )

    job = NodeTrainingJobBuilder.build_live(
        node=node,
        node_data=view,
        config=config,
    )

    assert job.features == b""
    assert job.targets == {}
    assert job.aux == {}
    assert job.overlays == {}
    assert job.has_live_objects()
    assert not job.has_payload()


def test_node_training_job_to_dict_round_trip_excludes_live_objects():
    view = _make_regression_context()
    node = NodeSpec(name="linreg", estimator_class=LinearRegression)
    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="r2",
            greater_is_better=True,
        ),
        verbosity=0,
    )

    job = NodeTrainingJobBuilder.build_serialized(
        node=node,
        node_data=view,
        config=config,
    )

    restored = type(job).from_dict(job.to_dict())

    assert restored.job_id == job.job_id
    assert restored.node_spec == job.node_spec
    assert restored.config == job.config
    assert restored.feature_names == job.feature_names
    assert restored.n_samples == job.n_samples
    assert restored.has_payload()
    assert not restored.has_live_objects()


def test_node_training_job_save_load_round_trip(tmp_path):
    view = _make_regression_context()
    node = NodeSpec(name="linreg", estimator_class=LinearRegression)
    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="r2",
            greater_is_better=True,
        ),
        verbosity=0,
    )

    job = NodeTrainingJobBuilder.build_serialized(
        node=node,
        node_data=view,
        config=config,
    )
    save_path = tmp_path / "job"
    job.save(save_path)
    restored = type(job).load(save_path)

    assert restored.to_dict() == job.to_dict()
    assert not restored.has_live_objects()


def test_build_serialized_preserves_row_selected_features():
    view = _make_regression_context().select_rows(np.array([0, 2, 4, 5]))
    node = NodeSpec(name="linreg", estimator_class=LinearRegression)
    config = RunConfig(
        cv=CVConfig(n_splits=2, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="r2",
            greater_is_better=True,
        ),
        verbosity=0,
    )

    job = NodeTrainingJobBuilder.build_serialized(
        node=node,
        node_data=view,
        config=config,
    )
    result = NodeTrainingJobRunner.run_serialized(
        job, RuntimeServices(search_backend=OptunaBackend())
    )
    reconstructed = NodeTrainingResultReconstructor.reconstruct(job, result)

    assert job.n_samples == 4
    assert reconstructed.oof_predictions.shape[0] == 4


def test_build_serialized_preserves_row_selected_overlays():
    view = (
        _make_regression_context()
        .with_overlay(
            "pred_upstream",
            np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        )
        .select_rows(np.array([0, 2, 4, 5]))
    )
    node = NodeSpec(name="linreg", estimator_class=LinearRegression)
    config = RunConfig(
        cv=CVConfig(n_splits=2, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="r2",
            greater_is_better=True,
        ),
        verbosity=0,
    )

    job = NodeTrainingJobBuilder.build_serialized(
        node=node,
        node_data=view,
        config=config,
    )
    result = NodeTrainingJobRunner.run_serialized(
        job, RuntimeServices(search_backend=OptunaBackend())
    )
    reconstructed = NodeTrainingResultReconstructor.reconstruct(job, result)

    assert job.n_samples == 4
    assert reconstructed.oof_predictions.shape[0] == 4


class _FailingDispatcher(LocalTrainingDispatcher):
    def __init__(self):
        super().__init__(n_workers=2, backend="threading")

    def _get_executor(self):
        class _FailingExecutor:
            n_workers = 2
            def map(self, fn, items):
                raise ValueError("boom")
        return _FailingExecutor()


class _DeepcopyBackend(SearchBackend):
    def __init__(
        self,
        direction: str = "maximize",
        random_state: int | None = 7,
        marker: str = "kept",
    ) -> None:
        super().__init__(direction=direction, random_state=random_state)
        self.marker = marker

    def optimize(self, *args, **kwargs):
        raise NotImplementedError

    def suggest_params(self, *args, **kwargs):
        return {}

    def tell(self, *args, **kwargs):
        return None

    def get_state(self):
        return {"marker": self.marker}

    def load_state(self, state):
        self.marker = state["marker"]


class _CustomInitBackend(SearchBackend):
    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token

    def optimize(self, *args, **kwargs):
        raise NotImplementedError

    def suggest_params(self, *args, **kwargs):
        return {}

    def tell(self, *args, **kwargs):
        return None

    def get_state(self):
        return {"token": self.token}

    def load_state(self, state):
        self.token = state["token"]


def test_dispatch_raises_when_parallel_execution_fails():
    view = _make_regression_context()
    node = NodeSpec(name="linreg", estimator_class=LinearRegression)
    graph = GraphSpec()
    graph.add_node(node)
    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="r2",
            greater_is_better=True,
        ),
        verbosity=0,
    )
    job = NodeTrainingJobBuilder.build_serialized(
        node=node,
        node_data=view,
        config=config,
    )
    services = RuntimeServices(search_backend=OptunaBackend())

    with pytest.raises(RuntimeError, match="Parallel node dispatch failed") as exc_info:
        _FailingDispatcher().dispatch([job], services)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_make_worker_services_preserves_non_optuna_backend():
    backend = _DeepcopyBackend(marker="custom")
    worker_services = NodeTrainingJobRunner.make_worker_services(
        RuntimeServices(search_backend=backend)
    )

    assert isinstance(worker_services.search_backend, _DeepcopyBackend)
    assert worker_services.search_backend is not backend
    assert worker_services.search_backend.marker == "custom"


def test_make_worker_services_clones_backends_with_custom_constructor():
    backend = _CustomInitBackend(token="secret")
    worker_services = NodeTrainingJobRunner.make_worker_services(
        RuntimeServices(search_backend=backend)
    )

    assert isinstance(worker_services.search_backend, _CustomInitBackend)
    assert worker_services.search_backend is not backend
    assert worker_services.search_backend.token == "secret"


def test_make_worker_services_clones_populated_optuna_backend():
    backend = OptunaBackend(
        direction="minimize", random_state=0, show_progress_bar=False,
    )
    search_space = SearchSpace()
    search_space.add_float("x", 0.0, 1.0)
    backend.optimize(
        lambda params: params["x"] ** 2,
        search_space,
        n_trials=2,
        study_name="clone",
    )

    worker_services = NodeTrainingJobRunner.make_worker_services(
        RuntimeServices(search_backend=backend)
    )

    assert isinstance(worker_services.search_backend, OptunaBackend)
    assert worker_services.search_backend is not backend
    assert worker_services.search_backend.study is not None
    assert len(worker_services.search_backend.study.trials) == 2
    assert worker_services.search_backend.study.best_params == backend.study.best_params


def test_optimization_summary_round_trip_preserves_trials():
    optimization_result = OptimizationResult(
        best_params={"alpha": 1.0},
        best_value=0.1,
        trials=[
            TrialResult(params={"alpha": 1.0}, value=0.1, trial_id=1),
            TrialResult(params={"alpha": 2.0}, value=0.2, trial_id=2),
        ],
        n_trials=2,
        study_name="dispatch",
    )

    summary = NodeTrainingJobRunner.serialize_optimization_result(
        optimization_result
    )
    restored = NodeTrainingJobRunner.deserialize_optimization_result(summary)

    assert restored is not None
    assert restored.n_trials == 2
    assert [trial.params for trial in restored.trials] == [
        {"alpha": 1.0},
        {"alpha": 2.0},
    ]


def test_node_training_result_to_dict_and_save_round_trip(tmp_path):
    view = _make_regression_context()
    node = NodeSpec(name="linreg", estimator_class=LinearRegression)
    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="r2",
            greater_is_better=True,
        ),
        verbosity=0,
    )
    services = RuntimeServices(search_backend=OptunaBackend())

    job = NodeTrainingJobBuilder.build_serialized(
        node=node,
        node_data=view,
        config=config,
    )
    result = LocalTrainingDispatcher().dispatch([job], services)[0]

    restored = type(result).from_dict(result.to_dict())
    assert restored.to_dict() == result.to_dict()

    save_path = tmp_path / "result"
    result.save(save_path)
    loaded = type(result).load(save_path)
    assert loaded.to_dict() == result.to_dict()

    reconstructed = NodeTrainingResultReconstructor.reconstruct(job, loaded)
    assert reconstructed.node_name == "linreg"
    assert len(reconstructed.models) == 3


def test_build_for_dispatch_chooses_live_job_without_parallel_executor():
    view = _make_regression_context()
    node = NodeSpec(name="linreg", estimator_class=LinearRegression)
    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="r2",
            greater_is_better=True,
        ),
        verbosity=0,
    )
    services = RuntimeServices(
        search_backend=OptunaBackend(),
        training_dispatcher=LocalTrainingDispatcher(),
    )

    job = NodeTrainingJobBuilder.build_for_dispatch(
        node=node,
        node_data=view,
        config=config,
        services=services,
    )

    assert job.has_live_objects()
    assert not job.has_payload()


# ---------------------------------------------------------------------------
# validate_dispatchable
# ---------------------------------------------------------------------------

class TestValidateDispatchable:
    """Tests for the pre-flight dispatch validation."""

    def test_valid_graph_returns_empty(self):
        graph = GraphSpec()
        graph.add_node(NodeSpec(name="lr", estimator_class=LinearRegression))
        config = RunConfig(
            cv=CVConfig(n_splits=2, strategy=CVStrategy.RANDOM),
            tuning=TuningConfig(n_trials=5, metric="neg_mean_squared_error"),
            verbosity=0,
        )
        warnings = validate_dispatchable(graph, config)
        assert warnings == []

    def test_conditional_node_warns(self):
        graph = GraphSpec()
        graph.add_node(NodeSpec(
            name="cond",
            estimator_class=LinearRegression,
            condition=lambda data: True,
        ))
        config = RunConfig(
            cv=CVConfig(n_splits=2, strategy=CVStrategy.RANDOM),
            tuning=TuningConfig(n_trials=5, metric="neg_mean_squared_error"),
            verbosity=0,
        )
        warnings = validate_dispatchable(graph, config)
        assert any(
            w.node_name == "cond" and w.category == "condition"
            for w in warnings
        )

    def test_missing_plugin_warns(self):
        graph = GraphSpec()
        graph.add_node(NodeSpec(
            name="lr",
            estimator_class=LinearRegression,
            plugins=["nonexistent_plugin"],
        ))
        config = RunConfig(
            cv=CVConfig(n_splits=2, strategy=CVStrategy.RANDOM),
            tuning=TuningConfig(n_trials=5, metric="neg_mean_squared_error"),
            verbosity=0,
        )
        empty_registry: dict = {}
        warnings = validate_dispatchable(graph, config, plugin_registry=empty_registry)
        assert any(
            w.category == "plugin" and "nonexistent_plugin" in w.message
            for w in warnings
        )

# ---------------------------------------------------------------------------
# Worker services cache propagation
# ---------------------------------------------------------------------------

class TestWorkerServicesCachePropagation:
    """Tests for FitCache propagation to workers."""

    def test_shared_cache_passed_to_workers(self, tmp_path):
        from sklearn_meta.persistence.cache import FitCache

        cache = FitCache(cache_dir=str(tmp_path / "shared"))
        base = RuntimeServices(search_backend=OptunaBackend(), fit_cache=cache)
        worker = NodeTrainingJobRunner.make_worker_services(base)

        assert worker.fit_cache is not None
        assert worker.fit_cache.cache_dir == cache.cache_dir
        assert worker.fit_cache is not cache  # new instance

    def test_non_shared_cache_dropped_from_workers(self):
        from sklearn_meta.persistence.cache import FitCache

        cache = FitCache()  # default temp dir -> not shared
        base = RuntimeServices(search_backend=OptunaBackend(), fit_cache=cache)
        worker = NodeTrainingJobRunner.make_worker_services(base)

        assert worker.fit_cache is None

# ---------------------------------------------------------------------------
# Frame serialization
# ---------------------------------------------------------------------------

class TestFrameSerialization:
    """Tests for the tagged binary frame format."""

    def test_numeric_frame_round_trip(self):
        from sklearn_meta.execution.training import (
            _FMT_NUMPY_DF,
            _FMT_TAG_LEN,
            _serialize_frame,
            _deserialize_frame,
        )

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        data = _serialize_frame(df)
        assert data[:_FMT_TAG_LEN] == _FMT_NUMPY_DF
        result = _deserialize_frame(data)
        pd.testing.assert_frame_equal(df, result)

    def test_mixed_dtype_frame_round_trip(self):
        from sklearn_meta.execution.training import (
            _FMT_PARQUET,
            _FMT_PICKLE,
            _FMT_TAG_LEN,
            _serialize_frame,
            _deserialize_frame,
        )

        df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        data = _serialize_frame(df)
        assert data[:_FMT_TAG_LEN] in (_FMT_PARQUET, _FMT_PICKLE)
        result = _deserialize_frame(data)
        pd.testing.assert_frame_equal(df, result)

    def test_no_pickle_protocol_byte_at_start(self):
        from sklearn_meta.execution.training import _serialize_frame

        df = pd.DataFrame({"a": [1.0, 2.0]})
        data = _serialize_frame(df)
        # Pickle protocol 2+ starts with 0x80
        assert data[0:1] != b"\x80"


class TestSchemaVersionChecks:
    """Tests for schema version validation on deserialization."""

    def test_job_from_dict_accepts_current_version(self):
        """Current schema version deserializes successfully."""
        view = _make_regression_context()
        node = NodeSpec(name="linreg", estimator_class=LinearRegression)
        config = RunConfig(
            cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
            tuning=TuningConfig(
                strategy=OptimizationStrategy.NONE,
                n_trials=1,
                metric="r2",
                greater_is_better=True,
            ),
            verbosity=0,
        )
        job = NodeTrainingJobBuilder.build_serialized(node=node, node_data=view, config=config)
        payload = job.to_dict()
        assert payload["schema_version"] == 1
        restored = NodeTrainingJob.from_dict(payload)
        assert restored.job_id == job.job_id

    def test_job_from_dict_rejects_unsupported_version(self):
        """Unsupported schema version raises SchemaVersionError."""
        from sklearn_meta.execution.training import SchemaVersionError

        payload = {"object_type": "node_training_job", "schema_version": 999, "job_id": "x"}
        with pytest.raises(SchemaVersionError, match="999") as exc_info:
            NodeTrainingJob.from_dict(payload)
        assert exc_info.value.found_version == 999
        assert exc_info.value.supported_versions == {1}

    def test_job_from_dict_accepts_missing_version(self):
        """Missing schema_version is accepted for backward compat."""
        view = _make_regression_context()
        node = NodeSpec(name="linreg", estimator_class=LinearRegression)
        config = RunConfig(
            cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
            tuning=TuningConfig(
                strategy=OptimizationStrategy.NONE,
                n_trials=1,
                metric="r2",
                greater_is_better=True,
            ),
            verbosity=0,
        )
        job = NodeTrainingJobBuilder.build_serialized(node=node, node_data=view, config=config)
        payload = job.to_dict()
        del payload["schema_version"]
        restored = NodeTrainingJob.from_dict(payload)
        assert restored.job_id == job.job_id

    def test_result_from_dict_rejects_unsupported_version(self):
        """Unsupported schema version on result raises SchemaVersionError."""
        from sklearn_meta.execution.training import SchemaVersionError

        payload = {"object_type": "node_training_result", "schema_version": 999, "job_id": "x", "node_name": "n", "mean_score": 0.0, "fit_time": 0.0}
        with pytest.raises(SchemaVersionError, match="999"):
            NodeTrainingResult.from_dict(payload)


class TestPayloadSummary:
    """Tests for payload_summary() methods."""

    def test_job_payload_summary_contains_expected_keys(self):
        view = _make_regression_context()
        node = NodeSpec(name="linreg", estimator_class=LinearRegression)
        config = RunConfig(
            cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
            tuning=TuningConfig(
                strategy=OptimizationStrategy.NONE,
                n_trials=1,
                metric="r2",
                greater_is_better=True,
            ),
            verbosity=0,
        )
        job = NodeTrainingJobBuilder.build_serialized(node=node, node_data=view, config=config)
        summary = job.payload_summary()

        assert summary["node_name"] == "linreg"
        assert summary["n_samples"] == 6
        assert summary["n_features"] == 2
        assert summary["has_payload"] is True
        assert summary["has_live_objects"] is False
        assert "job_id" in summary

    def test_result_payload_summary_contains_expected_keys(self):
        view = _make_regression_context()
        node = NodeSpec(name="linreg", estimator_class=LinearRegression)
        config = RunConfig(
            cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
            tuning=TuningConfig(
                strategy=OptimizationStrategy.NONE,
                n_trials=1,
                metric="r2",
                greater_is_better=True,
            ),
            verbosity=0,
        )
        services = RuntimeServices(search_backend=OptunaBackend())
        job = NodeTrainingJobBuilder.build_serialized(node=node, node_data=view, config=config)
        result = LocalTrainingDispatcher().dispatch([job], services)[0]
        summary = result.payload_summary()

        assert summary["node_name"] == "linreg"
        assert summary["n_folds"] == 3
        assert summary["n_fold_models"] == 3
        assert "mean_score" in summary
        assert "best_params" in summary


# ---------------------------------------------------------------------------
# DispatchListener
# ---------------------------------------------------------------------------

class TestDispatchListener:
    """Tests for DispatchListener integration."""

    def test_listener_receives_sequential_dispatch_events(self):
        """Listener gets on_dispatch_start, on_job_complete, on_dispatch_complete."""

        class _RecordingListener:
            def __init__(self):
                self.events = []

            def on_dispatch_start(self, jobs):
                self.events.append(("start", len(jobs)))

            def on_job_complete(self, job, result):
                self.events.append(("complete", result.node_name))

            def on_dispatch_complete(self, jobs, results):
                self.events.append(("done", len(results)))

        view = _make_regression_context()
        node = NodeSpec(name="linreg", estimator_class=LinearRegression)
        config = RunConfig(
            cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
            tuning=TuningConfig(
                strategy=OptimizationStrategy.NONE,
                n_trials=1,
                metric="r2",
                greater_is_better=True,
            ),
            verbosity=0,
        )

        listener = _RecordingListener()
        services = RuntimeServices(
            search_backend=OptunaBackend(),
            dispatch_listener=listener,
        )
        job = NodeTrainingJobBuilder.build_serialized(
            node=node, node_data=view, config=config,
        )
        LocalTrainingDispatcher().dispatch([job], services)

        assert listener.events[0] == ("start", 1)
        assert listener.events[1] == ("complete", "linreg")
        assert listener.events[2] == ("done", 1)

    def test_no_listener_does_not_raise(self):
        """Dispatch works fine without a listener."""
        view = _make_regression_context()
        node = NodeSpec(name="linreg", estimator_class=LinearRegression)
        config = RunConfig(
            cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
            tuning=TuningConfig(
                strategy=OptimizationStrategy.NONE,
                n_trials=1,
                metric="r2",
                greater_is_better=True,
            ),
            verbosity=0,
        )
        services = RuntimeServices(search_backend=OptunaBackend())
        job = NodeTrainingJobBuilder.build_serialized(
            node=node, node_data=view, config=config,
        )
        results = LocalTrainingDispatcher().dispatch([job], services)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Worker services resolution
# ---------------------------------------------------------------------------

class TestResolveWorkerServices:
    """Tests for _resolve_worker_services dispatcher delegation."""

    def test_delegates_to_dispatcher_make_worker_services(self):
        """If dispatcher has make_worker_services, it's used."""
        from sklearn_meta.execution.training import _resolve_worker_services

        class _CustomDispatcher:
            requires_serialized_jobs = True

            def dispatch(self, jobs, services):
                return []

            def make_worker_services(self, base):
                return RuntimeServices(
                    search_backend=base.search_backend,
                    audit_logger=base.audit_logger,
                )

        base = RuntimeServices(
            search_backend=OptunaBackend(),
            training_dispatcher=_CustomDispatcher(),
            audit_logger=None,
        )
        worker = _resolve_worker_services(base)
        # Custom factory preserves search_backend directly (not cloned)
        assert worker.search_backend is base.search_backend

    def test_falls_back_to_default_without_make_worker_services(self):
        """Without make_worker_services, uses NodeTrainingJobRunner default."""
        from sklearn_meta.execution.training import _resolve_worker_services

        class _MinimalDispatcher:
            requires_serialized_jobs = True

            def dispatch(self, jobs, services):
                return []

        base = RuntimeServices(
            search_backend=OptunaBackend(),
            training_dispatcher=_MinimalDispatcher(),
        )
        worker = _resolve_worker_services(base)
        # Default clones the backend
        assert worker.search_backend is not base.search_backend
        assert worker.training_dispatcher is None
