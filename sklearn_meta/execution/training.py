"""Node-level training dispatch helpers."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Protocol

import joblib
import numpy as np
import pandas as pd

from sklearn_meta.artifacts.training import NodeRunResult, QuantileNodeRunResult
from sklearn_meta.data.view import DataView
from sklearn_meta.engine.cv import CVEngine
from sklearn_meta.engine.quantile_trainer import QuantileNodeTrainer
from sklearn_meta.engine.search import SearchService
from sklearn_meta.engine.selection import FeatureSelectionService
from sklearn_meta.engine.trainer import StandardNodeTrainer
from sklearn_meta.runtime.config import CVFold, CVResult, FoldResult, RunConfig
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.search.backends.base import (
    OptimizationResult,
    TrialResult,
    TrialState,
)
from sklearn_meta.spec.dependency import DependencyType
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec

if TYPE_CHECKING:
    from sklearn_meta.execution.local import LocalExecutor


class SchemaVersionError(ValueError):
    """Raised when a serialized object has an unsupported schema version."""

    def __init__(self, object_type: str, found: Any, supported: set[int]) -> None:
        self.object_type = object_type
        self.found_version = found
        self.supported_versions = supported
        super().__init__(
            f"{object_type} schema version {found!r} is not supported. "
            f"Supported versions: {sorted(supported)}. "
            f"This usually means the coordinator and worker are running "
            f"different versions of sklearn-meta."
        )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_array(array: np.ndarray) -> bytes:
    payload = io.BytesIO()
    np.save(payload, np.asarray(array), allow_pickle=False)
    return payload.getvalue()


def _deserialize_array(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def _serialize_model(model: Any) -> bytes:
    payload = io.BytesIO()
    joblib.dump(model, payload)
    return payload.getvalue()


def _deserialize_model(data: bytes) -> Any:
    return joblib.load(io.BytesIO(data))


# Frame serialization format tags (8 bytes each, null-padded)
_FMT_NUMPY_DF = b"numpydf\x00"
_FMT_FEATHER = b"feather\x00"
_FMT_PARQUET = b"parquet\x00"
_FMT_PICKLE = b"pickle\x00\x00"
_FMT_TAG_LEN = 8


def _serialize_frame(frame: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to a tagged binary format.

    Format: 8-byte tag + raw payload.  Uses a NumPy structured-array
    encoding for all-numeric frames, then parquet, then pickle as a
    last resort.
    """
    all_numeric = all(
        pd.api.types.is_numeric_dtype(dt) for dt in frame.dtypes
    )
    if all_numeric:
        payload = io.BytesIO()
        np.save(payload, frame.to_records(index=False), allow_pickle=False)
        return _FMT_NUMPY_DF + payload.getvalue()

    payload = io.BytesIO()
    try:
        frame.to_parquet(payload, index=False)
        return _FMT_PARQUET + payload.getvalue()
    except Exception:
        return _FMT_PICKLE + pickle.dumps(frame)


def _deserialize_frame(data: bytes) -> pd.DataFrame:
    """Deserialize a DataFrame from the tagged binary format."""
    if len(data) >= _FMT_TAG_LEN:
        tag = data[:_FMT_TAG_LEN]
        payload = data[_FMT_TAG_LEN:]

        if tag == _FMT_NUMPY_DF:
            records = np.load(io.BytesIO(payload), allow_pickle=False)
            return pd.DataFrame.from_records(records)
        if tag == _FMT_FEATHER:
            return pd.read_feather(io.BytesIO(payload))
        if tag == _FMT_PARQUET:
            return pd.read_parquet(io.BytesIO(payload))
        if tag == _FMT_PICKLE:
            return pickle.loads(payload)  # noqa: S301

    # Legacy fallback: pickle-wrapped dict format
    wrapper = pickle.loads(data)  # noqa: S301
    fmt = wrapper["format"]
    payload = wrapper["payload"]
    if fmt == "feather":
        return pd.read_feather(io.BytesIO(payload))
    if fmt == "parquet":
        return pd.read_parquet(io.BytesIO(payload))
    if fmt == "pickle":
        return pickle.loads(payload)  # noqa: S301
    raise ValueError(f"Unknown frame format: {fmt}")


def _serialize_optimization_summary(
    optimization_result: Optional[OptimizationResult],
) -> Optional[Dict[str, Any]]:
    if optimization_result is None:
        return None

    trials = []
    for trial in optimization_result.trials:
        state = (
            trial.state.value
            if isinstance(trial.state, TrialState)
            else trial.state
        )
        trials.append({
            "params": dict(trial.params),
            "value": trial.value,
            "trial_id": trial.trial_id,
            "duration": trial.duration,
            "user_attrs": dict(trial.user_attrs),
            "state": state,
        })

    return {
        "n_trials": optimization_result.n_trials,
        "best_value": optimization_result.best_value,
        "study_name": optimization_result.study_name,
        "trials": trials,
    }


def _deserialize_optimization_summary(
    summary: Optional[Dict[str, Any]],
) -> Optional[OptimizationResult]:
    if summary is None:
        return None

    trials = []
    for payload in summary.get("trials", []):
        state = payload.get("state", TrialState.COMPLETE.value)
        if not isinstance(state, TrialState):
            state = TrialState(state)
        trials.append(
            TrialResult(
                params=dict(payload.get("params", {})),
                value=payload["value"],
                trial_id=payload.get("trial_id", 0),
                duration=payload.get("duration", 0.0),
                user_attrs=dict(payload.get("user_attrs", {})),
                state=state,
            )
        )

    return OptimizationResult(
        best_params={},
        best_value=summary.get("best_value", 0.0),
        trials=trials,
        n_trials=summary.get("n_trials", len(trials)),
        study_name=summary.get("study_name", ""),
    )


def _serialize_fold_result(fold_result: FoldResult) -> Dict[str, Any]:
    return {
        "fold_idx": fold_result.fold.fold_idx,
        "repeat_idx": fold_result.fold.repeat_idx,
        "train_indices": _serialize_array(fold_result.fold.train_indices),
        "val_indices": _serialize_array(fold_result.fold.val_indices),
        "val_predictions": _serialize_array(fold_result.val_predictions),
        "val_score": fold_result.val_score,
        "train_score": fold_result.train_score,
        "fit_time": fold_result.fit_time,
        "predict_time": fold_result.predict_time,
        "params": fold_result.params,
    }


def _deserialize_fold_result(
    payload: Dict[str, Any],
    model: Any,
) -> FoldResult:
    fold = CVFold(
        fold_idx=payload["fold_idx"],
        train_indices=_deserialize_array(payload["train_indices"]),
        val_indices=_deserialize_array(payload["val_indices"]),
        repeat_idx=payload.get("repeat_idx", 0),
    )
    return FoldResult(
        fold=fold,
        model=model,
        val_predictions=_deserialize_array(payload["val_predictions"]),
        val_score=payload["val_score"],
        train_score=payload.get("train_score"),
        fit_time=payload.get("fit_time", 0.0),
        predict_time=payload.get("predict_time", 0.0),
        params=payload.get("params", {}),
    )


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _encode_bytes(data: Optional[bytes]) -> Optional[str]:
    if data is None:
        return None
    return base64.b64encode(data).decode("ascii")


def _decode_bytes(data: Optional[str]) -> Optional[bytes]:
    if data is None:
        return None
    return base64.b64decode(data.encode("ascii"))


def _encode_bytes_dict(payload: Dict[str, bytes]) -> Dict[str, str]:
    return {
        key: _encode_bytes(value) or ""
        for key, value in payload.items()
    }


def _decode_bytes_dict(payload: Dict[str, str]) -> Dict[str, bytes]:
    return {
        key: _decode_bytes(value) or b""
        for key, value in payload.items()
    }


def _encode_bytes_list(items: List[bytes]) -> List[str]:
    return [_encode_bytes(item) or "" for item in items]


def _decode_bytes_list(items: List[str]) -> List[bytes]:
    return [_decode_bytes(item) or b"" for item in items]


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

@dataclass
class DispatchWarning:
    """A warning from pre-flight dispatch validation."""

    node_name: str
    category: str  # "serialization", "import", "condition", "distillation", "plugin"
    message: str


def validate_dispatchable(
    graph: GraphSpec,
    config: RunConfig,
    *,
    plugin_registry: Optional[Any] = None,
) -> List[DispatchWarning]:
    """Check whether a graph and config can be dispatched to remote workers.

    Returns a list of warnings. An empty list means the graph is fully
    dispatchable. Warnings do not prevent dispatch — conditional and
    distilled nodes automatically fall back to inline execution.

    Args:
        graph: The graph to validate.
        config: The run configuration.
        plugin_registry: Optional plugin registry to check plugin names against.
    """
    warnings: List[DispatchWarning] = []

    # Config round-trip
    try:
        RunConfig.from_dict(config.to_dict())
    except Exception as exc:
        warnings.append(DispatchWarning(
            node_name="<config>",
            category="serialization",
            message=f"RunConfig round-trip failed: {exc}",
        ))

    for node in graph.nodes.values():
        # Node spec round-trip
        try:
            NodeSpec.from_dict(node.to_dict())
        except Exception as exc:
            warnings.append(DispatchWarning(
                node_name=node.name,
                category="serialization",
                message=f"NodeSpec round-trip failed: {exc}",
            ))

        # Estimator class importability
        from sklearn_meta.spec._resolve import get_class_path

        class_path = get_class_path(node.estimator_class)
        try:
            from sklearn_meta.spec._resolve import resolve_class_path

            resolved = resolve_class_path(class_path)
            if resolved is not node.estimator_class:
                warnings.append(DispatchWarning(
                    node_name=node.name,
                    category="import",
                    message=f"Class path '{class_path}' resolves to a different object",
                ))
        except ImportError as exc:
            warnings.append(DispatchWarning(
                node_name=node.name,
                category="import",
                message=f"Cannot import '{class_path}': {exc}",
            ))

        # fit_params serializability
        if node.fit_params:
            try:
                json.dumps(node.fit_params, default=str)
            except (TypeError, ValueError) as exc:
                warnings.append(DispatchWarning(
                    node_name=node.name,
                    category="serialization",
                    message=f"fit_params not JSON-serializable: {exc}",
                ))

        # Conditional node warning
        if node.is_conditional:
            warnings.append(DispatchWarning(
                node_name=node.name,
                category="condition",
                message="Conditional nodes cannot be dispatched (will run inline)",
            ))

        # Distilled node warning
        if node.is_distilled:
            warnings.append(DispatchWarning(
                node_name=node.name,
                category="distillation",
                message="Distilled nodes cannot be dispatched (will run inline)",
            ))

        # Plugin reference check
        if node.plugins and plugin_registry is not None:
            for plugin_name in node.plugins:
                if plugin_name not in plugin_registry:
                    warnings.append(DispatchWarning(
                        node_name=node.name,
                        category="plugin",
                        message=f"Plugin '{plugin_name}' not found in registry",
                    ))

    return warnings


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class NodeTrainingJob:
    SCHEMA_VERSION = 1

    job_id: str
    node_spec: Dict[str, Any]
    config: Dict[str, Any]
    plugin_names: List[str]
    features: bytes
    targets: Dict[str, bytes]
    overlays: Dict[str, bytes]
    aux: Dict[str, bytes]
    groups: Optional[bytes]
    feature_names: List[str]
    n_samples: int

    # Live objects for in-process fast path (not serialized for remote).
    _live_node: Optional[Any] = field(default=None, repr=False, compare=False)
    _live_data: Optional[Any] = field(default=None, repr=False, compare=False)
    _live_config: Optional[Any] = field(default=None, repr=False, compare=False)

    def has_payload(self) -> bool:
        """Whether this job carries serialized data for out-of-process execution."""
        return bool(self.features)

    def has_live_objects(self) -> bool:
        """Whether this job carries live objects for in-process execution."""
        return (
            self._live_node is not None
            and self._live_data is not None
            and self._live_config is not None
        )

    def validate_for_serialized_execution(self) -> None:
        """Ensure the job can run without coordinator-resident live objects."""
        if not self.has_payload():
            raise ValueError(
                f"NodeTrainingJob '{self.job_id}' does not include serialized payload "
                "required for out-of-process execution."
            )

    def validate_for_live_execution(self) -> None:
        """Ensure the job can run directly on coordinator-resident live objects."""
        if not self.has_live_objects():
            raise ValueError(
                f"NodeTrainingJob '{self.job_id}' does not include live objects "
                "required for in-process execution."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this job to a JSON-safe dictionary.

        Live in-memory objects are intentionally excluded.
        """
        return {
            "object_type": "node_training_job",
            "schema_version": self.SCHEMA_VERSION,
            "job_id": self.job_id,
            "node_spec": self.node_spec,
            "config": self.config,
            "plugin_names": list(self.plugin_names),
            "features": _encode_bytes(self.features) or "",
            "targets": _encode_bytes_dict(self.targets),
            "overlays": _encode_bytes_dict(self.overlays),
            "aux": _encode_bytes_dict(self.aux),
            "groups": _encode_bytes(self.groups),
            "feature_names": list(self.feature_names),
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeTrainingJob":
        """Reconstruct a serialized node training job."""
        object_type = data.get("object_type")
        if object_type not in (None, "node_training_job"):
            raise ValueError(
                f"Expected object_type='node_training_job', got {object_type!r}"
            )
        version = data.get("schema_version")
        if version is not None and version not in {cls.SCHEMA_VERSION}:
            raise SchemaVersionError(
                "NodeTrainingJob", version, {cls.SCHEMA_VERSION}
            )
        return cls(
            job_id=data["job_id"],
            node_spec=dict(data["node_spec"]),
            config=dict(data["config"]),
            plugin_names=list(data.get("plugin_names", [])),
            features=_decode_bytes(data.get("features", "")) or b"",
            targets=_decode_bytes_dict(data.get("targets", {})),
            overlays=_decode_bytes_dict(data.get("overlays", {})),
            aux=_decode_bytes_dict(data.get("aux", {})),
            groups=_decode_bytes(data.get("groups")),
            feature_names=list(data.get("feature_names", [])),
            n_samples=data["n_samples"],
        )

    def save(self, path: str | Path) -> None:
        """Save this job to a directory containing ``manifest.json``.

        Cloud dispatchers can use this to externalize job payloads to
        shared storage (e.g. S3, GCS) instead of shipping bytes inline.
        Workers reconstruct via :meth:`load`.
        """
        from sklearn_meta.persistence.manifest import write_manifest

        write_manifest(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "NodeTrainingJob":
        """Load a job from a directory previously created by :meth:`save`.

        This is the recommended way for cloud workers to receive jobs
        via shared filesystem or object storage.
        """
        from sklearn_meta.persistence.manifest import read_manifest

        return cls.from_dict(read_manifest(Path(path)))

    def payload_summary(self) -> Dict[str, Any]:
        """Lightweight metadata about this job (no binary data).

        Useful for logging, monitoring dashboards, and debugging
        in cloud dispatch pipelines.
        """
        return {
            "job_id": self.job_id,
            "node_name": self.node_spec.get("name", ""),
            "n_samples": self.n_samples,
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "has_payload": self.has_payload(),
            "has_live_objects": self.has_live_objects(),
            "n_targets": len(self.targets),
            "n_overlays": len(self.overlays),
            "plugin_names": list(self.plugin_names),
        }


@dataclass
class NodeTrainingResult:
    SCHEMA_VERSION = 1

    job_id: str
    node_name: str
    best_params: Dict[str, Any]
    selected_features: Optional[List[str]]
    oof_predictions: bytes
    fold_models: List[bytes]
    optimization_summary: Optional[Dict[str, Any]]
    cv_scores: List[float]
    mean_score: float
    fit_time: float
    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    repeat_oof: Optional[bytes] = None
    is_quantile: bool = False
    oof_quantile_predictions: Optional[bytes] = None
    quantile_models: Optional[Dict[str, List[bytes]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this result to a JSON-safe dictionary."""
        return {
            "object_type": "node_training_result",
            "schema_version": self.SCHEMA_VERSION,
            "job_id": self.job_id,
            "node_name": self.node_name,
            "best_params": dict(self.best_params),
            "selected_features": (
                list(self.selected_features)
                if self.selected_features is not None else None
            ),
            "oof_predictions": _encode_bytes(self.oof_predictions) or "",
            "fold_models": _encode_bytes_list(self.fold_models),
            "optimization_summary": self.optimization_summary,
            "cv_scores": list(self.cv_scores),
            "mean_score": self.mean_score,
            "fit_time": self.fit_time,
            "fold_results": [
                {
                    **payload,
                    "train_indices": _encode_bytes(payload["train_indices"]) or "",
                    "val_indices": _encode_bytes(payload["val_indices"]) or "",
                    "val_predictions": _encode_bytes(payload["val_predictions"]) or "",
                }
                for payload in self.fold_results
            ],
            "repeat_oof": _encode_bytes(self.repeat_oof),
            "is_quantile": self.is_quantile,
            "oof_quantile_predictions": _encode_bytes(
                self.oof_quantile_predictions
            ),
            "quantile_models": (
                {
                    str(tau): _encode_bytes_list(models)
                    for tau, models in self.quantile_models.items()
                }
                if self.quantile_models is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeTrainingResult":
        """Reconstruct a serialized node training result."""
        object_type = data.get("object_type")
        if object_type not in (None, "node_training_result"):
            raise ValueError(
                f"Expected object_type='node_training_result', got {object_type!r}"
            )
        version = data.get("schema_version")
        if version is not None and version not in {cls.SCHEMA_VERSION}:
            raise SchemaVersionError(
                "NodeTrainingResult", version, {cls.SCHEMA_VERSION}
            )
        fold_results = []
        for payload in data.get("fold_results", []):
            fold_results.append({
                **payload,
                "train_indices": _decode_bytes(payload["train_indices"]) or b"",
                "val_indices": _decode_bytes(payload["val_indices"]) or b"",
                "val_predictions": _decode_bytes(payload["val_predictions"]) or b"",
            })

        quantile_models = data.get("quantile_models")
        return cls(
            job_id=data["job_id"],
            node_name=data["node_name"],
            best_params=dict(data.get("best_params", {})),
            selected_features=(
                list(data["selected_features"])
                if data.get("selected_features") is not None else None
            ),
            oof_predictions=_decode_bytes(data.get("oof_predictions", "")) or b"",
            fold_models=_decode_bytes_list(data.get("fold_models", [])),
            optimization_summary=data.get("optimization_summary"),
            cv_scores=list(data.get("cv_scores", [])),
            mean_score=data["mean_score"],
            fit_time=data["fit_time"],
            fold_results=fold_results,
            repeat_oof=_decode_bytes(data.get("repeat_oof")),
            is_quantile=data.get("is_quantile", False),
            oof_quantile_predictions=_decode_bytes(
                data.get("oof_quantile_predictions")
            ),
            quantile_models=(
                {
                    str(tau): _decode_bytes_list(models)
                    for tau, models in quantile_models.items()
                }
                if quantile_models is not None else None
            ),
        )

    def save(self, path: str | Path) -> None:
        """Save this result to a directory containing ``manifest.json``.

        Cloud dispatchers can use this to externalize result payloads to
        shared storage (e.g. S3, GCS) instead of shipping bytes inline.
        Coordinators reconstruct via :meth:`load`.
        """
        from sklearn_meta.persistence.manifest import write_manifest

        write_manifest(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "NodeTrainingResult":
        """Load a result from a directory previously created by :meth:`save`.

        This is the recommended way for coordinators to receive results
        from cloud workers via shared filesystem or object storage.
        """
        from sklearn_meta.persistence.manifest import read_manifest

        return cls.from_dict(read_manifest(Path(path)))

    def payload_summary(self) -> Dict[str, Any]:
        """Lightweight metadata about this result (no binary data)."""
        return {
            "job_id": self.job_id,
            "node_name": self.node_name,
            "mean_score": self.mean_score,
            "n_folds": len(self.fold_results),
            "n_fold_models": len(self.fold_models),
            "best_params": dict(self.best_params),
            "selected_features": self.selected_features,
            "is_quantile": self.is_quantile,
        }


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class TrainingDispatcher(Protocol):
    """Protocol for dispatching full node-training jobs.

    Third-party cloud dispatcher packages should implement this protocol.
    See NodeTrainingJobRunner for the worker-side entry point.
    """

    @property
    def requires_serialized_jobs(self) -> bool:
        """Whether dispatch() expects jobs with serialized payloads.

        Return True for out-of-process / remote dispatch (the common case).
        Return False only if the dispatcher will run jobs in the same process
        and can use live Python objects directly.
        """
        ...

    def dispatch(
        self,
        jobs: List[NodeTrainingJob],
        services: RuntimeServices,
    ) -> List[NodeTrainingResult]:
        """Execute a batch of independent node training jobs."""
        ...

    def make_worker_services(self, base: RuntimeServices) -> RuntimeServices:
        """Build services for an isolated worker.

        Override to customize worker service setup (e.g. distributed
        search backend storage, remote logging).

        The default implementation (on NodeTrainingJobRunner) clones
        the search backend, preserves shared caches and plugins, and
        drops the dispatcher.
        """
        ...


class DispatchListener(Protocol):
    """Optional observer for training dispatch lifecycle events.

    Implement this protocol to receive progress updates during dispatch.
    All methods are optional -- implement only the hooks you need.
    """

    def on_dispatch_start(
        self, jobs: List[NodeTrainingJob],
    ) -> None:
        """Called before dispatching a batch of jobs."""
        ...

    def on_job_complete(
        self, job: NodeTrainingJob, result: NodeTrainingResult,
    ) -> None:
        """Called when a single job completes successfully."""
        ...

    def on_dispatch_complete(
        self, jobs: List[NodeTrainingJob], results: List[NodeTrainingResult],
    ) -> None:
        """Called after all jobs in a batch complete."""
        ...


# ---------------------------------------------------------------------------
# Job builder
# ---------------------------------------------------------------------------

class NodeTrainingJobBuilder:
    """Build serialized node-training jobs from live coordinator state."""

    @staticmethod
    def is_dispatchable(node: NodeSpec, graph: GraphSpec) -> bool:
        """Whether *node* can be dispatched remotely.

        Checks only graph-level constraints (DISTILL edges).
        Caller is responsible for checking node-level conditions
        (``is_conditional``, ``is_distilled``) before calling this.
        """
        return not any(
            edge.dep_type == DependencyType.DISTILL
            for edge in graph.get_upstream(node.name)
        )

    @staticmethod
    def build(
        node: NodeSpec,
        node_data: DataView,
        config: RunConfig,
        include_payload: bool = True,
    ) -> NodeTrainingJob:
        feature_cols = list(node_data.feature_cols)
        n_samples = node_data.n_rows
        features = b""
        targets: Dict[str, bytes] = {}
        overlays: Dict[str, bytes] = {}
        aux: Dict[str, bytes] = {}
        groups = None

        if include_payload:
            if node_data.row_sel is not None:
                features_df = node_data.dataset.frame.iloc[
                    node_data.row_sel
                ][feature_cols].copy()
            else:
                features_df = node_data.dataset.frame[feature_cols].copy()
            features = _serialize_frame(features_df)
            targets = {
                name: _serialize_array(node_data.resolve_channel(ref))
                for name, ref in node_data.targets.items()
            }
            overlays = {
                name: _serialize_array(
                    values[node_data.row_sel]
                    if node_data.row_sel is not None else values
                )
                for name, values in node_data.overlays.items()
            }
            aux = {
                name: _serialize_array(node_data.resolve_channel(ref))
                for name, ref in node_data.aux.items()
            }
            if node_data.groups is not None:
                groups = _serialize_array(node_data.resolve_channel(node_data.groups))

        if include_payload:
            node_spec = node.to_dict()
            config_dict = config.to_dict()
            job_id = _stable_hash({
                "node_name": node.name,
                "node_spec": node_spec,
                "config": config_dict,
                "feature_names": feature_cols,
                "target_names": sorted(targets),
                "overlay_names": sorted(overlays),
                "aux_names": sorted(aux),
                "n_samples": n_samples,
            })
        else:
            # Minimal metadata for live in-process jobs — skip expensive
            # to_dict() serialization since it won't be shipped.
            node_spec = {"name": node.name}
            config_dict = {}
            job_id = f"{node.name}_{id(node)}"

        return NodeTrainingJob(
            job_id=job_id,
            node_spec=node_spec,
            config=config_dict,
            plugin_names=list(node.plugins),
            features=features,
            targets=targets,
            overlays=overlays,
            aux=aux,
            groups=groups,
            feature_names=feature_cols,
            n_samples=n_samples,
            _live_node=node if not include_payload else None,
            _live_data=node_data if not include_payload else None,
            _live_config=config if not include_payload else None,
        )

    @classmethod
    def build_live(
        cls,
        node: NodeSpec,
        node_data: DataView,
        config: RunConfig,
    ) -> NodeTrainingJob:
        """Build a job intended for single-process in-memory execution."""
        return cls.build(
            node=node,
            node_data=node_data,
            config=config,
            include_payload=False,
        )

    @classmethod
    def build_serialized(
        cls,
        node: NodeSpec,
        node_data: DataView,
        config: RunConfig,
    ) -> NodeTrainingJob:
        """Build a job intended for worker or out-of-process execution."""
        return cls.build(
            node=node,
            node_data=node_data,
            config=config,
            include_payload=True,
        )

    @classmethod
    def build_for_dispatch(
        cls,
        node: NodeSpec,
        node_data: DataView,
        config: RunConfig,
        services: RuntimeServices,
    ) -> NodeTrainingJob:
        """Build the appropriate job shape for the configured dispatcher."""
        if _requires_serialized_dispatch_payload(services):
            return cls.build_serialized(node=node, node_data=node_data, config=config)
        return cls.build_live(node=node, node_data=node_data, config=config)


# ---------------------------------------------------------------------------
# Result reconstruction
# ---------------------------------------------------------------------------

class NodeTrainingResultReconstructor:
    """Reconstruct public training artifacts from serialized node results."""

    @staticmethod
    def reconstruct(
        job: NodeTrainingJob,
        result: NodeTrainingResult,
    ) -> NodeRunResult:
        del job  # Reserved for future compatibility checks.

        fold_models = [_deserialize_model(data) for data in result.fold_models]
        fold_results = [
            _deserialize_fold_result(payload, model)
            for payload, model in zip(result.fold_results, fold_models)
        ]
        cv_result = CVResult(
            fold_results=fold_results,
            oof_predictions=_deserialize_array(result.oof_predictions),
            node_name=result.node_name,
            repeat_oof=(
                _deserialize_array(result.repeat_oof)
                if result.repeat_oof is not None else None
            ),
        )

        optimization_result = _deserialize_optimization_summary(
            result.optimization_summary
        )
        if optimization_result is not None:
            optimization_result.best_params = dict(result.best_params)

        if result.is_quantile:
            quantile_models: Dict[float, List[Any]] = {}
            for tau_str, model_payloads in (result.quantile_models or {}).items():
                quantile_models[float(tau_str)] = [
                    _deserialize_model(payload) for payload in model_payloads
                ]
            return QuantileNodeRunResult(
                node_name=result.node_name,
                cv_result=cv_result,
                best_params=result.best_params,
                quantile_models=quantile_models,
                oof_quantile_predictions=(
                    _deserialize_array(result.oof_quantile_predictions)
                    if result.oof_quantile_predictions is not None else None
                ),
                selected_features=result.selected_features,
                optimization_result=optimization_result,
            )

        return NodeRunResult(
            node_name=result.node_name,
            cv_result=cv_result,
            best_params=result.best_params,
            selected_features=result.selected_features,
            optimization_result=optimization_result,
        )


def reconstruct_node_result(
    job: NodeTrainingJob,
    result: NodeTrainingResult,
) -> NodeRunResult:
    """Backward-compatible wrapper around the public reconstructor."""
    return NodeTrainingResultReconstructor.reconstruct(job, result)


# ---------------------------------------------------------------------------
# Local dispatcher
# ---------------------------------------------------------------------------

def get_trainer(node: NodeSpec):
    """Choose trainer based on node type."""
    from sklearn_meta.spec.quantile import QuantileNodeSpec

    if isinstance(node, QuantileNodeSpec):
        return QuantileNodeTrainer()
    return StandardNodeTrainer()


def _requires_serialized_dispatch_payload(services: RuntimeServices) -> bool:
    dispatcher = services.training_dispatcher
    if dispatcher is None:
        return False
    return bool(getattr(dispatcher, "requires_serialized_jobs", True))


def _resolve_worker_services(base: RuntimeServices) -> RuntimeServices:
    """Build worker services, delegating to the dispatcher if it provides a factory."""
    dispatcher = base.training_dispatcher
    factory = getattr(dispatcher, "make_worker_services", None)
    if callable(factory):
        return factory(base)
    return NodeTrainingJobRunner.make_worker_services(base)


def _notify_dispatch_start(
    services: RuntimeServices,
    jobs: List[NodeTrainingJob],
) -> None:
    listener = services.dispatch_listener
    if listener is not None:
        listener.on_dispatch_start(jobs)


def _notify_job_complete(
    services: RuntimeServices,
    job: NodeTrainingJob,
    result: NodeTrainingResult,
) -> None:
    listener = services.dispatch_listener
    if listener is not None:
        listener.on_job_complete(job, result)


def _notify_dispatch_complete(
    services: RuntimeServices,
    jobs: List[NodeTrainingJob],
    results: List[NodeTrainingResult],
) -> None:
    listener = services.dispatch_listener
    if listener is not None:
        listener.on_dispatch_complete(jobs, results)


def _fit_and_package(
    job_id: str,
    node: NodeSpec,
    data: DataView,
    config: RunConfig,
    services: RuntimeServices,
) -> NodeTrainingResult:
    """Run trainer on resolved objects and package the result."""
    start_time = time.time()

    cv_engine = CVEngine(config.cv)
    search_service = SearchService(services.search_backend)
    selection_service = None
    if config.feature_selection is not None and config.feature_selection.enabled:
        selection_service = FeatureSelectionService(config.feature_selection)

    trainer = get_trainer(node)
    result = trainer.fit_node(
        node, data, config, services,
        cv_engine, search_service, selection_service,
    )
    return _package_result(job_id, result, time.time() - start_time)


class NodeTrainingJobRunner:
    """Execute a single node-training job and package the result."""

    serialize_optimization_result = staticmethod(_serialize_optimization_summary)
    deserialize_optimization_result = staticmethod(
        _deserialize_optimization_summary
    )

    @staticmethod
    def make_worker_services(base: RuntimeServices) -> RuntimeServices:
        """Build isolated services suitable for worker execution.

        Shared-filesystem caches (created with an explicit ``cache_dir``)
        are propagated to workers as new instances pointing at the same
        directory.  Non-shared caches are dropped.
        """
        fit_cache = None
        if base.fit_cache is not None and base.fit_cache.is_shared:
            from sklearn_meta.persistence.cache import FitCache

            fit_cache = FitCache(
                cache_dir=str(base.fit_cache.cache_dir),
                max_size_mb=base.fit_cache.max_size_mb,
                enabled=base.fit_cache.enabled,
            )
        return RuntimeServices(
            search_backend=base.search_backend.clone(),
            training_dispatcher=None,
            plugin_registry=base.plugin_registry,
            audit_logger=None,
            fit_cache=fit_cache,
        )

    @classmethod
    def run(
        cls,
        job: NodeTrainingJob,
        services: RuntimeServices,
    ) -> NodeTrainingResult:
        """Run a job using the most appropriate execution mode for ``services``."""
        if _requires_serialized_dispatch_payload(services):
            return cls.run_serialized(job, services)
        if job.has_live_objects():
            return cls.run_live(job, services)
        return cls.run_serialized(job, services)

    @staticmethod
    def run_live(
        job: NodeTrainingJob,
        services: RuntimeServices,
    ) -> NodeTrainingResult:
        """Run a job directly against in-memory live objects."""
        job.validate_for_live_execution()
        return _fit_and_package(
            job.job_id, job._live_node, job._live_data, job._live_config, services,
        )

    @classmethod
    def run_serialized(
        cls,
        job: NodeTrainingJob,
        base_services: RuntimeServices,
    ) -> NodeTrainingResult:
        """Run a job from serialized payloads using isolated worker services."""
        job.validate_for_serialized_execution()
        return _fit_and_package(
            job.job_id,
            NodeSpec.from_dict(job.node_spec),
            _reconstruct_data_view(job),
            RunConfig.from_dict(job.config),
            _resolve_worker_services(base_services),
        )


class LocalTrainingDispatcher:
    """Run node-training jobs locally with optional parallelism.

    Args:
        n_workers: Number of parallel workers. 1 = sequential (default).
                   -1 = use all CPUs.
        backend: Joblib backend ("threading", "loky", "multiprocessing").
        prefer: Joblib preference ("threads", "processes").
    """

    def __init__(
        self,
        n_workers: int = 1,
        backend: str = "threading",
        prefer: str = "processes",
    ) -> None:
        self._n_workers = n_workers
        self._backend = backend
        self._prefer = prefer
        self._executor: Optional[LocalExecutor] = None

    @property
    def requires_serialized_jobs(self) -> bool:
        return self._n_workers != 1

    def dispatch(
        self,
        jobs: List[NodeTrainingJob],
        services: RuntimeServices,
    ) -> List[NodeTrainingResult]:
        if not jobs:
            return []

        _notify_dispatch_start(services, jobs)

        if self.requires_serialized_jobs:
            executor = self._get_executor()
            runner = _RemoteJobRunner(services)
            try:
                results = list(executor.map(runner, jobs))
            except Exception as exc:
                raise RuntimeError(
                    "Parallel node dispatch failed"
                ) from exc
            for job, result in zip(jobs, results):
                _notify_job_complete(services, job, result)
            _notify_dispatch_complete(services, jobs, results)
            return results

        # Fast path: use live objects when available (in-process only).
        results: List[NodeTrainingResult] = []
        for job in jobs:
            result = NodeTrainingJobRunner.run(job, services)
            results.append(result)
            _notify_job_complete(services, job, result)
        _notify_dispatch_complete(services, jobs, results)
        return results

    def _get_executor(self) -> "LocalExecutor":
        if self._executor is None:
            from sklearn_meta.execution.local import LocalExecutor
            self._executor = LocalExecutor(
                n_workers=self._n_workers,
                backend=self._backend,
                prefer=self._prefer,
            )
        return self._executor



class _RemoteJobRunner:
    """Picklable callable for executor.map."""

    def __init__(self, services: RuntimeServices) -> None:
        self._services = services

    def __call__(self, job: NodeTrainingJob) -> NodeTrainingResult:
        return NodeTrainingJobRunner.run_serialized(job, self._services)


def _reconstruct_data_view(job: NodeTrainingJob) -> DataView:
    features = _deserialize_frame(job.features)
    data = DataView.from_X(features)
    for name, values in job.targets.items():
        data = data.bind_target(_deserialize_array(values), name=name)
    if job.groups is not None:
        data = data.bind_groups(_deserialize_array(job.groups))
    for name, values in job.aux.items():
        data = data.with_aux(name, _deserialize_array(values))
    for name, values in job.overlays.items():
        data = data.with_overlay(name, _deserialize_array(values))
    return data


def _package_result(
    job_id: str,
    result: NodeRunResult,
    fit_time: float,
) -> NodeTrainingResult:
    optimization_summary = _serialize_optimization_summary(
        result.optimization_result
    )
    fold_results = [
        _serialize_fold_result(fold_result)
        for fold_result in result.cv_result.fold_results
    ]
    node_training_result = NodeTrainingResult(
        job_id=job_id,
        node_name=result.node_name,
        best_params=dict(result.best_params),
        selected_features=(
            list(result.selected_features)
            if result.selected_features is not None else None
        ),
        oof_predictions=_serialize_array(result.oof_predictions),
        fold_models=[_serialize_model(model) for model in result.models],
        optimization_summary=optimization_summary,
        cv_scores=list(result.cv_result.val_scores),
        mean_score=result.mean_score,
        fit_time=fit_time,
        fold_results=fold_results,
        repeat_oof=(
            _serialize_array(result.cv_result.repeat_oof)
            if result.cv_result.repeat_oof is not None else None
        ),
    )

    if isinstance(result, QuantileNodeRunResult):
        node_training_result.is_quantile = True
        node_training_result.oof_quantile_predictions = (
            _serialize_array(result.oof_quantile_predictions)
            if result.oof_quantile_predictions is not None else None
        )
        node_training_result.quantile_models = {
            str(tau): [_serialize_model(model) for model in models]
            for tau, models in result.quantile_models.items()
        }

    return node_training_result
