"""
sklearn-meta: Meta-Learning Library

A flexible meta-learning library for tuning and training sklearn-compatible ML models
with arbitrary dependencies (stacking, feature chains, conditional execution).
"""

from __future__ import annotations

# Spec
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.spec.dependency import DependencyType, DependencyEdge
from sklearn_meta.spec.distillation import DistillationConfig
from sklearn_meta.spec.builder import GraphBuilder
from sklearn_meta.spec.quantile import (
    JointQuantileConfig,
    JointQuantileGraphSpec,
    OrderConstraint,
    QuantileScalingConfig,
)

# Data
from sklearn_meta.data.record import DatasetRecord
from sklearn_meta.data.view import DataView

# Runtime
from sklearn_meta.runtime.config import (
    RunConfig,
    TuningConfig,
    CVConfig,
    CVStrategy,
    FeatureSelectionConfig,
    RunConfigBuilder,
)
from sklearn_meta.runtime.services import RuntimeServices

# Engine
from sklearn_meta.engine.runner import GraphRunner
from sklearn_meta.execution.training import (
    DispatchListener,
    DispatchWarning,
    LocalTrainingDispatcher,
    NodeTrainingJob,
    NodeTrainingJobBuilder,
    NodeTrainingJobRunner,
    NodeTrainingResult,
    NodeTrainingResultReconstructor,
    SchemaVersionError,
    TrainingDispatcher,
    validate_dispatchable,
)

# Artifacts
from sklearn_meta.artifacts.training import TrainingRun, NodeRunResult
from sklearn_meta.artifacts.inference import InferenceGraph, JointQuantileInferenceGraph

# Reusable (unchanged)
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.search.backends.optuna import OptunaBackend
from sklearn_meta.meta.reparameterization import (
    Reparameterization,
    LogProductReparameterization,
    LinearReparameterization,
    RatioReparameterization,
    ReparameterizedSpace,
)
from sklearn_meta.meta.prebaked import get_prebaked_reparameterization
from sklearn_meta.meta.correlation import CorrelationAnalyzer, HyperparameterCorrelation
from sklearn_meta.audit.logger import AuditLogger
from sklearn_meta.persistence.cache import FitCache

# Convenience helpers
from sklearn_meta.convenience import ComparisonResult, compare, tune, cross_validate, stack


__version__ = "0.2.0"


import typing

import pandas as pd


@typing.overload
def fit(
    graph: GraphSpec,
    data: DataView,
    config: RunConfig,
    services: RuntimeServices | None = ...,
) -> TrainingRun: ...


@typing.overload
def fit(
    graph: GraphSpec,
    data: DataView,
    config: RunConfig,
    *,
    services: RuntimeServices | None = ...,
) -> TrainingRun: ...


@typing.overload
def fit(
    graph: GraphSpec,
    X: typing.Any,
    y: typing.Any,
    config: RunConfig,
    *,
    groups: typing.Any = ...,
    services: RuntimeServices | None = ...,
) -> TrainingRun: ...


def _coerce_feature_frame(X: typing.Any) -> pd.DataFrame:
    """Coerce array-like feature inputs into a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, (str, bytes)):
        raise TypeError(
            "Expected a DataView or array-like features as the second argument, "
            f"got {type(X).__name__}."
        )

    try:
        frame = pd.DataFrame(X)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(
            "Expected a DataView or array-like features as the second argument, "
            f"got {type(X).__name__}."
        ) from exc

    if frame.shape[1] == 0:
        raise TypeError(
            "Expected a DataView or 2D array-like features as the second argument, "
            f"got {type(X).__name__}."
        )

    if isinstance(frame.columns, pd.RangeIndex):
        frame.columns = [f"x{i}" for i in range(frame.shape[1])]
    return frame


def fit(
    graph: GraphSpec,
    data_or_X: DataView | typing.Any,
    y_or_config: RunConfig | typing.Any = None,
    config: RunConfig | RuntimeServices | None = None,
    *,
    groups: typing.Any = None,
    services: RuntimeServices | None = None,
    **aux: typing.Any,
) -> TrainingRun:
    """Convenience function: fit a graph on data with the given config.

    Accepts either a pre-built :class:`DataView` **or** raw array-like
    ``X``/``y``::

        # With DataView (original)
        fit(graph, data_view, config)

        # With raw arrays (new shorthand)
        fit(graph, X_train, y_train, config)
        fit(graph, X_train, y_train, config, groups=patient_ids)
    """
    if isinstance(data_or_X, DataView):
        resolved_data = data_or_X
        if not isinstance(y_or_config, RunConfig):
            raise TypeError(
                "When passing a DataView, the third argument must be a RunConfig."
            )
        resolved_config = y_or_config
        if config is None:
            resolved_services = services
        else:
            if services is not None:
                raise TypeError(
                    "RuntimeServices was provided both positionally and by keyword."
                )
            resolved_services = typing.cast(RuntimeServices, config)
        if groups is not None or aux:
            raise TypeError(
                "groups and **aux keyword arguments are only supported when "
                "passing raw X/y.  Use DataView.from_Xy() to bind groups."
            )
    else:
        resolved_X = _coerce_feature_frame(data_or_X)
        resolved_data = DataView.from_Xy(
            resolved_X, y=y_or_config, groups=groups, **aux,
        )
        if not isinstance(config, RunConfig):
            raise TypeError(
                "When passing raw X/y, the fourth argument must be a RunConfig."
            )
        resolved_config = config
        resolved_services = services

    resolved_services = resolved_services or RuntimeServices.default()
    return GraphRunner(resolved_services).fit(graph, resolved_data, resolved_config)


__all__ = [
    # Spec
    "GraphSpec",
    "NodeSpec",
    "OutputType",
    "DependencyType",
    "DependencyEdge",
    "DistillationConfig",
    "GraphBuilder",
    "JointQuantileConfig",
    "JointQuantileGraphSpec",
    "OrderConstraint",
    "QuantileScalingConfig",
    # Data
    "DatasetRecord",
    "DataView",
    # Runtime
    "RunConfig",
    "TuningConfig",
    "CVConfig",
    "CVStrategy",
    "FeatureSelectionConfig",
    "RunConfigBuilder",
    "RuntimeServices",
    # Engine
    "GraphRunner",
    "NodeTrainingJob",
    "NodeTrainingResult",
    "NodeTrainingJobBuilder",
    "NodeTrainingJobRunner",
    "NodeTrainingResultReconstructor",
    "TrainingDispatcher",
    "LocalTrainingDispatcher",
    "DispatchListener",
    "DispatchWarning",
    "SchemaVersionError",
    "validate_dispatchable",
    # Artifacts
    "TrainingRun",
    "NodeRunResult",
    "InferenceGraph",
    "JointQuantileInferenceGraph",
    # Search
    "SearchSpace",
    "OptunaBackend",
    # Meta-learning
    "CorrelationAnalyzer",
    "HyperparameterCorrelation",
    "Reparameterization",
    "LogProductReparameterization",
    "LinearReparameterization",
    "RatioReparameterization",
    "ReparameterizedSpace",
    "get_prebaked_reparameterization",
    # Persistence
    "FitCache",
    # Audit
    "AuditLogger",
    # Convenience
    "fit",
    "ComparisonResult",
    "compare",
    "tune",
    "cross_validate",
    "stack",
]
