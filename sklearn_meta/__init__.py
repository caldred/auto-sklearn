"""
sklearn-meta: Meta-Learning Library

A flexible meta-learning library for tuning and training sklearn-compatible ML models
with arbitrary dependencies (stacking, feature chains, conditional execution).
"""

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


__version__ = "0.2.0"


def fit(
    graph: GraphSpec,
    data: DataView,
    config: RunConfig,
    services: RuntimeServices | None = None,
) -> TrainingRun:
    """Convenience function: fit a graph with config and optional services."""
    services = services or RuntimeServices.default()
    return GraphRunner(services).fit(graph, data, config)


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
]
