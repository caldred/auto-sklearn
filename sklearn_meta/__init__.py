"""
sklearn-meta: Meta-Learning Library

A flexible meta-learning library for tuning and training sklearn-compatible ML models
with arbitrary dependencies (stacking, feature chains, conditional execution).
"""

from sklearn_meta.api import GraphBuilder
from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVFold, NestedCVFold
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.node import ModelNode
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.model.dependency import DependencyType, DependencyEdge
from sklearn_meta.core.tuning.orchestrator import TuningOrchestrator, TuningConfig
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.search.backends.optuna import OptunaBackend
from sklearn_meta.selection.selector import FeatureSelector, FeatureSelectionConfig
from sklearn_meta.persistence.cache import FitCache
from sklearn_meta.audit.logger import AuditLogger

# Meta-learning components
from sklearn_meta.meta.correlation import CorrelationAnalyzer, HyperparameterCorrelation
from sklearn_meta.meta.reparameterization import (
    Reparameterization,
    LogProductReparameterization,
    LinearReparameterization,
    RatioReparameterization,
    ReparameterizedSpace,
)
from sklearn_meta.meta.prebaked import get_prebaked_reparameterization

__version__ = "0.1.0"

__all__ = [
    # API
    "GraphBuilder",
    # Data
    "DataContext",
    "CVConfig",
    "CVFold",
    "NestedCVFold",
    "DataManager",
    # Model
    "ModelNode",
    "ModelGraph",
    "DependencyType",
    "DependencyEdge",
    # Tuning
    "TuningOrchestrator",
    "TuningConfig",
    "OptimizationStrategy",
    # Search
    "SearchSpace",
    "OptunaBackend",
    # Selection
    "FeatureSelector",
    "FeatureSelectionConfig",
    # Persistence
    "FitCache",
    # Audit
    "AuditLogger",
    # Meta-learning
    "CorrelationAnalyzer",
    "HyperparameterCorrelation",
    "Reparameterization",
    "LogProductReparameterization",
    "LinearReparameterization",
    "RatioReparameterization",
    "ReparameterizedSpace",
    "get_prebaked_reparameterization",
]
