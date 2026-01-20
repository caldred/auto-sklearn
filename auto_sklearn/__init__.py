"""
Auto-sklearn Meta-Learning Library

A flexible meta-learning library for tuning and training sklearn-compatible ML models
with arbitrary dependencies (stacking, feature chains, conditional execution).
"""

from auto_sklearn.api import GraphBuilder
from auto_sklearn.core.data.context import DataContext
from auto_sklearn.core.data.cv import CVConfig, CVFold, NestedCVFold
from auto_sklearn.core.data.manager import DataManager
from auto_sklearn.core.model.node import ModelNode
from auto_sklearn.core.model.graph import ModelGraph
from auto_sklearn.core.model.dependency import DependencyType, DependencyEdge
from auto_sklearn.core.tuning.orchestrator import TuningOrchestrator, TuningConfig
from auto_sklearn.search.space import SearchSpace

# Meta-learning components
from auto_sklearn.meta.correlation import CorrelationAnalyzer, HyperparameterCorrelation
from auto_sklearn.meta.reparameterization import (
    Reparameterization,
    LogProductReparameterization,
    LinearReparameterization,
    RatioReparameterization,
    ReparameterizedSpace,
)
from auto_sklearn.meta.prebaked import get_prebaked_reparameterization

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
    # Search
    "SearchSpace",
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
