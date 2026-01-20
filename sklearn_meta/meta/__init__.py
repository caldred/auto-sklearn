"""Meta-learning components for hyperparameter optimization."""

from sklearn_meta.meta.correlation import (
    CorrelationAnalyzer,
    HyperparameterCorrelation,
    CorrelationType,
)
from sklearn_meta.meta.reparameterization import (
    Reparameterization,
    LinearReparameterization,
    LogProductReparameterization,
    RatioReparameterization,
    ReparameterizedSpace,
)
from sklearn_meta.meta.prebaked import (
    get_prebaked_reparameterization,
    PREBAKED_REGISTRY,
    register_prebaked,
)

__all__ = [
    "CorrelationAnalyzer",
    "HyperparameterCorrelation",
    "CorrelationType",
    "Reparameterization",
    "LinearReparameterization",
    "LogProductReparameterization",
    "RatioReparameterization",
    "ReparameterizedSpace",
    "get_prebaked_reparameterization",
    "PREBAKED_REGISTRY",
    "register_prebaked",
]
