"""XGBoost-specific plugins."""

from auto_sklearn.plugins.xgboost.multiplier import XGBMultiplierPlugin
from auto_sklearn.plugins.xgboost.importance import XGBImportancePlugin

__all__ = ["XGBMultiplierPlugin", "XGBImportancePlugin"]
