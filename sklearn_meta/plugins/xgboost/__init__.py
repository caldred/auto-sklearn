"""XGBoost-specific plugins."""

from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin
from sklearn_meta.plugins.xgboost.importance import XGBImportancePlugin

__all__ = ["XGBMultiplierPlugin", "XGBImportancePlugin"]
