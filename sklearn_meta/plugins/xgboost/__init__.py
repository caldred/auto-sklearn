"""XGBoost-specific plugins."""

XGBOOST_CLASS_NAMES = frozenset({"XGBClassifier", "XGBRegressor", "XGBRanker"})

from sklearn_meta.plugins.xgboost.importance import XGBImportancePlugin  # noqa: E402
from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin  # noqa: E402

__all__ = ["XGBMultiplierPlugin", "XGBImportancePlugin", "XGBOOST_CLASS_NAMES"]
