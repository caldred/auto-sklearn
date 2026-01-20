"""ImportanceExtractor: Feature importance extraction for various model types."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class FeatureImportance(Protocol):
    """Protocol for models that can report feature importance."""

    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        importance_type: str = "gain",
    ) -> Dict[str, float]:
        """Get feature importance from a model."""
        ...


class ImportanceExtractor(ABC):
    """
    Abstract base class for feature importance extractors.

    Each extractor handles a specific type of model and knows how to
    extract feature importance scores from it.
    """

    @abstractmethod
    def applies_to(self, model: Any) -> bool:
        """
        Check if this extractor can handle the given model.

        Args:
            model: The fitted model.

        Returns:
            True if this extractor can extract importance from the model.
        """
        pass

    @abstractmethod
    def extract(
        self,
        model: Any,
        feature_names: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract feature importance from a model.

        Args:
            model: The fitted model.
            feature_names: List of feature names.
            **kwargs: Additional arguments (e.g., importance_type).

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        pass

    def normalize(self, importance: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize importance scores to sum to 1.

        Args:
            importance: Raw importance scores.

        Returns:
            Normalized importance scores.
        """
        total = sum(importance.values())
        if total == 0:
            return {k: 0.0 for k in importance}
        return {k: v / total for k, v in importance.items()}


class TreeImportanceExtractor(ImportanceExtractor):
    """
    Extractor for tree-based models.

    Handles XGBoost, LightGBM, CatBoost, and sklearn tree ensembles.
    """

    def applies_to(self, model: Any) -> bool:
        """Check if model is a tree-based estimator."""
        # Check for sklearn tree models
        if hasattr(model, "feature_importances_"):
            return True

        # Check for XGBoost
        if hasattr(model, "get_booster"):
            return True

        # Check for LightGBM
        if hasattr(model, "booster_") and hasattr(model, "feature_importances_"):
            return True

        # Check for CatBoost
        if hasattr(model, "get_feature_importance"):
            return True

        return False

    def extract(
        self,
        model: Any,
        feature_names: List[str],
        importance_type: str = "gain",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract feature importance from a tree model.

        Args:
            model: Fitted tree-based model.
            feature_names: List of feature names.
            importance_type: Type of importance ("gain", "weight", "cover").

        Returns:
            Dictionary of feature importances.
        """
        # XGBoost
        if hasattr(model, "get_booster"):
            return self._extract_xgboost(model, feature_names, importance_type)

        # CatBoost
        if hasattr(model, "get_feature_importance"):
            return self._extract_catboost(model, feature_names, importance_type)

        # LightGBM or sklearn
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))

        return {name: 0.0 for name in feature_names}

    def _extract_xgboost(
        self,
        model: Any,
        feature_names: List[str],
        importance_type: str,
    ) -> Dict[str, float]:
        """Extract importance from XGBoost model."""
        booster = model.get_booster()

        # Map importance_type to XGBoost types
        xgb_type_map = {
            "gain": "total_gain",
            "weight": "weight",
            "cover": "cover",
            "total_gain": "total_gain",
            "total_cover": "total_cover",
        }
        xgb_type = xgb_type_map.get(importance_type, "total_gain")

        try:
            score = booster.get_score(importance_type=xgb_type)
        except Exception as e:
            logger.warning(f"Requested importance_type '{xgb_type}' failed, using default: {e}")
            score = booster.get_score()

        # Map XGBoost feature names (f0, f1, ...) to actual names
        importance = {}
        for i, name in enumerate(feature_names):
            xgb_name = f"f{i}"
            importance[name] = score.get(xgb_name, score.get(name, 0.0))

        return importance

    def _extract_catboost(
        self,
        model: Any,
        feature_names: List[str],
        importance_type: str,
    ) -> Dict[str, float]:
        """Extract importance from CatBoost model."""
        # Map to CatBoost types
        cb_type_map = {
            "gain": "FeatureImportance",
            "weight": "FeatureImportance",
            "shap": "ShapValues",
        }
        cb_type = cb_type_map.get(importance_type, "FeatureImportance")

        try:
            importances = model.get_feature_importance(type=cb_type)
        except Exception as e:
            logger.warning(f"Requested CatBoost importance_type '{cb_type}' failed, using default: {e}")
            importances = model.get_feature_importance()

        return dict(zip(feature_names, importances))


class LinearImportanceExtractor(ImportanceExtractor):
    """
    Extractor for linear models.

    Uses absolute coefficient magnitude as importance.
    """

    def applies_to(self, model: Any) -> bool:
        """Check if model is a linear estimator."""
        return hasattr(model, "coef_")

    def extract(
        self,
        model: Any,
        feature_names: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract feature importance from coefficient magnitude.

        Args:
            model: Fitted linear model.
            feature_names: List of feature names.

        Returns:
            Dictionary of absolute coefficient magnitudes.
        """
        coefs = np.abs(model.coef_).flatten()

        if len(coefs) != len(feature_names):
            # Multi-class: average across classes
            coefs = np.abs(model.coef_).mean(axis=0)

        return dict(zip(feature_names, coefs))


class PermutationImportanceExtractor(ImportanceExtractor):
    """
    Fallback extractor using permutation importance.

    Works for any model with predict or predict_proba methods.
    """

    def __init__(
        self,
        n_repeats: int = 5,
        random_state: int = 42,
        scoring: Optional[str] = None,
    ) -> None:
        """
        Initialize the permutation importance extractor.

        Args:
            n_repeats: Number of times to permute each feature.
            random_state: Random seed for reproducibility.
            scoring: Scoring function (default: model's default scorer).
        """
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring

    def applies_to(self, model: Any) -> bool:
        """Check if model can be used for permutation importance."""
        return hasattr(model, "predict") or hasattr(model, "predict_proba")

    def extract(
        self,
        model: Any,
        feature_names: List[str],
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract feature importance using permutation.

        Args:
            model: Fitted model.
            feature_names: List of feature names.
            X_val: Validation features (required).
            y_val: Validation target (required).

        Returns:
            Dictionary of permutation importances.
        """
        if X_val is None or y_val is None:
            raise ValueError(
                "PermutationImportanceExtractor requires X_val and y_val"
            )

        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=self.scoring,
        )

        return dict(zip(feature_names, result.importances_mean))


class ImportanceRegistry:
    """
    Registry of importance extractors.

    Automatically selects the appropriate extractor for a given model.
    """

    def __init__(self) -> None:
        """Initialize with default extractors."""
        self._extractors: List[ImportanceExtractor] = [
            TreeImportanceExtractor(),
            LinearImportanceExtractor(),
        ]
        self._fallback = PermutationImportanceExtractor()

    def register(self, extractor: ImportanceExtractor, priority: int = -1) -> None:
        """
        Register a custom extractor.

        Args:
            extractor: The extractor to register.
            priority: Position in the priority list (-1 for end).
        """
        if priority == -1:
            self._extractors.append(extractor)
        else:
            self._extractors.insert(priority, extractor)

    def get_extractor(self, model: Any) -> ImportanceExtractor:
        """
        Get the appropriate extractor for a model.

        Args:
            model: The fitted model.

        Returns:
            The best matching extractor.
        """
        for extractor in self._extractors:
            if extractor.applies_to(model):
                return extractor
        return self._fallback

    def extract_importance(
        self,
        model: Any,
        feature_names: List[str],
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract importance using the best available extractor.

        Args:
            model: Fitted model.
            feature_names: List of feature names.
            X_val: Validation features (for permutation importance).
            y_val: Validation target (for permutation importance).

        Returns:
            Dictionary of feature importances.
        """
        extractor = self.get_extractor(model)

        if isinstance(extractor, PermutationImportanceExtractor):
            return extractor.extract(
                model, feature_names, X_val=X_val, y_val=y_val, **kwargs
            )

        return extractor.extract(model, feature_names, **kwargs)


# Global registry instance
importance_registry = ImportanceRegistry()
