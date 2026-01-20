"""XGBImportancePlugin: XGBoost-specific feature importance extraction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from sklearn_meta.plugins.base import ModelPlugin
from sklearn_meta.selection.importance import ImportanceExtractor

if TYPE_CHECKING:
    from sklearn_meta.core.data.context import DataContext
    from sklearn_meta.core.model.node import ModelNode

logger = logging.getLogger(__name__)


class XGBImportanceExtractor(ImportanceExtractor):
    """
    XGBoost-specific importance extractor.

    Supports multiple importance types:
    - total_gain: Total gain of splits using this feature
    - weight: Number of times feature is used in splits
    - cover: Average coverage of splits using this feature
    - gain: Average gain per split using this feature
    """

    def __init__(self, importance_type: str = "total_gain") -> None:
        """
        Initialize the extractor.

        Args:
            importance_type: Type of importance to extract.
        """
        self.importance_type = importance_type

    def applies_to(self, model: Any) -> bool:
        """Check if model is XGBoost."""
        return hasattr(model, "get_booster")

    def extract(
        self,
        model: Any,
        feature_names: List[str],
        importance_type: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract feature importance from XGBoost model.

        Args:
            model: Fitted XGBoost model.
            feature_names: List of feature names.
            importance_type: Override importance type.

        Returns:
            Dictionary of feature importances.
        """
        imp_type = importance_type or self.importance_type
        booster = model.get_booster()

        try:
            score = booster.get_score(importance_type=imp_type)
        except Exception as e:
            # Fall back to default
            logger.warning(f"Requested XGBoost importance_type '{imp_type}' failed, using default: {e}")
            score = booster.get_score()

        # Map XGBoost feature names to actual names
        importance = {}
        for i, name in enumerate(feature_names):
            # XGBoost uses f0, f1, ... by default
            xgb_name = f"f{i}"
            importance[name] = score.get(xgb_name, score.get(name, 0.0))

        return importance


class XGBImportancePlugin(ModelPlugin):
    """
    XGBoost importance plugin.

    Provides XGBoost-specific feature importance extraction and
    optional feature selection based on importance.
    """

    def __init__(
        self,
        importance_type: str = "total_gain",
        prune_zero_importance: bool = False,
    ) -> None:
        """
        Initialize the importance plugin.

        Args:
            importance_type: Type of importance to use.
            prune_zero_importance: Whether to warn about zero-importance features.
        """
        self.importance_type = importance_type
        self.prune_zero_importance = prune_zero_importance
        self._extractor = XGBImportanceExtractor(importance_type)

    @property
    def name(self) -> str:
        return "xgb_importance"

    def applies_to(self, estimator_class: Type) -> bool:
        """Check if estimator is XGBoost."""
        class_name = estimator_class.__name__
        return class_name in ("XGBClassifier", "XGBRegressor", "XGBRanker")

    def post_fit(
        self,
        model: Any,
        node: ModelNode,
        ctx: DataContext,
    ) -> Any:
        """
        Extract and cache feature importance after fitting.

        Args:
            model: Fitted XGBoost model.
            node: The model node.
            ctx: Data context.

        Returns:
            The model (unmodified).
        """
        # Extract importance
        feature_names = list(ctx.X.columns)
        importance = self._extractor.extract(
            model, feature_names, self.importance_type
        )

        # Store in model for later retrieval
        if not hasattr(model, "_sklearn_meta_meta"):
            model._sklearn_meta_meta = {}

        model._sklearn_meta_meta["feature_importance"] = importance
        model._sklearn_meta_meta["importance_type"] = self.importance_type

        # Warn about zero-importance features if enabled
        if self.prune_zero_importance:
            zero_features = [f for f, v in importance.items() if v == 0]
            if zero_features:
                import warnings
                warnings.warn(
                    f"Found {len(zero_features)} features with zero importance: "
                    f"{zero_features[:5]}..."
                )

        return model

    def get_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Get feature importance from a fitted model.

        Args:
            model: Fitted XGBoost model.
            feature_names: Optional feature names override.

        Returns:
            Dictionary of feature importances.
        """
        # Check for cached importance
        if hasattr(model, "_sklearn_meta_meta"):
            cached = model._sklearn_meta_meta.get("feature_importance")
            if cached is not None:
                return cached

        # Extract fresh
        if feature_names is None:
            # Try to get from model
            try:
                feature_names = model.get_booster().feature_names
            except Exception:
                raise ValueError("feature_names required for importance extraction")

        return self._extractor.extract(model, feature_names)

    def get_top_features(
        self,
        model: Any,
        n: int = 10,
        feature_names: Optional[List[str]] = None,
    ) -> List[tuple]:
        """
        Get top N most important features.

        Args:
            model: Fitted XGBoost model.
            n: Number of top features to return.
            feature_names: Optional feature names.

        Returns:
            List of (feature_name, importance) tuples.
        """
        importance = self.get_importance(model, feature_names)
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]

    def __repr__(self) -> str:
        return f"XGBImportancePlugin(importance_type={self.importance_type})"
