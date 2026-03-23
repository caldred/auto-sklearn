"""FeatureSelectionService: Feature selection decoupled from DataContext."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from sklearn_meta.data.record import DEFAULT_TARGET_KEY
from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import FeatureSelectionConfig
from sklearn_meta.selection.selector import FeatureSelector, FeatureSelectionResult
from sklearn_meta.engine.estimator_factory import create_estimator
from sklearn_meta.spec.node import NodeSpec

logger = logging.getLogger(__name__)


class FeatureSelectionService:
    """Applies feature selection for a node using the existing FeatureSelector."""

    def __init__(self, config: FeatureSelectionConfig) -> None:
        self.config = config

    def apply(
        self,
        node: NodeSpec,
        data: DataView,
        best_params: Dict[str, Any],
        target_key: str = DEFAULT_TARGET_KEY,
    ) -> Tuple[FeatureSelectionResult, DataView]:
        """Apply feature selection and return result + updated view.

        Args:
            node: The node to select features for.
            data: DataView containing features and targets.
            best_params: Best hyperparameters found so far.
            target_key: Key into ``batch.targets`` to use as *y*.

        Returns:
            Tuple of (FeatureSelectionResult, DataView with updated feature_cols).
        """
        selector = FeatureSelector(self.config)

        # Materialize for the selector
        batch = data.materialize()
        model = create_estimator(node, best_params)
        feature_cols = list(batch.X.columns)
        y = batch.targets.get(target_key, batch.y)

        result = selector.select(
            model=model,
            X=batch.X,
            y=y,
            feature_cols=feature_cols,
        )

        # Update the view's feature_cols if features were selected
        updated_view = data
        if result.selected_features:
            updated_view = data.select_features(result.selected_features)

        return result, updated_view
