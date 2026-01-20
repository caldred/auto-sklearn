"""FeatureSelector: Orchestrator for feature selection workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from sklearn_meta.selection.importance import ImportanceRegistry, PermutationImportanceExtractor
from sklearn_meta.selection.shadow import ShadowFeatureSelector, ShadowResult

if TYPE_CHECKING:
    from sklearn_meta.core.data.context import DataContext
    from sklearn_meta.core.model.node import ModelNode


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""

    enabled: bool = True
    method: str = "shadow"  # "shadow", "permutation", "threshold"
    n_shadows: int = 5
    threshold_mult: float = 1.414
    threshold_percentile: float = 10.0  # For threshold method
    retune_after_pruning: bool = True
    min_features: int = 1
    max_features: Optional[int] = None
    random_state: int = 42


@dataclass
class FeatureSelectionResult:
    """Result from feature selection."""

    selected_features: List[str]
    dropped_features: List[str]
    importances: Dict[str, float]
    method_used: str
    details: Dict[str, Any] = field(default_factory=dict)


class FeatureSelector:
    """
    Orchestrator for feature selection workflows.

    Supports multiple selection methods:
    - shadow: Entropy-matched shadow feature pruning
    - permutation: Permutation importance-based selection
    - threshold: Simple importance threshold

    Integrates with TuningOrchestrator for:
    - Pre-tuning feature selection
    - Post-tuning refinement
    - Retune-after-pruning workflow
    """

    def __init__(self, config: FeatureSelectionConfig) -> None:
        """
        Initialize the feature selector.

        Args:
            config: Feature selection configuration.
        """
        self.config = config
        self._importance_registry = ImportanceRegistry()

    def select(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> FeatureSelectionResult:
        """
        Select features using the configured method.

        Args:
            model: Unfitted sklearn-compatible estimator.
            X: Training features.
            y: Training target.
            feature_cols: List of feature columns to evaluate.
            X_val: Validation features (for permutation method).
            y_val: Validation target (for permutation method).

        Returns:
            FeatureSelectionResult with selected features and metadata.
        """
        if not self.config.enabled:
            feature_cols = feature_cols or list(X.columns)
            return FeatureSelectionResult(
                selected_features=feature_cols,
                dropped_features=[],
                importances={},
                method_used="none",
            )

        if self.config.method == "shadow":
            return self._select_shadow(model, X, y, feature_cols)
        elif self.config.method == "permutation":
            return self._select_permutation(model, X, y, feature_cols, X_val, y_val)
        elif self.config.method == "threshold":
            return self._select_threshold(model, X, y, feature_cols)
        else:
            raise ValueError(f"Unknown selection method: {self.config.method}")

    def _select_shadow(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]],
    ) -> FeatureSelectionResult:
        """Select features using shadow feature method."""
        selector = ShadowFeatureSelector(
            n_shadows=self.config.n_shadows,
            threshold_mult=self.config.threshold_mult,
            random_state=self.config.random_state,
        )

        result = selector.fit_select(model, X, y, feature_cols)

        # Apply min/max constraints
        selected = self._apply_constraints(
            result.features_to_keep,
            result.features_to_drop,
            result.feature_importances,
        )

        return FeatureSelectionResult(
            selected_features=selected,
            dropped_features=[f for f in (feature_cols or list(X.columns)) if f not in selected],
            importances=result.feature_importances,
            method_used="shadow",
            details={
                "shadow_importances": result.shadow_importances,
                "feature_to_shadow": result.feature_to_shadow,
                "threshold_used": result.threshold_used,
            },
        )

    def _select_permutation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> FeatureSelectionResult:
        """Select features using permutation importance."""
        feature_cols = feature_cols or list(X.columns)

        # Fit model first
        model.fit(X, y)

        # Use validation set if available, otherwise use training set
        if X_val is not None and y_val is not None:
            X_eval, y_eval = X_val, y_val
        else:
            X_eval, y_eval = X, y

        # Get permutation importance
        extractor = PermutationImportanceExtractor(
            random_state=self.config.random_state
        )
        importances = extractor.extract(model, feature_cols, X_val=X_eval, y_val=y_eval)

        # Select features above threshold
        threshold = np.percentile(
            list(importances.values()),
            self.config.threshold_percentile,
        )

        to_keep = [f for f, imp in importances.items() if imp >= threshold]
        to_drop = [f for f, imp in importances.items() if imp < threshold]

        selected = self._apply_constraints(to_keep, to_drop, importances)

        return FeatureSelectionResult(
            selected_features=selected,
            dropped_features=[f for f in feature_cols if f not in selected],
            importances=importances,
            method_used="permutation",
            details={"threshold": threshold},
        )

    def _select_threshold(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]],
    ) -> FeatureSelectionResult:
        """Select features using simple importance threshold."""
        feature_cols = feature_cols or list(X.columns)

        # Fit model
        model.fit(X, y)

        # Get importance
        extractor = self._importance_registry.get_extractor(model)
        importances = extractor.extract(model, feature_cols)

        # Select features above percentile threshold
        threshold = np.percentile(
            list(importances.values()),
            self.config.threshold_percentile,
        )

        to_keep = [f for f, imp in importances.items() if imp >= threshold]
        to_drop = [f for f, imp in importances.items() if imp < threshold]

        selected = self._apply_constraints(to_keep, to_drop, importances)

        return FeatureSelectionResult(
            selected_features=selected,
            dropped_features=[f for f in feature_cols if f not in selected],
            importances=importances,
            method_used="threshold",
            details={"threshold": threshold},
        )

    def _apply_constraints(
        self,
        to_keep: List[str],
        to_drop: List[str],
        importances: Dict[str, float],
    ) -> List[str]:
        """Apply min/max feature constraints."""
        all_features = to_keep + to_drop

        # Sort by importance (descending)
        sorted_features = sorted(
            all_features,
            key=lambda f: importances.get(f, 0),
            reverse=True,
        )

        # Apply min constraint
        n_keep = max(len(to_keep), self.config.min_features)

        # Apply max constraint
        if self.config.max_features is not None:
            n_keep = min(n_keep, self.config.max_features)

        return sorted_features[:n_keep]

    def select_for_node(
        self,
        node: ModelNode,
        ctx: DataContext,
        params: Dict[str, Any],
    ) -> FeatureSelectionResult:
        """
        Select features for a specific model node.

        Args:
            node: The model node.
            ctx: Data context.
            params: Model parameters.

        Returns:
            FeatureSelectionResult for this node.
        """
        # Create model instance
        model = node.create_estimator(params)

        # Get feature columns
        feature_cols = node.feature_cols or list(ctx.X.columns)

        return self.select(
            model=model,
            X=ctx.X[feature_cols],
            y=ctx.y,
            feature_cols=feature_cols,
        )

    def __repr__(self) -> str:
        return (
            f"FeatureSelector(method={self.config.method}, "
            f"enabled={self.config.enabled})"
        )
