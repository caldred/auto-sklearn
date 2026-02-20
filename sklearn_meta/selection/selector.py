"""FeatureSelector: Orchestrator for feature selection workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

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
    n_shadows: int = 5  # Number of shadow rounds (per-round fraction is 1 / n_shadows)
    threshold_mult: float = 1.414
    threshold_percentile: float = 10.0  # For threshold method
    retune_after_pruning: bool = True
    min_features: int = 1
    max_features: Optional[int] = None
    random_state: int = 42
    feature_groups: Optional[Dict[str, List[str]]] = None


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
        feature_cols = feature_cols or list(X.columns)

        if not self.config.enabled:
            return FeatureSelectionResult(
                selected_features=feature_cols,
                dropped_features=[],
                importances={},
                method_used="none",
            )

        grouping = self._resolve_feature_groups(feature_cols)

        if self.config.method == "shadow":
            return self._select_shadow(model, X, y, feature_cols, grouping)
        elif self.config.method == "permutation":
            return self._select_permutation(
                model, X, y, feature_cols, grouping, X_val, y_val
            )
        elif self.config.method == "threshold":
            return self._select_threshold(model, X, y, feature_cols, grouping)
        else:
            raise ValueError(f"Unknown selection method: {self.config.method}")

    def _select_shadow(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        grouping: _FeatureGrouping,
    ) -> FeatureSelectionResult:
        """Select features using shadow feature method."""
        selector = ShadowFeatureSelector(
            n_shadows=self.config.n_shadows,
            threshold_mult=self.config.threshold_mult,
            random_state=self.config.random_state,
        )

        if grouping.has_explicit_groups:
            result = selector.fit_select_grouped(
                model=model,
                X=X,
                y=y,
                group_to_features=grouping.group_to_features,
                feature_cols=feature_cols,
            )
        else:
            result = selector.fit_select(model, X, y, feature_cols)

        # Apply min/max constraints
        selected = self._apply_constraints(
            result.features_to_keep,
            result.features_to_drop,
            result.feature_importances,
            grouping=grouping,
        )

        importances = result.feature_importances
        details: Dict[str, Any] = {
            "shadow_importances": result.shadow_importances,
            "feature_to_shadow": result.feature_to_shadow,
            "threshold_used": result.threshold_used,
        }

        if grouping.has_explicit_groups:
            group_importances, averaged_importances = self._compute_group_importances(
                result.feature_importances,
                grouping,
            )
            group_thresholds = self._compute_shadow_group_thresholds(result, grouping)
            to_keep_groups = [
                group
                for group, importance in group_importances.items()
                if importance >= group_thresholds[group]
            ]
            to_drop_groups = [
                group for group in group_importances if group not in to_keep_groups
            ]
            selected = self._apply_constraints(
                self._expand_groups(to_keep_groups, grouping),
                self._expand_groups(to_drop_groups, grouping),
                averaged_importances,
                grouping=grouping,
            )
            importances = averaged_importances
            details["group_importances"] = group_importances
            details["group_thresholds"] = group_thresholds

        return FeatureSelectionResult(
            selected_features=selected,
            dropped_features=[f for f in feature_cols if f not in selected],
            importances=importances,
            method_used="shadow",
            details=details,
        )

    def _select_permutation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        grouping: _FeatureGrouping,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> FeatureSelectionResult:
        """Select features using permutation importance."""
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
        (
            importances_to_use,
            to_keep,
            to_drop,
            threshold,
            group_details,
        ) = self._group_aware_percentile_selection(importances, grouping)

        selected = self._apply_constraints(
            to_keep,
            to_drop,
            importances_to_use,
            grouping=grouping,
        )

        details: Dict[str, Any] = {"threshold": threshold}
        details.update(group_details)

        return FeatureSelectionResult(
            selected_features=selected,
            dropped_features=[f for f in feature_cols if f not in selected],
            importances=importances_to_use,
            method_used="permutation",
            details=details,
        )

    def _select_threshold(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        grouping: _FeatureGrouping,
    ) -> FeatureSelectionResult:
        """Select features using simple importance threshold."""
        # Fit model
        model.fit(X, y)

        # Get importance
        extractor = self._importance_registry.get_extractor(model)
        importances = extractor.extract(model, feature_cols)

        # Select features above percentile threshold
        (
            importances_to_use,
            to_keep,
            to_drop,
            threshold,
            group_details,
        ) = self._group_aware_percentile_selection(importances, grouping)

        selected = self._apply_constraints(
            to_keep,
            to_drop,
            importances_to_use,
            grouping=grouping,
        )

        details: Dict[str, Any] = {"threshold": threshold}
        details.update(group_details)

        return FeatureSelectionResult(
            selected_features=selected,
            dropped_features=[f for f in feature_cols if f not in selected],
            importances=importances_to_use,
            method_used="threshold",
            details=details,
        )

    def _apply_constraints(
        self,
        to_keep: List[str],
        to_drop: List[str],
        importances: Dict[str, float],
        grouping: Optional[_FeatureGrouping] = None,
    ) -> List[str]:
        """Apply min/max feature constraints."""
        if grouping is None or not grouping.has_explicit_groups:
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

        # Group-aware constraints: groups are always added/removed atomically.
        group_importances, _ = self._compute_group_importances(importances, grouping)
        keep_groups = self._features_to_groups(to_keep, grouping)
        drop_groups = self._features_to_groups(to_drop, grouping)

        sorted_keep_groups = sorted(
            keep_groups,
            key=lambda g: group_importances.get(g, 0.0),
            reverse=True,
        )
        sorted_drop_groups = sorted(
            [g for g in drop_groups if g not in keep_groups],
            key=lambda g: group_importances.get(g, 0.0),
            reverse=True,
        )

        selected_groups = list(sorted_keep_groups)
        selected_count = sum(len(grouping.group_to_features[g]) for g in selected_groups)
        target_count = max(len(to_keep), self.config.min_features)
        if self.config.max_features is not None:
            target_count = min(target_count, self.config.max_features)

        for group in sorted_drop_groups:
            if selected_count >= target_count:
                break
            selected_groups.append(group)
            selected_count += len(grouping.group_to_features[group])

        selected_features: List[str] = []
        for group in selected_groups:
            selected_features.extend(grouping.group_to_features[group])
        return selected_features

    def _resolve_feature_groups(self, feature_cols: List[str]) -> _FeatureGrouping:
        """Build a validated grouping map for the active feature columns."""
        configured_groups = self.config.feature_groups or {}
        feature_set = set(feature_cols)

        group_to_features: Dict[str, List[str]] = {}
        feature_to_group: Dict[str, str] = {}

        for group_name, group_features in configured_groups.items():
            active_group_features = [f for f in group_features if f in feature_set]
            if not active_group_features:
                continue
            if len(set(active_group_features)) != len(active_group_features):
                raise ValueError(
                    f"Feature group '{group_name}' contains duplicate features"
                )
            for feature in active_group_features:
                if feature in feature_to_group:
                    raise ValueError(
                        f"Feature '{feature}' appears in multiple feature_groups"
                    )
                feature_to_group[feature] = group_name
            group_to_features[group_name] = [
                feature for feature in feature_cols if feature in active_group_features
            ]

        for feature in feature_cols:
            if feature not in feature_to_group:
                group_name = f"__single__::{feature}"
                feature_to_group[feature] = group_name
                group_to_features[group_name] = [feature]

        has_explicit_groups = any(
            len(features) > 1 for features in group_to_features.values()
        )
        return _FeatureGrouping(
            feature_to_group=feature_to_group,
            group_to_features=group_to_features,
            has_explicit_groups=has_explicit_groups,
        )

    def _compute_group_importances(
        self,
        importances: Dict[str, float],
        grouping: _FeatureGrouping,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Average feature importances by group."""
        group_importances: Dict[str, float] = {}
        averaged_feature_importances: Dict[str, float] = {}

        for group_name, features in grouping.group_to_features.items():
            group_score = float(np.mean([importances.get(f, 0.0) for f in features]))
            group_importances[group_name] = group_score
            for feature in features:
                averaged_feature_importances[feature] = group_score

        return group_importances, averaged_feature_importances

    def _compute_shadow_group_thresholds(
        self,
        result: ShadowResult,
        grouping: _FeatureGrouping,
    ) -> Dict[str, float]:
        """Compute a shadow threshold for each group as mean member threshold."""
        group_thresholds: Dict[str, float] = {}
        for group_name, features in grouping.group_to_features.items():
            feature_thresholds = []
            for feature in features:
                shadow_name = result.feature_to_shadow.get(feature)
                shadow_imp = result.shadow_importances.get(shadow_name, 0.0)
                feature_thresholds.append(self.config.threshold_mult * shadow_imp)
            group_thresholds[group_name] = (
                float(np.mean(feature_thresholds)) if feature_thresholds else 0.0
            )
        return group_thresholds

    def _group_aware_percentile_selection(
        self,
        importances: Dict[str, float],
        grouping: _FeatureGrouping,
    ) -> tuple[Dict[str, float], List[str], List[str], float, Dict[str, Any]]:
        """Apply threshold selection on per-feature or per-group importances."""
        if not grouping.has_explicit_groups:
            threshold = float(
                np.percentile(
                    list(importances.values()),
                    self.config.threshold_percentile,
                )
            )
            to_keep = [f for f, imp in importances.items() if imp >= threshold]
            to_drop = [f for f, imp in importances.items() if imp < threshold]
            return importances, to_keep, to_drop, threshold, {}

        group_importances, averaged_feature_importances = self._compute_group_importances(
            importances,
            grouping,
        )
        threshold = float(
            np.percentile(
                list(group_importances.values()),
                self.config.threshold_percentile,
            )
        )
        to_keep_groups = [
            group for group, imp in group_importances.items() if imp >= threshold
        ]
        to_drop_groups = [
            group for group, imp in group_importances.items() if imp < threshold
        ]
        to_keep = self._expand_groups(to_keep_groups, grouping)
        to_drop = self._expand_groups(to_drop_groups, grouping)
        return (
            averaged_feature_importances,
            to_keep,
            to_drop,
            threshold,
            {"group_importances": group_importances},
        )

    def _expand_groups(
        self,
        groups: List[str],
        grouping: _FeatureGrouping,
    ) -> List[str]:
        """Expand group names to member feature names."""
        features: List[str] = []
        for group in groups:
            features.extend(grouping.group_to_features[group])
        return features

    def _features_to_groups(
        self,
        features: List[str],
        grouping: _FeatureGrouping,
    ) -> Set[str]:
        """Get unique groups represented by feature names."""
        return {grouping.feature_to_group[f] for f in features if f in grouping.feature_to_group}

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


@dataclass
class _FeatureGrouping:
    """Internal representation of active feature grouping."""

    feature_to_group: Dict[str, str]
    group_to_features: Dict[str, List[str]]
    has_explicit_groups: bool
