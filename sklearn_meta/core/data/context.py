"""DataContext: Immutable container for a dataset snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataContext:
    """
    Immutable container for a dataset snapshot.

    This class represents the data at any point in the ML pipeline, including
    the original features, target, and any derived data from upstream models.

    Attributes:
        X: Feature DataFrame.
        y: Target Series (optional for prediction-only contexts).
        groups: Group labels for group-aware CV (optional).
        base_margin: Base margin for stacking models like XGBoost (optional).
        indices: Original indices for subset tracking (optional).
        upstream_outputs: Dictionary mapping node names to their outputs.
        metadata: Additional metadata for the context.
    """

    X: pd.DataFrame
    y: Optional[pd.Series] = None
    groups: Optional[pd.Series] = None
    base_margin: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    upstream_outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data consistency."""
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have same length. Got X: {len(self.X)}, y: {len(self.y)}"
            )
        if self.groups is not None and len(self.X) != len(self.groups):
            raise ValueError(
                f"X and groups must have same length. Got X: {len(self.X)}, groups: {len(self.groups)}"
            )
        if self.base_margin is not None and len(self.X) != len(self.base_margin):
            raise ValueError(
                f"X and base_margin must have same length. Got X: {len(self.X)}, base_margin: {len(self.base_margin)}"
            )

        # Warn about NaN values (don't error - user may handle it in preprocessing)
        if self.X.isnull().any().any():
            import warnings
            nan_count = self.X.isnull().sum().sum()
            warnings.warn(f"X contains {nan_count} NaN values")

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.X)

    @property
    def n_features(self) -> int:
        """Number of features in the dataset."""
        return self.X.shape[1]

    @property
    def feature_names(self) -> list[str]:
        """List of feature names."""
        return list(self.X.columns)

    def with_X(self, X: pd.DataFrame) -> DataContext:
        """Create a new context with updated features."""
        return DataContext(
            X=X,
            y=self.y,
            groups=self.groups,
            base_margin=self.base_margin,
            indices=self.indices,
            upstream_outputs=self.upstream_outputs,
            metadata=self.metadata,
        )

    def with_y(self, y: pd.Series) -> DataContext:
        """Create a new context with updated target."""
        return DataContext(
            X=self.X,
            y=y,
            groups=self.groups,
            base_margin=self.base_margin,
            indices=self.indices,
            upstream_outputs=self.upstream_outputs,
            metadata=self.metadata,
        )

    def with_indices(self, indices: np.ndarray) -> DataContext:
        """Create a new context with subset indices."""
        return DataContext(
            X=self.X.iloc[indices].reset_index(drop=True),
            y=self.y.iloc[indices].reset_index(drop=True) if self.y is not None else None,
            groups=self.groups.iloc[indices].reset_index(drop=True) if self.groups is not None else None,
            base_margin=self.base_margin[indices] if self.base_margin is not None else None,
            indices=indices,
            upstream_outputs={k: v[indices] for k, v in self.upstream_outputs.items()},
            metadata=self.metadata,
        )

    def with_upstream_output(self, node_name: str, output: np.ndarray) -> DataContext:
        """Create a new context with an additional upstream output."""
        new_outputs = dict(self.upstream_outputs)
        new_outputs[node_name] = output
        return DataContext(
            X=self.X,
            y=self.y,
            groups=self.groups,
            base_margin=self.base_margin,
            indices=self.indices,
            upstream_outputs=new_outputs,
            metadata=self.metadata,
        )

    def with_base_margin(self, base_margin: np.ndarray) -> DataContext:
        """Create a new context with base margin for stacking."""
        return DataContext(
            X=self.X,
            y=self.y,
            groups=self.groups,
            base_margin=base_margin,
            indices=self.indices,
            upstream_outputs=self.upstream_outputs,
            metadata=self.metadata,
        )

    def with_metadata(self, key: str, value: Any) -> DataContext:
        """Create a new context with additional metadata."""
        new_metadata = dict(self.metadata)
        new_metadata[key] = value
        return DataContext(
            X=self.X,
            y=self.y,
            groups=self.groups,
            base_margin=self.base_margin,
            indices=self.indices,
            upstream_outputs=self.upstream_outputs,
            metadata=new_metadata,
        )

    def augment_with_predictions(
        self, predictions: Dict[str, np.ndarray], prefix: str = "pred_"
    ) -> DataContext:
        """
        Create a new context with predictions added as features.

        This is used for stacking, where base model predictions become
        features for the meta-learner.
        """
        X_augmented = self.X.copy()
        for node_name, preds in predictions.items():
            # Validate predictions shape
            if len(preds) != len(self.X):
                raise ValueError(
                    f"Predictions for '{node_name}' have {len(preds)} samples but X has {len(self.X)}"
                )
            col_name = f"{prefix}{node_name}"
            if preds.ndim == 1:
                X_augmented[col_name] = preds
            else:
                # Multi-class probabilities
                for i in range(preds.shape[1]):
                    X_augmented[f"{col_name}_{i}"] = preds[:, i]

        return DataContext(
            X=X_augmented,
            y=self.y,
            groups=self.groups,
            base_margin=self.base_margin,
            indices=self.indices,
            upstream_outputs=self.upstream_outputs,
            metadata=self.metadata,
        )

    def copy(self) -> DataContext:
        """Create a shallow copy of the context."""
        return DataContext(
            X=self.X.copy(),
            y=self.y.copy() if self.y is not None else None,
            groups=self.groups.copy() if self.groups is not None else None,
            base_margin=self.base_margin.copy() if self.base_margin is not None else None,
            indices=self.indices.copy() if self.indices is not None else None,
            upstream_outputs=dict(self.upstream_outputs),
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        return (
            f"DataContext(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"has_y={self.y is not None}, has_groups={self.groups is not None}, "
            f"n_upstream={len(self.upstream_outputs)})"
        )
