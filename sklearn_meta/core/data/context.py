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

    Holds a single DataFrame with column roles declared via feature_cols,
    target_col, and group_col. Identity columns (player_id, game_id, etc.)
    are preserved in df but excluded from feature_cols.

    Attributes:
        df: Single DataFrame with ALL columns.
        feature_cols: Which columns are features (tuple for frozen dataclass).
        target_col: Which column is the target (optional).
        group_col: Which column is the group (optional).
        base_margin: XGBoost base margin (stays separate, not a natural column).
        indices: Original indices for subset tracking (optional).
        metadata: Additional metadata for the context.
    """

    df: pd.DataFrame
    feature_cols: tuple[str, ...] = ()
    target_col: Optional[str] = None
    group_col: Optional[str] = None
    base_margin: Optional[np.ndarray] = None
    soft_targets: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data consistency."""
        # Validate feature_cols exist in df
        missing_features = set(self.feature_cols) - set(self.df.columns)
        if missing_features:
            raise ValueError(
                f"feature_cols not found in df: {missing_features}"
            )

        # Validate target_col exists in df
        if self.target_col is not None and self.target_col not in self.df.columns:
            raise ValueError(
                f"target_col '{self.target_col}' not found in df columns"
            )

        # Validate group_col exists in df
        if self.group_col is not None and self.group_col not in self.df.columns:
            raise ValueError(
                f"group_col '{self.group_col}' not found in df columns"
            )

        # Validate base_margin length
        if self.base_margin is not None and len(self.df) != len(self.base_margin):
            raise ValueError(
                f"df and base_margin must have same length. "
                f"Got df: {len(self.df)}, base_margin: {len(self.base_margin)}"
            )

        # Validate soft_targets length
        if self.soft_targets is not None and len(self.df) != len(self.soft_targets):
            raise ValueError(
                f"df and soft_targets must have same length. "
                f"Got df: {len(self.df)}, soft_targets: {len(self.soft_targets)}"
            )

    @classmethod
    def from_Xy(
        cls,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
        base_margin: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataContext:
        """Construct a DataContext from separate X, y, groups."""
        df = X.copy()
        feature_cols = tuple(X.columns)

        target_col = None
        if y is not None:
            if len(X) != len(y):
                raise ValueError(
                    f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
                )
            target_name = y.name if y.name else "__target__"
            # Avoid collision with feature column names
            target_col = target_name if target_name not in df.columns else "__target__"
            df[target_col] = y.values

        group_col = None
        if groups is not None:
            if len(X) != len(groups):
                raise ValueError(
                    f"X and groups must have same length. "
                    f"Got X: {len(X)}, groups: {len(groups)}"
                )
            group_name = groups.name if groups.name else "__groups__"
            group_col = group_name if group_name not in df.columns else "__groups__"
            df[group_col] = groups.values

        return cls(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            base_margin=base_margin,
            indices=indices,
            metadata=metadata or {},
        )

    @property
    def X(self) -> pd.DataFrame:
        """Feature DataFrame."""
        return self.df[list(self.feature_cols)]

    @property
    def y(self) -> Optional[pd.Series]:
        """Target Series."""
        if self.target_col is None:
            return None
        return self.df[self.target_col]

    @property
    def groups(self) -> Optional[pd.Series]:
        """Group labels Series."""
        if self.group_col is None:
            return None
        return self.df[self.group_col]

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.df)

    @property
    def n_features(self) -> int:
        """Number of features in the dataset."""
        return len(self.feature_cols)

    @property
    def feature_names(self) -> list[str]:
        """List of feature names."""
        return list(self.feature_cols)

    def with_feature_cols(self, feature_cols: list[str]) -> DataContext:
        """Create a new context with updated feature columns."""
        return DataContext(
            df=self.df,
            feature_cols=tuple(feature_cols),
            target_col=self.target_col,
            group_col=self.group_col,
            base_margin=self.base_margin,
            soft_targets=self.soft_targets,
            indices=self.indices,
            metadata=self.metadata,
        )

    def with_target_col(self, col_name: str) -> DataContext:
        """Create a new context pointing to a different target column."""
        return DataContext(
            df=self.df,
            feature_cols=self.feature_cols,
            target_col=col_name,
            group_col=self.group_col,
            base_margin=self.base_margin,
            soft_targets=self.soft_targets,
            indices=self.indices,
            metadata=self.metadata,
        )

    def with_columns(self, as_features: bool = False, **cols: Any) -> DataContext:
        """
        Add columns to df.

        Args:
            as_features: If True, also extend feature_cols with the new columns.
            **cols: Column name -> values mapping.

        Returns:
            New DataContext with the columns added.
        """
        new_df = self.df.copy()
        new_feature_cols = list(self.feature_cols)

        for col_name, values in cols.items():
            new_df[col_name] = values
            if as_features and col_name not in new_feature_cols:
                new_feature_cols.append(col_name)

        return DataContext(
            df=new_df,
            feature_cols=tuple(new_feature_cols),
            target_col=self.target_col,
            group_col=self.group_col,
            base_margin=self.base_margin,
            soft_targets=self.soft_targets,
            indices=self.indices,
            metadata=self.metadata,
        )

    def with_indices(self, indices: np.ndarray) -> DataContext:
        """Create a new context with subset indices."""
        return DataContext(
            df=self.df.iloc[indices].reset_index(drop=True),
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            group_col=self.group_col,
            base_margin=self.base_margin[indices] if self.base_margin is not None else None,
            soft_targets=self.soft_targets[indices] if self.soft_targets is not None else None,
            indices=indices,
            metadata=self.metadata,
        )

    def with_base_margin(self, base_margin: np.ndarray) -> DataContext:
        """Create a new context with base margin for stacking."""
        return DataContext(
            df=self.df,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            group_col=self.group_col,
            base_margin=base_margin,
            soft_targets=self.soft_targets,
            indices=self.indices,
            metadata=self.metadata,
        )

    def with_soft_targets(self, soft_targets: np.ndarray) -> DataContext:
        """Create a new context with soft targets for distillation."""
        return DataContext(
            df=self.df,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            group_col=self.group_col,
            base_margin=self.base_margin,
            soft_targets=soft_targets,
            indices=self.indices,
            metadata=self.metadata,
        )

    def with_metadata(self, key: str, value: Any) -> DataContext:
        """Create a new context with additional metadata."""
        new_metadata = dict(self.metadata)
        new_metadata[key] = value
        return DataContext(
            df=self.df,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            group_col=self.group_col,
            base_margin=self.base_margin,
            soft_targets=self.soft_targets,
            indices=self.indices,
            metadata=new_metadata,
        )

    def with_y(self, y: pd.Series) -> DataContext:
        """Create a new context with updated target."""
        target_name = y.name if y.name else "__target__"
        target_col = target_name if target_name not in self.feature_cols else "__target__"
        new_df = self.df.copy()
        # Remove old target column if different and not a feature
        if (self.target_col is not None
                and self.target_col != target_col
                and self.target_col in new_df.columns
                and self.target_col not in self.feature_cols):
            new_df = new_df.drop(columns=[self.target_col])
        new_df[target_col] = y.values

        return DataContext(
            df=new_df,
            feature_cols=self.feature_cols,
            target_col=target_col,
            group_col=self.group_col,
            base_margin=self.base_margin,
            soft_targets=self.soft_targets,
            indices=self.indices,
            metadata=self.metadata,
        )

    def augment_with_predictions(
        self, predictions: Dict[str, np.ndarray], prefix: str = "pred_"
    ) -> DataContext:
        """
        Create a new context with predictions added as features.

        This is used for stacking, where base model predictions become
        features for the meta-learner.
        """
        new_df = self.df.copy()
        new_feature_cols = list(self.feature_cols)

        for node_name, preds in predictions.items():
            if len(preds) != len(self.df):
                raise ValueError(
                    f"Predictions for '{node_name}' have {len(preds)} samples "
                    f"but df has {len(self.df)}"
                )
            col_name = f"{prefix}{node_name}"
            if preds.ndim == 1:
                new_df[col_name] = preds
                if col_name not in new_feature_cols:
                    new_feature_cols.append(col_name)
            else:
                for i in range(preds.shape[1]):
                    sub_col = f"{col_name}_{i}"
                    new_df[sub_col] = preds[:, i]
                    if sub_col not in new_feature_cols:
                        new_feature_cols.append(sub_col)

        return DataContext(
            df=new_df,
            feature_cols=tuple(new_feature_cols),
            target_col=self.target_col,
            group_col=self.group_col,
            base_margin=self.base_margin,
            soft_targets=self.soft_targets,
            indices=self.indices,
            metadata=self.metadata,
        )

    def copy(self) -> DataContext:
        """Create a shallow copy of the context."""
        return DataContext(
            df=self.df.copy(deep=False),
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            group_col=self.group_col,
            base_margin=self.base_margin,
            soft_targets=self.soft_targets,
            indices=self.indices,
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        return (
            f"DataContext(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"has_target={self.target_col is not None}, "
            f"has_groups={self.group_col is not None})"
        )
