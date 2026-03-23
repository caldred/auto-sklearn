"""DataView: Lazy, declarative view over a DatasetRecord."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn_meta.data.batch import MaterializedBatch
from sklearn_meta.data.record import ChannelRef, DatasetRecord, DEFAULT_TARGET_KEY, RowSelector


def _coerce_array(value) -> np.ndarray:
    """Coerce array-like inputs to np.ndarray immediately."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, pd.Series):
        return value.values
    return np.asarray(value)


@dataclass(frozen=True)
class DataView:
    """
    Lazy, declarative view over a DatasetRecord.

    All mutating operations return new DataView instances.
    No data is copied until materialize() is called.
    """

    dataset: DatasetRecord
    row_sel: Optional[RowSelector] = None
    feature_cols: Tuple[str, ...] = ()
    targets: Mapping[str, ChannelRef] = field(default_factory=dict)
    groups: Optional[ChannelRef] = None
    aux: Mapping[str, ChannelRef] = field(default_factory=dict)
    overlays: Mapping[str, np.ndarray] = field(default_factory=dict)

    # --- Lazy operations (return new DataView, no data copy) ---

    def select_rows(self, indices: np.ndarray) -> DataView:
        """Select rows by integer indices. Composes with existing row_sel."""
        if self.row_sel is not None:
            indices = self.row_sel[indices]
        return replace(self, row_sel=indices)

    def select_features(self, cols: Sequence[str]) -> DataView:
        """Restrict feature columns to a subset.

        Overlay names in *cols* are silently ignored since overlays are
        always appended during materialization.
        """
        overlay_names = set(self.overlays)
        base_cols = tuple(c for c in cols if c not in overlay_names)
        return replace(self, feature_cols=base_cols)

    def with_overlay(self, name: str, values: np.ndarray) -> DataView:
        """Add or replace an overlay. Values must be full-dataset-length."""
        values = _coerce_array(values)
        if values.shape[0] != self.dataset.n_rows:
            raise ValueError(
                f"Overlay '{name}' length ({values.shape[0]}) must match "
                f"dataset length ({self.dataset.n_rows})"
            )
        new_overlays = dict(self.overlays)
        new_overlays[name] = values
        return replace(self, overlays=new_overlays)

    def with_overlays(self, predictions: Dict[str, np.ndarray]) -> DataView:
        """Add multiple overlays at once."""
        new_overlays = dict(self.overlays)
        for name, values in predictions.items():
            values = _coerce_array(values)
            if values.shape[0] != self.dataset.n_rows:
                raise ValueError(
                    f"Overlay '{name}' length ({values.shape[0]}) must match "
                    f"dataset length ({self.dataset.n_rows})"
                )
            new_overlays[name] = values
        return replace(self, overlays=new_overlays)

    def bind_target(self, target: ChannelRef, name: str = DEFAULT_TARGET_KEY) -> DataView:
        """Bind a named target channel."""
        if isinstance(target, np.ndarray):
            pass  # already ndarray
        elif isinstance(target, (pd.Series, list)):
            target = _coerce_array(target)
        new_targets = dict(self.targets)
        new_targets[name] = target
        return replace(self, targets=new_targets)

    def bind_groups(self, groups: ChannelRef) -> DataView:
        """Bind groups for CV splitting."""
        if not isinstance(groups, str):
            groups = _coerce_array(groups)
        return replace(self, groups=groups)

    def with_aux(self, key: str, value: ChannelRef) -> DataView:
        """Add an auxiliary channel."""
        if not isinstance(value, str):
            value = _coerce_array(value)
        new_aux = dict(self.aux)
        new_aux[key] = value
        return replace(self, aux=new_aux)

    # --- Target access ---

    @property
    def target(self) -> Optional[ChannelRef]:
        """Default target (convenience for single-target nodes)."""
        return self.targets.get(DEFAULT_TARGET_KEY)

    # --- Materialization ---

    def materialize(self) -> MaterializedBatch:
        """Resolve all lazy references into concrete arrays."""
        # 1. Slice DataFrame to selected rows
        if self.row_sel is not None:
            df_slice = self.dataset.frame.iloc[self.row_sel]
            row_ids = np.asarray(self.dataset.row_ids[self.row_sel])
        else:
            df_slice = self.dataset.frame
            row_ids = np.asarray(self.dataset.row_ids)

        # 2. Extract feature columns
        X = df_slice[list(self.feature_cols)].copy()

        # 3. Append overlay columns (row-selected from full-length arrays)
        for overlay_name, overlay_values in self.overlays.items():
            if self.row_sel is not None:
                sliced = overlay_values[self.row_sel]
            else:
                sliced = overlay_values
            if sliced.ndim == 1:
                X[overlay_name] = sliced
            else:
                for i in range(sliced.shape[1]):
                    X[f"{overlay_name}_{i}"] = sliced[:, i]

        # 4. Resolve each target
        resolved_targets: Dict[str, np.ndarray] = {}
        for tname, tref in self.targets.items():
            resolved_targets[tname] = self.resolve_channel(tref)

        # 5. Resolve aux channels
        resolved_aux: Dict[str, np.ndarray] = {}
        for aname, aref in self.aux.items():
            resolved_aux[aname] = self.resolve_channel(aref)

        return MaterializedBatch(
            X=X,
            row_ids=row_ids,
            targets=resolved_targets,
            aux=resolved_aux,
        )

    def resolve_channel(self, ref: ChannelRef) -> np.ndarray:
        """Resolve a ChannelRef to a concrete numpy array for current rows."""
        if isinstance(ref, str):
            # Column name in the dataset frame
            col_data = self.dataset.frame[ref].values
        elif isinstance(ref, np.ndarray):
            col_data = ref
        else:
            col_data = np.asarray(ref)

        if self.row_sel is not None:
            return col_data[self.row_sel]
        return col_data

    # --- Properties ---

    @property
    def effective_row_ids(self) -> np.ndarray:
        if self.row_sel is not None:
            return np.asarray(self.dataset.row_ids[self.row_sel])
        return np.asarray(self.dataset.row_ids)

    @property
    def n_rows(self) -> int:
        if self.row_sel is not None:
            return len(self.row_sel)
        return self.dataset.n_rows

    @property
    def n_features(self) -> int:
        return len(self.feature_cols) + len(self.overlays)

    # --- Factories ---

    @classmethod
    def from_Xy(
        cls,
        X: pd.DataFrame,
        y=None,
        groups=None,
        **aux,
    ) -> DataView:
        """Create a DataView from separate X, y, groups, and aux arrays."""
        record = DatasetRecord.from_frame(X)
        feature_cols = tuple(X.columns)

        targets: Dict[str, ChannelRef] = {}
        if y is not None:
            targets[DEFAULT_TARGET_KEY] = _coerce_array(y)

        resolved_groups = None
        if groups is not None:
            resolved_groups = _coerce_array(groups)

        resolved_aux: Dict[str, ChannelRef] = {}
        for key, value in aux.items():
            resolved_aux[key] = _coerce_array(value)

        return cls(
            dataset=record,
            feature_cols=feature_cols,
            targets=targets,
            groups=resolved_groups,
            aux=resolved_aux,
        )

    @classmethod
    def from_X(cls, X: pd.DataFrame) -> DataView:
        """Create a DataView from features only."""
        return cls.from_Xy(X)

    def __repr__(self) -> str:
        return (
            f"DataView(n_rows={self.n_rows}, n_features={self.n_features}, "
            f"has_target={DEFAULT_TARGET_KEY in self.targets}, "
            f"has_groups={self.groups is not None})"
        )
