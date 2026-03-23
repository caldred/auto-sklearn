"""DatasetRecord: Immutable base table with stable row identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Union

import numpy as np
import pandas as pd


ChannelRef = Union[str, np.ndarray]
RowSelector = np.ndarray  # integer index array into DatasetRecord
DEFAULT_TARGET_KEY = "__default__"


@dataclass(frozen=True)
class DatasetRecord:
    """
    Immutable base table holding the full dataset once.

    The frame is never copied — views and slices are deferred
    to DataView.materialize().
    """

    frame: pd.DataFrame
    row_ids: pd.Index
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.row_ids) != len(self.frame):
            raise ValueError(
                f"row_ids length ({len(self.row_ids)}) must match "
                f"frame length ({len(self.frame)})"
            )

    @classmethod
    def from_frame(cls, df: pd.DataFrame, metadata: Mapping[str, Any] | None = None) -> DatasetRecord:
        """Create a DatasetRecord from a DataFrame, using its index as row_ids."""
        return cls(frame=df, row_ids=df.index, metadata=metadata or {})

    @property
    def n_rows(self) -> int:
        return len(self.frame)
