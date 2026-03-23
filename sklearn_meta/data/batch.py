"""MaterializedBatch: Concrete data ready for model fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn_meta.data.record import DEFAULT_TARGET_KEY


@dataclass
class MaterializedBatch:
    """
    Concrete data ready for model fitting.

    Produced by DataView.materialize(). Contains fully resolved arrays.
    """

    X: pd.DataFrame
    row_ids: np.ndarray
    targets: Dict[str, np.ndarray]
    aux: Dict[str, np.ndarray]

    @property
    def y(self) -> Optional[np.ndarray]:
        """Shortcut for targets['__default__']. Not a second source of truth."""
        return self.targets.get(DEFAULT_TARGET_KEY)

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def feature_names(self) -> List[str]:
        return list(self.X.columns)

    def __repr__(self) -> str:
        return (
            f"MaterializedBatch(n_samples={self.n_samples}, "
            f"n_features={len(self.feature_names)}, "
            f"targets={list(self.targets.keys())})"
        )
