"""Cross-validation configuration and fold management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class CVStrategy(Enum):
    """Cross-validation splitting strategies."""

    GROUP = "group"
    STRATIFIED = "stratified"
    RANDOM = "random"
    TIME_SERIES = "time_series"


@dataclass
class CVFold:
    """
    Represents a single cross-validation fold.

    Attributes:
        fold_idx: Index of this fold (0-based).
        train_indices: Indices of training samples.
        val_indices: Indices of validation samples.
        repeat_idx: Index of the repeat (for repeated CV).
    """

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    repeat_idx: int = 0

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train_indices)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.val_indices)

    def __repr__(self) -> str:
        return (
            f"CVFold(fold={self.fold_idx}, repeat={self.repeat_idx}, "
            f"n_train={self.n_train}, n_val={self.n_val})"
        )


@dataclass
class NestedCVFold:
    """
    Represents a nested cross-validation fold structure.

    Used for proper hyperparameter tuning within CV to avoid data leakage.
    The outer fold is used for final evaluation, while inner folds are
    used for hyperparameter tuning.

    Attributes:
        outer_fold: The outer CV fold for final evaluation.
        inner_folds: List of inner CV folds for hyperparameter tuning.
    """

    outer_fold: CVFold
    inner_folds: List[CVFold]

    @property
    def fold_idx(self) -> int:
        """Index of the outer fold."""
        return self.outer_fold.fold_idx

    @property
    def n_inner_folds(self) -> int:
        """Number of inner folds."""
        return len(self.inner_folds)

    def __repr__(self) -> str:
        return (
            f"NestedCVFold(outer={self.outer_fold}, "
            f"n_inner_folds={self.n_inner_folds})"
        )


@dataclass
class CVConfig:
    """
    Cross-validation configuration.

    Supports various CV strategies and nested CV for proper hyperparameter tuning.

    Attributes:
        n_splits: Number of CV folds.
        n_repeats: Number of times to repeat the CV (for repeated CV).
        strategy: CV splitting strategy.
        shuffle: Whether to shuffle before splitting.
        random_state: Random seed for reproducibility.
        inner_cv: Configuration for inner CV (for nested CV).
    """

    n_splits: int = 5
    n_repeats: int = 1
    strategy: CVStrategy = CVStrategy.GROUP
    shuffle: bool = True
    random_state: int = 42
    inner_cv: Optional[CVConfig] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {self.n_repeats}")
        if isinstance(self.strategy, str):
            self.strategy = CVStrategy(self.strategy)

    @property
    def is_nested(self) -> bool:
        """Whether this is a nested CV configuration."""
        return self.inner_cv is not None

    @property
    def total_folds(self) -> int:
        """Total number of folds including repeats."""
        return self.n_splits * self.n_repeats

    def with_inner_cv(
        self,
        n_splits: int = 3,
        strategy: Optional[CVStrategy] = None,
    ) -> CVConfig:
        """Create a new config with inner CV for nested cross-validation."""
        inner_strategy = strategy if strategy is not None else self.strategy
        return CVConfig(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            strategy=self.strategy,
            shuffle=self.shuffle,
            random_state=self.random_state,
            inner_cv=CVConfig(
                n_splits=n_splits,
                n_repeats=1,
                strategy=inner_strategy,
                shuffle=self.shuffle,
                random_state=self.random_state + 1,
            ),
        )

    def __repr__(self) -> str:
        nested_str = f", inner_cv={self.inner_cv}" if self.is_nested else ""
        return (
            f"CVConfig(n_splits={self.n_splits}, n_repeats={self.n_repeats}, "
            f"strategy={self.strategy.value}{nested_str})"
        )


@dataclass
class FoldResult:
    """
    Result from fitting a model on a single fold.

    Attributes:
        fold: The CV fold this result is from.
        model: The fitted model.
        val_predictions: Predictions on the validation set.
        val_score: Validation score (e.g., accuracy, AUC).
        train_score: Training score (optional).
        fit_time: Time to fit the model in seconds.
        predict_time: Time to predict in seconds.
        params: Parameters used for this fold.
    """

    fold: CVFold
    model: object
    val_predictions: np.ndarray
    val_score: float
    train_score: Optional[float] = None
    fit_time: float = 0.0
    predict_time: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def fold_idx(self) -> int:
        """Index of the fold."""
        return self.fold.fold_idx

    def __repr__(self) -> str:
        return (
            f"FoldResult(fold={self.fold_idx}, val_score={self.val_score:.4f}, "
            f"fit_time={self.fit_time:.2f}s)"
        )


@dataclass
class CVResult:
    """
    Aggregated results from cross-validation.

    Attributes:
        fold_results: List of per-fold results.
        oof_predictions: Out-of-fold predictions for all samples.
        node_name: Name of the model node.
    """

    fold_results: List[FoldResult]
    oof_predictions: np.ndarray
    node_name: str

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return len(self.fold_results)

    @property
    def val_scores(self) -> np.ndarray:
        """Array of validation scores."""
        return np.array([r.val_score for r in self.fold_results])

    @property
    def mean_score(self) -> float:
        """Mean validation score across folds."""
        return float(np.mean(self.val_scores))

    @property
    def std_score(self) -> float:
        """Standard deviation of validation scores."""
        return float(np.std(self.val_scores))

    @property
    def total_fit_time(self) -> float:
        """Total time to fit all folds."""
        return sum(r.fit_time for r in self.fold_results)

    @property
    def models(self) -> List[object]:
        """List of fitted models from all folds."""
        return [r.model for r in self.fold_results]

    def __repr__(self) -> str:
        return (
            f"CVResult(node={self.node_name}, n_folds={self.n_folds}, "
            f"mean_score={self.mean_score:.4f} +/- {self.std_score:.4f})"
        )
