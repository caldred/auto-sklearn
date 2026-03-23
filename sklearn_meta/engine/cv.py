"""CVEngine: Cross-validation fold creation and OOF routing."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from sklearn_meta.data.record import DEFAULT_TARGET_KEY
from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import (
    CVConfig,
    CVFold,
    CVResult,
    CVStrategy,
    FoldResult,
)

logger = logging.getLogger(__name__)


class CVEngine:
    """Cross-validation fold creation, lazy splitting, and OOF routing."""

    def __init__(self, cv_config: CVConfig) -> None:
        self.cv_config = cv_config

    def create_folds(
        self, data: DataView, target_key: Optional[str] = DEFAULT_TARGET_KEY
    ) -> List[CVFold]:
        """Create CV folds from the data view.

        Args:
            data: DataView containing features and optionally targets.
            target_key: Key into ``batch.targets`` to use as *y* for
                stratified splitting.  Pass ``None`` to skip target
                resolution (only valid for non-STRATIFIED strategies).
        """
        batch = data.materialize()

        # Resolve the target array
        if target_key is not None:
            y = batch.targets.get(target_key)
        else:
            y = None

        # y is only *required* for STRATIFIED splitting; the other
        # splitters (KFold, GroupKFold, TimeSeriesSplit) accept y but
        # do not use it.
        if self.cv_config.strategy == CVStrategy.STRATIFIED and y is None:
            raise ValueError(
                "Cannot create folds without target variable y "
                "when using STRATIFIED strategy"
            )

        n_samples = batch.n_samples
        if n_samples < self.cv_config.n_splits:
            raise ValueError(
                f"Dataset has {n_samples} samples but n_splits={self.cv_config.n_splits}. "
                f"Reduce n_splits or use more data."
            )

        # Resolve groups
        groups = None
        if data.groups is not None:
            groups = data.resolve_channel(data.groups)

        effective_strategy = self.cv_config.strategy
        if effective_strategy == CVStrategy.GROUP and groups is None:
            logger.warning(
                "CVConfig strategy is GROUP but no groups provided. "
                "Falling back to RANDOM."
            )
            effective_strategy = CVStrategy.RANDOM

        splitter = self._create_splitter(effective_strategy)
        folds = []

        X = batch.X
        if effective_strategy == CVStrategy.GROUP:
            split_iter = splitter.split(X, y, groups=groups)
        else:
            split_iter = splitter.split(X, y)

        fold_idx = 0
        repeat_idx = 0
        for train_idx, val_idx in split_iter:
            folds.append(
                CVFold(
                    fold_idx=fold_idx % self.cv_config.n_splits,
                    train_indices=np.array(train_idx),
                    val_indices=np.array(val_idx),
                    repeat_idx=repeat_idx,
                )
            )
            fold_idx += 1
            if fold_idx % self.cv_config.n_splits == 0:
                repeat_idx += 1

        return folds

    def split_for_fold(self, data: DataView, fold: CVFold) -> Tuple[DataView, DataView]:
        """Return lazy train/val views for a fold."""
        train_view = data.select_rows(fold.train_indices)
        val_view = data.select_rows(fold.val_indices)
        return train_view, val_view

    def route_oof_predictions(
        self,
        data: DataView,
        fold_results: List[FoldResult],
    ) -> np.ndarray:
        """Combine per-fold predictions into OOF predictions.

        For repeated CV (n_repeats > 1), averages across repeats using
        sum + count arrays (fixes the silent-overwrite bug in v1).
        """
        if not fold_results:
            raise ValueError("fold_results cannot be empty")

        n_samples = data.n_rows
        first_preds = fold_results[0].val_predictions

        if first_preds.ndim == 1:
            oof_sum = np.zeros(n_samples, dtype=np.float64)
            oof_count = np.zeros(n_samples, dtype=np.int32)
        else:
            oof_sum = np.zeros((n_samples, first_preds.shape[1]), dtype=np.float64)
            oof_count = np.zeros(n_samples, dtype=np.int32)

        for result in fold_results:
            oof_sum[result.fold.val_indices] += result.val_predictions
            oof_count[result.fold.val_indices] += 1

        # Avoid division by zero for samples never in validation
        safe_count = np.maximum(oof_count, 1)
        if oof_sum.ndim == 1:
            oof = oof_sum / safe_count
        else:
            oof = oof_sum / safe_count[:, np.newaxis]

        return oof

    def aggregate_cv_result(
        self,
        node_name: str,
        fold_results: List[FoldResult],
        data: DataView,
    ) -> CVResult:
        """Aggregate fold results into a CVResult."""
        oof_predictions = self.route_oof_predictions(data, fold_results)

        # Build repeat_oof if n_repeats > 1
        repeat_oof = None
        n_repeats = self.cv_config.n_repeats
        if n_repeats > 1:
            n_samples = data.n_rows
            first_preds = fold_results[0].val_predictions
            if first_preds.ndim == 1:
                repeat_oof = np.zeros((n_repeats, n_samples), dtype=np.float64)
            else:
                repeat_oof = np.zeros(
                    (n_repeats, n_samples, first_preds.shape[1]), dtype=np.float64
                )

            for result in fold_results:
                repeat_oof[result.fold.repeat_idx][result.fold.val_indices] = (
                    result.val_predictions
                )

        return CVResult(
            fold_results=fold_results,
            oof_predictions=oof_predictions,
            node_name=node_name,
            repeat_oof=repeat_oof,
        )

    def _create_splitter(self, strategy: Optional[CVStrategy] = None):
        """Create the appropriate sklearn splitter."""
        if strategy is None:
            strategy = self.cv_config.strategy
        n_splits = self.cv_config.n_splits
        n_repeats = self.cv_config.n_repeats
        random_state = self.cv_config.random_state

        if strategy == CVStrategy.GROUP:
            if n_repeats > 1:
                raise ValueError("Repeated GroupKFold is not supported")
            return GroupKFold(n_splits=n_splits)

        elif strategy == CVStrategy.STRATIFIED:
            if n_repeats > 1:
                return RepeatedStratifiedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=random_state,
                )
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=self.cv_config.shuffle,
                random_state=random_state,
            )

        elif strategy == CVStrategy.RANDOM:
            shuffle = self.cv_config.shuffle
            rs = random_state if shuffle else None
            if n_repeats > 1:
                return RepeatedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=rs,
                )
            return KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=rs,
            )

        elif strategy == CVStrategy.TIME_SERIES:
            if n_repeats > 1:
                raise ValueError("Repeated TimeSeriesSplit is not supported")
            return TimeSeriesSplit(n_splits=n_splits)

        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")

    def __repr__(self) -> str:
        return f"CVEngine(cv_config={self.cv_config})"
