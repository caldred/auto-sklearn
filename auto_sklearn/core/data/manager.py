"""DataManager: Handles all splitting and CV logic."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from auto_sklearn.core.data.context import DataContext
from auto_sklearn.core.data.cv import (
    CVConfig,
    CVFold,
    CVResult,
    CVStrategy,
    FoldResult,
    NestedCVFold,
)


class DataManager:
    """
    Handles all splitting and CV logic.

    This class is responsible for:
    - Creating CV folds (including nested CV for proper hyperparameter tuning)
    - Aligning data contexts to specific folds
    - Routing out-of-fold predictions
    """

    def __init__(self, cv_config: CVConfig) -> None:
        """
        Initialize the DataManager.

        Args:
            cv_config: Cross-validation configuration.
        """
        self.cv_config = cv_config

    def create_folds(self, ctx: DataContext) -> List[CVFold]:
        """
        Create CV folds from the data context.

        Args:
            ctx: Data context containing X, y, and optionally groups.

        Returns:
            List of CVFold objects.
        """
        if ctx.y is None:
            raise ValueError("Cannot create folds without target variable y")

        splitter = self._create_splitter(ctx)
        folds = []

        # Get split iterator
        if self.cv_config.strategy == CVStrategy.GROUP:
            if ctx.groups is None:
                raise ValueError("Group CV requires groups in DataContext")
            split_iter = splitter.split(ctx.X, ctx.y, groups=ctx.groups)
        else:
            split_iter = splitter.split(ctx.X, ctx.y)

        # Create folds
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

    def create_nested_folds(self, ctx: DataContext) -> List[NestedCVFold]:
        """
        Create nested CV folds for proper hyperparameter tuning.

        The outer folds are used for final evaluation, while inner folds
        within each outer training set are used for hyperparameter tuning.

        Args:
            ctx: Data context containing X, y, and optionally groups.

        Returns:
            List of NestedCVFold objects.
        """
        if not self.cv_config.is_nested:
            raise ValueError("CVConfig must have inner_cv for nested CV")

        outer_folds = self.create_folds(ctx)
        nested_folds = []

        for outer_fold in outer_folds:
            # Create context for outer training set
            train_ctx = ctx.with_indices(outer_fold.train_indices)

            # Create inner folds on the outer training set
            inner_manager = DataManager(self.cv_config.inner_cv)
            inner_folds_raw = inner_manager.create_folds(train_ctx)

            # Remap inner fold indices to original indices
            inner_folds = []
            for inner_fold in inner_folds_raw:
                inner_folds.append(
                    CVFold(
                        fold_idx=inner_fold.fold_idx,
                        train_indices=outer_fold.train_indices[inner_fold.train_indices],
                        val_indices=outer_fold.train_indices[inner_fold.val_indices],
                        repeat_idx=inner_fold.repeat_idx,
                    )
                )

            nested_folds.append(NestedCVFold(outer_fold=outer_fold, inner_folds=inner_folds))

        return nested_folds

    def align_to_fold(
        self, ctx: DataContext, fold: CVFold
    ) -> Tuple[DataContext, DataContext]:
        """
        Split a context into train and validation contexts for a fold.

        Args:
            ctx: Full data context.
            fold: CV fold defining the split.

        Returns:
            Tuple of (train_context, val_context).
        """
        train_ctx = self._subset_context(ctx, fold.train_indices)
        val_ctx = self._subset_context(ctx, fold.val_indices)
        return train_ctx, val_ctx

    def route_oof_predictions(
        self,
        ctx: DataContext,
        fold_results: List[FoldResult],
    ) -> np.ndarray:
        """
        Combine per-fold predictions into out-of-fold predictions.

        This ensures each sample's prediction comes from a model that
        didn't see that sample during training.

        Args:
            ctx: Original data context.
            fold_results: Results from each CV fold.

        Returns:
            Array of OOF predictions with shape (n_samples,) or (n_samples, n_classes).
        """
        # Determine output shape from first result
        first_preds = fold_results[0].val_predictions
        if first_preds.ndim == 1:
            oof = np.zeros(ctx.n_samples, dtype=first_preds.dtype)
        else:
            oof = np.zeros((ctx.n_samples, first_preds.shape[1]), dtype=first_preds.dtype)

        # Fill in predictions from each fold
        for result in fold_results:
            oof[result.fold.val_indices] = result.val_predictions

        return oof

    def aggregate_cv_result(
        self,
        node_name: str,
        fold_results: List[FoldResult],
        ctx: DataContext,
    ) -> CVResult:
        """
        Aggregate fold results into a CVResult.

        Args:
            node_name: Name of the model node.
            fold_results: Results from each CV fold.
            ctx: Original data context.

        Returns:
            Aggregated CV result with OOF predictions.
        """
        oof_predictions = self.route_oof_predictions(ctx, fold_results)
        return CVResult(
            fold_results=fold_results,
            oof_predictions=oof_predictions,
            node_name=node_name,
        )

    def _create_splitter(self, ctx: DataContext):
        """Create the appropriate sklearn splitter."""
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
            if n_repeats > 1:
                return RepeatedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=random_state,
                )
            return KFold(
                n_splits=n_splits,
                shuffle=self.cv_config.shuffle,
                random_state=random_state,
            )

        elif strategy == CVStrategy.TIME_SERIES:
            if n_repeats > 1:
                raise ValueError("Repeated TimeSeriesSplit is not supported")
            return TimeSeriesSplit(n_splits=n_splits)

        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")

    def _subset_context(self, ctx: DataContext, indices: np.ndarray) -> DataContext:
        """Create a subset of the context for given indices."""
        X_subset = ctx.X.iloc[indices].reset_index(drop=True)
        y_subset = ctx.y.iloc[indices].reset_index(drop=True) if ctx.y is not None else None
        groups_subset = (
            ctx.groups.iloc[indices].reset_index(drop=True)
            if ctx.groups is not None
            else None
        )
        base_margin_subset = (
            ctx.base_margin[indices] if ctx.base_margin is not None else None
        )
        upstream_subset = {k: v[indices] for k, v in ctx.upstream_outputs.items()}

        return DataContext(
            X=X_subset,
            y=y_subset,
            groups=groups_subset,
            base_margin=base_margin_subset,
            indices=indices,
            upstream_outputs=upstream_subset,
            metadata=ctx.metadata,
        )

    def __repr__(self) -> str:
        return f"DataManager(cv_config={self.cv_config})"
