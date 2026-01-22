"""JointQuantileOrchestrator: Training coordinator for joint quantile regression."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVFold, CVResult, FoldResult
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.dependency import DependencyType
from sklearn_meta.core.model.joint_quantile_graph import JointQuantileGraph
from sklearn_meta.core.model.quantile_node import QuantileModelNode
from sklearn_meta.core.tuning.orchestrator import TuningConfig

if TYPE_CHECKING:
    from sklearn_meta.audit.logger import AuditLogger
    from sklearn_meta.execution.base import Executor
    from sklearn_meta.plugins.registry import PluginRegistry
    from sklearn_meta.search.backends.base import OptimizationResult, SearchBackend


@dataclass
class FittedQuantileNode:
    """
    Result of fitting a single quantile node.

    Stores models for all quantile levels and their OOF predictions.

    Attributes:
        node: The original QuantileModelNode definition.
        quantile_models: Dict mapping tau -> list of fold models.
        oof_quantile_predictions: OOF predictions for all quantiles.
                                 Shape: (n_samples, n_quantiles).
        best_params: Best hyperparameters (tuned at median).
        optimization_result: Full optimization results (optional).
    """

    node: QuantileModelNode
    quantile_models: Dict[float, List[Any]]
    oof_quantile_predictions: np.ndarray
    best_params: Dict[str, Any]
    optimization_result: Optional[OptimizationResult] = None

    @property
    def quantile_levels(self) -> List[float]:
        """Get the quantile levels."""
        return sorted(self.quantile_models.keys())

    @property
    def n_quantiles(self) -> int:
        """Number of quantile levels."""
        return len(self.quantile_models)

    @property
    def n_folds(self) -> int:
        """Number of CV folds."""
        tau = list(self.quantile_models.keys())[0]
        return len(self.quantile_models[tau])

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict all quantile levels for input features.

        Args:
            X: Input features.

        Returns:
            Predictions of shape (n_samples, n_quantiles).
        """
        predictions = np.zeros((len(X), self.n_quantiles))

        for q_idx, tau in enumerate(self.quantile_levels):
            # Average predictions across all fold models
            fold_preds = []
            for model in self.quantile_models[tau]:
                fold_preds.append(model.predict(X))
            predictions[:, q_idx] = np.mean(fold_preds, axis=0)

        return predictions

    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict median (or closest quantile to 0.5).

        Args:
            X: Input features.

        Returns:
            Median predictions of shape (n_samples,).
        """
        median_tau = min(self.quantile_levels, key=lambda x: abs(x - 0.5))
        fold_preds = []
        for model in self.quantile_models[median_tau]:
            fold_preds.append(model.predict(X))
        return np.mean(fold_preds, axis=0)


@dataclass
class JointQuantileFitResult:
    """
    Result of fitting a JointQuantileGraph.

    Attributes:
        graph: The original JointQuantileGraph.
        fitted_nodes: Dict mapping property names to FittedQuantileNode.
        tuning_config: Configuration used for tuning.
        total_time: Total time taken in seconds.
    """

    graph: JointQuantileGraph
    fitted_nodes: Dict[str, FittedQuantileNode]
    tuning_config: TuningConfig
    total_time: float = 0.0

    def get_node(self, property_name: str) -> FittedQuantileNode:
        """Get a fitted node by property name."""
        return self.fitted_nodes[property_name]

    def __repr__(self) -> str:
        return (
            f"JointQuantileFitResult(properties={list(self.fitted_nodes.keys())}, "
            f"time={self.total_time:.1f}s)"
        )


class JointQuantileOrchestrator:
    """
    Training coordinator for joint quantile regression.

    Handles:
    - Sequential fitting of quantile models in chain order
    - Conditioning on actual Y values during training
    - Hyperparameter tuning at median quantile
    - Training models for all quantile levels

    The key difference from standard TuningOrchestrator is that during
    training, downstream nodes condition on ACTUAL Y values from upstream
    properties (not predictions), ensuring proper density estimation.

    Example:
        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=dm,
            search_backend=optuna_backend,
            tuning_config=config,
        )

        result = orchestrator.fit(ctx, targets={
            "price": y_price,
            "volume": y_volume,
            "volatility": y_volatility,
        })
    """

    def __init__(
        self,
        graph: JointQuantileGraph,
        data_manager: DataManager,
        search_backend: SearchBackend,
        tuning_config: TuningConfig,
        executor: Optional[Executor] = None,
        plugin_registry: Optional[PluginRegistry] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            graph: JointQuantileGraph to train.
            data_manager: Handles CV splitting and data routing.
            search_backend: Backend for hyperparameter optimization.
            tuning_config: Tuning configuration.
            executor: Optional executor for parallelization.
            plugin_registry: Optional registry of model plugins.
            audit_logger: Optional logger for auditing.
        """
        self.graph = graph
        self.data_manager = data_manager
        self.search_backend = search_backend
        self.tuning_config = tuning_config
        self.executor = executor
        self.plugin_registry = plugin_registry
        self.audit_logger = audit_logger

        # Validate graph
        self.graph.validate()

    def fit(
        self,
        ctx: DataContext,
        targets: Dict[str, pd.Series],
    ) -> JointQuantileFitResult:
        """
        Fit the joint quantile model.

        Args:
            ctx: Data context with input features X.
            targets: Dict mapping property names to target Series.
                    e.g., {"price": y_price, "volume": y_volume}

        Returns:
            JointQuantileFitResult with all fitted nodes.
        """
        start_time = time.time()

        # Validate targets
        for prop_name in self.graph.property_order:
            if prop_name not in targets:
                raise ValueError(f"Missing target for property: {prop_name}")
            if len(targets[prop_name]) != ctx.n_samples:
                raise ValueError(
                    f"Target '{prop_name}' length mismatch: "
                    f"expected {ctx.n_samples}, got {len(targets[prop_name])}"
                )

        fitted_nodes: Dict[str, FittedQuantileNode] = {}
        oof_cache: Dict[str, np.ndarray] = {}

        # Fit nodes in topological order (which is the property order)
        for prop_name in self.graph.property_order:
            if self.tuning_config.verbose >= 1:
                logger.info(f"Fitting quantile models for property: {prop_name}")

            node = self.graph.get_quantile_node(prop_name)

            # Prepare context with conditioning features
            prop_ctx = self._prepare_conditional_context(
                ctx, prop_name, targets, fitted_nodes
            )

            # Create context with appropriate target
            prop_ctx = prop_ctx.with_y(targets[prop_name])

            # Fit the node
            fitted_node = self._fit_quantile_node(node, prop_ctx)
            fitted_nodes[prop_name] = fitted_node

            # Cache median OOF predictions for conditioning downstream nodes
            median_idx = fitted_node.quantile_levels.index(
                min(fitted_node.quantile_levels, key=lambda x: abs(x - 0.5))
            )
            oof_cache[prop_name] = fitted_node.oof_quantile_predictions[:, median_idx]

        total_time = time.time() - start_time

        return JointQuantileFitResult(
            graph=self.graph,
            fitted_nodes=fitted_nodes,
            tuning_config=self.tuning_config,
            total_time=total_time,
        )

    def _prepare_conditional_context(
        self,
        ctx: DataContext,
        current_prop: str,
        targets: Dict[str, pd.Series],
        fitted_nodes: Dict[str, FittedQuantileNode],
    ) -> DataContext:
        """
        Prepare context by adding actual Y values from upstream properties.

        During training, we condition on ACTUAL values, not predictions.
        This ensures proper density estimation.

        Args:
            ctx: Original data context.
            current_prop: Current property being fitted.
            targets: All target values.
            fitted_nodes: Already fitted nodes.

        Returns:
            Context with conditioning features added.
        """
        # Get upstream properties
        upstream_props = self.graph.get_conditioning_properties(current_prop)

        if not upstream_props:
            return ctx

        # Add actual Y values as features
        X_augmented = ctx.X.copy()
        for upstream_prop in upstream_props:
            col_name = f"cond_{upstream_prop}"
            X_augmented[col_name] = targets[upstream_prop].values

        return ctx.with_X(X_augmented)

    def _fit_quantile_node(
        self,
        node: QuantileModelNode,
        ctx: DataContext,
    ) -> FittedQuantileNode:
        """
        Fit a quantile node with all quantile levels.

        1. Optimize hyperparameters at median quantile
        2. Train models for all quantile levels with best params

        Args:
            node: QuantileModelNode to fit.
            ctx: Data context with features and target.

        Returns:
            FittedQuantileNode with all trained models.
        """
        # Step 1: Optimize hyperparameters at median quantile
        median_tau = node.median_quantile

        if node.has_search_space:
            best_params, opt_result = self._optimize_at_quantile(
                node, ctx, median_tau
            )
        else:
            best_params = dict(node.fixed_params)
            opt_result = None

        if self.tuning_config.verbose >= 1:
            logger.info(f"  Best params: {best_params}")

        # Step 2: Train models for all quantile levels
        quantile_models: Dict[float, List[Any]] = {}
        oof_predictions_list = []

        for tau in node.quantile_levels:
            if self.tuning_config.verbose >= 2:
                logger.info(f"  Training quantile {tau:.2f}")

            # Get parameters for this quantile
            params = node.get_params_for_quantile(tau, best_params)

            # Cross-validate at this quantile
            fold_models, fold_oof = self._cross_validate_quantile(
                node, ctx, params, tau
            )

            quantile_models[tau] = fold_models
            oof_predictions_list.append(fold_oof)

        # Stack OOF predictions: (n_samples, n_quantiles)
        oof_quantile_predictions = np.column_stack(oof_predictions_list)

        return FittedQuantileNode(
            node=node,
            quantile_models=quantile_models,
            oof_quantile_predictions=oof_quantile_predictions,
            best_params=best_params,
            optimization_result=opt_result,
        )

    def _optimize_at_quantile(
        self,
        node: QuantileModelNode,
        ctx: DataContext,
        tau: float,
    ) -> Tuple[Dict[str, Any], OptimizationResult]:
        """
        Optimize hyperparameters at a specific quantile level.

        Args:
            node: QuantileModelNode to optimize.
            ctx: Data context.
            tau: Quantile level for optimization.

        Returns:
            Tuple of (best_params, optimization_result).
        """
        search_space = node.search_space

        def objective(params: Dict[str, Any]) -> float:
            # Merge with fixed params and quantile-specific params
            all_params = node.get_params_for_quantile(tau, params)

            # Cross-validate
            _, oof_preds = self._cross_validate_quantile(node, ctx, all_params, tau)

            # Calculate pinball loss
            loss = self._pinball_loss(ctx.y.values, oof_preds, tau)
            return loss

        opt_result = self.search_backend.optimize(
            objective=objective,
            search_space=search_space,
            n_trials=self.tuning_config.n_trials,
            timeout=self.tuning_config.timeout,
            study_name=f"{node.name}_tau{tau:.2f}_tuning",
        )

        # Merge best params with fixed params
        best_params = dict(node.fixed_params)
        best_params.update(opt_result.best_params)

        return best_params, opt_result

    def _cross_validate_quantile(
        self,
        node: QuantileModelNode,
        ctx: DataContext,
        params: Dict[str, Any],
        tau: float,
    ) -> Tuple[List[Any], np.ndarray]:
        """
        Cross-validate a quantile model.

        Args:
            node: QuantileModelNode.
            ctx: Data context.
            params: Complete parameters including quantile settings.
            tau: Quantile level.

        Returns:
            Tuple of (list of fold models, OOF predictions).
        """
        cv_config = self.tuning_config.cv_config or CVConfig()
        dm = DataManager(cv_config)
        folds = dm.create_folds(ctx)

        # Parallel fold fitting if executor available with multiple workers
        if self.executor is not None and self.executor.n_workers > 1:
            def fit_fold_task(fold: CVFold) -> Tuple[Any, np.ndarray, CVFold]:
                model, val_preds = self._fit_fold_quantile(node, ctx, fold, params)
                return (model, val_preds, fold)

            results = self.executor.map(fit_fold_task, folds)

            fold_models = []
            fold_results = []
            for model, val_preds, fold in results:
                fold_models.append(model)
                fold_results.append(
                    FoldResult(
                        fold=fold,
                        model=model,
                        val_predictions=val_preds,
                        val_score=-self._pinball_loss(
                            ctx.y.iloc[fold.val_indices].values, val_preds, tau
                        ),
                        fit_time=0,
                        predict_time=0,
                        params=params,
                    )
                )
        else:
            # Sequential fallback
            fold_models = []
            fold_results = []

            for fold in folds:
                model, val_preds = self._fit_fold_quantile(node, ctx, fold, params)
                fold_models.append(model)
                fold_results.append(
                    FoldResult(
                        fold=fold,
                        model=model,
                        val_predictions=val_preds,
                        val_score=-self._pinball_loss(
                            ctx.y.iloc[fold.val_indices].values, val_preds, tau
                        ),
                        fit_time=0,
                        predict_time=0,
                        params=params,
                    )
                )

        # Combine OOF predictions
        oof_predictions = dm.route_oof_predictions(ctx, fold_results)

        return fold_models, oof_predictions

    def _fit_fold_quantile(
        self,
        node: QuantileModelNode,
        ctx: DataContext,
        fold: CVFold,
        params: Dict[str, Any],
    ) -> Tuple[Any, np.ndarray]:
        """
        Fit a quantile model on a single fold.

        Args:
            node: QuantileModelNode.
            ctx: Data context.
            fold: CV fold.
            params: Model parameters.

        Returns:
            Tuple of (fitted model, validation predictions).
        """
        cv_config = self.tuning_config.cv_config or CVConfig()
        dm = DataManager(cv_config)
        train_ctx, val_ctx = dm.align_to_fold(ctx, fold)

        # Create and fit model
        model = node.estimator_class(**params)

        # Apply plugin fit param modifications
        fit_params = dict(node.fit_params)
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                fit_params = plugin.modify_fit_params(fit_params, train_ctx)

        model.fit(train_ctx.X, train_ctx.y, **fit_params)

        # Apply plugin post-fit modifications
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                model = plugin.post_fit(model, node, train_ctx)

        # Get validation predictions
        val_predictions = model.predict(val_ctx.X)

        return model, val_predictions

    def _pinball_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tau: float,
    ) -> float:
        """
        Calculate pinball loss for quantile regression.

        Loss = (tau - 1) * (y - y_pred) if y < y_pred
             = tau * (y - y_pred) if y >= y_pred

        Args:
            y_true: True values.
            y_pred: Predicted values.
            tau: Quantile level.

        Returns:
            Mean pinball loss.
        """
        residual = y_true - y_pred
        loss = np.where(
            residual >= 0,
            tau * residual,
            (tau - 1) * residual,
        )
        return np.mean(loss)

    def __repr__(self) -> str:
        return (
            f"JointQuantileOrchestrator(graph={self.graph}, "
            f"n_trials={self.tuning_config.n_trials})"
        )
