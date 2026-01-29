"""TuningOrchestrator: Main coordinator for tuning and training."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

import numpy as np

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVFold, CVResult, FoldResult
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.dependency import DependencyType
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.model.node import ModelNode, OutputType
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.search.backends.base import OptimizationResult, SearchBackend
from sklearn_meta.meta.reparameterization import Reparameterization, ReparameterizedSpace
from sklearn_meta.meta.prebaked import get_prebaked_reparameterization
from sklearn_meta.selection.selector import FeatureSelector, FeatureSelectionResult
from sklearn_meta.persistence.cache import FitCache

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sklearn_meta.audit.logger import AuditLogger
    from sklearn_meta.execution.base import Executor
    from sklearn_meta.plugins.registry import PluginRegistry
    from sklearn_meta.selection.selector import FeatureSelectionConfig


@dataclass
class TuningConfig:
    """
    Configuration for the tuning process.

    Attributes:
        strategy: How to optimize (layer-by-layer, full-graph, etc.).
        n_trials: Number of optimization trials per node.
        timeout: Optional timeout in seconds for optimization.
        early_stopping_rounds: Stop if no improvement for this many trials.
        cv_config: Cross-validation configuration.
        metric: Scoring metric name (e.g., "accuracy", "roc_auc").
        greater_is_better: Whether higher metric values are better.
        feature_selection: Optional feature selection configuration.
        use_reparameterization: Whether to apply reparameterization transforms.
        custom_reparameterizations: Custom reparameterization transforms to apply.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        tuning_n_estimators: n_estimators to use during tuning (faster trials).
            If set, learning_rate is scaled proportionally for the final model.
        final_n_estimators: n_estimators to use for final model. If tuning_n_estimators
            is set, learning_rate is scaled by (tuning_n_estimators / final_n_estimators).
        estimator_scaling_search: If True, search for optimal n_estimators/learning_rate
            scaling after tuning. Tests multipliers [2, 5, 10, 20] with early stopping
            if performance degrades. Overrides tuning_n_estimators/final_n_estimators.
        estimator_scaling_factors: Custom scaling factors to search (default: [2, 5, 10, 20]).
    """

    strategy: OptimizationStrategy = OptimizationStrategy.LAYER_BY_LAYER
    n_trials: int = 100
    timeout: Optional[float] = None
    early_stopping_rounds: Optional[int] = None
    cv_config: Optional[CVConfig] = None
    metric: str = "neg_mean_squared_error"
    greater_is_better: bool = False
    feature_selection: Optional[FeatureSelectionConfig] = None
    use_reparameterization: bool = False
    custom_reparameterizations: Optional[List[Reparameterization]] = None
    verbose: int = 1
    tuning_n_estimators: Optional[int] = None
    final_n_estimators: Optional[int] = None
    estimator_scaling_search: bool = False
    estimator_scaling_factors: Optional[List[int]] = None


@dataclass
class FittedNode:
    """
    Result of fitting a single node.

    Attributes:
        node: The original node definition.
        cv_result: Cross-validation results.
        best_params: Best hyperparameters found.
        optimization_result: Full optimization results.
        selected_features: Features selected (if feature selection was used).
    """

    node: ModelNode
    cv_result: CVResult
    best_params: Dict[str, Any]
    optimization_result: Optional[OptimizationResult] = None
    selected_features: Optional[List[str]] = None

    @property
    def oof_predictions(self) -> np.ndarray:
        """Out-of-fold predictions."""
        return self.cv_result.oof_predictions

    @property
    def models(self) -> List[Any]:
        """Fitted models from all folds."""
        return self.cv_result.models

    @property
    def mean_score(self) -> float:
        """Mean CV score."""
        return self.cv_result.mean_score


@dataclass
class FittedGraph:
    """
    Result of fitting the entire model graph.

    Attributes:
        graph: The original model graph.
        fitted_nodes: Dictionary mapping node names to fitted results.
        tuning_config: Configuration used for tuning.
        total_time: Total time taken in seconds.
    """

    graph: ModelGraph
    fitted_nodes: Dict[str, FittedNode]
    tuning_config: TuningConfig
    total_time: float = 0.0

    def get_node(self, name: str) -> FittedNode:
        """Get a fitted node by name."""
        return self.fitted_nodes[name]

    def get_oof_predictions(self, name: str) -> np.ndarray:
        """Get OOF predictions for a node."""
        return self.fitted_nodes[name].oof_predictions

    def predict(self, X, node_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using the fitted graph.

        Args:
            X: Input features.
            node_name: Specific node to predict from (default: first leaf node).

        Returns:
            Predictions array.
        """
        if node_name is None:
            leaves = self.graph.get_leaf_nodes()
            if not leaves:
                raise ValueError("Graph has no leaf nodes")
            node_name = leaves[0]

        return self._predict_node(X, node_name)

    def _predict_node(
        self, X, node_name: str, cache: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Recursively predict through the graph."""
        if cache is None:
            cache = {}

        if node_name in cache:
            return cache[node_name]

        fitted = self.fitted_nodes[node_name]

        # Get upstream predictions and augment features
        upstream_edges = self.graph.get_upstream(node_name)
        X_augmented = X.copy() if hasattr(X, "copy") else X

        for edge in upstream_edges:
            upstream_preds = self._predict_node(X, edge.source, cache)

            # Handle all dependency types that inject features
            if edge.dep_type in (
                DependencyType.PREDICTION,
                DependencyType.PROBA,
                DependencyType.FEATURE,
            ):
                col_name = edge.feature_name
                if upstream_preds.ndim == 1:
                    X_augmented[col_name] = upstream_preds
                else:
                    for i in range(upstream_preds.shape[1]):
                        X_augmented[f"{col_name}_{i}"] = upstream_preds[:, i]
            elif edge.dep_type == DependencyType.TRANSFORM:
                # TRANSFORM replaces features entirely
                X_augmented = upstream_preds
            elif edge.dep_type in (DependencyType.BASE_MARGIN, DependencyType.DISTILL):
                # BASE_MARGIN is handled via fit_params at training time;
                # DISTILL only affects training loss, not inference.
                pass

        # Filter to selected features if feature selection was used
        if fitted.selected_features is not None:
            X_augmented = X_augmented[fitted.selected_features]

        # Ensemble predictions from all fold models
        predictions = []
        for model in fitted.models:
            if fitted.node.output_type == OutputType.PROBA:
                predictions.append(model.predict_proba(X_augmented))
            else:
                predictions.append(model.predict(X_augmented))

        # Average predictions
        result = np.mean(predictions, axis=0)
        cache[node_name] = result
        return result

    def __repr__(self) -> str:
        return (
            f"FittedGraph(nodes={len(self.fitted_nodes)}, "
            f"time={self.total_time:.1f}s)"
        )


class TuningOrchestrator:
    """
    Main coordinator for tuning and training model graphs.

    This class handles:
    - Layer-wise optimization of model hyperparameters
    - Cross-validation with proper data handling
    - Stacking with leakage-free OOF predictions
    - Plugin lifecycle hooks
    """

    def __init__(
        self,
        graph: ModelGraph,
        data_manager: DataManager,
        search_backend: SearchBackend,
        tuning_config: TuningConfig,
        executor: Optional[Executor] = None,
        plugin_registry: Optional[PluginRegistry] = None,
        audit_logger: Optional[AuditLogger] = None,
        fit_cache: Optional[FitCache] = None,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            graph: Model graph to tune and train.
            data_manager: Handles CV splitting and data routing.
            search_backend: Backend for hyperparameter optimization.
            tuning_config: Tuning configuration.
            executor: Optional executor for parallelization.
            plugin_registry: Optional registry of model plugins.
            audit_logger: Optional logger for auditing.
            fit_cache: Optional cache for model fitting results.
        """
        self.graph = graph
        self.data_manager = data_manager
        self.search_backend = search_backend
        self.tuning_config = tuning_config
        self.executor = executor
        self.plugin_registry = plugin_registry
        self.audit_logger = audit_logger
        self.fit_cache = fit_cache

        # Validate graph
        self.graph.validate()

    def fit(self, ctx: DataContext) -> FittedGraph:
        """
        Fit the entire model graph.

        Args:
            ctx: Data context with features and target.

        Returns:
            FittedGraph with all fitted nodes.
        """
        start_time = time.time()
        fitted_nodes: Dict[str, FittedNode] = {}

        if self.tuning_config.strategy == OptimizationStrategy.LAYER_BY_LAYER:
            fitted_nodes = self._fit_layer_by_layer(ctx)
        elif self.tuning_config.strategy == OptimizationStrategy.GREEDY:
            fitted_nodes = self._fit_greedy(ctx)
        elif self.tuning_config.strategy == OptimizationStrategy.NONE:
            fitted_nodes = self._fit_no_tuning(ctx)
        elif self.tuning_config.strategy == OptimizationStrategy.FULL_GRAPH:
            raise NotImplementedError(
                "FULL_GRAPH strategy is not yet implemented. "
                "Use LAYER_BY_LAYER (recommended) or GREEDY instead."
            )
        else:
            raise ValueError(
                f"Unsupported optimization strategy: {self.tuning_config.strategy}"
            )

        total_time = time.time() - start_time

        return FittedGraph(
            graph=self.graph,
            fitted_nodes=fitted_nodes,
            tuning_config=self.tuning_config,
            total_time=total_time,
        )

    def _fit_layer_by_layer(self, ctx: DataContext) -> Dict[str, FittedNode]:
        """Fit models layer by layer."""
        fitted_nodes: Dict[str, FittedNode] = {}
        oof_cache: Dict[str, np.ndarray] = {}

        layers = self.graph.get_layers()

        for layer_idx, layer in enumerate(layers):
            if self.tuning_config.verbose >= 1:
                logger.info(f"Fitting layer {layer_idx + 1}/{len(layers)}: {layer}")

            # Prepare context with upstream OOF predictions
            layer_ctx = self._prepare_context_with_oof(ctx, layer, oof_cache)

            # Filter nodes that should run
            nodes_to_fit = []
            for node_name in layer:
                node = self.graph.get_node(node_name)
                if node.is_conditional and not node.should_run(layer_ctx):
                    continue
                nodes_to_fit.append((node_name, node))

            # Parallel node fitting within layer if executor available
            if (self.executor is not None and
                self.executor.n_workers > 1 and
                len(nodes_to_fit) > 1):

                def fit_node_task(item: Tuple[str, ModelNode]) -> Tuple[str, FittedNode]:
                    name, node = item
                    node_ctx = layer_ctx
                    if node.is_distilled:
                        node_ctx = self._inject_soft_targets(node_ctx, node, oof_cache)
                    return (name, self._fit_node(node, node_ctx))

                results = self.executor.map(fit_node_task, nodes_to_fit)
                for name, fitted in results:
                    fitted_nodes[name] = fitted
                    oof_cache[name] = fitted.oof_predictions
            else:
                # Sequential fallback
                for node_name, node in nodes_to_fit:
                    node_ctx = layer_ctx
                    if node.is_distilled:
                        node_ctx = self._inject_soft_targets(node_ctx, node, oof_cache)
                    fitted = self._fit_node(node, node_ctx)
                    fitted_nodes[node_name] = fitted
                    oof_cache[node_name] = fitted.oof_predictions

        return fitted_nodes

    def _fit_greedy(self, ctx: DataContext) -> Dict[str, FittedNode]:
        """Fit models one at a time in topological order."""
        fitted_nodes: Dict[str, FittedNode] = {}
        oof_cache: Dict[str, np.ndarray] = {}

        for node_name in self.graph.topological_order():
            node = self.graph.get_node(node_name)

            # Prepare context with upstream OOF predictions
            node_ctx = self._prepare_context_with_oof(ctx, [node_name], oof_cache)

            if node.is_conditional and not node.should_run(node_ctx):
                continue

            if self.tuning_config.verbose >= 1:
                logger.info(f"Fitting node: {node_name}")

            if node.is_distilled:
                node_ctx = self._inject_soft_targets(node_ctx, node, oof_cache)

            fitted = self._fit_node(node, node_ctx)
            fitted_nodes[node_name] = fitted
            oof_cache[node_name] = fitted.oof_predictions

        return fitted_nodes

    def _fit_no_tuning(self, ctx: DataContext) -> Dict[str, FittedNode]:
        """Fit models without hyperparameter tuning."""
        fitted_nodes: Dict[str, FittedNode] = {}
        oof_cache: Dict[str, np.ndarray] = {}

        for node_name in self.graph.topological_order():
            node = self.graph.get_node(node_name)
            node_ctx = self._prepare_context_with_oof(ctx, [node_name], oof_cache)

            if node.is_conditional and not node.should_run(node_ctx):
                continue

            if node.is_distilled:
                node_ctx = self._inject_soft_targets(node_ctx, node, oof_cache)

            # Use fixed params only
            cv_result = self._cross_validate(node, node_ctx, node.fixed_params)
            fitted = FittedNode(
                node=node,
                cv_result=cv_result,
                best_params=node.fixed_params,
            )

            fitted_nodes[node_name] = fitted
            oof_cache[node_name] = fitted.oof_predictions

        return fitted_nodes

    def _fit_node(self, node: ModelNode, ctx: DataContext) -> FittedNode:
        """Fit a single node with hyperparameter optimization."""
        # Apply plugin modifications to search space
        search_space = node.search_space
        if self.plugin_registry and search_space:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                search_space = plugin.modify_search_space(search_space, node)

        # Apply reparameterization if enabled
        reparam_space = None
        if self.tuning_config.use_reparameterization and search_space:
            reparams = []
            if self.tuning_config.custom_reparameterizations:
                reparams.extend(self.tuning_config.custom_reparameterizations)
            # Get prebaked reparameterizations for this model
            prebaked = get_prebaked_reparameterization(
                node.estimator_class,
                search_space.parameter_names
            )
            reparams.extend(prebaked)

            if reparams:
                reparam_space = ReparameterizedSpace(search_space, reparams)
                search_space = reparam_space.build_transformed_space()

        # Optimize if search space exists
        if search_space and len(search_space) > 0:
            best_params, opt_result = self._optimize_node(
                node, ctx, search_space, reparam_space
            )
        else:
            best_params = dict(node.fixed_params)
            opt_result = None

        # Apply plugin post-tune modifications
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                best_params = plugin.post_tune(best_params, node, ctx)

        # Apply feature selection if configured
        selected_features = None
        if (self.tuning_config.feature_selection and
                self.tuning_config.feature_selection.enabled):
            selector = FeatureSelector(self.tuning_config.feature_selection)
            selection_result = selector.select_for_node(node, ctx, best_params)
            selected_features = selection_result.selected_features

            # Log feature selection results
            n_original = len(ctx.feature_cols)
            n_selected = len(selected_features) if selected_features else n_original
            dropped = set(ctx.feature_cols) - set(selected_features) if selected_features else set()
            logger.info(f"Feature selection for '{node.name}': {n_selected}/{n_original} features kept")
            if dropped:
                logger.info(f"  Dropped: {sorted(dropped)}")

            # Filter context to selected features
            if selected_features:
                ctx = ctx.with_feature_cols(selected_features)

            # Retune if configured
            if self.tuning_config.feature_selection.retune_after_pruning:
                logger.info(f"Re-tuning '{node.name}' with selected features...")
                if search_space and len(search_space) > 0:
                    best_params, opt_result = self._optimize_node(
                        node, ctx, search_space, reparam_space
                    )

        # Search for optimal n_estimators scaling if configured
        cv_result = None
        if self.tuning_config.estimator_scaling_search:
            best_params, cv_result = self._search_estimator_scaling(node, ctx, best_params)
        # Or use fixed scaling if configured
        elif (self.tuning_config.tuning_n_estimators is not None and
                self.tuning_config.final_n_estimators is not None):
            tuning_n = self.tuning_config.tuning_n_estimators
            final_n = self.tuning_config.final_n_estimators
            scale_factor = tuning_n / final_n

            best_params = dict(best_params)  # Copy to avoid mutation
            best_params["n_estimators"] = final_n

            # Scale learning_rate proportionally
            if "learning_rate" in best_params:
                old_lr = best_params["learning_rate"]
                best_params["learning_rate"] = old_lr * scale_factor
                logger.info(
                    f"Scaled for final model: n_estimators {tuning_n} -> {final_n}, "
                    f"learning_rate {old_lr:.6f} -> {best_params['learning_rate']:.6f}"
                )

        # Final CV with best params (skip if already done in scaling search)
        if cv_result is None:
            cv_result = self._cross_validate(node, ctx, best_params)

        return FittedNode(
            node=node,
            cv_result=cv_result,
            best_params=best_params,
            optimization_result=opt_result,
            selected_features=selected_features,
        )

    def _search_estimator_scaling(
        self,
        node: ModelNode,
        ctx: DataContext,
        best_params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], CVResult]:
        """Search for optimal n_estimators/learning_rate scaling.

        Tests scaling factors [1, 2, 5, 10, 20] (or custom with 1 prepended),
        scaling n_estimators up and learning_rate down proportionally.
        Stops early if performance degrades.

        Args:
            node: The model node.
            ctx: Data context.
            best_params: Best parameters from tuning.

        Returns:
            Tuple of (parameters with optimal scaling, CV result for best scaling).
        """
        scaling_factors = [1] + (self.tuning_config.estimator_scaling_factors or [2, 5, 10, 20])

        base_n_estimators = best_params.get("n_estimators", 100)
        base_lr = best_params.get("learning_rate")

        if base_lr is None:
            logger.warning("No learning_rate in params, skipping estimator scaling search")
            cv_result = self._cross_validate(node, ctx, best_params)
            return best_params, cv_result

        logger.info(
            f"Estimator scaling search: base n_estimators={base_n_estimators}, "
            f"lr={base_lr:.6f}"
        )

        best_scale = 1
        best_score = None
        best_scaled_params = dict(best_params)
        best_cv_result = None

        for scale in scaling_factors:
            scaled_params = dict(best_params)
            scaled_params["n_estimators"] = base_n_estimators * scale
            scaled_params["learning_rate"] = base_lr / scale

            cv_result = self._cross_validate(node, ctx, scaled_params)
            score = cv_result.mean_score

            # Check if better (accounting for metric direction)
            if best_score is None:
                is_better = True  # First run (1x baseline)
            elif self.tuning_config.greater_is_better:
                is_better = score > best_score
            else:
                is_better = score < best_score

            status = "(baseline)" if scale == 1 else ("(better)" if is_better else "(worse)")
            logger.info(
                f"  {scale}x: n_estimators={scaled_params['n_estimators']}, "
                f"lr={scaled_params['learning_rate']:.6f}, score={score:.5f} {status}"
            )

            if is_better:
                best_scale = scale
                best_score = score
                best_scaled_params = scaled_params
                best_cv_result = cv_result
            elif scale > 1:
                # Early stopping: performance degraded (only after baseline)
                logger.info(f"  Stopping search (performance degraded at {scale}x)")
                break

        logger.info(
            f"Best scaling: {best_scale}x (n_estimators={best_scaled_params['n_estimators']}, "
            f"lr={best_scaled_params['learning_rate']:.6f}, score={best_score:.5f})"
        )

        return best_scaled_params, best_cv_result

    def _optimize_node(
        self,
        node: ModelNode,
        ctx: DataContext,
        search_space,
        reparam_space: Optional[ReparameterizedSpace] = None,
    ) -> Tuple[Dict[str, Any], OptimizationResult]:
        """Run hyperparameter optimization for a node."""
        # Get the original search space for type conversion
        original_space = (
            reparam_space.original_space if reparam_space else search_space
        )

        def convert_param_types(params: Dict[str, Any]) -> Dict[str, Any]:
            """Convert parameters back to their original types."""
            from sklearn_meta.search.parameter import IntParameter

            converted = {}
            for name, value in params.items():
                param = original_space.get_parameter(name)
                if param is not None and isinstance(param, IntParameter):
                    converted[name] = int(round(value))
                else:
                    converted[name] = value
            return converted

        def objective(params: Dict[str, Any]) -> float:
            # Inverse transform if reparameterized
            if reparam_space:
                params = reparam_space.inverse_transform(params)
                params = convert_param_types(params)

            # Merge with fixed params
            all_params = dict(node.fixed_params)
            all_params.update(params)

            # Override n_estimators for faster tuning if configured
            if self.tuning_config.tuning_n_estimators is not None:
                all_params["n_estimators"] = self.tuning_config.tuning_n_estimators

            # Cross-validate
            cv_result = self._cross_validate(node, ctx, all_params)

            # Return appropriate value based on optimization direction
            score = cv_result.mean_score
            if self.tuning_config.greater_is_better:
                return -score  # Minimize negative score
            return score

        opt_result = self.search_backend.optimize(
            objective=objective,
            search_space=search_space,
            n_trials=self.tuning_config.n_trials,
            timeout=self.tuning_config.timeout,
            study_name=f"{node.name}_tuning",
            early_stopping_rounds=self.tuning_config.early_stopping_rounds,
        )

        # Inverse transform best params if reparameterized
        best_params_transformed = opt_result.best_params
        if reparam_space:
            best_params_transformed = reparam_space.inverse_transform(best_params_transformed)
            best_params_transformed = convert_param_types(best_params_transformed)

        # Merge best params with fixed params
        best_params = dict(node.fixed_params)
        best_params.update(best_params_transformed)

        return best_params, opt_result

    def _cross_validate(
        self, node: ModelNode, ctx: DataContext, params: Dict[str, Any]
    ) -> CVResult:
        """Run cross-validation for a node with given parameters."""
        cv_config = self.tuning_config.cv_config or CVConfig()
        dm = DataManager(cv_config)
        folds = dm.create_folds(ctx)

        # Parallel fold fitting if executor available with multiple workers
        if self.executor is not None and self.executor.n_workers > 1:
            # Create a function that captures the fixed arguments
            def fit_fold_task(fold: CVFold) -> FoldResult:
                return self._fit_fold(node, ctx, fold, params)

            fold_results = self.executor.map(fit_fold_task, folds)
        else:
            # Sequential fallback
            fold_results = []
            for fold in folds:
                result = self._fit_fold(node, ctx, fold, params)
                fold_results.append(result)

        # Audit logging (after parallel section)
        if self.audit_logger:
            for result in fold_results:
                self.audit_logger.log_fold(
                    node_name=node.name,
                    fold=result.fold,
                    score=result.val_score,
                    fit_time=result.fit_time,
                    params=params,
                )

        return dm.aggregate_cv_result(node.name, fold_results, ctx)

    def _fit_fold(
        self,
        node: ModelNode,
        ctx: DataContext,
        fold: CVFold,
        params: Dict[str, Any],
    ) -> FoldResult:
        """Fit a model on a single CV fold."""
        cv_config = self.tuning_config.cv_config or CVConfig()
        dm = DataManager(cv_config)
        train_ctx, val_ctx = dm.align_to_fold(ctx, fold)

        # Call plugin on_fold_start hooks
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                plugin.on_fold_start(fold.fold_idx, node, train_ctx)

        # Check cache before fitting
        cache_key = None
        if self.fit_cache:
            cache_key = self.fit_cache.cache_key(node, params, train_ctx)
            cached_model = self.fit_cache.get(cache_key)
            if cached_model is not None:
                # Use cached model
                model = cached_model
                fit_time = 0.0

                # Get predictions and score
                pred_start = time.time()
                val_predictions = node.get_output(model, val_ctx.X)
                predict_time = time.time() - pred_start

                val_score = self._calculate_score(val_ctx.y, val_predictions, node)

                return FoldResult(
                    fold=fold,
                    model=model,
                    val_predictions=val_predictions,
                    val_score=val_score,
                    fit_time=fit_time,
                    predict_time=predict_time,
                    params=params,
                )

        # Create and fit model
        model = node.create_estimator(params)

        # Inject distillation objective if applicable
        if node.is_distilled and train_ctx.soft_targets is not None:
            from sklearn_meta.core.model.distillation import build_distillation_objective
            custom_obj = build_distillation_objective(
                train_ctx.soft_targets, node.distillation_config
            )
            model.set_params(objective=custom_obj)

        # Apply plugin fit param modifications
        fit_params = dict(node.fit_params)
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                fit_params = plugin.modify_fit_params(fit_params, train_ctx)

        start_time = time.time()
        model.fit(train_ctx.X, train_ctx.y, **fit_params)
        fit_time = time.time() - start_time

        # Apply plugin post-fit modifications
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                model = plugin.post_fit(model, node, train_ctx)

        # Store in cache after fitting
        if self.fit_cache and cache_key:
            self.fit_cache.put(cache_key, model)

        # Get predictions and score
        pred_start = time.time()
        val_predictions = node.get_output(model, val_ctx.X)
        predict_time = time.time() - pred_start

        # Calculate score
        val_score = self._calculate_score(val_ctx.y, val_predictions, node)

        # Call plugin on_fold_end hooks
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                plugin.on_fold_end(fold.fold_idx, model, val_score, node)

        return FoldResult(
            fold=fold,
            model=model,
            val_predictions=val_predictions,
            val_score=val_score,
            fit_time=fit_time,
            predict_time=predict_time,
            params=params,
        )

    def _calculate_score(self, y_true, y_pred, node=None) -> float:
        """Calculate the evaluation score."""
        from sklearn.metrics import get_scorer

        metric = self.tuning_config.metric
        scorer = get_scorer(metric)

        # Create a dummy estimator wrapper for the scorer
        # The scorer expects (estimator, X, y) but we already have predictions
        class _PredictionWrapper:
            """Wrapper that returns pre-computed predictions."""

            def __init__(self, predictions):
                self._predictions = predictions

            def predict(self, X):
                # For classification with probability outputs, convert to class labels
                preds = self._predictions
                if preds.ndim > 1:
                    return np.argmax(preds, axis=1)
                return preds

            def predict_proba(self, X):
                return self._predictions

            def decision_function(self, X):
                # For binary classification probas, return positive class probability
                preds = self._predictions
                if preds.ndim > 1 and preds.shape[1] == 2:
                    return preds[:, 1]
                return preds

        wrapper = _PredictionWrapper(y_pred)

        # Set _estimator_type and classes_ so sklearn's scorer recognizes the
        # wrapper as a classifier (needed for metrics like neg_log_loss that
        # use predict_proba)
        if node is not None:
            estimator_type = getattr(node.estimator_class, "_estimator_type", None)
            if estimator_type:
                wrapper._estimator_type = estimator_type
            if estimator_type == "classifier":
                wrapper.classes_ = np.unique(y_true)

        # Use scorer properly - this respects _sign, _kwargs, and response_method
        # We pass y_true as X since the wrapper ignores it
        return scorer(wrapper, y_true, y_true)

    def _inject_soft_targets(
        self,
        ctx: DataContext,
        node: ModelNode,
        oof_cache: Dict[str, np.ndarray],
    ) -> DataContext:
        """Inject teacher soft targets into context for distillation."""
        # Find teacher from DISTILL edge
        teacher_name = None
        for edge in self.graph.get_upstream(node.name):
            if edge.dep_type == DependencyType.DISTILL:
                teacher_name = edge.source
                break

        if teacher_name is None:
            raise ValueError(
                f"Node '{node.name}' has distillation_config but no DISTILL "
                f"edge in the graph"
            )

        if teacher_name not in oof_cache:
            raise ValueError(
                f"Teacher '{teacher_name}' OOF predictions not found. "
                f"Ensure the teacher is fitted before the student."
            )

        teacher_oof = oof_cache[teacher_name]

        # Extract positive-class probability
        if teacher_oof.ndim == 2 and teacher_oof.shape[1] == 2:
            soft_targets = teacher_oof[:, 1]
        elif teacher_oof.ndim == 1:
            soft_targets = teacher_oof
        else:
            raise ValueError(
                f"Teacher '{teacher_name}' OOF has unexpected shape "
                f"{teacher_oof.shape}. Expected (n,) or (n, 2)."
            )

        return ctx.with_soft_targets(soft_targets)

    def _prepare_context_with_oof(
        self,
        ctx: DataContext,
        node_names: List[str],
        oof_cache: Dict[str, np.ndarray],
    ) -> DataContext:
        """Prepare context by adding OOF predictions from upstream nodes."""
        # Collect all upstream nodes for the target nodes
        all_upstream = set()
        for node_name in node_names:
            for edge in self.graph.get_upstream(node_name):
                if edge.dep_type == DependencyType.DISTILL:
                    continue  # handled per-node, not as features
                all_upstream.add(edge)

        if not all_upstream:
            return ctx

        # Add upstream predictions to context
        predictions = {}
        for edge in all_upstream:
            if edge.source in oof_cache:
                predictions[edge.feature_name] = oof_cache[edge.source]

        if predictions:
            return ctx.augment_with_predictions(predictions, prefix="")

        return ctx

    def __repr__(self) -> str:
        return (
            f"TuningOrchestrator(graph={self.graph}, "
            f"strategy={self.tuning_config.strategy.value})"
        )
