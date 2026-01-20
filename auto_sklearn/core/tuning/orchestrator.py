"""TuningOrchestrator: Main coordinator for tuning and training."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from auto_sklearn.core.data.context import DataContext
from auto_sklearn.core.data.cv import CVConfig, CVFold, CVResult, FoldResult
from auto_sklearn.core.data.manager import DataManager
from auto_sklearn.core.model.dependency import DependencyType
from auto_sklearn.core.model.graph import ModelGraph
from auto_sklearn.core.model.node import ModelNode, OutputType
from auto_sklearn.core.tuning.strategy import OptimizationStrategy
from auto_sklearn.search.backends.base import OptimizationResult, SearchBackend

if TYPE_CHECKING:
    from auto_sklearn.audit.logger import AuditLogger
    from auto_sklearn.execution.base import Executor
    from auto_sklearn.plugins.registry import PluginRegistry
    from auto_sklearn.selection.selector import FeatureSelectionConfig


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
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
    """

    strategy: OptimizationStrategy = OptimizationStrategy.LAYER_BY_LAYER
    n_trials: int = 100
    timeout: Optional[float] = None
    early_stopping_rounds: Optional[int] = None
    cv_config: Optional[CVConfig] = None
    metric: str = "neg_mean_squared_error"
    greater_is_better: bool = False
    feature_selection: Optional[FeatureSelectionConfig] = None
    verbose: int = 1


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

            if edge.dep_type == DependencyType.PREDICTION:
                col_name = edge.feature_name
                if upstream_preds.ndim == 1:
                    X_augmented[col_name] = upstream_preds
                else:
                    for i in range(upstream_preds.shape[1]):
                        X_augmented[f"{col_name}_{i}"] = upstream_preds[:, i]

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
                print(f"Fitting layer {layer_idx + 1}/{len(layers)}: {layer}")

            # Prepare context with upstream OOF predictions
            layer_ctx = self._prepare_context_with_oof(ctx, layer, oof_cache)

            # Fit all nodes in this layer
            for node_name in layer:
                node = self.graph.get_node(node_name)

                if node.is_conditional and not node.should_run(layer_ctx):
                    continue

                fitted = self._fit_node(node, layer_ctx)
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
                print(f"Fitting node: {node_name}")

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

        # Optimize if search space exists
        if search_space and len(search_space) > 0:
            best_params, opt_result = self._optimize_node(node, ctx, search_space)
        else:
            best_params = dict(node.fixed_params)
            opt_result = None

        # Apply plugin post-tune modifications
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                best_params = plugin.post_tune(best_params, node, ctx)

        # Final CV with best params
        cv_result = self._cross_validate(node, ctx, best_params)

        return FittedNode(
            node=node,
            cv_result=cv_result,
            best_params=best_params,
            optimization_result=opt_result,
        )

    def _optimize_node(
        self, node: ModelNode, ctx: DataContext, search_space
    ) -> tuple[Dict[str, Any], OptimizationResult]:
        """Run hyperparameter optimization for a node."""

        def objective(params: Dict[str, Any]) -> float:
            # Merge with fixed params
            all_params = dict(node.fixed_params)
            all_params.update(params)

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
        )

        # Merge best params with fixed params
        best_params = dict(node.fixed_params)
        best_params.update(opt_result.best_params)

        return best_params, opt_result

    def _cross_validate(
        self, node: ModelNode, ctx: DataContext, params: Dict[str, Any]
    ) -> CVResult:
        """Run cross-validation for a node with given parameters."""
        cv_config = self.tuning_config.cv_config or CVConfig()
        dm = DataManager(cv_config)
        folds = dm.create_folds(ctx)

        fold_results = []
        for fold in folds:
            result = self._fit_fold(node, ctx, fold, params)
            fold_results.append(result)

            if self.audit_logger:
                self.audit_logger.log_fold(
                    node_name=node.name,
                    fold=fold,
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

        # Create and fit model
        model = node.create_estimator(params)

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

        # Get predictions and score
        pred_start = time.time()
        val_predictions = node.get_output(model, val_ctx.X)
        predict_time = time.time() - pred_start

        # Calculate score
        val_score = self._calculate_score(val_ctx.y, val_predictions)

        return FoldResult(
            fold=fold,
            model=model,
            val_predictions=val_predictions,
            val_score=val_score,
            fit_time=fit_time,
            predict_time=predict_time,
            params=params,
        )

    def _calculate_score(self, y_true, y_pred) -> float:
        """Calculate the evaluation score."""
        from sklearn.metrics import get_scorer

        metric = self.tuning_config.metric

        # Handle common metrics
        if metric == "neg_mean_squared_error":
            from sklearn.metrics import mean_squared_error
            return -mean_squared_error(y_true, y_pred)
        elif metric == "accuracy":
            from sklearn.metrics import accuracy_score
            if y_pred.ndim > 1:
                y_pred = np.argmax(y_pred, axis=1)
            return accuracy_score(y_true, y_pred)
        elif metric == "roc_auc":
            from sklearn.metrics import roc_auc_score
            if y_pred.ndim > 1:
                return roc_auc_score(y_true, y_pred[:, 1])
            return roc_auc_score(y_true, y_pred)
        else:
            # Try sklearn scorer
            scorer = get_scorer(metric)
            return scorer._score_func(y_true, y_pred)

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
