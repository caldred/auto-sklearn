"""StandardNodeTrainer: Per-node fitting logic extracted from TuningOrchestrator."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import get_scorer

from sklearn_meta.data.view import DataView
from sklearn_meta.engine._metrics import log_feature_selection
from sklearn_meta.engine.cv import CVEngine
from sklearn_meta.engine.distillation import build_distillation_objective
from sklearn_meta.engine.estimator_scaling import EstimatorScaler, supports_param
from sklearn_meta.engine.search import SearchService
from sklearn_meta.engine.selection import FeatureSelectionService
from sklearn_meta.runtime.config import (
    CVFold,
    CVResult,
    FoldResult,
    RunConfig,
)
from sklearn_meta.artifacts.training import NodeRunResult
from sklearn_meta.search.backends.base import OptimizationResult
from sklearn_meta.engine.estimator_factory import create_estimator, get_output
from sklearn_meta.spec.node import NodeSpec

if TYPE_CHECKING:
    from typing import Protocol, runtime_checkable

    from sklearn_meta.runtime.services import RuntimeServices
else:
    from typing import runtime_checkable

    try:
        from typing import Protocol
    except ImportError:
        from typing_extensions import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction wrappers (for sklearn scorer compatibility)
# ---------------------------------------------------------------------------

class _PredictionWrapper(BaseEstimator):
    """Wraps pre-computed predictions so sklearn scorers can consume them."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        preds = self._predictions
        if preds.ndim > 1:
            return np.argmax(preds, axis=1)
        return preds

    def predict_proba(self, X: Any) -> np.ndarray:  # noqa: N803
        return self._predictions

    def decision_function(self, X: Any) -> np.ndarray:  # noqa: N803
        preds = self._predictions
        if preds.ndim > 1 and preds.shape[1] == 2:
            return preds[:, 1]
        return preds


class _ClassifierPredictionWrapper(ClassifierMixin, _PredictionWrapper):
    """Classifier wrapper that sklearn scorers recognize."""

    def __init__(self, predictions: np.ndarray, classes: np.ndarray) -> None:
        super().__init__(predictions)
        self.classes_ = classes

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        preds = self._predictions
        if preds.ndim > 1:
            return self.classes_[np.argmax(preds, axis=1)]
        return preds


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class NodeTrainer(Protocol):
    """Protocol for node-level training strategies."""

    def fit_node(
        self,
        node: NodeSpec,
        data: DataView,
        config: RunConfig,
        services: RuntimeServices,
        cv_engine: CVEngine,
        search_service: SearchService,
        selection_service: Optional[FeatureSelectionService],
    ) -> NodeRunResult:
        ...


# ---------------------------------------------------------------------------
# StandardNodeTrainer
# ---------------------------------------------------------------------------

class StandardNodeTrainer:
    """Default implementation of :class:`NodeTrainer`.

    Extracts per-node fitting logic that was previously embedded inside
    ``TuningOrchestrator._fit_node`` and ``_fit_fold``.
    """

    def __init__(self) -> None:
        # Cache of folds per data view (keyed by id(data)).
        self._fold_cache: Dict[int, List[CVFold]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_node(
        self,
        node: NodeSpec,
        data: DataView,
        config: RunConfig,
        services: RuntimeServices,
        cv_engine: CVEngine,
        search_service: SearchService,
        selection_service: Optional[FeatureSelectionService],
    ) -> NodeRunResult:
        """Fit a single node through the full tuning pipeline.

        Steps:
        1. Plugin search-space modifications
        2. Reparameterisation setup
        3. Hyperparameter optimisation (if search space exists)
        4. Plugin post-tune hooks
        5. Feature selection (optional)
        6. Re-tune after pruning (optional)
        7. Estimator scaling (optional)
        8. Final cross-validation with best params
        """

        search_space = node.search_space

        # --- 1. Plugin search-space modifications ---
        plugins = self._get_plugins(node, services)
        for plugin in plugins:
            if search_space is not None:
                search_space = plugin.modify_search_space(search_space, node)

        # --- 2. Reparameterisation ---
        reparam_space = search_service.build_reparameterized_space(
            node, search_space, config.reparameterization,
        )

        # --- 3. Optimisation ---
        opt_result: Optional[OptimizationResult] = None

        if node.has_search_space and search_space is not None:
            tuning_n_estimators: Optional[int] = None
            if config.estimator_scaling is not None:
                tuning_n_estimators = config.estimator_scaling.tuning_n_estimators

            def _cv_fn(params: Dict[str, Any]) -> CVResult:
                return self._cross_validate(
                    node, data, params, cv_engine, services, config,
                )

            best_params, opt_result = search_service.optimize_node(
                node=node,
                search_space=search_space,
                reparam_space=reparam_space,
                cross_validate_fn=_cv_fn,
                n_trials=config.tuning.n_trials,
                timeout=config.tuning.timeout,
                early_stopping_rounds=config.tuning.early_stopping_rounds,
                greater_is_better=config.tuning.greater_is_better,
                tuning_n_estimators=tuning_n_estimators,
            )
        else:
            best_params = dict(node.fixed_params)

        # --- 4. Plugin post-tune hooks ---
        for plugin in plugins:
            best_params = plugin.post_tune(best_params, node, data)

        # --- 5. Feature selection ---
        selected_features: Optional[List[str]] = None

        if (
            selection_service is not None
            and config.feature_selection is not None
            and config.feature_selection.enabled
        ):
            fs_result, data = selection_service.apply(node, data, best_params)
            selected_features = fs_result.selected_features

            batch = data.materialize()
            log_feature_selection(
                logger,
                node.name,
                batch.feature_names,
                selected_features,
            )

            # --- 6. Re-tune after pruning ---
            if config.feature_selection.retune_after_pruning and node.has_search_space and search_space is not None:
                def _cv_fn_retune(params: Dict[str, Any]) -> CVResult:
                    return self._cross_validate(
                        node, data, params, cv_engine, services, config,
                    )

                tuning_n_est: Optional[int] = None
                if config.estimator_scaling is not None:
                    tuning_n_est = config.estimator_scaling.tuning_n_estimators

                best_params, opt_result = search_service.optimize_node(
                    node=node,
                    search_space=search_space,
                    reparam_space=reparam_space,
                    cross_validate_fn=_cv_fn_retune,
                    n_trials=config.tuning.n_trials,
                    timeout=config.tuning.timeout,
                    early_stopping_rounds=config.tuning.early_stopping_rounds,
                    greater_is_better=config.tuning.greater_is_better,
                    tuning_n_estimators=tuning_n_est,
                )

        # --- 7. Estimator scaling ---
        if config.estimator_scaling is not None:
            scaler = EstimatorScaler(
                config.estimator_scaling,
                greater_is_better=config.tuning.greater_is_better,
            )

            if config.estimator_scaling.scaling_search:
                def _cv_fn_scale(params: Dict[str, Any]) -> CVResult:
                    return self._cross_validate(
                        node, data, params, cv_engine, services, config,
                    )

                best_params, scaling_cv = scaler.search_scaling(
                    node, data, best_params, _cv_fn_scale,
                )
            elif (
                config.estimator_scaling.final_n_estimators is not None
                and supports_param(node.estimator_class, "n_estimators")
            ):
                best_params = scaler.apply_fixed_scaling(node, best_params)

        # --- 8. Final CV ---
        cv_result = self._cross_validate(
            node, data, best_params, cv_engine, services, config,
        )

        return NodeRunResult(
            node_name=node.name,
            cv_result=cv_result,
            best_params=best_params,
            selected_features=selected_features,
            optimization_result=opt_result,
        )

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(
        self,
        node: NodeSpec,
        data: DataView,
        params: Dict[str, Any],
        cv_engine: CVEngine,
        services: RuntimeServices,
        config: RunConfig,
    ) -> CVResult:
        """Run cross-validation for *node* with the given *params*."""

        # Cache folds per data view
        data_id = id(data)
        if data_id not in self._fold_cache:
            self._fold_cache[data_id] = cv_engine.create_folds(data)
        folds = self._fold_cache[data_id]

        fold_results: List[FoldResult] = []
        for fold in folds:
            result = self._fit_fold(
                node, data, fold, params, cv_engine, services, config,
            )
            fold_results.append(result)

            # Audit logging
            if services.audit_logger is not None:
                services.audit_logger.log_fold(
                    node_name=node.name,
                    fold=fold,
                    score=result.val_score,
                    fit_time=result.fit_time,
                    params=params,
                )

        return cv_engine.aggregate_cv_result(node.name, fold_results, data)

    # ------------------------------------------------------------------
    # Single fold
    # ------------------------------------------------------------------

    def _fit_fold(
        self,
        node: NodeSpec,
        data: DataView,
        fold: CVFold,
        params: Dict[str, Any],
        cv_engine: CVEngine,
        services: RuntimeServices,
        config: RunConfig,
    ) -> FoldResult:
        """Fit *node* on a single CV fold and return the :class:`FoldResult`."""

        # Split
        train_view, val_view = cv_engine.split_for_fold(data, fold)
        train_batch = train_view.materialize()
        val_batch = val_view.materialize()

        # --- Retrieve plugins early (needed before model creation) ---
        plugins = self._get_plugins(node, services)

        # --- Cache check ---
        cache_key: Optional[str] = None
        if services.fit_cache is not None:
            cache_key = services.fit_cache.cache_key(node, params, train_view)
            cached_model = services.fit_cache.get(cache_key)
            if cached_model is not None:
                val_predictions = get_output(node, cached_model, val_batch.X)
                score = self._calculate_score(
                    val_batch.y, val_predictions, node, config,
                )
                return FoldResult(
                    fold=fold,
                    model=cached_model,
                    val_predictions=val_predictions,
                    val_score=score,
                    params=params,
                )

        # --- Plugin modify_params (before model creation) ---
        for plugin in plugins:
            params = plugin.modify_params(params, node)

        # --- Create estimator ---
        model = create_estimator(node, params)

        # --- Distillation ---
        if node.is_distilled and "soft_targets" in train_batch.aux:
            soft_targets = train_batch.aux["soft_targets"]
            custom_obj = build_distillation_objective(
                soft_targets, node.distillation_config,
            )
            model.set_params(objective=custom_obj)

        # --- Build fit kwargs ---
        fit_kwargs: Dict[str, Any] = dict(node.fit_params)

        # Translate standard aux keys to estimator-specific fit kwargs
        if "sample_weight" in train_batch.aux:
            fit_kwargs["sample_weight"] = train_batch.aux["sample_weight"]
        if "base_margin" in train_batch.aux:
            fit_kwargs["base_margin"] = train_batch.aux["base_margin"]

        # --- Plugin on_fold_start (receives DataView) ---
        for plugin in plugins:
            plugin.on_fold_start(fold.fold_idx, node, train_view)

        # --- Plugin modify_fit_params (receives MaterializedBatch) ---
        for plugin in plugins:
            fit_kwargs = plugin.modify_fit_params(fit_kwargs, train_batch)

        # --- Plugin pre_fit (receives MaterializedBatch) ---
        for plugin in plugins:
            model = plugin.pre_fit(model, node, train_batch)

        # --- Fit ---
        t0 = time.perf_counter()
        model.fit(train_batch.X, train_batch.y, **fit_kwargs)
        fit_time = time.perf_counter() - t0

        # --- Plugin post_fit (receives MaterializedBatch) ---
        for plugin in plugins:
            model = plugin.post_fit(model, node, train_batch)

        # --- Store in cache ---
        if services.fit_cache is not None and cache_key is not None:
            services.fit_cache.put(cache_key, model)

        # --- Predict + score ---
        t1 = time.perf_counter()
        val_predictions = get_output(node, model, val_batch.X)
        predict_time = time.perf_counter() - t1

        score = self._calculate_score(
            val_batch.y, val_predictions, node, config,
        )

        # --- Plugin on_fold_end ---
        for plugin in plugins:
            plugin.on_fold_end(fold.fold_idx, model, score, node)

        return FoldResult(
            fold=fold,
            model=model,
            val_predictions=val_predictions,
            val_score=score,
            fit_time=fit_time,
            predict_time=predict_time,
            params=params,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_score(
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray,
        node: NodeSpec,
        config: RunConfig,
    ) -> float:
        """Score predictions using the metric specified in *config*."""

        if y_true is None:
            return 0.0

        scorer = get_scorer(config.tuning.metric)

        # Wrap predictions so the sklearn scorer can call predict / predict_proba.
        from sklearn_meta.spec.node import OutputType

        if node.output_type == OutputType.PROBA:
            wrapper = _ClassifierPredictionWrapper(y_pred, np.unique(y_true))
        else:
            wrapper = _PredictionWrapper(y_pred)

        # Sklearn scorer signature: scorer(estimator, X, y_true)
        # We pass y_true as X since the wrapper ignores it.
        return float(scorer(wrapper, y_true, y_true))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_plugins(node: NodeSpec, services: RuntimeServices) -> list:
        """Return the list of plugins applicable to *node*."""
        if services.plugin_registry is None:
            return []
        if node.plugins:
            return services.plugin_registry.get_plugins_for_names(node.plugins)
        return services.plugin_registry.get_plugins_for(node.estimator_class)

