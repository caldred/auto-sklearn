"""QuantileNodeTrainer: Handles fitting quantile regression nodes."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from sklearn_meta.artifacts.training import QuantileNodeRunResult
from sklearn_meta.data.record import DEFAULT_TARGET_KEY
from sklearn_meta.data.view import DataView
from sklearn_meta.engine._metrics import pinball_loss, log_feature_selection
from sklearn_meta.engine.cv import CVEngine
from sklearn_meta.engine.selection import FeatureSelectionService
from sklearn_meta.runtime.config import CVResult, FoldResult, RunConfig
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.spec.quantile import QuantileNodeSpec

if TYPE_CHECKING:
    from sklearn_meta.engine.search import SearchService
    from sklearn_meta.search.backends.base import OptimizationResult

logger = logging.getLogger(__name__)


class QuantileNodeTrainer:
    """Trains a single QuantileNodeSpec across all quantile levels.

    Workflow:
        1. Optimize hyperparameters at the median quantile.
        2. Optionally apply feature selection (and retune on narrowed space).
        3. Train models for every quantile level using the best params.
        4. Assemble a QuantileNodeRunResult with OOF predictions and models.
    """

    def fit_node(
        self,
        node: QuantileNodeSpec,
        data: DataView,
        config: RunConfig,
        services: RuntimeServices,
        cv_engine: CVEngine,
        search_service: SearchService,
        selection_service: Optional[FeatureSelectionService] = None,
    ) -> QuantileNodeRunResult:
        """Fit a quantile node across all quantile levels.

        Args:
            node: The quantile node specification to fit.
            data: DataView containing features and target.
            config: Run configuration (tuning, CV, feature selection).
            services: Runtime services (search backend, plugins, etc.).
            cv_engine: Cross-validation engine for fold creation/routing.
            search_service: Hyperparameter search service.
            selection_service: Optional feature selection service.

        Returns:
            QuantileNodeRunResult with fitted models and predictions.
        """
        # Resolve the named-target key for this quantile node (once, then
        # thread through all downstream calls).
        target_key = self._resolve_target_key(node, data)

        # -----------------------------------------------------------------
        # Step 1: Optimize hyperparameters at the median quantile
        # -----------------------------------------------------------------
        median_tau = node.median_quantile
        opt_result: Optional[OptimizationResult] = None

        if node.has_search_space:
            best_params, opt_result = self._optimize_at_median(
                node, data, median_tau, config, services, cv_engine,
                target_key=target_key,
            )
        else:
            best_params = dict(node.fixed_params)

        logger.info("Best params for '%s': %s", node.name, best_params)

        # -----------------------------------------------------------------
        # Step 2: Feature selection (optional)
        # -----------------------------------------------------------------
        selected_features: Optional[List[str]] = None

        if (
            selection_service is not None
            and config.feature_selection is not None
            and config.feature_selection.enabled
            and node.has_search_space
        ):
            fs_result, data = selection_service.apply(
                node, data, best_params, target_key=target_key,
            )
            selected_features = fs_result.selected_features

            batch_for_log = data.materialize()
            log_feature_selection(
                logger, node.name,
                tuple(batch_for_log.feature_names),
                selected_features if selected_features else [],
            )

            # Optionally retune on narrowed space after feature pruning
            if config.feature_selection.retune_after_pruning and node.search_space is not None:
                narrowed_space = node.search_space.narrow_around(
                    center=best_params,
                    factor=0.5,
                    regularization_bias=0.25,
                )
                if narrowed_space and len(narrowed_space) > 0:
                    logger.info(
                        "Re-tuning '%s' at median quantile after feature selection",
                        node.name,
                    )
                    best_params, opt_result = self._optimize_at_median(
                        node, data, median_tau, config, services, cv_engine,
                        target_key=target_key,
                        search_space_override=narrowed_space,
                    )

        # -----------------------------------------------------------------
        # Step 3: Train all quantile levels
        # -----------------------------------------------------------------
        quantile_models: Dict[float, List[Any]] = {}
        oof_predictions_list: List[np.ndarray] = []
        median_fold_results: Optional[List[FoldResult]] = None

        for tau in node.quantile_levels:
            logger.debug("Training quantile %.2f for '%s'", tau, node.name)

            params = node.get_params_for_quantile(tau, best_params)
            fold_models, oof_preds, fold_results = self._cross_validate_quantile(
                node, data, params, tau, cv_engine, services,
                target_key=target_key,
            )

            quantile_models[tau] = fold_models
            oof_predictions_list.append(oof_preds)

            # Keep fold results from the median quantile for the CVResult
            if tau == median_tau:
                median_fold_results = fold_results

        # Stack OOF predictions: (n_samples, n_quantiles)
        oof_quantile_predictions = np.column_stack(oof_predictions_list)

        # -----------------------------------------------------------------
        # Step 4: Build median CVResult
        # -----------------------------------------------------------------
        if median_fold_results is None:
            # Fallback: use the first quantile's fold results
            raise RuntimeError(
                f"Median quantile {median_tau} not found in node.quantile_levels"
            )

        cv_result = cv_engine.aggregate_cv_result(
            node_name=node.name,
            fold_results=median_fold_results,
            data=data,
        )

        return QuantileNodeRunResult(
            node_name=node.name,
            cv_result=cv_result,
            best_params=best_params,
            quantile_models=quantile_models,
            oof_quantile_predictions=oof_quantile_predictions,
            selected_features=selected_features,
            optimization_result=opt_result,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_target_key(self, node: QuantileNodeSpec, data: DataView) -> str:
        """Return the target key to use for *node*.

        Prefers ``node.property_name`` if it exists in ``data.targets``,
        falls back to ``"__default__"``, and raises if neither is present.
        """
        if node.property_name in data.targets:
            return node.property_name
        if DEFAULT_TARGET_KEY in data.targets:
            return DEFAULT_TARGET_KEY
        raise ValueError(
            f"QuantileNodeSpec '{node.name}' expects target "
            f"'{node.property_name}' but it is not present in data.targets "
            f"(available: {list(data.targets.keys())})"
        )

    def _optimize_at_median(
        self,
        node: QuantileNodeSpec,
        data: DataView,
        tau: float,
        config: RunConfig,
        services: RuntimeServices,
        cv_engine: CVEngine,
        target_key: str,
        search_space_override: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], OptimizationResult]:
        """Optimize hyperparameters at a specific quantile level.

        Uses the search backend directly with a pinball-loss objective.

        Returns:
            Tuple of (best_params, OptimizationResult).
        """
        search_space = search_space_override or node.search_space

        # Resolve y_true once — it's invariant across trials.
        y_true = data.resolve_channel(data.targets[target_key])

        def objective(params: Dict[str, Any]) -> float:
            all_params = node.get_params_for_quantile(tau, params)
            _, oof_preds, _ = self._cross_validate_quantile(
                node, data, all_params, tau, cv_engine, services,
                target_key=target_key,
            )
            return pinball_loss(y_true, oof_preds, tau)

        opt_result = services.search_backend.optimize(
            objective=objective,
            search_space=search_space,
            n_trials=config.tuning.n_trials,
            timeout=config.tuning.timeout,
            study_name=f"{node.name}_tau{tau:.2f}_tuning",
            early_stopping_rounds=config.tuning.early_stopping_rounds,
        )

        best_params = dict(node.fixed_params)
        best_params.update(opt_result.best_params)

        return best_params, opt_result

    def _cross_validate_quantile(
        self,
        node: QuantileNodeSpec,
        data: DataView,
        params: Dict[str, Any],
        tau: float,
        cv_engine: CVEngine,
        services: RuntimeServices,
        target_key: str = DEFAULT_TARGET_KEY,
    ) -> Tuple[List[Any], np.ndarray, List[FoldResult]]:
        """Cross-validate a quantile model at a single quantile level.

        Args:
            node: QuantileNodeSpec.
            data: DataView with features and target.
            params: Complete model parameters (including quantile settings).
            tau: Quantile level.
            cv_engine: CV engine for fold creation and OOF routing.
            services: Runtime services (plugins, etc.).
            target_key: Target key for resolving y from batch targets.

        Returns:
            Tuple of (fold_models, oof_predictions, fold_results).
        """
        folds = cv_engine.create_folds(data, target_key=target_key)

        fold_models: List[Any] = []
        fold_results: List[FoldResult] = []

        for fold in folds:
            model, val_preds, y_val = self._fit_fold_quantile(
                node, data, fold, params, cv_engine, services,
                target_key=target_key,
            )
            fold_models.append(model)

            # Compute per-fold pinball loss for val_score
            val_score = -pinball_loss(y_val, val_preds, tau)

            fold_results.append(
                FoldResult(
                    fold=fold,
                    model=model,
                    val_predictions=val_preds,
                    val_score=val_score,
                    params=params,
                )
            )

        oof_predictions = cv_engine.route_oof_predictions(data, fold_results)

        return fold_models, oof_predictions, fold_results

    def _fit_fold_quantile(
        self,
        node: QuantileNodeSpec,
        data: DataView,
        fold: Any,
        params: Dict[str, Any],
        cv_engine: CVEngine,
        services: RuntimeServices,
        target_key: str = DEFAULT_TARGET_KEY,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Fit a quantile model on a single CV fold.

        Args:
            node: QuantileNodeSpec.
            data: Full DataView.
            fold: CVFold for this split.
            params: Complete model parameters.
            cv_engine: CV engine for splitting.
            services: Runtime services (plugins, etc.).
            target_key: Target key for resolving y from batch targets.

        Returns:
            Tuple of (fitted model, validation predictions, y_val).
        """
        train_view, val_view = cv_engine.split_for_fold(data, fold)
        train_batch = train_view.materialize()
        val_batch = val_view.materialize()

        # Plugin modify_params (before model creation)
        if services.plugin_registry is not None:
            params = services.plugin_registry.apply_modify_params(
                node.estimator_class, params, node,
            )

        # Create model
        model = node.estimator_class(**params)

        # Prepare fit params, applying plugin modifications
        fit_params = dict(node.fit_params)
        if services.plugin_registry is not None:
            fit_params = services.plugin_registry.apply_modify_fit_params(
                node.estimator_class, fit_params, train_batch,
            )

        # Plugin pre_fit (after creation, before fit)
        if services.plugin_registry is not None:
            model = services.plugin_registry.apply_pre_fit(
                node.estimator_class, model, node, train_batch,
            )

        # Fit using the named target
        y_train = train_batch.targets.get(target_key)
        model.fit(train_batch.X, y_train, **fit_params)

        # Post-fit plugin hook
        if services.plugin_registry is not None:
            model = services.plugin_registry.apply_post_fit(
                node.estimator_class, model, node, train_batch,
            )

        # Validation predictions and target
        val_predictions = model.predict(val_batch.X)
        y_val = val_batch.targets.get(target_key)

        return model, val_predictions, y_val
