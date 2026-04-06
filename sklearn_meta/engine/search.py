"""SearchService: Hyperparameter optimization wrapper."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn_meta.meta.prebaked import get_prebaked_reparameterization
from sklearn_meta.meta.reparameterization import Reparameterization, ReparameterizedSpace
from sklearn_meta.runtime.config import ReparameterizationConfig
from sklearn_meta.search.backends.base import OptimizationResult, SearchBackend
from sklearn_meta.search.parameter import IntParameter
from sklearn_meta.spec.node import NodeSpec

logger = logging.getLogger(__name__)


class SearchService:
    """Manages hyperparameter optimization including reparameterization."""

    def __init__(self, backend: SearchBackend) -> None:
        self.backend = backend

    def build_reparameterized_space(
        self,
        node: NodeSpec,
        search_space,
        reparam_config: Optional[ReparameterizationConfig],
    ) -> Optional[ReparameterizedSpace]:
        """Build a reparameterized search space if configured."""
        if reparam_config is None or not reparam_config.enabled or search_space is None:
            return None

        reparams: List[Reparameterization] = list(reparam_config.custom_reparameterizations)
        if reparam_config.use_prebaked:
            prebaked = get_prebaked_reparameterization(
                node.estimator_class,
                search_space.parameter_names,
            )
            reparams.extend(prebaked)

        if not reparams:
            return None

        return ReparameterizedSpace(search_space, reparams)

    def optimize_node(
        self,
        node: NodeSpec,
        search_space,
        reparam_space: Optional[ReparameterizedSpace],
        cross_validate_fn: Callable[[Dict[str, Any]], Any],
        n_trials: int,
        timeout: Optional[float],
        early_stopping_rounds: Optional[int],
        greater_is_better: bool,
        tuning_n_estimators: Optional[int] = None,
        seed_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], OptimizationResult]:
        """Run hyperparameter optimization for a node."""
        from sklearn_meta.engine.estimator_scaling import supports_param

        effective_space = search_space
        if reparam_space:
            effective_space = reparam_space.build_transformed_space()

        original_space = (
            reparam_space.original_space if reparam_space else search_space
        )

        _use_tuning_n_estimators = (
            tuning_n_estimators is not None
            and supports_param(node.estimator_class, "n_estimators")
        )

        def convert_param_types(params: Dict[str, Any]) -> Dict[str, Any]:
            converted = {}
            for name, value in params.items():
                param = original_space.get_parameter(name)
                if param is not None and isinstance(param, IntParameter):
                    converted[name] = int(round(value))
                else:
                    converted[name] = value
            return converted

        def objective(params: Dict[str, Any]) -> float:
            if reparam_space:
                params = reparam_space.inverse_transform(params)
                params = convert_param_types(params)

            all_params = dict(node.fixed_params)
            all_params.update(params)

            if _use_tuning_n_estimators:
                all_params["n_estimators"] = tuning_n_estimators

            cv_result = cross_validate_fn(all_params)
            score = cv_result.mean_score
            if greater_is_better:
                return -score
            return score

        prepared_seed_params = self._prepare_seed_params(
            effective_space=effective_space,
            original_space=original_space,
            reparam_space=reparam_space,
            seed_params=seed_params,
        )

        opt_result = self.backend.optimize(
            objective=objective,
            search_space=effective_space,
            n_trials=n_trials,
            timeout=timeout,
            study_name=f"{node.name}_tuning",
            early_stopping_rounds=early_stopping_rounds,
            seed_params=prepared_seed_params,
        )

        best_params_transformed = opt_result.best_params
        if reparam_space:
            best_params_transformed = reparam_space.inverse_transform(best_params_transformed)
            best_params_transformed = convert_param_types(best_params_transformed)

        best_params = dict(node.fixed_params)
        best_params.update(best_params_transformed)
        if _use_tuning_n_estimators:
            best_params["n_estimators"] = tuning_n_estimators

        return best_params, opt_result

    @staticmethod
    def _prepare_seed_params(
        *,
        effective_space,
        original_space,
        reparam_space: Optional[ReparameterizedSpace],
        seed_params: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Filter and transform warm-start parameter sets for the backend."""

        if not seed_params:
            return None

        prepared: List[Dict[str, Any]] = []
        original_param_names = set(original_space.parameter_names)
        effective_param_names = set(effective_space.parameter_names)

        for params in seed_params:
            filtered = {
                name: value
                for name, value in params.items()
                if name in original_param_names
            }
            if not filtered:
                continue
            transformed = (
                reparam_space.forward_transform(filtered)
                if reparam_space is not None
                else filtered
            )
            transformed = {
                name: value
                for name, value in transformed.items()
                if name in effective_param_names
            }
            if transformed:
                prepared.append(transformed)

        return prepared or None
