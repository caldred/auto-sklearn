"""EstimatorScaler: Post-tuning estimator scaling for boosting models."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def supports_param(estimator_class, param_name: str) -> bool:
    """Check if an estimator class accepts a given parameter in __init__."""
    sig = inspect.signature(estimator_class.__init__)
    params = sig.parameters
    return param_name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


@dataclass
class EstimatorScalingConfig:
    """Configuration for estimator scaling."""

    tuning_n_estimators: Optional[int] = None
    final_n_estimators: Optional[int] = None
    scaling_search: bool = False
    scaling_factors: Optional[List[float]] = None
    scaling_estimators: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tuning_n_estimators": self.tuning_n_estimators,
            "final_n_estimators": self.final_n_estimators,
            "scaling_search": self.scaling_search,
            "scaling_factors": self.scaling_factors,
            "scaling_estimators": self.scaling_estimators,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EstimatorScalingConfig":
        return cls(
            tuning_n_estimators=data.get("tuning_n_estimators"),
            final_n_estimators=data.get("final_n_estimators"),
            scaling_search=data.get("scaling_search", False),
            scaling_factors=data.get("scaling_factors"),
            scaling_estimators=data.get("scaling_estimators"),
        )


class EstimatorScaler:
    """Handles n_estimators/learning_rate scaling for boosting models."""

    def __init__(self, config: EstimatorScalingConfig, greater_is_better: bool = False):
        self.config = config
        self.greater_is_better = greater_is_better

    def apply_fixed_scaling(self, node, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fixed n_estimators scaling with proportional learning_rate adjustment."""
        if not supports_param(node.estimator_class, "n_estimators"):
            logger.warning(
                f"Skipping n_estimators scaling for '{node.name}': "
                f"{node.estimator_class.__name__} does not support n_estimators"
            )
            return best_params

        tuning_n = self.config.tuning_n_estimators
        final_n = self.config.final_n_estimators
        scale_factor = tuning_n / final_n

        best_params = dict(best_params)
        best_params["n_estimators"] = final_n

        if "learning_rate" in best_params:
            old_lr = best_params["learning_rate"]
            best_params["learning_rate"] = old_lr * scale_factor
            logger.info(
                f"Scaled for final model: n_estimators {tuning_n} -> {final_n}, "
                f"learning_rate {old_lr:.6f} -> {best_params['learning_rate']:.6f}"
            )

        return best_params

    def _resolve_search_n_estimators(self, base_n_estimators: int) -> List[int]:
        resolved = [base_n_estimators]

        scaling_factors = self.config.scaling_factors
        scaling_estimators = self.config.scaling_estimators
        if scaling_factors is None and scaling_estimators is None:
            scaling_factors = [2, 5, 10, 20]

        for factor in scaling_factors or []:
            candidate = int(round(base_n_estimators * float(factor)))
            if candidate > base_n_estimators:
                resolved.append(candidate)

        for n_estimators in scaling_estimators or []:
            candidate = int(n_estimators)
            if candidate > base_n_estimators:
                resolved.append(candidate)

        return [base_n_estimators] + sorted(set(resolved) - {base_n_estimators})

    def search_scaling(
        self,
        node,
        ctx,
        best_params: Dict[str, Any],
        cross_validate_fn: Callable[[Dict[str, Any]], Any],
    ) -> Tuple[Dict[str, Any], Any]:
        """Search for optimal n_estimators/learning_rate scaling."""
        if ("n_estimators" not in best_params
                and not supports_param(node.estimator_class, "n_estimators")):
            logger.warning(
                f"Skipping estimator scaling search for '{node.name}': "
                f"{node.estimator_class.__name__} does not support n_estimators"
            )
            return best_params, None

        base_n_estimators = best_params.get(
            "n_estimators",
            self.config.tuning_n_estimators or 100,
        )
        base_lr = best_params.get("learning_rate")

        if base_lr is None:
            logger.warning("No learning_rate in params, skipping estimator scaling search")
            return best_params, None

        search_n_estimators = self._resolve_search_n_estimators(base_n_estimators)

        logger.info(
            f"Estimator scaling search: base n_estimators={base_n_estimators}, "
            f"lr={base_lr:.6f}"
        )

        best_n_estimators = base_n_estimators
        best_score = None
        best_scaled_params = dict(best_params)
        best_cv_result = None

        for n_estimators in search_n_estimators:
            scale = n_estimators / base_n_estimators
            scaled_params = dict(best_params)
            scaled_params["n_estimators"] = n_estimators
            scaled_params["learning_rate"] = base_lr / scale

            cv_result = cross_validate_fn(scaled_params)
            score = cv_result.mean_score

            if best_score is None:
                is_better = True
            elif self.greater_is_better:
                is_better = score > best_score
            else:
                is_better = score < best_score

            status = "(baseline)" if n_estimators == base_n_estimators else (
                "(better)" if is_better else "(worse)"
            )
            logger.info(
                f"  {scale:.3g}x: n_estimators={scaled_params['n_estimators']}, "
                f"lr={scaled_params['learning_rate']:.6f}, score={score:.5f} {status}"
            )

            if is_better:
                best_n_estimators = n_estimators
                best_score = score
                best_scaled_params = scaled_params
                best_cv_result = cv_result
            elif n_estimators > base_n_estimators:
                logger.info(
                    "  Stopping search "
                    f"(performance degraded at n_estimators={n_estimators})"
                )
                break

        logger.info(
            f"Best scaling: n_estimators={best_n_estimators} "
            f"(n_estimators={best_scaled_params['n_estimators']}, "
            f"lr={best_scaled_params['learning_rate']:.6f}, score={best_score:.5f})"
        )

        return best_scaled_params, best_cv_result
