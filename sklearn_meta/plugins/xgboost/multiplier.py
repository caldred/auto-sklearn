"""XGBMultiplierPlugin: Learning rate / n_estimators multiplier tuning for XGBoost."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

import numpy as np

from sklearn_meta.plugins.base import ModelPlugin

if TYPE_CHECKING:
    from sklearn_meta.core.data.context import DataContext
    from sklearn_meta.core.model.node import ModelNode


class XGBMultiplierPlugin(ModelPlugin):
    """
    XGBoost multiplier tuning plugin.

    After initial hyperparameter optimization, this plugin searches over
    learning_rate / n_estimators multipliers to find a better trade-off.

    The idea is that the optimal ratio between learning_rate and n_estimators
    might differ from what was found in the initial search, especially when
    the search space for these parameters is coarse.
    """

    def __init__(
        self,
        multipliers: Optional[List[float]] = None,
        cv_folds: int = 3,
        metric: str = "rmse",
        enable_post_tune: bool = True,
    ) -> None:
        """
        Initialize the multiplier plugin.

        Args:
            multipliers: List of multipliers to try (default: [0.5, 1.0, 2.0]).
            cv_folds: Number of CV folds for evaluation.
            metric: XGBoost evaluation metric.
            enable_post_tune: Whether to enable post-tune optimization.
        """
        self.multipliers = multipliers or [0.5, 0.75, 1.0, 1.5, 2.0]
        self.cv_folds = cv_folds
        self.metric = metric
        self.enable_post_tune = enable_post_tune

    @property
    def name(self) -> str:
        return "xgb_multiplier"

    def applies_to(self, estimator_class: Type) -> bool:
        """Check if estimator is XGBoost."""
        # Check class name (works even if xgboost is not imported)
        class_name = estimator_class.__name__
        if class_name in ("XGBClassifier", "XGBRegressor", "XGBRanker"):
            return True

        # Check for get_booster method (XGBoost specific)
        return hasattr(estimator_class, "get_booster")

    def modify_fit_params(
        self,
        params: Dict[str, Any],
        ctx: DataContext,
    ) -> Dict[str, Any]:
        """
        Modify fit parameters for XGBoost.

        Adds early stopping configuration if not present.
        """
        # Don't modify if already configured
        if "early_stopping_rounds" in params:
            return params

        params = dict(params)
        params.setdefault("verbose", False)

        return params

    def post_tune(
        self,
        best_params: Dict[str, Any],
        node: ModelNode,
        ctx: DataContext,
    ) -> Dict[str, Any]:
        """
        Search over learning_rate/n_estimators multipliers.

        Args:
            best_params: Best parameters from initial tuning.
            node: The model node.
            ctx: Data context.

        Returns:
            Refined parameters with optimal multiplier applied.
        """
        if not self.enable_post_tune:
            return best_params

        # Get base values
        base_lr = best_params.get("learning_rate", 0.1)
        base_n_estimators = best_params.get("n_estimators", 100)

        # Skip if these params weren't tuned
        if base_lr is None or base_n_estimators is None:
            return best_params

        best_score = float("inf")
        best_multiplier = 1.0

        for multiplier in self.multipliers:
            # Adjust learning rate and n_estimators inversely
            test_params = dict(best_params)
            test_params["learning_rate"] = base_lr * multiplier
            test_params["n_estimators"] = int(base_n_estimators / multiplier)

            # Ensure valid values
            test_params["n_estimators"] = max(10, test_params["n_estimators"])

            # Evaluate with CV
            score = self._evaluate_params(node, ctx, test_params)

            if score < best_score:
                best_score = score
                best_multiplier = multiplier

        # Apply best multiplier
        best_params = dict(best_params)
        best_params["learning_rate"] = base_lr * best_multiplier
        best_params["n_estimators"] = max(
            10, int(base_n_estimators / best_multiplier)
        )

        return best_params

    def _evaluate_params(
        self,
        node: ModelNode,
        ctx: DataContext,
        params: Dict[str, Any],
    ) -> float:
        """Evaluate parameters using CV."""
        try:
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score

            # Create model
            model = node.create_estimator(params)

            # Use simple CV for speed
            scores = cross_val_score(
                model,
                ctx.X,
                ctx.y,
                cv=self.cv_folds,
                scoring="neg_mean_squared_error",
            )

            return -np.mean(scores)

        except Exception:
            return float("inf")

    def __repr__(self) -> str:
        return f"XGBMultiplierPlugin(multipliers={self.multipliers})"
