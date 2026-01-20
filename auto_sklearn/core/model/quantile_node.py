"""QuantileModelNode: Model node for quantile regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from auto_sklearn.core.model.node import ModelNode, OutputType

if TYPE_CHECKING:
    from auto_sklearn.core.data.context import DataContext
    from auto_sklearn.search.space import SearchSpace


# Default quantile levels: 19 levels from 0.05 to 0.95
DEFAULT_QUANTILE_LEVELS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]


@dataclass
class QuantileScalingConfig:
    """
    Configuration for scaling parameters by quantile level.

    Some hyperparameters (like regularization) may need different values
    at extreme quantiles compared to the median. This config defines
    how to scale parameters based on distance from the median (tau=0.5).

    Attributes:
        base_params: Base hyperparameters used for all quantiles.
        scaling_rules: Rules for scaling parameters at tail quantiles.
                      Format: {"param_name": {"base": value, "tail_multiplier": mult}}
                      The tail_multiplier is applied proportionally to |tau - 0.5|.

    Example:
        config = QuantileScalingConfig(
            base_params={"n_estimators": 100, "max_depth": 6},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
                "reg_alpha": {"base": 0.1, "tail_multiplier": 1.5},
            }
        )
        # At tau=0.1 (tail distance = 0.4, max distance = 0.45):
        # reg_lambda = 1.0 * (1 + (0.4/0.45) * 2.0) = 2.78
    """

    base_params: Dict[str, Any] = field(default_factory=dict)
    scaling_rules: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get_params_for_quantile(self, tau: float) -> Dict[str, Any]:
        """
        Get scaled parameters for a specific quantile level.

        Args:
            tau: Quantile level (0 < tau < 1).

        Returns:
            Dictionary of scaled parameters.
        """
        params = dict(self.base_params)

        # Distance from median, normalized by max distance (0.45 for 0.05-0.95 range)
        tail_distance = abs(tau - 0.5)
        max_distance = 0.45  # Assuming standard 0.05-0.95 range
        normalized_distance = tail_distance / max_distance

        for param_name, rule in self.scaling_rules.items():
            base_value = rule.get("base", 1.0)
            tail_multiplier = rule.get("tail_multiplier", 1.0)

            # Scale factor: 1 at median, up to (1 + tail_multiplier) at extremes
            scale_factor = 1.0 + normalized_distance * (tail_multiplier - 1.0)
            params[param_name] = base_value * scale_factor

        return params


@dataclass
class QuantileModelNode(ModelNode):
    """
    Model node specialized for quantile regression.

    Extends ModelNode to handle multiple quantile levels, each requiring
    a separate model with quantile-specific objective function.

    Attributes:
        property_name: Name of the target property this node predicts.
        quantile_levels: List of quantile levels to model (default: 19 levels).
        quantile_scaling: Optional config for scaling params by quantile.
        xgboost_objective: XGBoost objective name for quantile regression.

    Note:
        The estimator_class should support quantile regression, typically via
        XGBoost with objective='reg:quantileerror' and quantile_alpha parameter.
    """

    property_name: str = ""
    quantile_levels: List[float] = field(default_factory=lambda: list(DEFAULT_QUANTILE_LEVELS))
    quantile_scaling: Optional[QuantileScalingConfig] = None
    xgboost_objective: str = "reg:quantileerror"

    def __post_init__(self) -> None:
        """Validate node configuration."""
        # Set output type to QUANTILES
        self.output_type = OutputType.QUANTILES

        # Validate property_name
        if not self.property_name:
            raise ValueError("property_name is required for QuantileModelNode")

        # Validate quantile levels
        if not self.quantile_levels:
            raise ValueError("quantile_levels cannot be empty")

        for tau in self.quantile_levels:
            if not 0 < tau < 1:
                raise ValueError(f"Quantile level must be in (0, 1), got {tau}")

        # Sort quantile levels
        self.quantile_levels = sorted(self.quantile_levels)

        # Set default name if not provided
        if not self.name:
            self.name = f"quantile_{self.property_name}"

        # Call parent validation (skipping estimator_class check for now)
        if self.estimator_class:
            if not hasattr(self.estimator_class, "fit"):
                raise ValueError(
                    f"Estimator {self.estimator_class} must have a 'fit' method"
                )

    def create_estimator_for_quantile(
        self,
        tau: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create an estimator configured for a specific quantile level.

        Args:
            tau: Quantile level (0 < tau < 1).
            params: Additional hyperparameters to use.

        Returns:
            Configured estimator instance for quantile regression.
        """
        # Start with fixed params
        all_params = dict(self.fixed_params)

        # Apply quantile scaling if configured
        if self.quantile_scaling:
            scaled_params = self.quantile_scaling.get_params_for_quantile(tau)
            all_params.update(scaled_params)

        # Apply user-provided params
        if params:
            all_params.update(params)

        # Set quantile-specific parameters for XGBoost
        all_params["objective"] = self.xgboost_objective
        all_params["quantile_alpha"] = tau

        return self.estimator_class(**all_params)

    def get_params_for_quantile(
        self,
        tau: float,
        tuned_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get complete parameter dict for a specific quantile level.

        Args:
            tau: Quantile level.
            tuned_params: Tuned hyperparameters (from optimization at median).

        Returns:
            Complete parameter dictionary.
        """
        # Start with fixed params
        all_params = dict(self.fixed_params)

        # Apply tuned params
        if tuned_params:
            all_params.update(tuned_params)

        # Apply quantile scaling
        if self.quantile_scaling:
            scaled_params = self.quantile_scaling.get_params_for_quantile(tau)
            all_params.update(scaled_params)

        # Set quantile-specific parameters
        all_params["objective"] = self.xgboost_objective
        all_params["quantile_alpha"] = tau

        return all_params

    @property
    def median_quantile(self) -> float:
        """Get the median quantile level (or closest to 0.5)."""
        return min(self.quantile_levels, key=lambda x: abs(x - 0.5))

    @property
    def n_quantiles(self) -> int:
        """Number of quantile levels."""
        return len(self.quantile_levels)

    def __repr__(self) -> str:
        class_name = self.estimator_class.__name__ if self.estimator_class else "None"
        return (
            f"QuantileModelNode(name={self.name!r}, property={self.property_name!r}, "
            f"estimator={class_name}, n_quantiles={self.n_quantiles})"
        )
