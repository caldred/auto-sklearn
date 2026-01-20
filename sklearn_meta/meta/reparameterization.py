"""
Hyperparameter Reparameterization.

This module provides transformations that convert correlated hyperparameter
spaces into more orthogonal representations. The key insight is that many
hyperparameters have functional relationships:

- learning_rate × n_estimators ≈ constant (total learning budget)
- L1_weight + L2_weight ≈ total regularization
- dropout1 + dropout2 + ... ≈ total dropout

By reparameterizing to:
1. A "total effect" dimension (strength, budget, etc.)
2. A "distribution" dimension (ratio, allocation, etc.)

We get a more orthogonal search space that's easier to optimize.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn_meta.search.space import SearchSpace
    from sklearn_meta.search.parameter import SearchParameter


@dataclass
class ReparameterizationResult:
    """Result of applying a reparameterization to sample."""

    original_params: Dict[str, float]
    transformed_params: Dict[str, float]
    reparameterization_name: str


class Reparameterization(ABC):
    """
    Abstract base class for hyperparameter reparameterizations.

    A reparameterization transforms a set of correlated parameters into
    a more orthogonal representation, then transforms back to the original
    parameter space for model training.
    """

    def __init__(self, name: str, original_params: List[str]) -> None:
        """
        Initialize the reparameterization.

        Args:
            name: Name of this reparameterization.
            original_params: List of original parameter names.
        """
        self.name = name
        self.original_params = original_params

    @property
    @abstractmethod
    def transformed_params(self) -> List[str]:
        """Names of the transformed parameters."""
        pass

    @abstractmethod
    def forward(self, original: Dict[str, float]) -> Dict[str, float]:
        """
        Transform from original to reparameterized space.

        Args:
            original: Dictionary of original parameter values.

        Returns:
            Dictionary of transformed parameter values.
        """
        pass

    @abstractmethod
    def inverse(self, transformed: Dict[str, float]) -> Dict[str, float]:
        """
        Transform from reparameterized back to original space.

        Args:
            transformed: Dictionary of transformed parameter values.

        Returns:
            Dictionary of original parameter values.
        """
        pass

    @abstractmethod
    def get_transformed_bounds(
        self,
        original_bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float, bool]]:
        """
        Compute bounds for transformed parameters.

        Args:
            original_bounds: Bounds for original params as (low, high) tuples.

        Returns:
            Bounds for transformed params as (low, high, log_scale) tuples.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.original_params} -> {self.transformed_params})"


class LogProductReparameterization(Reparameterization):
    """
    Reparameterization for multiplicative tradeoffs.

    Transforms params where p1 × p2 ≈ constant into:
    - log_product: log(p1 × p2) - the total "budget"
    - log_ratio: log(p1 / p2) - how the budget is split

    Common use case: learning_rate × n_estimators in gradient boosting.

    Mathematical basis:
    - log(p1) + log(p2) = log_product
    - log(p1) - log(p2) = log_ratio
    - Therefore: log(p1) = (log_product + log_ratio) / 2
    -            log(p2) = (log_product - log_ratio) / 2
    """

    def __init__(
        self,
        name: str,
        param1: str,
        param2: str,
        product_name: Optional[str] = None,
        ratio_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the log-product reparameterization.

        Args:
            name: Name of this reparameterization.
            param1: First parameter name.
            param2: Second parameter name.
            product_name: Name for the product parameter.
            ratio_name: Name for the ratio parameter.
        """
        super().__init__(name, [param1, param2])
        self.param1 = param1
        self.param2 = param2
        self._product_name = product_name or f"{param1}_{param2}_budget"
        self._ratio_name = ratio_name or f"{param1}_{param2}_ratio"

    @property
    def transformed_params(self) -> List[str]:
        return [self._product_name, self._ratio_name]

    def forward(self, original: Dict[str, float]) -> Dict[str, float]:
        p1 = original[self.param1]
        p2 = original[self.param2]

        # Use log transform for numerical stability
        log_p1 = np.log(p1 + 1e-10)
        log_p2 = np.log(p2 + 1e-10)

        return {
            self._product_name: log_p1 + log_p2,  # log(p1 * p2)
            self._ratio_name: log_p1 - log_p2,    # log(p1 / p2)
        }

    def inverse(self, transformed: Dict[str, float]) -> Dict[str, float]:
        log_product = transformed[self._product_name]
        log_ratio = transformed[self._ratio_name]

        log_p1 = (log_product + log_ratio) / 2
        log_p2 = (log_product - log_ratio) / 2

        return {
            self.param1: np.exp(log_p1),
            self.param2: np.exp(log_p2),
        }

    def get_transformed_bounds(
        self,
        original_bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float, bool]]:
        low1, high1 = original_bounds[self.param1]
        low2, high2 = original_bounds[self.param2]

        # Product bounds
        log_low = np.log(low1 + 1e-10) + np.log(low2 + 1e-10)
        log_high = np.log(high1) + np.log(high2)

        # Ratio bounds (can go negative to positive)
        ratio_low = np.log(low1 + 1e-10) - np.log(high2)
        ratio_high = np.log(high1) - np.log(low2 + 1e-10)

        return {
            self._product_name: (log_low, log_high, False),
            self._ratio_name: (ratio_low, ratio_high, False),
        }


class LinearReparameterization(Reparameterization):
    """
    Reparameterization for additive relationships.

    Transforms params where p1 + p2 ≈ constant effect into:
    - total: p1 + p2 - the total effect
    - ratio: p1 / (p1 + p2) - allocation between params

    Common use case: L1 and L2 regularization weights.
    """

    def __init__(
        self,
        name: str,
        params: List[str],
        weights: Optional[List[float]] = None,
        total_name: Optional[str] = None,
        ratio_prefix: str = "ratio",
    ) -> None:
        """
        Initialize the linear reparameterization.

        Args:
            name: Name of this reparameterization.
            params: List of parameter names.
            weights: Optional weights for each param in the total.
            total_name: Name for the total parameter.
            ratio_prefix: Prefix for ratio parameters.
        """
        super().__init__(name, params)
        self.weights = weights or [1.0] * len(params)
        self._total_name = total_name or "_".join(params) + "_total"
        self._ratio_prefix = ratio_prefix

    @property
    def transformed_params(self) -> List[str]:
        # Total + (n-1) ratios for n params
        ratios = [f"{self._ratio_prefix}_{p}" for p in self.original_params[:-1]]
        return [self._total_name] + ratios

    def forward(self, original: Dict[str, float]) -> Dict[str, float]:
        values = [original[p] * w for p, w in zip(self.original_params, self.weights)]
        total = sum(values)

        result = {self._total_name: total}

        # Compute cumulative ratios
        if total > 1e-10:
            cumsum = 0
            for i, p in enumerate(self.original_params[:-1]):
                cumsum += values[i]
                result[f"{self._ratio_prefix}_{p}"] = cumsum / total
        else:
            for p in self.original_params[:-1]:
                result[f"{self._ratio_prefix}_{p}"] = 1.0 / len(self.original_params)

        return result

    def inverse(self, transformed: Dict[str, float]) -> Dict[str, float]:
        total = transformed[self._total_name]

        # Decode ratios to get fractions
        fractions = []
        prev_ratio = 0

        for p in self.original_params[:-1]:
            ratio = transformed[f"{self._ratio_prefix}_{p}"]
            fractions.append(ratio - prev_ratio)
            prev_ratio = ratio

        fractions.append(1.0 - prev_ratio)  # Last param gets remainder

        # Convert to original params
        result = {}
        for i, p in enumerate(self.original_params):
            result[p] = (total * fractions[i]) / self.weights[i]

        return result

    def get_transformed_bounds(
        self,
        original_bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float, bool]]:
        # Total bounds
        min_total = sum(
            b[0] * w for b, w in zip(
                [original_bounds[p] for p in self.original_params],
                self.weights
            )
        )
        max_total = sum(
            b[1] * w for b, w in zip(
                [original_bounds[p] for p in self.original_params],
                self.weights
            )
        )

        result = {self._total_name: (min_total, max_total, False)}

        # Ratios are always 0 to 1
        for p in self.original_params[:-1]:
            result[f"{self._ratio_prefix}_{p}"] = (0.0, 1.0, False)

        return result


class RatioReparameterization(Reparameterization):
    """
    Simple two-parameter ratio reparameterization.

    Transforms (p1, p2) where total = p1 + p2 matters into:
    - total: p1 + p2
    - ratio: p1 / (p1 + p2)

    Simpler than LinearReparameterization for the common 2-param case.
    """

    def __init__(
        self,
        name: str,
        param1: str,
        param2: str,
        total_name: Optional[str] = None,
        ratio_name: Optional[str] = None,
    ) -> None:
        super().__init__(name, [param1, param2])
        self.param1 = param1
        self.param2 = param2
        self._total_name = total_name or f"{param1}_{param2}_total"
        self._ratio_name = ratio_name or f"{param1}_ratio"

    @property
    def transformed_params(self) -> List[str]:
        return [self._total_name, self._ratio_name]

    def forward(self, original: Dict[str, float]) -> Dict[str, float]:
        p1 = original[self.param1]
        p2 = original[self.param2]
        total = p1 + p2

        return {
            self._total_name: total,
            self._ratio_name: p1 / (total + 1e-10),
        }

    def inverse(self, transformed: Dict[str, float]) -> Dict[str, float]:
        total = transformed[self._total_name]
        ratio = transformed[self._ratio_name]

        return {
            self.param1: total * ratio,
            self.param2: total * (1 - ratio),
        }

    def get_transformed_bounds(
        self,
        original_bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float, bool]]:
        low1, high1 = original_bounds[self.param1]
        low2, high2 = original_bounds[self.param2]

        return {
            self._total_name: (low1 + low2, high1 + high2, False),
            self._ratio_name: (0.0, 1.0, False),
        }


class ReparameterizedSpace:
    """
    A search space with reparameterizations applied.

    This class wraps a SearchSpace and applies one or more reparameterizations,
    providing methods to sample in the transformed space and convert back to
    the original parameter space.
    """

    def __init__(
        self,
        original_space: SearchSpace,
        reparameterizations: List[Reparameterization],
    ) -> None:
        """
        Initialize the reparameterized space.

        Args:
            original_space: The original search space.
            reparameterizations: List of reparameterizations to apply.
        """
        self.original_space = original_space
        self.reparameterizations = reparameterizations

        # Track which params are reparameterized
        self._reparam_params = set()
        for reparam in reparameterizations:
            self._reparam_params.update(reparam.original_params)

    def build_transformed_space(self) -> SearchSpace:
        """
        Build a new SearchSpace in the transformed coordinates.

        Returns:
            SearchSpace with transformed parameters.
        """
        from sklearn_meta.search.space import SearchSpace

        new_space = SearchSpace()

        # Add untransformed params
        for param in self.original_space:
            if param.name not in self._reparam_params:
                new_space.add_parameter(param)

        # Add transformed params from reparameterizations
        for reparam in self.reparameterizations:
            # Get original bounds
            original_bounds = {}
            for pname in reparam.original_params:
                param = self.original_space.get_parameter(pname)
                if param is not None and hasattr(param, 'low'):
                    original_bounds[pname] = (param.low, param.high)
                else:
                    # Default bounds for categorical or unknown
                    original_bounds[pname] = (0.0, 1.0)

            # Get transformed bounds
            transformed_bounds = reparam.get_transformed_bounds(original_bounds)

            # Add to new space
            for tname, (low, high, log) in transformed_bounds.items():
                new_space.add_float(tname, low, high, log=log)

        return new_space

    def sample_and_transform(self, trial) -> Dict[str, Any]:
        """
        Sample from transformed space and convert to original params.

        Args:
            trial: Optuna trial for sampling.

        Returns:
            Dictionary of original parameter values.
        """
        transformed_space = self.build_transformed_space()
        transformed_sample = transformed_space.sample_optuna(trial)

        return self.inverse_transform(transformed_sample)

    def inverse_transform(self, transformed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert transformed parameters back to original space.

        Args:
            transformed: Dictionary of transformed parameter values.

        Returns:
            Dictionary of original parameter values.
        """
        result = {}

        # Copy untransformed params
        for pname in transformed:
            is_from_reparam = any(
                pname in reparam.transformed_params
                for reparam in self.reparameterizations
            )
            if not is_from_reparam:
                result[pname] = transformed[pname]

        # Apply inverse reparameterizations
        for reparam in self.reparameterizations:
            reparam_values = {
                k: transformed[k]
                for k in reparam.transformed_params
                if k in transformed
            }
            if len(reparam_values) == len(reparam.transformed_params):
                original = reparam.inverse(reparam_values)
                result.update(original)

        return result

    def forward_transform(self, original: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert original parameters to transformed space.

        Args:
            original: Dictionary of original parameter values.

        Returns:
            Dictionary of transformed parameter values.
        """
        result = {}

        # Copy untransformed params
        for pname, value in original.items():
            if pname not in self._reparam_params:
                result[pname] = value

        # Apply forward reparameterizations
        for reparam in self.reparameterizations:
            reparam_values = {
                k: original[k]
                for k in reparam.original_params
                if k in original
            }
            if len(reparam_values) == len(reparam.original_params):
                transformed = reparam.forward(reparam_values)
                result.update(transformed)

        return result

    def __repr__(self) -> str:
        return (
            f"ReparameterizedSpace(original={len(self.original_space)} params, "
            f"reparameterizations={len(self.reparameterizations)})"
        )
