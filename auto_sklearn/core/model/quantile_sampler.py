"""Quantile sampling strategies for joint quantile regression."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class SamplingStrategy(Enum):
    """Strategies for sampling from quantile distributions."""

    LINEAR_INTERPOLATION = "linear_interpolation"
    """Linear interpolation between quantile levels."""

    NORMAL = "normal"
    """Fit normal distribution to quantiles."""

    SKEW_NORMAL = "skew_normal"
    """Fit skew-normal distribution to quantiles."""

    JOHNSON_SU = "johnson_su"
    """Fit Johnson SU distribution to quantiles."""

    AUTO = "auto"
    """Automatically select best-fitting distribution."""


class QuantileSamplerBase(ABC):
    """
    Abstract base class for quantile samplers.

    Quantile samplers take quantile predictions (values at specific quantile
    levels) and generate samples from the implied distribution.
    """

    @abstractmethod
    def fit(
        self,
        quantile_levels: np.ndarray,
        quantile_values: np.ndarray,
    ) -> "QuantileSamplerBase":
        """
        Fit the sampler to quantile predictions.

        Args:
            quantile_levels: Array of quantile levels (e.g., [0.1, 0.5, 0.9]).
            quantile_values: Array of predicted values at each quantile level.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int, uniform_samples: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate samples from the fitted distribution.

        Args:
            n_samples: Number of samples to generate.
            uniform_samples: Optional pre-generated uniform samples for
                           consistent sampling paths across properties.

        Returns:
            Array of sampled values.
        """
        pass

    @abstractmethod
    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Percent point function (inverse CDF).

        Args:
            q: Quantile levels (values in [0, 1]).

        Returns:
            Values at the requested quantile levels.
        """
        pass


class LinearInterpolationSampler(QuantileSamplerBase):
    """
    Sample by linear interpolation between quantile levels.

    This is the simplest approach: interpolate linearly between
    predicted quantile values using np.interp.
    """

    def __init__(self) -> None:
        """Initialize the sampler."""
        self._quantile_levels: Optional[np.ndarray] = None
        self._quantile_values: Optional[np.ndarray] = None

    def fit(
        self,
        quantile_levels: np.ndarray,
        quantile_values: np.ndarray,
    ) -> "LinearInterpolationSampler":
        """Fit by storing the quantile levels and values."""
        self._quantile_levels = np.asarray(quantile_levels)
        self._quantile_values = np.asarray(quantile_values)
        return self

    def sample(self, n_samples: int, uniform_samples: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample by interpolating at uniform random quantile levels."""
        if self._quantile_levels is None:
            raise RuntimeError("Sampler must be fit before sampling")

        if uniform_samples is None:
            uniform_samples = np.random.uniform(0, 1, n_samples)

        return self.ppf(uniform_samples)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Interpolate to get values at requested quantile levels."""
        if self._quantile_levels is None:
            raise RuntimeError("Sampler must be fit before calling ppf")

        q = np.asarray(q)
        return np.interp(q, self._quantile_levels, self._quantile_values)


class ParametricSampler(QuantileSamplerBase):
    """
    Sample by fitting a parametric distribution to quantiles.

    Supports normal, skew-normal, and Johnson SU distributions.
    """

    def __init__(self, distribution: str = "normal") -> None:
        """
        Initialize the sampler.

        Args:
            distribution: Distribution type ('normal', 'skew_normal', 'johnson_su').
        """
        self._distribution = distribution
        self._params: Optional[Tuple] = None
        self._dist = None

    def fit(
        self,
        quantile_levels: np.ndarray,
        quantile_values: np.ndarray,
    ) -> "ParametricSampler":
        """Fit the parametric distribution to match the quantiles."""
        try:
            from scipy import stats
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError("scipy is required for parametric sampling")

        quantile_levels = np.asarray(quantile_levels)
        quantile_values = np.asarray(quantile_values)

        if self._distribution == "normal":
            self._dist = stats.norm
            # Fit normal: minimize squared error between predicted and actual quantiles
            def objective(params):
                loc, scale = params
                if scale <= 0:
                    return np.inf
                predicted = stats.norm.ppf(quantile_levels, loc=loc, scale=scale)
                return np.sum((predicted - quantile_values) ** 2)

            # Initial guess from median and IQR
            median_idx = np.argmin(np.abs(quantile_levels - 0.5))
            loc_init = quantile_values[median_idx]
            q25_idx = np.argmin(np.abs(quantile_levels - 0.25))
            q75_idx = np.argmin(np.abs(quantile_levels - 0.75))
            iqr = quantile_values[q75_idx] - quantile_values[q25_idx]
            scale_init = max(iqr / 1.35, 0.01)  # IQR = 1.35 * sigma for normal

            result = minimize(objective, [loc_init, scale_init], method="Nelder-Mead")
            self._params = tuple(result.x)

        elif self._distribution == "skew_normal":
            self._dist = stats.skewnorm

            def objective(params):
                a, loc, scale = params
                if scale <= 0:
                    return np.inf
                predicted = stats.skewnorm.ppf(quantile_levels, a, loc=loc, scale=scale)
                return np.sum((predicted - quantile_values) ** 2)

            # Initial guess
            median_idx = np.argmin(np.abs(quantile_levels - 0.5))
            loc_init = quantile_values[median_idx]
            q25_idx = np.argmin(np.abs(quantile_levels - 0.25))
            q75_idx = np.argmin(np.abs(quantile_levels - 0.75))
            iqr = quantile_values[q75_idx] - quantile_values[q25_idx]
            scale_init = max(iqr / 1.35, 0.01)

            result = minimize(objective, [0, loc_init, scale_init], method="Nelder-Mead")
            self._params = tuple(result.x)

        elif self._distribution == "johnson_su":
            self._dist = stats.johnsonsu

            def objective(params):
                a, b, loc, scale = params
                if b <= 0 or scale <= 0:
                    return np.inf
                try:
                    predicted = stats.johnsonsu.ppf(quantile_levels, a, b, loc=loc, scale=scale)
                    if np.any(np.isnan(predicted)):
                        return np.inf
                    return np.sum((predicted - quantile_values) ** 2)
                except Exception:
                    return np.inf

            # Initial guess
            median_idx = np.argmin(np.abs(quantile_levels - 0.5))
            loc_init = quantile_values[median_idx]
            q25_idx = np.argmin(np.abs(quantile_levels - 0.25))
            q75_idx = np.argmin(np.abs(quantile_levels - 0.75))
            iqr = quantile_values[q75_idx] - quantile_values[q25_idx]
            scale_init = max(iqr / 1.35, 0.01)

            result = minimize(
                objective, [0, 1, loc_init, scale_init], method="Nelder-Mead"
            )
            self._params = tuple(result.x)

        else:
            raise ValueError(f"Unknown distribution: {self._distribution}")

        return self

    def sample(self, n_samples: int, uniform_samples: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample from the fitted distribution."""
        if self._params is None:
            raise RuntimeError("Sampler must be fit before sampling")

        if uniform_samples is None:
            uniform_samples = np.random.uniform(0, 1, n_samples)

        return self.ppf(uniform_samples)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Get values at requested quantile levels."""
        if self._params is None or self._dist is None:
            raise RuntimeError("Sampler must be fit before calling ppf")

        q = np.asarray(q)
        return self._dist.ppf(q, *self._params)


class AutoSelectSampler(QuantileSamplerBase):
    """
    Automatically select the best-fitting distribution.

    Tries multiple distributions and selects the one with lowest
    error on the observed quantiles.
    """

    def __init__(self) -> None:
        """Initialize the sampler."""
        self._best_sampler: Optional[QuantileSamplerBase] = None
        self._best_distribution: Optional[str] = None

    def fit(
        self,
        quantile_levels: np.ndarray,
        quantile_values: np.ndarray,
    ) -> "AutoSelectSampler":
        """Fit multiple distributions and select the best one."""
        quantile_levels = np.asarray(quantile_levels)
        quantile_values = np.asarray(quantile_values)

        candidates = [
            ("linear", LinearInterpolationSampler()),
            ("normal", ParametricSampler("normal")),
            ("skew_normal", ParametricSampler("skew_normal")),
        ]

        best_error = np.inf
        best_sampler = None
        best_name = None

        for name, sampler in candidates:
            try:
                sampler.fit(quantile_levels, quantile_values)
                # Evaluate fit quality
                predicted = sampler.ppf(quantile_levels)
                error = np.mean((predicted - quantile_values) ** 2)

                if error < best_error:
                    best_error = error
                    best_sampler = sampler
                    best_name = name
            except Exception:
                continue

        if best_sampler is None:
            # Fallback to linear interpolation
            best_sampler = LinearInterpolationSampler()
            best_sampler.fit(quantile_levels, quantile_values)
            best_name = "linear"

        self._best_sampler = best_sampler
        self._best_distribution = best_name

        return self

    def sample(self, n_samples: int, uniform_samples: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample from the best-fit distribution."""
        if self._best_sampler is None:
            raise RuntimeError("Sampler must be fit before sampling")
        return self._best_sampler.sample(n_samples, uniform_samples)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Get values at requested quantile levels from best-fit distribution."""
        if self._best_sampler is None:
            raise RuntimeError("Sampler must be fit before calling ppf")
        return self._best_sampler.ppf(q)

    @property
    def selected_distribution(self) -> Optional[str]:
        """Name of the selected distribution."""
        return self._best_distribution


@dataclass
class QuantileSampler:
    """
    Main interface for sampling from quantile distributions.

    This class manages sampling across multiple properties while maintaining
    consistent sampling paths (same uniform samples across all properties).

    Attributes:
        strategy: Sampling strategy to use.
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.

    Example:
        sampler = QuantileSampler(
            strategy=SamplingStrategy.LINEAR_INTERPOLATION,
            n_samples=1000,
            random_state=42
        )

        # Sample from property 1
        samples_1 = sampler.sample_property(
            "price",
            quantile_levels=[0.1, 0.5, 0.9],
            quantile_predictions=[100, 150, 200]
        )

        # Sample from property 2 (uses same uniform samples)
        samples_2 = sampler.sample_property(
            "volume",
            quantile_levels=[0.1, 0.5, 0.9],
            quantile_predictions=[50, 100, 180]
        )
    """

    strategy: SamplingStrategy = SamplingStrategy.LINEAR_INTERPOLATION
    n_samples: int = 1000
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize random state and pre-generate uniform samples."""
        self._rng = np.random.RandomState(self.random_state)
        self._uniform_samples = self._rng.uniform(0, 1, self.n_samples)
        self._property_samplers: Dict[str, QuantileSamplerBase] = {}

    def reset_samples(self) -> None:
        """Regenerate uniform samples (for new inference batch)."""
        self._uniform_samples = self._rng.uniform(0, 1, self.n_samples)
        self._property_samplers.clear()

    def sample_property(
        self,
        property_name: str,
        quantile_levels: Union[List[float], np.ndarray],
        quantile_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Sample from the quantile distribution for a single property.

        Args:
            property_name: Name of the property (for tracking).
            quantile_levels: Quantile levels (e.g., [0.1, 0.5, 0.9]).
            quantile_predictions: Predicted values at each quantile level.
                                 Shape: (n_quantiles,) for single point,
                                        (n_data_points, n_quantiles) for batch.

        Returns:
            Sampled values. Shape: (n_samples,) for single point,
                                  (n_data_points, n_samples) for batch.
        """
        quantile_levels = np.asarray(quantile_levels)
        quantile_predictions = np.asarray(quantile_predictions)

        # Handle batch case
        if quantile_predictions.ndim == 2:
            n_data_points = quantile_predictions.shape[0]
            samples = np.zeros((n_data_points, self.n_samples))

            for i in range(n_data_points):
                sampler = self._create_sampler()
                sampler.fit(quantile_levels, quantile_predictions[i])
                samples[i] = sampler.sample(self.n_samples, self._uniform_samples)

            return samples

        # Single point case
        sampler = self._create_sampler()
        sampler.fit(quantile_levels, quantile_predictions)
        return sampler.sample(self.n_samples, self._uniform_samples)

    def sample_property_batched(
        self,
        property_name: str,
        quantile_levels: Union[List[float], np.ndarray],
        quantile_predictions: np.ndarray,
        batch_uniform_samples: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Efficiently sample for multiple data points using vectorized operations.

        For linear interpolation (the most common case), this uses numpy's
        vectorized interp across all data points simultaneously.

        Args:
            property_name: Name of the property.
            quantile_levels: Quantile levels.
            quantile_predictions: Shape (n_data_points, n_quantiles).
            batch_uniform_samples: Optional uniform samples of shape
                                  (n_data_points, n_samples). If None,
                                  uses the same samples for all points.

        Returns:
            Samples of shape (n_data_points, n_samples).
        """
        quantile_levels = np.asarray(quantile_levels)
        quantile_predictions = np.asarray(quantile_predictions)

        if quantile_predictions.ndim != 2:
            raise ValueError("quantile_predictions must be 2D (n_data_points, n_quantiles)")

        n_data_points = quantile_predictions.shape[0]

        if batch_uniform_samples is None:
            # Use same uniform samples for all data points
            batch_uniform_samples = np.tile(
                self._uniform_samples, (n_data_points, 1)
            )

        if self.strategy == SamplingStrategy.LINEAR_INTERPOLATION:
            # Vectorized linear interpolation
            samples = np.zeros((n_data_points, self.n_samples))
            for i in range(n_data_points):
                samples[i] = np.interp(
                    batch_uniform_samples[i],
                    quantile_levels,
                    quantile_predictions[i],
                )
            return samples
        else:
            # Fall back to per-point sampling for parametric distributions
            return self.sample_property(property_name, quantile_levels, quantile_predictions)

    def get_median(
        self,
        quantile_levels: Union[List[float], np.ndarray],
        quantile_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Get median predictions by interpolating to tau=0.5.

        Args:
            quantile_levels: Quantile levels.
            quantile_predictions: Shape (n_data_points, n_quantiles).

        Returns:
            Median values of shape (n_data_points,).
        """
        quantile_levels = np.asarray(quantile_levels)
        quantile_predictions = np.asarray(quantile_predictions)

        if quantile_predictions.ndim == 1:
            return np.interp(0.5, quantile_levels, quantile_predictions)

        n_data_points = quantile_predictions.shape[0]
        medians = np.zeros(n_data_points)
        for i in range(n_data_points):
            medians[i] = np.interp(0.5, quantile_levels, quantile_predictions[i])
        return medians

    def get_quantile(
        self,
        q: float,
        quantile_levels: Union[List[float], np.ndarray],
        quantile_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Get predictions at a specific quantile level by interpolation.

        Args:
            q: Quantile level (0 < q < 1).
            quantile_levels: Available quantile levels.
            quantile_predictions: Shape (n_data_points, n_quantiles).

        Returns:
            Values at quantile q, shape (n_data_points,).
        """
        quantile_levels = np.asarray(quantile_levels)
        quantile_predictions = np.asarray(quantile_predictions)

        if quantile_predictions.ndim == 1:
            return np.interp(q, quantile_levels, quantile_predictions)

        n_data_points = quantile_predictions.shape[0]
        values = np.zeros(n_data_points)
        for i in range(n_data_points):
            values[i] = np.interp(q, quantile_levels, quantile_predictions[i])
        return values

    @property
    def uniform_samples(self) -> np.ndarray:
        """Get the pre-generated uniform samples."""
        return self._uniform_samples.copy()

    def _create_sampler(self) -> QuantileSamplerBase:
        """Create a sampler based on the configured strategy."""
        if self.strategy == SamplingStrategy.LINEAR_INTERPOLATION:
            return LinearInterpolationSampler()
        elif self.strategy == SamplingStrategy.NORMAL:
            return ParametricSampler("normal")
        elif self.strategy == SamplingStrategy.SKEW_NORMAL:
            return ParametricSampler("skew_normal")
        elif self.strategy == SamplingStrategy.JOHNSON_SU:
            return ParametricSampler("johnson_su")
        elif self.strategy == SamplingStrategy.AUTO:
            return AutoSelectSampler()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
