"""Knowledge distillation configuration and validation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Type


@dataclass(frozen=True)
class DistillationConfig:
    """
    Configuration for knowledge distillation.

    Controls how a student node learns from a teacher's soft targets
    using a blended KL-divergence + cross-entropy loss.

    Attributes:
        temperature: Softens probability distributions before KL computation.
            Higher values produce softer distributions. Must be > 0.
        alpha: Blending weight between soft and hard losses.
            Loss = alpha * KL_soft + (1 - alpha) * CE_hard.
            Must be in [0, 1].
    """

    temperature: float = 3.0
    alpha: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}"
            )
        if not (0 <= self.alpha <= 1):
            raise ValueError(
                f"alpha must be in [0, 1], got {self.alpha}"
            )


def validate_distillation_estimator(estimator_class: Type) -> None:
    """
    Validate that an estimator class supports custom objectives.

    Checks whether the estimator's constructor accepts an ``objective``
    parameter, which is required for injecting the distillation loss.

    Args:
        estimator_class: The estimator class to validate.

    Raises:
        ValueError: If the estimator does not support custom objectives.
    """
    sig = inspect.signature(estimator_class.__init__)
    if "objective" not in sig.parameters:
        raise ValueError(
            f"Estimator {estimator_class.__name__} does not support custom "
            f"objectives (no 'objective' parameter in __init__). "
            f"Distillation requires XGBoost, LightGBM, or similar."
        )
