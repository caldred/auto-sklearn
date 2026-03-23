"""Optimization strategies for tuning."""

from __future__ import annotations

from enum import Enum


class OptimizationStrategy(Enum):
    """Strategies for optimizing model hyperparameters."""

    LAYER_BY_LAYER = "layer_by_layer"
    """Optimize each layer of the model graph sequentially."""

    GREEDY = "greedy"
    """Optimize models one at a time in topological order."""

    NONE = "none"
    """Don't optimize - use fixed parameters only."""
