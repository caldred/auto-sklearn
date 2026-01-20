"""Optimization strategies for tuning."""

from __future__ import annotations

from enum import Enum


class OptimizationStrategy(Enum):
    """Strategies for optimizing model hyperparameters."""

    LAYER_BY_LAYER = "layer_by_layer"
    """
    Optimize each layer of the model graph sequentially.

    This is the default strategy. Models in layer 0 (no dependencies)
    are optimized first, then layer 1, etc. This ensures that when
    optimizing a model, its dependencies are already tuned.
    """

    FULL_GRAPH = "full_graph"
    """
    Optimize all models jointly in a single search space.

    This can find better global optima but is more expensive and
    harder to scale. Useful for small graphs with strong interactions.
    """

    GREEDY = "greedy"
    """
    Optimize models one at a time in topological order.

    Similar to LAYER_BY_LAYER but doesn't parallelize within layers.
    Can be useful when memory is constrained.
    """

    NONE = "none"
    """
    Don't optimize - use fixed parameters only.

    Useful for evaluation runs with pre-tuned parameters.
    """
