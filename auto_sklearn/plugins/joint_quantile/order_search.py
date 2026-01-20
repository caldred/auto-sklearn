"""OrderSearchPlugin: Local search for optimal property ordering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from auto_sklearn.plugins.base import ModelPlugin

if TYPE_CHECKING:
    from auto_sklearn.core.data.context import DataContext
    from auto_sklearn.core.model.joint_quantile_graph import JointQuantileGraph
    from auto_sklearn.core.tuning.joint_quantile_orchestrator import (
        JointQuantileOrchestrator,
        JointQuantileFitResult,
    )


@dataclass
class OrderSearchConfig:
    """
    Configuration for order search.

    Attributes:
        max_iterations: Maximum iterations of local search.
        n_random_restarts: Number of random restarts from different orderings.
        score_fn: Custom scoring function. If None, uses mean pinball loss
                 across all properties and quantiles.
        verbose: Verbosity level.
    """

    max_iterations: int = 10
    n_random_restarts: int = 0
    score_fn: Optional[Callable[[JointQuantileFitResult, Dict[str, pd.Series]], float]] = None
    verbose: int = 1


@dataclass
class OrderSearchResult:
    """
    Result of order search.

    Attributes:
        best_order: The best property ordering found.
        best_score: Score of the best ordering.
        best_fit_result: Fit result for the best ordering.
        search_history: History of (order, score) tuples evaluated.
        n_iterations: Number of iterations performed.
        converged: Whether search converged (no improvement found).
    """

    best_order: List[str]
    best_score: float
    best_fit_result: JointQuantileFitResult
    search_history: List[Tuple[List[str], float]] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


class OrderSearchPlugin(ModelPlugin):
    """
    Plugin for searching optimal property ordering via local swap search.

    The order of properties in joint quantile regression affects model
    quality because later properties condition on earlier ones. This
    plugin implements hill-climbing search with adjacent swaps to find
    a good ordering.

    Algorithm:
    1. Start with initial ordering (from constraints or default)
    2. Evaluate all valid adjacent swaps
    3. Accept the swap that gives the best improvement
    4. Repeat until no improvement (local optimum)
    5. Optionally restart from random orderings

    Example:
        search_plugin = OrderSearchPlugin(config=OrderSearchConfig(
            max_iterations=20,
            n_random_restarts=3,
        ))

        result = search_plugin.search_order(
            graph=joint_quantile_graph,
            ctx=data_context,
            targets={"price": y_price, "volume": y_volume},
            orchestrator=orchestrator,
        )

        print(f"Best order: {result.best_order}")
        print(f"Best score: {result.best_score}")
    """

    def __init__(
        self,
        config: Optional[OrderSearchConfig] = None,
    ) -> None:
        """
        Initialize the plugin.

        Args:
            config: Order search configuration.
        """
        self.config = config or OrderSearchConfig()
        self._rng = np.random.RandomState()

    def applies_to(self, estimator_class) -> bool:
        """Check if this plugin applies to an estimator class."""
        # This plugin is specifically for JointQuantileGraph operations
        return False  # Not used via standard plugin hooks

    def search_order(
        self,
        graph: JointQuantileGraph,
        ctx: DataContext,
        targets: Dict[str, pd.Series],
        orchestrator: JointQuantileOrchestrator,
        random_state: Optional[int] = None,
    ) -> OrderSearchResult:
        """
        Search for the optimal property ordering.

        Args:
            graph: JointQuantileGraph to optimize.
            ctx: Data context with input features.
            targets: Target values for each property.
            orchestrator: Orchestrator for fitting.
            random_state: Random seed for reproducibility.

        Returns:
            OrderSearchResult with best ordering found.
        """
        if random_state is not None:
            self._rng = np.random.RandomState(random_state)

        # Run initial search from current order
        best_result = self._local_search(graph, ctx, targets, orchestrator)

        # Random restarts
        for restart_idx in range(self.config.n_random_restarts):
            if self.config.verbose >= 1:
                print(f"\nRandom restart {restart_idx + 1}/{self.config.n_random_restarts}")

            # Generate random valid order
            random_order = self._generate_random_order(graph)
            graph.set_order(random_order)

            restart_result = self._local_search(graph, ctx, targets, orchestrator)

            # Keep best result
            if restart_result.best_score < best_result.best_score:
                best_result = restart_result
                if self.config.verbose >= 1:
                    print(f"  New best score: {best_result.best_score:.6f}")

        # Restore best order
        graph.set_order(best_result.best_order)

        return best_result

    def _local_search(
        self,
        graph: JointQuantileGraph,
        ctx: DataContext,
        targets: Dict[str, pd.Series],
        orchestrator: JointQuantileOrchestrator,
    ) -> OrderSearchResult:
        """
        Run local search from current ordering.

        Args:
            graph: Graph with initial ordering.
            ctx: Data context.
            targets: Target values.
            orchestrator: Orchestrator for fitting.

        Returns:
            OrderSearchResult for this search run.
        """
        current_order = list(graph.property_order)
        search_history = []

        # Evaluate initial order
        fit_result = orchestrator.fit(ctx, targets)
        current_score = self._score_fit_result(fit_result, targets)
        search_history.append((list(current_order), current_score))

        best_order = list(current_order)
        best_score = current_score
        best_fit = fit_result

        if self.config.verbose >= 1:
            print(f"Initial order: {current_order}, score: {current_score:.6f}")

        n_iterations = 0
        converged = False

        for iteration in range(self.config.max_iterations):
            n_iterations += 1

            # Find all valid swaps
            valid_swaps = graph.get_valid_swaps()

            if not valid_swaps:
                converged = True
                break

            # Evaluate all valid swaps
            swap_results = []
            for swap in valid_swaps:
                i, j = swap

                # Create swapped order
                swapped_order = list(current_order)
                swapped_order[i], swapped_order[j] = swapped_order[j], swapped_order[i]

                # Fit with swapped order
                graph.set_order(swapped_order)
                swap_fit = orchestrator.fit(ctx, targets)
                swap_score = self._score_fit_result(swap_fit, targets)

                swap_results.append((swap, swapped_order, swap_score, swap_fit))

                if self.config.verbose >= 2:
                    print(f"  Swap {swap}: {swapped_order} -> {swap_score:.6f}")

            # Find best swap
            best_swap_idx = np.argmin([r[2] for r in swap_results])
            best_swap = swap_results[best_swap_idx]

            swap_pos, new_order, new_score, new_fit = best_swap

            # Check for improvement
            if new_score < current_score:
                current_order = new_order
                current_score = new_score
                search_history.append((list(current_order), current_score))

                if new_score < best_score:
                    best_order = list(current_order)
                    best_score = new_score
                    best_fit = new_fit

                if self.config.verbose >= 1:
                    print(
                        f"Iteration {iteration + 1}: Swap {swap_pos} -> "
                        f"{current_order}, score: {current_score:.6f}"
                    )

                # Update graph to use new order
                graph.set_order(current_order)
            else:
                # No improvement - local optimum reached
                converged = True
                if self.config.verbose >= 1:
                    print(f"Converged at iteration {iteration + 1}")
                break

        # Restore best order
        graph.set_order(best_order)

        return OrderSearchResult(
            best_order=best_order,
            best_score=best_score,
            best_fit_result=best_fit,
            search_history=search_history,
            n_iterations=n_iterations,
            converged=converged,
        )

    def _score_fit_result(
        self,
        fit_result: JointQuantileFitResult,
        targets: Dict[str, pd.Series],
    ) -> float:
        """
        Score a fit result.

        Uses custom score function if provided, otherwise uses mean
        pinball loss across all properties and quantiles.

        Args:
            fit_result: Result from fitting.
            targets: Target values.

        Returns:
            Score (lower is better).
        """
        if self.config.score_fn is not None:
            return self.config.score_fn(fit_result, targets)

        # Default: mean pinball loss across properties and quantiles
        total_loss = 0.0
        n_scores = 0

        for prop_name, fitted_node in fit_result.fitted_nodes.items():
            y_true = targets[prop_name].values
            oof_preds = fitted_node.oof_quantile_predictions

            for q_idx, tau in enumerate(fitted_node.quantile_levels):
                y_pred = oof_preds[:, q_idx]
                loss = self._pinball_loss(y_true, y_pred, tau)
                total_loss += loss
                n_scores += 1

        return total_loss / n_scores if n_scores > 0 else float("inf")

    def _pinball_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tau: float,
    ) -> float:
        """Calculate pinball loss for quantile regression."""
        residual = y_true - y_pred
        loss = np.where(
            residual >= 0,
            tau * residual,
            (tau - 1) * residual,
        )
        return np.mean(loss)

    def _generate_random_order(self, graph: JointQuantileGraph) -> List[str]:
        """
        Generate a random valid ordering.

        Respects fixed position constraints.

        Args:
            graph: JointQuantileGraph.

        Returns:
            Random valid ordering.
        """
        props = list(graph.property_order)
        constraints = graph.config.order_constraints

        if constraints is None:
            # No constraints - full shuffle
            self._rng.shuffle(props)
            return props

        # Respect fixed positions
        fixed = constraints.fixed_positions
        free_positions = [i for i in range(len(props)) if i not in fixed.values()]
        free_props = [p for p in props if p not in fixed]

        # Shuffle free properties
        self._rng.shuffle(free_props)

        # Build result
        result = [None] * len(props)
        for prop_name, pos in fixed.items():
            result[pos] = prop_name

        free_idx = 0
        for i in range(len(result)):
            if result[i] is None:
                result[i] = free_props[free_idx]
                free_idx += 1

        # Try to satisfy must_precede constraints
        if constraints.must_precede:
            result = self._fix_precedence_order(result, constraints.must_precede)

        return result

    def _fix_precedence_order(
        self,
        order: List[str],
        must_precede: List[Tuple[str, str]],
    ) -> List[str]:
        """Fix ordering to satisfy precedence constraints."""
        order = list(order)

        for _ in range(len(order) ** 2):
            changed = False
            for first, second in must_precede:
                if first not in order or second not in order:
                    continue

                idx_first = order.index(first)
                idx_second = order.index(second)

                if idx_first > idx_second:
                    # Move first before second
                    order.remove(first)
                    new_idx = order.index(second)
                    order.insert(new_idx, first)
                    changed = True

            if not changed:
                break

        return order

    def __repr__(self) -> str:
        return (
            f"OrderSearchPlugin(max_iterations={self.config.max_iterations}, "
            f"n_random_restarts={self.config.n_random_restarts})"
        )
