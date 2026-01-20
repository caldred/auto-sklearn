"""
Hyperparameter Correlation Analysis.

This module provides tools for discovering and modeling correlations between
hyperparameters. The key insight is that some hyperparameters provide similar
functional effects (e.g., different forms of regularization) and thus have
tradeoffs - an increase in one should lead to a decrease in another for
equivalent model behavior.

Detecting these correlations allows us to:
1. Reparameterize the search space into more orthogonal dimensions
2. Reduce the effective dimensionality of the search
3. Improve optimization efficiency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from auto_sklearn.search.backends.base import OptimizationResult


class CorrelationType(Enum):
    """Types of hyperparameter correlations."""

    SUBSTITUTE = "substitute"
    """
    Substitutes: Params that provide similar effects.
    Increasing one can compensate for decreasing another.
    Example: L1 and L2 regularization both reduce overfitting.
    """

    COMPLEMENT = "complement"
    """
    Complements: Params that work together.
    They tend to increase or decrease together for optimal results.
    Example: learning_rate and momentum in some optimization schemes.
    """

    TRADEOFF = "tradeoff"
    """
    Tradeoffs: Params with inverse relationship for constant effect.
    Example: learning_rate and n_estimators (higher LR needs fewer trees).
    """

    CONDITIONAL = "conditional"
    """
    Conditional: One param's optimal value depends on another.
    Example: max_depth optimal value depends on n_estimators.
    """


@dataclass
class HyperparameterCorrelation:
    """
    Represents a discovered correlation between hyperparameters.

    Attributes:
        params: List of correlated parameter names.
        correlation_type: Type of correlation.
        strength: Correlation strength (0 to 1).
        functional_form: Description of the functional relationship.
        transform: Optional callable to compute the "effective" value.
        inverse_transform: Optional callable to recover original params.
        confidence: Confidence in this correlation (based on sample size).
    """

    params: List[str]
    correlation_type: CorrelationType
    strength: float
    functional_form: str = ""
    transform: Optional[Callable] = None
    inverse_transform: Optional[Callable] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_pairwise(self) -> bool:
        """Whether this is a pairwise correlation."""
        return len(self.params) == 2

    def effective_value(self, param_values: Dict[str, float]) -> float:
        """
        Compute the effective combined value of correlated params.

        For example, for learning_rate and n_estimators with tradeoff
        correlation, this might return lr * n_estimators (total budget).
        """
        if self.transform is None:
            raise ValueError("No transform defined for this correlation")

        values = [param_values[p] for p in self.params]
        return self.transform(*values)

    def decompose(
        self, effective: float, ratio: float = 0.5
    ) -> Dict[str, float]:
        """
        Decompose an effective value back to original params.

        Args:
            effective: The effective combined value.
            ratio: How to split between params (0 to 1).

        Returns:
            Dictionary of original parameter values.
        """
        if self.inverse_transform is None:
            raise ValueError("No inverse transform defined")

        return self.inverse_transform(effective, ratio)

    def __repr__(self) -> str:
        return (
            f"HyperparameterCorrelation({self.params}, "
            f"type={self.correlation_type.value}, strength={self.strength:.2f})"
        )


class CorrelationAnalyzer:
    """
    Analyzes optimization history to discover hyperparameter correlations.

    This class examines the relationship between hyperparameters across
    optimization trials to identify:
    - Parameters that provide similar regularization effects
    - Parameters with tradeoff relationships
    - Parameters that should be tuned together vs independently
    """

    def __init__(
        self,
        min_trials: int = 20,
        significance_threshold: float = 0.1,
        correlation_threshold: float = 0.3,
    ) -> None:
        """
        Initialize the correlation analyzer.

        Args:
            min_trials: Minimum trials needed for reliable analysis.
            significance_threshold: P-value threshold for significance.
            correlation_threshold: Minimum correlation to report.
        """
        self.min_trials = min_trials
        self.significance_threshold = significance_threshold
        self.correlation_threshold = correlation_threshold

    def analyze(
        self,
        optimization_result: OptimizationResult,
        param_names: Optional[List[str]] = None,
    ) -> List[HyperparameterCorrelation]:
        """
        Analyze optimization history for correlations.

        Args:
            optimization_result: Result from hyperparameter optimization.
            param_names: Specific params to analyze (default: all).

        Returns:
            List of discovered correlations.
        """
        trials = [t for t in optimization_result.trials if t.is_complete]

        if len(trials) < self.min_trials:
            return []

        # Extract param values and scores
        if param_names is None:
            param_names = list(trials[0].params.keys())

        param_matrix = self._build_param_matrix(trials, param_names)
        scores = np.array([t.value for t in trials])

        correlations = []

        # Analyze pairwise correlations
        for i, p1 in enumerate(param_names):
            for j, p2 in enumerate(param_names):
                if i >= j:
                    continue

                corr = self._analyze_pair(
                    param_matrix[:, i],
                    param_matrix[:, j],
                    scores,
                    p1,
                    p2,
                )
                if corr is not None:
                    correlations.append(corr)

        # Analyze higher-order correlations (tradeoffs)
        correlations.extend(
            self._analyze_tradeoffs(param_matrix, scores, param_names)
        )

        return correlations

    def _build_param_matrix(
        self,
        trials: List,
        param_names: List[str],
    ) -> np.ndarray:
        """Build matrix of parameter values across trials."""
        n_trials = len(trials)
        n_params = len(param_names)
        matrix = np.zeros((n_trials, n_params))

        for i, trial in enumerate(trials):
            for j, name in enumerate(param_names):
                value = trial.params.get(name)
                if value is not None:
                    # Handle categorical params
                    if isinstance(value, (int, float)):
                        matrix[i, j] = float(value)
                    else:
                        matrix[i, j] = hash(str(value)) % 1000

        return matrix

    def _analyze_pair(
        self,
        values1: np.ndarray,
        values2: np.ndarray,
        scores: np.ndarray,
        name1: str,
        name2: str,
    ) -> Optional[HyperparameterCorrelation]:
        """Analyze correlation between a pair of parameters."""
        # Skip if either has no variance
        if np.std(values1) < 1e-10 or np.std(values2) < 1e-10:
            return None

        # Compute Pearson correlation between params
        param_corr = np.corrcoef(values1, values2)[0, 1]

        # Compute correlation of each param with score
        score_corr1 = np.corrcoef(values1, scores)[0, 1]
        score_corr2 = np.corrcoef(values2, scores)[0, 1]

        # Detect substitute relationship (similar effect on score)
        if (
            abs(param_corr) < 0.5  # Not strongly correlated with each other
            and np.sign(score_corr1) == np.sign(score_corr2)  # Same direction effect
            and abs(score_corr1) > 0.2
            and abs(score_corr2) > 0.2
        ):
            strength = min(abs(score_corr1), abs(score_corr2))
            if strength > self.correlation_threshold:
                return HyperparameterCorrelation(
                    params=[name1, name2],
                    correlation_type=CorrelationType.SUBSTITUTE,
                    strength=strength,
                    functional_form=f"{name1} + {name2} ≈ constant effect",
                    confidence=len(scores) / 100,
                )

        # Detect complement relationship (move together)
        if abs(param_corr) > 0.5:
            if param_corr > 0:
                corr_type = CorrelationType.COMPLEMENT
                form = f"{name1} ↑ with {name2} ↑"
            else:
                corr_type = CorrelationType.TRADEOFF
                form = f"{name1} ↑ with {name2} ↓"

            return HyperparameterCorrelation(
                params=[name1, name2],
                correlation_type=corr_type,
                strength=abs(param_corr),
                functional_form=form,
                confidence=len(scores) / 100,
            )

        return None

    def _analyze_tradeoffs(
        self,
        param_matrix: np.ndarray,
        scores: np.ndarray,
        param_names: List[str],
    ) -> List[HyperparameterCorrelation]:
        """
        Analyze for tradeoff relationships (product/ratio constant).

        Looks for pairs where p1 * p2 or p1 / p2 is approximately
        constant among good solutions.
        """
        correlations = []

        # Get indices of top 25% trials
        threshold = np.percentile(scores, 25)  # Lower is better
        good_mask = scores <= threshold

        if np.sum(good_mask) < 5:
            return correlations

        for i, p1 in enumerate(param_names):
            for j, p2 in enumerate(param_names):
                if i >= j:
                    continue

                v1 = param_matrix[good_mask, i]
                v2 = param_matrix[good_mask, j]

                # Skip if any zeros (can't compute product/ratio)
                if np.any(v1 == 0) or np.any(v2 == 0):
                    continue

                # Check if product is approximately constant
                product = v1 * v2
                cv_product = np.std(product) / (np.mean(product) + 1e-10)

                if cv_product < 0.3:  # Low coefficient of variation
                    mean_product = np.mean(product)

                    def make_transform(mean_prod):
                        return lambda x, y: x * y / mean_prod

                    def make_inverse(mean_prod):
                        def inverse(effective, ratio):
                            # effective * mean_prod = x * y
                            # ratio determines split: x = sqrt(eff * mean * ratio)
                            total = effective * mean_prod
                            x = np.sqrt(total * (ratio + 0.01))
                            y = total / (x + 1e-10)
                            return {param_names[i]: x, param_names[j]: y}
                        return inverse

                    correlations.append(
                        HyperparameterCorrelation(
                            params=[p1, p2],
                            correlation_type=CorrelationType.TRADEOFF,
                            strength=1 - cv_product,
                            functional_form=f"{p1} × {p2} ≈ {mean_product:.2f}",
                            transform=make_transform(mean_product),
                            inverse_transform=make_inverse(mean_product),
                            confidence=np.sum(good_mask) / len(scores),
                            metadata={"mean_product": mean_product},
                        )
                    )

        return correlations

    def suggest_reparameterization(
        self,
        correlations: List[HyperparameterCorrelation],
    ) -> Dict[str, Any]:
        """
        Suggest reparameterizations based on discovered correlations.

        Returns a dictionary with:
        - "substitutes": Groups of params that should be combined
        - "tradeoffs": Pairs that should use product/ratio parameterization
        - "independent": Params that can be tuned independently
        """
        substitutes = []
        tradeoffs = []
        seen_params = set()

        # Group substitutes
        for corr in correlations:
            if corr.correlation_type == CorrelationType.SUBSTITUTE:
                if corr.strength > 0.5:
                    substitutes.append(corr.params)
                    seen_params.update(corr.params)

        # Identify tradeoffs
        for corr in correlations:
            if corr.correlation_type == CorrelationType.TRADEOFF:
                if corr.strength > 0.5 and corr.transform is not None:
                    tradeoffs.append({
                        "params": corr.params,
                        "transform": corr.transform,
                        "inverse": corr.inverse_transform,
                        "form": corr.functional_form,
                    })
                    seen_params.update(corr.params)

        return {
            "substitutes": substitutes,
            "tradeoffs": tradeoffs,
            "correlation_details": correlations,
        }

    def __repr__(self) -> str:
        return (
            f"CorrelationAnalyzer(min_trials={self.min_trials}, "
            f"threshold={self.correlation_threshold})"
        )
