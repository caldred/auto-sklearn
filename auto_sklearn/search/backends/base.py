"""SearchBackend: Abstract base class for optimization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from auto_sklearn.search.space import SearchSpace


@dataclass
class TrialResult:
    """
    Result of a single optimization trial.

    Attributes:
        params: Parameters used in this trial.
        value: Objective value (score).
        trial_id: Unique identifier for this trial.
        duration: Time taken for this trial in seconds.
        user_attrs: Additional user-defined attributes.
        state: Trial state (e.g., "COMPLETE", "PRUNED", "FAIL").
    """

    params: Dict[str, Any]
    value: float
    trial_id: int = 0
    duration: float = 0.0
    user_attrs: Dict[str, Any] = field(default_factory=dict)
    state: str = "COMPLETE"

    @property
    def is_complete(self) -> bool:
        """Whether the trial completed successfully."""
        return self.state == "COMPLETE"


@dataclass
class OptimizationResult:
    """
    Result of hyperparameter optimization.

    Attributes:
        best_params: Best parameters found.
        best_value: Best objective value achieved.
        trials: List of all trial results.
        n_trials: Total number of trials run.
        study_name: Name of the optimization study.
    """

    best_params: Dict[str, Any]
    best_value: float
    trials: List[TrialResult]
    n_trials: int
    study_name: str = ""

    @property
    def best_trial(self) -> Optional[TrialResult]:
        """Get the best trial result."""
        complete_trials = [t for t in self.trials if t.is_complete]
        if not complete_trials:
            return None
        return min(complete_trials, key=lambda t: t.value)

    def get_param_history(self, param_name: str) -> List[Any]:
        """Get the history of values for a specific parameter."""
        return [t.params.get(param_name) for t in self.trials if param_name in t.params]

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(best_value={self.best_value:.4f}, "
            f"n_trials={self.n_trials})"
        )


class SearchBackend(ABC):
    """
    Abstract base class for hyperparameter optimization backends.

    Subclasses implement the actual optimization logic using different
    libraries (Optuna, Hyperopt, etc.).
    """

    def __init__(
        self,
        direction: str = "minimize",
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the search backend.

        Args:
            direction: Optimization direction ("minimize" or "maximize").
            random_state: Random seed for reproducibility.
        """
        if direction not in ("minimize", "maximize"):
            raise ValueError(f"direction must be 'minimize' or 'maximize', got {direction}")
        self.direction = direction
        self.random_state = random_state

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: SearchSpace,
        n_trials: int,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
        study_name: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            objective: Function that takes params dict and returns objective value.
            search_space: Search space to optimize over.
            n_trials: Number of trials to run.
            timeout: Optional timeout in seconds.
            callbacks: Optional list of callback functions.
            study_name: Optional name for the study.

        Returns:
            OptimizationResult with best parameters and all trial results.
        """
        pass

    @abstractmethod
    def suggest_params(self, search_space: SearchSpace) -> Dict[str, Any]:
        """
        Suggest a single set of parameters.

        Args:
            search_space: Search space to sample from.

        Returns:
            Dictionary of parameter values.
        """
        pass

    @abstractmethod
    def tell(self, params: Dict[str, Any], value: float) -> None:
        """
        Report the result of evaluating parameters.

        This is used for ask-tell style optimization.

        Args:
            params: Parameters that were evaluated.
            value: Objective value.
        """
        pass

    def supports_pruning(self) -> bool:
        """Whether this backend supports trial pruning."""
        return False

    def supports_parallel(self) -> bool:
        """Whether this backend supports parallel optimization."""
        return False

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state for serialization."""
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization."""
        pass
