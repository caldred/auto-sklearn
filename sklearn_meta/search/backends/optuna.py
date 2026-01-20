"""OptunaBackend: Optuna-based hyperparameter optimization."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import optuna
from optuna.samplers import TPESampler

from sklearn_meta.search.backends.base import (
    OptimizationResult,
    SearchBackend,
    TrialResult,
)

if TYPE_CHECKING:
    from sklearn_meta.search.space import SearchSpace


# Suppress Optuna's verbose logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaBackend(SearchBackend):
    """
    Optuna-based hyperparameter optimization backend.

    Uses Tree-structured Parzen Estimator (TPE) by default for efficient
    Bayesian optimization.
    """

    def __init__(
        self,
        direction: str = "minimize",
        random_state: Optional[int] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> None:
        """
        Initialize the Optuna backend.

        Args:
            direction: Optimization direction ("minimize" or "maximize").
            random_state: Random seed for reproducibility.
            sampler: Optional custom Optuna sampler.
            pruner: Optional Optuna pruner for early stopping.
        """
        super().__init__(direction=direction, random_state=random_state)

        self.sampler = sampler or TPESampler(seed=random_state)
        self.pruner = pruner

        self._study: Optional[optuna.Study] = None
        self._current_trial: Optional[optuna.Trial] = None

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
        Run hyperparameter optimization using Optuna.

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
        study_name = study_name or "sklearn_meta_study"

        self._study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
        )

        def optuna_objective(trial: optuna.Trial) -> float:
            self._current_trial = trial
            params = search_space.sample_optuna(trial)
            return objective(params)

        # Convert callbacks to Optuna format
        optuna_callbacks = None
        if callbacks:
            optuna_callbacks = [
                lambda study, trial: cb(
                    TrialResult(
                        params=trial.params,
                        value=trial.value if trial.value is not None else float("inf"),
                        trial_id=trial.number,
                        state=trial.state.name,
                    )
                )
                for cb in callbacks
            ]

        self._study.optimize(
            optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=optuna_callbacks,
            show_progress_bar=False,
        )

        # Convert Optuna trials to TrialResults
        trials = []
        for trial in self._study.trials:
            trials.append(
                TrialResult(
                    params=trial.params,
                    value=trial.value if trial.value is not None else float("inf"),
                    trial_id=trial.number,
                    duration=trial.duration.total_seconds() if trial.duration else 0.0,
                    user_attrs=dict(trial.user_attrs),
                    state=trial.state.name,
                )
            )

        return OptimizationResult(
            best_params=self._study.best_params,
            best_value=self._study.best_value,
            trials=trials,
            n_trials=len(self._study.trials),
            study_name=study_name,
        )

    def suggest_params(self, search_space: SearchSpace) -> Dict[str, Any]:
        """
        Suggest a single set of parameters using ask-tell interface.

        Args:
            search_space: Search space to sample from.

        Returns:
            Dictionary of parameter values.
        """
        if self._study is None:
            self._study = optuna.create_study(
                direction=self.direction,
                sampler=self.sampler,
            )

        trial = self._study.ask()
        self._current_trial = trial
        return search_space.sample_optuna(trial)

    def tell(self, params: Dict[str, Any], value: float) -> None:
        """
        Report the result of evaluating parameters.

        Args:
            params: Parameters that were evaluated.
            value: Objective value.
        """
        if self._study is None or self._current_trial is None:
            raise RuntimeError("Must call suggest_params before tell")

        self._study.tell(self._current_trial, value)
        self._current_trial = None

    def report_intermediate(self, value: float, step: int) -> bool:
        """
        Report intermediate value for pruning.

        Args:
            value: Intermediate objective value.
            step: Step number (e.g., epoch).

        Returns:
            True if trial should be pruned.
        """
        if self._current_trial is None:
            return False

        self._current_trial.report(value, step)
        return self._current_trial.should_prune()

    def supports_pruning(self) -> bool:
        """Whether this backend supports trial pruning."""
        return self.pruner is not None

    def supports_parallel(self) -> bool:
        """Whether this backend supports parallel optimization."""
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get the current state for serialization."""
        if self._study is None:
            return {}

        return {
            "study_name": self._study.study_name,
            "direction": self.direction,
            "trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "state": t.state.name,
                }
                for t in self._study.trials
            ],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization."""
        if not state:
            return

        self._study = optuna.create_study(
            study_name=state.get("study_name", "sklearn_meta_study"),
            direction=state.get("direction", self.direction),
            sampler=self.sampler,
            pruner=self.pruner,
        )

        # Replay trials
        for trial_data in state.get("trials", []):
            if trial_data["state"] == "COMPLETE" and trial_data["value"] is not None:
                self._study.add_trial(
                    optuna.trial.create_trial(
                        params=trial_data["params"],
                        values=[trial_data["value"]],
                        distributions={},  # Will be inferred
                        state=optuna.trial.TrialState.COMPLETE,
                    )
                )

    @property
    def study(self) -> Optional[optuna.Study]:
        """Access the underlying Optuna study."""
        return self._study

    def get_param_importances(self) -> Dict[str, float]:
        """
        Get parameter importances using fANOVA.

        Returns:
            Dictionary mapping parameter names to importance scores.
        """
        if self._study is None or len(self._study.trials) < 2:
            return {}

        try:
            return optuna.importance.get_param_importances(self._study)
        except Exception:
            return {}

    def __repr__(self) -> str:
        n_trials = len(self._study.trials) if self._study else 0
        return f"OptunaBackend(direction={self.direction}, n_trials={n_trials})"
