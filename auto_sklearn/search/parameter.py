"""SearchParameter: Backend-agnostic hyperparameter definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union


class SearchParameter(ABC):
    """Base class for search parameters."""

    def __init__(self, name: str) -> None:
        """
        Initialize a search parameter.

        Args:
            name: Parameter name.
        """
        self.name = name

    @abstractmethod
    def sample_optuna(self, trial) -> Any:
        """Sample a value using Optuna trial."""
        pass

    @abstractmethod
    def to_hyperopt(self) -> Any:
        """Convert to Hyperopt search space."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


@dataclass
class FloatParameter(SearchParameter):
    """
    Floating point parameter.

    Attributes:
        name: Parameter name.
        low: Lower bound.
        high: Upper bound.
        log: Whether to sample in log space.
        step: Optional step size for discrete sampling.
    """

    name: str
    low: float
    high: float
    log: bool = False
    step: Optional[float] = None

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        if self.log and self.low <= 0:
            raise ValueError(f"log scale requires positive low bound, got {self.low}")

    def sample_optuna(self, trial) -> float:
        """Sample using Optuna trial."""
        if self.step is not None:
            return trial.suggest_float(
                self.name, self.low, self.high, step=self.step, log=self.log
            )
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)

    def to_hyperopt(self) -> Any:
        """Convert to Hyperopt space."""
        from hyperopt import hp

        if self.log:
            return hp.loguniform(self.name, np.log(self.low), np.log(self.high))
        if self.step is not None:
            return hp.quniform(self.name, self.low, self.high, self.step)
        return hp.uniform(self.name, self.low, self.high)

    def __repr__(self) -> str:
        log_str = ", log" if self.log else ""
        step_str = f", step={self.step}" if self.step else ""
        return f"Float({self.name}: [{self.low}, {self.high}]{log_str}{step_str})"


@dataclass
class IntParameter(SearchParameter):
    """
    Integer parameter.

    Attributes:
        name: Parameter name.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        log: Whether to sample in log space.
        step: Optional step size.
    """

    name: str
    low: int
    high: int
    log: bool = False
    step: int = 1

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        if self.log and self.low <= 0:
            raise ValueError(f"log scale requires positive low bound, got {self.low}")

    def sample_optuna(self, trial) -> int:
        """Sample using Optuna trial."""
        return trial.suggest_int(
            self.name, self.low, self.high, step=self.step, log=self.log
        )

    def to_hyperopt(self) -> Any:
        """Convert to Hyperopt space."""
        from hyperopt import hp
        import numpy as np

        if self.log:
            # Hyperopt doesn't have direct log int, so use qloguniform
            return hp.qloguniform(self.name, np.log(self.low), np.log(self.high), self.step)
        return hp.quniform(self.name, self.low, self.high, self.step)

    def __repr__(self) -> str:
        log_str = ", log" if self.log else ""
        step_str = f", step={self.step}" if self.step > 1 else ""
        return f"Int({self.name}: [{self.low}, {self.high}]{log_str}{step_str})"


@dataclass
class CategoricalParameter(SearchParameter):
    """
    Categorical parameter.

    Attributes:
        name: Parameter name.
        choices: List of possible values.
    """

    name: str
    choices: List[Any]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("choices cannot be empty")

    def sample_optuna(self, trial) -> Any:
        """Sample using Optuna trial."""
        return trial.suggest_categorical(self.name, self.choices)

    def to_hyperopt(self) -> Any:
        """Convert to Hyperopt space."""
        from hyperopt import hp

        return hp.choice(self.name, self.choices)

    def __repr__(self) -> str:
        choices_str = ", ".join(str(c) for c in self.choices[:3])
        if len(self.choices) > 3:
            choices_str += ", ..."
        return f"Cat({self.name}: [{choices_str}])"


@dataclass
class ConditionalParameter(SearchParameter):
    """
    Conditional parameter that depends on another parameter's value.

    Attributes:
        name: Parameter name.
        parent_name: Name of the parent parameter.
        parent_value: Value of parent that activates this parameter.
        parameter: The actual parameter to use when active.
    """

    name: str
    parent_name: str
    parent_value: Any
    parameter: SearchParameter

    def sample_optuna(self, trial) -> Optional[Any]:
        """
        Sample using Optuna trial.

        Note: Conditional sampling must be handled by the caller.
        This method assumes the condition is met.
        """
        return self.parameter.sample_optuna(trial)

    def to_hyperopt(self) -> Any:
        """
        Convert to Hyperopt space.

        Note: Hyperopt handles conditionals differently.
        This returns the inner parameter's space.
        """
        return self.parameter.to_hyperopt()

    def __repr__(self) -> str:
        return f"Conditional({self.name} if {self.parent_name}={self.parent_value}: {self.parameter})"


def parse_shorthand(
    name: str, value: Union[Tuple, List]
) -> SearchParameter:
    """
    Parse shorthand parameter notation.

    Shorthand formats:
    - (low, high): Float or Int range (inferred from types)
    - (low, high, "log"): Float/Int with log scale
    - [a, b, c]: Categorical choices

    Args:
        name: Parameter name.
        value: Shorthand value.

    Returns:
        Appropriate SearchParameter instance.
    """
    if isinstance(value, list):
        return CategoricalParameter(name=name, choices=value)

    if isinstance(value, tuple):
        if len(value) < 2:
            raise ValueError(f"Tuple must have at least 2 elements: {value}")

        low, high = value[0], value[1]
        log = len(value) > 2 and value[2] == "log"

        if isinstance(low, int) and isinstance(high, int):
            return IntParameter(name=name, low=low, high=high, log=log)
        else:
            return FloatParameter(
                name=name, low=float(low), high=float(high), log=log
            )

    raise ValueError(f"Cannot parse shorthand value: {value}")
