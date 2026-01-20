"""SearchSpace: Backend-agnostic hyperparameter search space."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from sklearn_meta.search.parameter import (
    CategoricalParameter,
    ConditionalParameter,
    FloatParameter,
    IntParameter,
    SearchParameter,
    parse_shorthand,
)


class SearchSpace:
    """
    Backend-agnostic hyperparameter search space.

    This class provides a unified interface for defining search spaces
    that can be converted to different optimization backends (Optuna, Hyperopt, etc.).

    Example:
        space = SearchSpace()
        space.add_float("learning_rate", 0.001, 0.1, log=True)
        space.add_int("max_depth", 3, 10)
        space.add_categorical("booster", ["gbtree", "dart"])

        # Sample using Optuna
        params = space.sample_optuna(trial)

        # Or convert to Hyperopt space
        hp_space = space.to_hyperopt()
    """

    def __init__(self) -> None:
        """Initialize an empty search space."""
        self._parameters: Dict[str, SearchParameter] = {}

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
        step: Optional[float] = None,
    ) -> SearchSpace:
        """
        Add a floating point parameter.

        Args:
            name: Parameter name.
            low: Lower bound.
            high: Upper bound.
            log: Whether to sample in log space.
            step: Optional step size for discrete sampling.

        Returns:
            Self for chaining.
        """
        self._parameters[name] = FloatParameter(
            name=name, low=low, high=high, log=log, step=step
        )
        return self

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        log: bool = False,
        step: int = 1,
    ) -> SearchSpace:
        """
        Add an integer parameter.

        Args:
            name: Parameter name.
            low: Lower bound (inclusive).
            high: Upper bound (inclusive).
            log: Whether to sample in log space.
            step: Step size.

        Returns:
            Self for chaining.
        """
        self._parameters[name] = IntParameter(
            name=name, low=low, high=high, log=log, step=step
        )
        return self

    def add_categorical(self, name: str, choices: List[Any]) -> SearchSpace:
        """
        Add a categorical parameter.

        Args:
            name: Parameter name.
            choices: List of possible values.

        Returns:
            Self for chaining.
        """
        self._parameters[name] = CategoricalParameter(name=name, choices=choices)
        return self

    def add_conditional(
        self,
        name: str,
        parent_name: str,
        parent_value: Any,
        parameter: SearchParameter,
    ) -> SearchSpace:
        """
        Add a conditional parameter.

        Args:
            name: Parameter name.
            parent_name: Name of the parent parameter.
            parent_value: Value of parent that activates this parameter.
            parameter: The parameter to use when active.

        Returns:
            Self for chaining.
        """
        self._parameters[name] = ConditionalParameter(
            name=name,
            parent_name=parent_name,
            parent_value=parent_value,
            parameter=parameter,
        )
        return self

    def add_parameter(self, param: SearchParameter) -> SearchSpace:
        """
        Add a pre-constructed parameter.

        Args:
            param: The parameter to add.

        Returns:
            Self for chaining.
        """
        self._parameters[param.name] = param
        return self

    def add_from_shorthand(self, **kwargs) -> SearchSpace:
        """
        Add parameters using shorthand notation.

        Shorthand formats:
        - (low, high): Float or Int range (inferred from types)
        - (low, high, "log"): Float/Int with log scale
        - [a, b, c]: Categorical choices

        Example:
            space.add_from_shorthand(
                max_depth=(3, 10),
                learning_rate=(0.01, 0.3, "log"),
                booster=["gbtree", "dart"],
            )

        Returns:
            Self for chaining.
        """
        for name, value in kwargs.items():
            param = parse_shorthand(name, value)
            self._parameters[name] = param
        return self

    def sample_optuna(self, trial) -> Dict[str, Any]:
        """
        Sample all parameters using an Optuna trial.

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of parameter name to sampled value.
        """
        params = {}
        for name, param in self._parameters.items():
            if isinstance(param, ConditionalParameter):
                # Check if parent condition is met
                if param.parent_name in params:
                    if params[param.parent_name] == param.parent_value:
                        params[name] = param.sample_optuna(trial)
            else:
                params[name] = param.sample_optuna(trial)
        return params

    def to_hyperopt(self) -> Dict[str, Any]:
        """
        Convert to Hyperopt search space.

        Returns:
            Dictionary suitable for Hyperopt's fmin.
        """
        space = {}
        for name, param in self._parameters.items():
            if not isinstance(param, ConditionalParameter):
                space[name] = param.to_hyperopt()
        # Note: Conditional parameters need special handling in Hyperopt
        return space

    def get_parameter(self, name: str) -> Optional[SearchParameter]:
        """Get a parameter by name."""
        return self._parameters.get(name)

    def remove_parameter(self, name: str) -> SearchSpace:
        """Remove a parameter by name."""
        if name in self._parameters:
            del self._parameters[name]
        return self

    @property
    def parameter_names(self) -> List[str]:
        """List of all parameter names."""
        return list(self._parameters.keys())

    def __len__(self) -> int:
        """Number of parameters."""
        return len(self._parameters)

    def __contains__(self, name: str) -> bool:
        """Check if parameter exists."""
        return name in self._parameters

    def __iter__(self) -> Iterator[SearchParameter]:
        """Iterate over parameters."""
        return iter(self._parameters.values())

    def __repr__(self) -> str:
        params_str = ", ".join(repr(p) for p in self._parameters.values())
        return f"SearchSpace([{params_str}])"

    def copy(self) -> SearchSpace:
        """Create a copy of this search space."""
        new_space = SearchSpace()
        for param in self._parameters.values():
            new_space.add_parameter(param)
        return new_space

    def merge(self, other: SearchSpace) -> SearchSpace:
        """
        Merge with another search space.

        Parameters from the other space override this one on conflict.

        Args:
            other: Search space to merge in.

        Returns:
            Self for chaining.
        """
        for param in other:
            self._parameters[param.name] = param
        return self

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> SearchSpace:
        """
        Create a search space from a dictionary.

        Dictionary format:
        {
            "param_name": {
                "type": "float"|"int"|"categorical",
                "low": ...,
                "high": ...,
                "log": ...,
                "choices": [...],
            }
        }

        Args:
            config: Dictionary configuration.

        Returns:
            Configured SearchSpace.
        """
        space = cls()
        for name, spec in config.items():
            if isinstance(spec, (tuple, list)):
                space.add_parameter(parse_shorthand(name, spec))
            elif isinstance(spec, dict):
                param_type = spec.get("type", "float")
                if param_type == "float":
                    space.add_float(
                        name,
                        spec["low"],
                        spec["high"],
                        log=spec.get("log", False),
                        step=spec.get("step"),
                    )
                elif param_type == "int":
                    space.add_int(
                        name,
                        spec["low"],
                        spec["high"],
                        log=spec.get("log", False),
                        step=spec.get("step", 1),
                    )
                elif param_type == "categorical":
                    space.add_categorical(name, spec["choices"])
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
        return space
