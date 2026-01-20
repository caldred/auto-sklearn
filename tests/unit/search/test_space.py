"""Tests for SearchSpace."""

import pytest
import numpy as np

from auto_sklearn.search.space import SearchSpace
from auto_sklearn.search.parameter import (
    CategoricalParameter,
    ConditionalParameter,
    FloatParameter,
    IntParameter,
)


class TestSearchSpaceCreation:
    """Tests for SearchSpace creation and basic operations."""

    def test_empty_space(self):
        """Verify empty space has no parameters."""
        space = SearchSpace()

        assert len(space) == 0
        assert space.parameter_names == []

    def test_add_float(self):
        """Verify add_float adds parameter."""
        space = SearchSpace()
        space.add_float("learning_rate", 0.001, 0.1)

        assert len(space) == 1
        assert "learning_rate" in space
        assert isinstance(space.get_parameter("learning_rate"), FloatParameter)

    def test_add_float_with_options(self):
        """Verify add_float supports all options."""
        space = SearchSpace()
        space.add_float("lr", 0.001, 0.1, log=True, step=0.01)

        param = space.get_parameter("lr")
        assert param.log is True
        assert param.step == 0.01

    def test_add_int(self):
        """Verify add_int adds parameter."""
        space = SearchSpace()
        space.add_int("n_estimators", 10, 100)

        assert len(space) == 1
        assert "n_estimators" in space
        assert isinstance(space.get_parameter("n_estimators"), IntParameter)

    def test_add_int_with_options(self):
        """Verify add_int supports all options."""
        space = SearchSpace()
        space.add_int("n", 1, 1000, log=True, step=5)

        param = space.get_parameter("n")
        assert param.log is True
        assert param.step == 5

    def test_add_categorical(self):
        """Verify add_categorical adds parameter."""
        space = SearchSpace()
        space.add_categorical("solver", ["adam", "sgd"])

        assert len(space) == 1
        assert "solver" in space
        assert isinstance(space.get_parameter("solver"), CategoricalParameter)

    def test_add_parameter(self):
        """Verify add_parameter adds pre-constructed parameter."""
        space = SearchSpace()
        param = FloatParameter(name="x", low=0.0, high=1.0)
        space.add_parameter(param)

        assert len(space) == 1
        assert space.get_parameter("x") is param


class TestSearchSpaceChaining:
    """Tests for method chaining."""

    def test_add_float_returns_self(self):
        """Verify add_float returns self for chaining."""
        space = SearchSpace()
        result = space.add_float("x", 0.0, 1.0)

        assert result is space

    def test_add_int_returns_self(self):
        """Verify add_int returns self for chaining."""
        space = SearchSpace()
        result = space.add_int("n", 1, 10)

        assert result is space

    def test_add_categorical_returns_self(self):
        """Verify add_categorical returns self for chaining."""
        space = SearchSpace()
        result = space.add_categorical("x", ["a", "b"])

        assert result is space

    def test_chained_adds(self):
        """Verify methods can be chained."""
        space = (
            SearchSpace()
            .add_float("lr", 0.001, 0.1, log=True)
            .add_int("n", 10, 100)
            .add_categorical("solver", ["adam", "sgd"])
        )

        assert len(space) == 3


class TestSearchSpaceShorthand:
    """Tests for shorthand notation."""

    def test_add_from_shorthand_int(self):
        """Verify shorthand creates int parameter."""
        space = SearchSpace()
        space.add_from_shorthand(n=(10, 100))

        param = space.get_parameter("n")
        assert isinstance(param, IntParameter)
        assert param.low == 10
        assert param.high == 100

    def test_add_from_shorthand_float(self):
        """Verify shorthand creates float parameter."""
        space = SearchSpace()
        space.add_from_shorthand(x=(0.1, 1.0))

        param = space.get_parameter("x")
        assert isinstance(param, FloatParameter)

    def test_add_from_shorthand_log(self):
        """Verify shorthand supports log scale."""
        space = SearchSpace()
        space.add_from_shorthand(lr=(0.001, 0.1, "log"))

        param = space.get_parameter("lr")
        assert isinstance(param, FloatParameter)
        assert param.log is True

    def test_add_from_shorthand_categorical(self):
        """Verify shorthand creates categorical from list."""
        space = SearchSpace()
        space.add_from_shorthand(solver=["adam", "sgd"])

        param = space.get_parameter("solver")
        assert isinstance(param, CategoricalParameter)

    def test_add_from_shorthand_multiple(self):
        """Verify shorthand supports multiple parameters."""
        space = SearchSpace()
        space.add_from_shorthand(
            n=(10, 100),
            lr=(0.001, 0.1, "log"),
            solver=["adam", "sgd"],
        )

        assert len(space) == 3


class TestSearchSpaceConditional:
    """Tests for conditional parameters."""

    def test_add_conditional(self):
        """Verify add_conditional adds parameter."""
        space = SearchSpace()
        space.add_categorical("optimizer", ["adam", "sgd"])

        inner = FloatParameter(name="momentum", low=0.0, high=1.0)
        space.add_conditional(
            name="momentum",
            parent_name="optimizer",
            parent_value="sgd",
            parameter=inner,
        )

        assert len(space) == 2
        assert "momentum" in space


class TestSearchSpaceSampling:
    """Tests for sampling parameters."""

    def test_sample_optuna(self, mock_trial):
        """Verify Optuna sampling returns all parameters."""
        space = (
            SearchSpace()
            .add_float("lr", 0.001, 0.1)
            .add_int("n", 10, 100)
            .add_categorical("solver", ["adam", "sgd"])
        )

        params = space.sample_optuna(mock_trial)

        assert "lr" in params
        assert "n" in params
        assert "solver" in params

    def test_sample_optuna_values_in_range(self, mock_trial):
        """Verify sampled values are in valid ranges."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)
        space.add_int("n", 10, 100)

        params = space.sample_optuna(mock_trial)

        assert 0.0 <= params["x"] <= 1.0
        assert 10 <= params["n"] <= 100

    def test_sample_optuna_conditional_active(self, mock_trial):
        """Verify conditional parameter sampled when active."""
        space = SearchSpace()
        space.add_categorical("optimizer", ["adam", "sgd"])

        inner = FloatParameter(name="momentum", low=0.0, high=1.0)
        space.add_conditional(
            name="momentum",
            parent_name="optimizer",
            parent_value="sgd",
            parameter=inner,
        )

        # Force sgd selection
        mock_trial._params["optimizer"] = "sgd"
        params = space.sample_optuna(mock_trial)

        if params["optimizer"] == "sgd":
            assert "momentum" in params

    def test_sample_optuna_conditional_inactive(self, mock_trial):
        """Verify conditional parameter not sampled when inactive."""
        space = SearchSpace()
        space.add_categorical("optimizer", ["adam", "sgd"])

        inner = FloatParameter(name="momentum", low=0.0, high=1.0)
        space.add_conditional(
            name="momentum",
            parent_name="optimizer",
            parent_value="sgd",
            parameter=inner,
        )

        # Force adam selection
        mock_trial._params["optimizer"] = "adam"
        params = space.sample_optuna(mock_trial)

        if params["optimizer"] == "adam":
            assert "momentum" not in params


class TestSearchSpaceOperations:
    """Tests for space operations."""

    def test_get_parameter_exists(self):
        """Verify get_parameter returns parameter."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)

        param = space.get_parameter("x")

        assert param is not None
        assert param.name == "x"

    def test_get_parameter_not_exists(self):
        """Verify get_parameter returns None for missing."""
        space = SearchSpace()

        param = space.get_parameter("nonexistent")

        assert param is None

    def test_remove_parameter(self):
        """Verify remove_parameter removes parameter."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)
        space.add_float("y", 0.0, 1.0)

        space.remove_parameter("x")

        assert len(space) == 1
        assert "x" not in space
        assert "y" in space

    def test_remove_parameter_returns_self(self):
        """Verify remove_parameter returns self."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)

        result = space.remove_parameter("x")

        assert result is space

    def test_parameter_names(self):
        """Verify parameter_names returns all names."""
        space = (
            SearchSpace()
            .add_float("a", 0.0, 1.0)
            .add_int("b", 1, 10)
            .add_categorical("c", ["x", "y"])
        )

        names = space.parameter_names

        assert set(names) == {"a", "b", "c"}


class TestSearchSpaceCopy:
    """Tests for space copying."""

    def test_copy_creates_new_space(self):
        """Verify copy creates new space."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)

        copy = space.copy()

        assert copy is not space
        assert len(copy) == 1

    def test_copy_independent(self):
        """Verify copy is independent of original."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)

        copy = space.copy()
        copy.add_float("y", 0.0, 1.0)

        assert len(space) == 1
        assert len(copy) == 2


class TestSearchSpaceMerge:
    """Tests for space merging."""

    def test_merge_adds_params(self):
        """Verify merge adds parameters from other space."""
        space1 = SearchSpace()
        space1.add_float("x", 0.0, 1.0)

        space2 = SearchSpace()
        space2.add_float("y", 0.0, 1.0)

        space1.merge(space2)

        assert len(space1) == 2
        assert "x" in space1
        assert "y" in space1

    def test_merge_overrides_on_conflict(self):
        """Verify merge overrides on name conflict."""
        space1 = SearchSpace()
        space1.add_float("x", 0.0, 1.0)

        space2 = SearchSpace()
        space2.add_float("x", 0.0, 10.0)

        space1.merge(space2)

        param = space1.get_parameter("x")
        assert param.high == 10.0

    def test_merge_returns_self(self):
        """Verify merge returns self."""
        space1 = SearchSpace()
        space2 = SearchSpace()

        result = space1.merge(space2)

        assert result is space1


class TestSearchSpaceFromDict:
    """Tests for from_dict class method."""

    def test_from_dict_shorthand(self):
        """Verify from_dict parses shorthand."""
        config = {
            "n": (10, 100),
            "lr": (0.001, 0.1, "log"),
            "solver": ["adam", "sgd"],
        }

        space = SearchSpace.from_dict(config)

        assert len(space) == 3
        assert isinstance(space.get_parameter("n"), IntParameter)
        assert isinstance(space.get_parameter("lr"), FloatParameter)
        assert isinstance(space.get_parameter("solver"), CategoricalParameter)

    def test_from_dict_explicit(self):
        """Verify from_dict parses explicit dict format."""
        config = {
            "x": {"type": "float", "low": 0.001, "high": 1.0, "log": True},
            "n": {"type": "int", "low": 1, "high": 10},
            "solver": {"type": "categorical", "choices": ["a", "b"]},
        }

        space = SearchSpace.from_dict(config)

        assert len(space) == 3

        x_param = space.get_parameter("x")
        assert isinstance(x_param, FloatParameter)
        assert x_param.log is True


class TestSearchSpaceContains:
    """Tests for __contains__."""

    def test_contains_true(self):
        """Verify __contains__ returns True for existing param."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)

        assert "x" in space

    def test_contains_false(self):
        """Verify __contains__ returns False for missing param."""
        space = SearchSpace()

        assert "x" not in space


class TestSearchSpaceIteration:
    """Tests for iteration."""

    def test_iter_returns_parameters(self):
        """Verify iteration returns parameter objects."""
        space = (
            SearchSpace()
            .add_float("a", 0.0, 1.0)
            .add_int("b", 1, 10)
        )

        params = list(space)

        assert len(params) == 2
        assert all(hasattr(p, "name") for p in params)

    def test_iter_names(self):
        """Verify iteration covers all parameters."""
        space = (
            SearchSpace()
            .add_float("a", 0.0, 1.0)
            .add_int("b", 1, 10)
        )

        names = {p.name for p in space}

        assert names == {"a", "b"}


class TestSearchSpaceRepr:
    """Tests for repr."""

    def test_repr(self):
        """Verify repr is informative."""
        space = (
            SearchSpace()
            .add_float("lr", 0.001, 0.1)
            .add_int("n", 10, 100)
        )

        repr_str = repr(space)

        assert "SearchSpace" in repr_str
        assert "lr" in repr_str
        assert "n" in repr_str
