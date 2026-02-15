"""Tests for SearchParameter classes."""

import pytest
import numpy as np

from sklearn_meta.search.parameter import (
    CategoricalParameter,
    ConditionalParameter,
    FloatParameter,
    IntParameter,
    SearchParameter,
    parse_shorthand,
)


class TestFloatParameter:
    """Tests for FloatParameter."""

    def test_basic_creation(self):
        """Verify basic float parameter creation."""
        param = FloatParameter(name="learning_rate", low=0.01, high=0.3)

        assert param.name == "learning_rate"
        assert param.low == 0.01
        assert param.high == 0.3
        assert param.log is False
        assert param.step is None

    def test_log_scale_creation(self):
        """Verify log scale parameter creation."""
        param = FloatParameter(name="lr", low=0.001, high=1.0, log=True)

        assert param.log is True

    def test_step_creation(self):
        """Verify step parameter creation."""
        param = FloatParameter(name="alpha", low=0.0, high=1.0, step=0.1)

        assert param.step == 0.1

    def test_low_greater_than_high_raises(self):
        """Verify low >= high raises error."""
        with pytest.raises(ValueError, match="must be less than"):
            FloatParameter(name="x", low=1.0, high=0.5)

    def test_low_equals_high_raises(self):
        """Verify low == high raises error."""
        with pytest.raises(ValueError, match="must be less than"):
            FloatParameter(name="x", low=1.0, high=1.0)

    def test_log_with_negative_low_raises(self):
        """Verify log scale with non-positive low raises error."""
        with pytest.raises(ValueError, match="positive"):
            FloatParameter(name="x", low=-0.1, high=1.0, log=True)

    def test_log_with_zero_low_raises(self):
        """Verify log scale with zero low raises error."""
        with pytest.raises(ValueError, match="positive"):
            FloatParameter(name="x", low=0.0, high=1.0, log=True)

    def test_sample_optuna(self, mock_trial):
        """Verify Optuna sampling returns value in range."""
        param = FloatParameter(name="x", low=0.0, high=1.0)

        value = param.sample_optuna(mock_trial)

        assert 0.0 <= value <= 1.0

    def test_sample_optuna_log(self, mock_trial):
        """Verify Optuna sampling works with log scale."""
        param = FloatParameter(name="x", low=0.001, high=1.0, log=True)

        value = param.sample_optuna(mock_trial)

        assert 0.001 <= value <= 1.0

    def test_repr(self):
        """Verify repr is informative."""
        param = FloatParameter(name="lr", low=0.001, high=0.1, log=True)

        repr_str = repr(param)

        assert "Float" in repr_str
        assert "lr" in repr_str
        assert "0.001" in repr_str
        assert "0.1" in repr_str
        assert "log" in repr_str


class TestIntParameter:
    """Tests for IntParameter."""

    def test_basic_creation(self):
        """Verify basic int parameter creation."""
        param = IntParameter(name="n_estimators", low=10, high=100)

        assert param.name == "n_estimators"
        assert param.low == 10
        assert param.high == 100
        assert param.log is False
        assert param.step == 1

    def test_log_scale_creation(self):
        """Verify log scale parameter creation."""
        param = IntParameter(name="n", low=1, high=1000, log=True)

        assert param.log is True

    def test_step_creation(self):
        """Verify step parameter creation."""
        param = IntParameter(name="n", low=10, high=100, step=5)

        assert param.step == 5

    def test_low_greater_than_high_raises(self):
        """Verify low >= high raises error."""
        with pytest.raises(ValueError, match="must be less than"):
            IntParameter(name="x", low=100, high=10)

    def test_log_with_non_positive_low_raises(self):
        """Verify log scale with non-positive low raises error."""
        with pytest.raises(ValueError, match="positive"):
            IntParameter(name="x", low=0, high=10, log=True)

    def test_sample_optuna(self, mock_trial):
        """Verify Optuna sampling returns int in range."""
        param = IntParameter(name="x", low=10, high=100)

        value = param.sample_optuna(mock_trial)

        assert isinstance(value, (int, np.integer))
        assert 10 <= value <= 100

    def test_repr(self):
        """Verify repr is informative."""
        param = IntParameter(name="n", low=10, high=100, step=5)

        repr_str = repr(param)

        assert "Int" in repr_str
        assert "n" in repr_str
        assert "10" in repr_str
        assert "100" in repr_str
        assert "step=5" in repr_str


class TestCategoricalParameter:
    """Tests for CategoricalParameter."""

    def test_basic_creation(self):
        """Verify basic categorical parameter creation."""
        param = CategoricalParameter(name="solver", choices=["adam", "sgd", "rmsprop"])

        assert param.name == "solver"
        assert param.choices == ["adam", "sgd", "rmsprop"]

    def test_empty_choices_raises(self):
        """Verify empty choices raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            CategoricalParameter(name="x", choices=[])

    def test_sample_optuna(self, mock_trial):
        """Verify Optuna sampling returns valid choice."""
        choices = ["a", "b", "c"]
        param = CategoricalParameter(name="x", choices=choices)

        value = param.sample_optuna(mock_trial)

        assert value in choices

    def test_repr(self):
        """Verify repr is informative."""
        param = CategoricalParameter(name="solver", choices=["a", "b"])

        repr_str = repr(param)

        assert "Cat" in repr_str
        assert "solver" in repr_str
        assert "a" in repr_str
        assert "b" in repr_str

    def test_repr_truncates_long_choices(self):
        """Verify repr truncates long choice lists."""
        param = CategoricalParameter(name="x", choices=["a", "b", "c", "d", "e"])

        repr_str = repr(param)

        assert "..." in repr_str


class TestConditionalParameter:
    """Tests for ConditionalParameter."""

    def test_basic_creation(self):
        """Verify basic conditional parameter creation."""
        inner = FloatParameter(name="momentum", low=0.0, high=1.0)
        param = ConditionalParameter(
            name="momentum",
            parent_name="optimizer",
            parent_value="sgd",
            parameter=inner,
        )

        assert param.name == "momentum"
        assert param.parent_name == "optimizer"
        assert param.parent_value == "sgd"
        assert param.parameter is inner

    def test_sample_optuna(self, mock_trial):
        """Verify Optuna sampling delegates to inner parameter."""
        inner = FloatParameter(name="momentum", low=0.0, high=1.0)
        param = ConditionalParameter(
            name="momentum",
            parent_name="optimizer",
            parent_value="sgd",
            parameter=inner,
        )

        value = param.sample_optuna(mock_trial)

        assert 0.0 <= value <= 1.0

    def test_repr(self):
        """Verify repr is informative."""
        inner = FloatParameter(name="momentum", low=0.0, high=1.0)
        param = ConditionalParameter(
            name="momentum",
            parent_name="optimizer",
            parent_value="sgd",
            parameter=inner,
        )

        repr_str = repr(param)

        assert "Conditional" in repr_str
        assert "momentum" in repr_str
        assert "optimizer" in repr_str
        assert "sgd" in repr_str


class TestParseShorthand:
    """Tests for parse_shorthand function."""

    def test_parse_int_range(self):
        """Verify parsing integer range."""
        param = parse_shorthand("n", (10, 100))

        assert isinstance(param, IntParameter)
        assert param.low == 10
        assert param.high == 100

    def test_parse_float_range(self):
        """Verify parsing float range."""
        param = parse_shorthand("x", (0.1, 1.0))

        assert isinstance(param, FloatParameter)
        assert param.low == 0.1
        assert param.high == 1.0

    def test_parse_float_log_scale(self):
        """Verify parsing float with log scale."""
        param = parse_shorthand("lr", (0.001, 0.1, "log"))

        assert isinstance(param, FloatParameter)
        assert param.log is True

    def test_parse_int_log_scale(self):
        """Verify parsing int with log scale."""
        param = parse_shorthand("n", (1, 1000, "log"))

        assert isinstance(param, IntParameter)
        assert param.log is True

    def test_parse_categorical_list(self):
        """Verify parsing categorical from list."""
        param = parse_shorthand("solver", ["adam", "sgd"])

        assert isinstance(param, CategoricalParameter)
        assert param.choices == ["adam", "sgd"]

    def test_parse_mixed_type_becomes_float(self):
        """Verify mixed int/float range becomes float."""
        param = parse_shorthand("x", (1, 10.0))

        assert isinstance(param, FloatParameter)

    def test_parse_too_short_tuple_raises(self):
        """Verify tuple with < 2 elements raises error."""
        with pytest.raises(ValueError, match="at least 2"):
            parse_shorthand("x", (1,))

    def test_parse_invalid_type_raises(self):
        """Verify invalid type raises error."""
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_shorthand("x", "invalid")


class TestSearchParameterBase:
    """Tests for SearchParameter abstract base class."""

    def test_parameter_has_name(self):
        """Verify all parameters have name attribute."""
        params = [
            FloatParameter(name="a", low=0.0, high=1.0),
            IntParameter(name="b", low=1, high=10),
            CategoricalParameter(name="c", choices=["x", "y"]),
        ]

        for param in params:
            assert hasattr(param, "name")
            assert isinstance(param.name, str)

    def test_parameter_has_sample_optuna(self):
        """Verify all parameters have sample_optuna method."""
        params = [
            FloatParameter(name="a", low=0.0, high=1.0),
            IntParameter(name="b", low=1, high=10),
            CategoricalParameter(name="c", choices=["x", "y"]),
        ]

        for param in params:
            assert hasattr(param, "sample_optuna")
            assert callable(param.sample_optuna)

