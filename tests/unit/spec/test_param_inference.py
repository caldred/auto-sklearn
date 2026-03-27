"""Tests for NodeBuilder.param() shorthand behavior."""

import pytest

from sklearn_meta.spec.builder import GraphBuilder
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.search.parameter import FloatParameter, IntParameter, CategoricalParameter


def _get_space(node_builder) -> SearchSpace:
    """Extract the SearchSpace from a NodeBuilder (returned by add_model/param/etc)."""
    return node_builder._search_space


class TestParamInfersInt:
    """Count-like integer ranges infer IntParameter."""

    def test_basic_int_bounds_become_int(self):
        b = GraphBuilder("t").add_model("m", object).param("x", 3, 10)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], IntParameter)
        assert space._parameters["x"].low == 3
        assert space._parameters["x"].high == 10

    def test_int_bounds_with_integer_step_become_int(self):
        b = GraphBuilder("t").add_model("m", object).param("x", 10, 100, step=5)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], IntParameter)
        assert space._parameters["x"].step == 5

    def test_int_bounds_with_integral_float_step_become_int(self):
        b = GraphBuilder("t").add_model("m", object).param("x", 3, 10, step=2.0)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], IntParameter)
        assert space._parameters["x"].step == 2

    def test_int_bounds_with_log_become_int(self):
        b = GraphBuilder("t").add_model("m", object).param("x", 3, 10, log=True)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], IntParameter)
        assert space._parameters["x"].log is True


class TestParamInfersFloat:
    """Continuous ranges stay FloatParameter."""

    def test_int_bounds_with_fractional_step_stay_float(self):
        b = GraphBuilder("t").add_model("m", object).param("x", 3, 10, step=0.5)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], FloatParameter)
        assert space._parameters["x"].step == 0.5

    def test_unit_interval_int_bounds_stay_float(self):
        b = GraphBuilder("t").add_model("m", object).param("x", 0, 1)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], FloatParameter)
        assert space._parameters["x"].low == 0.0
        assert space._parameters["x"].high == 1.0

    def test_float_bounds(self):
        b = GraphBuilder("t").add_model("m", object).param("x", 0.01, 0.3)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], FloatParameter)

    def test_mixed_bounds(self):
        b = GraphBuilder("t").add_model("m", object).param("x", 3, 10.0)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], FloatParameter)

    def test_float_with_log(self):
        b = GraphBuilder("t").add_model("m", object).param("x", 0.01, 0.3, log=True)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], FloatParameter)
        assert space._parameters["x"].log is True


class TestParamInfersCategorical:
    """When first arg is a list, .param() creates CategoricalParameter."""

    def test_list_of_strings(self):
        b = GraphBuilder("t").add_model("m", object).param("x", ["a", "b", "c"])
        space = _get_space(b)
        assert isinstance(space._parameters["x"], CategoricalParameter)
        assert space._parameters["x"].choices == ["a", "b", "c"]

    def test_list_rejects_high(self):
        with pytest.raises(TypeError, match="does not accept"):
            GraphBuilder("t").add_model("m", object).param("x", ["a", "b"], high=10)

    def test_list_rejects_log(self):
        with pytest.raises(TypeError, match="does not accept"):
            GraphBuilder("t").add_model("m", object).param("x", ["a", "b"], log=True)

    def test_list_rejects_step(self):
        with pytest.raises(TypeError, match="does not accept"):
            GraphBuilder("t").add_model("m", object).param("x", ["a", "b"], step=1)


class TestParamEdgeCases:
    """Edge cases for type inference."""

    def test_bool_bounds_are_not_int(self):
        b = GraphBuilder("t").add_model("m", object).param("x", False, True)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], FloatParameter)

    def test_existing_int_param_unchanged(self):
        b = GraphBuilder("t").add_model("m", object).int_param("x", 3, 10)
        space = _get_space(b)
        assert isinstance(space._parameters["x"], IntParameter)

    def test_existing_cat_param_unchanged(self):
        b = GraphBuilder("t").add_model("m", object).cat_param("x", ["a", "b"])
        space = _get_space(b)
        assert isinstance(space._parameters["x"], CategoricalParameter)
