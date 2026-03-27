"""Tests for SearchSpace."""

import pytest

from sklearn_meta.search.space import SearchSpace
from sklearn_meta.search.parameter import (
    CategoricalParameter,
    ConditionalParameter,
    FloatParameter,
    IntParameter,
)


class TestSearchSpaceCreation:
    """Tests for SearchSpace creation and basic operations."""

    def test_add_float_with_options(self):
        """Verify add_float supports all options."""
        space = SearchSpace()
        space.add_float("lr", 0.001, 0.1, log=True, step=0.01)

        param = space.get_parameter("lr")
        assert param.log is True
        assert param.step == 0.01

    def test_add_int_with_options(self):
        """Verify add_int supports all options."""
        space = SearchSpace()
        space.add_int("n", 1, 1000, log=True, step=5)

        param = space.get_parameter("n")
        assert param.log is True
        assert param.step == 5

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


class TestSearchSpaceSampling:
    """Tests for sampling parameters."""

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

class TestSearchSpaceCopy:
    """Tests for space copying."""

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

    def test_from_dict_conditional(self):
        """Verify from_dict parses conditional parameters."""
        config = {
            "optimizer": {"type": "categorical", "name": "optimizer", "choices": ["adam", "sgd"]},
            "momentum": {
                "type": "conditional",
                "name": "momentum",
                "parent_name": "optimizer",
                "parent_value": "sgd",
                "parameter": {
                    "type": "float",
                    "name": "momentum",
                    "low": 0.0,
                    "high": 1.0,
                    "log": False,
                    "step": None,
                },
            },
        }

        space = SearchSpace.from_dict(config)

        momentum = space.get_parameter("momentum")
        assert isinstance(momentum, ConditionalParameter)
        assert momentum.parent_name == "optimizer"


class TestSearchSpaceToDict:
    """Tests for to_dict serialization."""

    def test_to_dict_preserves_conditional_sampling_order(self):
        class _FixedTrial:
            def suggest_categorical(self, name, choices):
                return "sgd"

            def suggest_float(self, name, low, high, step=None, log=False):
                return 0.5

        space = SearchSpace()
        space.add_categorical("optimizer", ["adam", "sgd"])
        space.add_conditional(
            name="momentum",
            parent_name="optimizer",
            parent_value="sgd",
            parameter=FloatParameter(name="momentum", low=0.0, high=1.0),
        )

        restored = SearchSpace.from_dict(space.to_dict())
        params = restored.sample_optuna(_FixedTrial())

        assert params == {"optimizer": "sgd", "momentum": 0.5}


class TestSearchSpaceNarrowAround:
    """Tests for SearchSpace.narrow_around."""

    def test_float_linear_narrows_symmetrically(self):
        """Float param with linear scale narrows to factor-fraction of range around center."""
        space = SearchSpace().add_float("x", 0.0, 100.0)
        center = {"x": 50.0}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("x")
        # half_width = 100 * 0.5 / 2 = 25
        assert param.low == pytest.approx(25.0)
        assert param.high == pytest.approx(75.0)

    def test_float_log_narrows_by_factor(self):
        """Float param with log scale narrows by multiplying/dividing by (1+factor)."""
        space = SearchSpace().add_float("lr", 0.001, 1.0, log=True)
        center = {"lr": 0.1}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("lr")
        expected_low = 0.1 / 1.5
        expected_high = 0.1 * 1.5
        assert param.low == pytest.approx(expected_low)
        assert param.high == pytest.approx(expected_high)
        assert param.log is True

    def test_int_param_narrows(self):
        """Int param narrows similarly to float with integer bounds."""
        space = SearchSpace().add_int("depth", 1, 21)
        center = {"depth": 11}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("depth")
        # original_range = 20, half_width = int(20 * 0.5 / 2) = 5
        assert isinstance(param, IntParameter)
        assert param.low == 6
        assert param.high == 16

    def test_regularization_param_biases_lower(self):
        """Regularization params should have wider range below center than above."""
        space = SearchSpace().add_float("reg_lambda", 0.0, 10.0)
        center = {"reg_lambda": 5.0}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("reg_lambda")
        # half_width = 10 * 0.5 / 2 = 2.5
        # reg: new_low = max(0, 5 - 2.5*2) = 0.0, new_high = min(10, 5 + 2.5*0.5) = 6.25
        assert param.low == pytest.approx(0.0)
        assert param.high == pytest.approx(6.25)
        # Range below center is wider than range above center
        assert (center["reg_lambda"] - param.low) > (param.high - center["reg_lambda"])

    def test_regularization_log_scale_biases_lower(self):
        """Log-scale regularization params use regularization_bias for lower bound."""
        space = SearchSpace().add_float("reg_alpha", 0.001, 10.0, log=True)
        center = {"reg_alpha": 1.0}

        narrowed = space.narrow_around(center, factor=0.5, regularization_bias=0.25)

        param = narrowed.get_parameter("reg_alpha")
        # reg + log: new_low = max(0.001, 1.0 * 0.25) = 0.25
        #            new_high = min(10.0, 1.0 * 1.5) = 1.5
        assert param.low == pytest.approx(0.25)
        assert param.high == pytest.approx(1.5)

    def test_categorical_kept_as_is(self):
        """Categorical parameters should be passed through unchanged."""
        space = SearchSpace().add_categorical("solver", ["adam", "sgd", "lbfgs"])
        center = {"solver": "adam"}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("solver")
        assert isinstance(param, CategoricalParameter)
        assert param.choices == ["adam", "sgd", "lbfgs"]

    def test_collapsed_range_falls_back_to_original(self):
        """When narrowed bounds collapse (low >= high), fall back to original bounds."""
        space = SearchSpace().add_float("x", 0.0, 100.0)
        # Center at the boundary with a tiny factor so that rounding causes collapse
        # With center=100 and factor=0.5: new_low = max(0, 100-25) = 75
        # new_high = min(100, 100+25) = 100 -- this doesn't collapse.
        # Use center=0 and factor very small: new_low = max(0, 0-eps) = 0
        # new_high = min(100, 0+eps) = eps -- also doesn't collapse.
        # Force collapse: low=5.0, high=5.1, center at 5.0, factor=0.001
        space2 = SearchSpace().add_float("y", 5.0, 5.1)
        center = {"y": 5.0}
        # range=0.1, half_width=0.1*0.001/2 = 0.00005
        # new_low = max(5.0, 5.0 - 0.00005) = 5.0
        # new_high = min(5.1, 5.0 + 0.00005) = 5.00005
        # 5.0 < 5.00005 => doesn't collapse either.
        # Actually use: center at low bound exactly with log-scale to force collapse
        # Simpler: just test that when new_low >= new_high, we get original
        space3 = SearchSpace().add_float("z", 0.0, 100.0)
        # We need center_val - half_width >= center_val + half_width, impossible for positive half_width.
        # Let's use a regularization param with center near high bound:
        # reg: new_low = center - half_width*2, new_high = center + half_width*0.5
        # If center=99, range=100, factor=0.5, half_width=25
        # new_low = max(0, 99-50)=49, new_high = min(100, 99+12.5)=100 => valid.
        # If center=100 (at boundary): new_low=max(0,100-50)=50, new_high=min(100,100+12.5)=100 => valid.
        # For collapse: int param with very small range.
        space4 = SearchSpace().add_int("n", 10, 12)
        center4 = {"n": 10}
        # range=2, half_width=int(2*0.5/2)=0 -> or 1 (clamp)
        # half_width = int(1) or 1 = max(int(0.5), 1) ... code says: int(2*0.5/2) or 1 = int(0.5) or 1 = 0 or 1 = 1
        # new_low = max(10, 10-1)=10, new_high = min(12, 10+1)=11 => valid (10<11)

        # The most reliable way: use a reg int param with center at lower bound.
        space5 = SearchSpace().add_int("reg_lambda", 1, 3)
        center5 = {"reg_lambda": 1}
        # range=2, half_width=int(2*0.5/2) or 1 = 0 or 1 = 1
        # reg: new_low = max(1, 1-2)=1, new_high = min(3, 1+0)=1
        # 1 >= 1 => collapse! Falls back to 1, 3
        narrowed = space5.narrow_around(center5, factor=0.5)
        param = narrowed.get_parameter("reg_lambda")
        assert param.low == 1
        assert param.high == 3

    def test_missing_center_keeps_original(self):
        """Params not in center dict should keep original bounds."""
        space = (
            SearchSpace()
            .add_float("x", 0.0, 100.0)
            .add_float("y", 0.0, 50.0)
        )
        center = {"x": 50.0}  # y not in center

        narrowed = space.narrow_around(center, factor=0.5)

        y_param = narrowed.get_parameter("y")
        assert y_param.low == pytest.approx(0.0)
        assert y_param.high == pytest.approx(50.0)

    def test_center_at_low_boundary(self):
        """Center at the low boundary should produce valid narrowed range."""
        space = SearchSpace().add_float("x", 0.0, 100.0)
        center = {"x": 0.0}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("x")
        # half_width = 25, new_low = max(0, 0-25) = 0, new_high = min(100, 0+25) = 25
        assert param.low == pytest.approx(0.0)
        assert param.high == pytest.approx(25.0)
        assert param.low < param.high

    def test_center_at_high_boundary(self):
        """Center at the high boundary should produce valid narrowed range."""
        space = SearchSpace().add_float("x", 0.0, 100.0)
        center = {"x": 100.0}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("x")
        # half_width = 25, new_low = max(0, 100-25) = 75, new_high = min(100, 100+25) = 100
        assert param.low == pytest.approx(75.0)
        assert param.high == pytest.approx(100.0)
        assert param.low < param.high

    def test_returns_new_space(self):
        """narrow_around should return a new SearchSpace, not mutate the original."""
        space = SearchSpace().add_float("x", 0.0, 100.0)
        center = {"x": 50.0}

        narrowed = space.narrow_around(center, factor=0.5)

        assert narrowed is not space
        # Original unchanged
        orig = space.get_parameter("x")
        assert orig.low == pytest.approx(0.0)
        assert orig.high == pytest.approx(100.0)

    def test_preserves_float_step(self):
        """Narrowing should preserve the step attribute on float params."""
        space = SearchSpace().add_float("x", 0.0, 100.0, step=0.1)
        center = {"x": 50.0}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("x")
        assert param.step == pytest.approx(0.1)

    def test_preserves_int_step_and_log(self):
        """Narrowing should preserve step and log attributes on int params."""
        space = SearchSpace().add_int("n", 10, 1000, log=True, step=5)
        center = {"n": 100}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("n")
        assert param.log is True
        assert param.step == 5

    def test_custom_regularization_params(self):
        """Custom regularization_params list should be used instead of defaults."""
        space = SearchSpace().add_float("my_reg", 0.0, 10.0)
        center = {"my_reg": 5.0}

        narrowed_default = space.narrow_around(center, factor=0.5)
        narrowed_custom = space.narrow_around(
            center, factor=0.5, regularization_params=["my_reg"]
        )

        default_param = narrowed_default.get_parameter("my_reg")
        custom_param = narrowed_custom.get_parameter("my_reg")

        # Default treats "my_reg" as a normal param (symmetric narrowing)
        # half_width = 10 * 0.5 / 2 = 2.5
        # new_low = max(0, 5 - 2.5) = 2.5, new_high = min(10, 5 + 2.5) = 7.5
        assert default_param.low == pytest.approx(2.5)
        assert default_param.high == pytest.approx(7.5)

        # Custom treats "my_reg" as regularization (biased lower)
        # half_width = 2.5, new_low = max(0, 5-5) = 0, new_high = min(10, 5+1.25) = 6.25
        assert custom_param.low == pytest.approx(0.0)
        assert custom_param.high == pytest.approx(6.25)

    def test_log_scale_clamped_to_original_bounds(self):
        """Log-scale narrowing should not exceed original bounds."""
        space = SearchSpace().add_float("lr", 0.01, 0.1, log=True)
        center = {"lr": 0.09}

        narrowed = space.narrow_around(center, factor=0.5)

        param = narrowed.get_parameter("lr")
        # new_high = min(0.1, 0.09 * 1.5) = min(0.1, 0.135) = 0.1
        assert param.high == pytest.approx(0.1)
        assert param.low >= 0.01


class TestSearchSpaceRoundTripValues:
    """Tests that serialization round-trip preserves attribute values."""

    def test_float_parameter_values_preserved(self):
        """FloatParameter low, high, log, step all survive round-trip."""
        space = SearchSpace().add_float("lr", 0.001, 0.3, log=True, step=0.001)

        restored = SearchSpace.from_dict(space.to_dict())

        param = restored.get_parameter("lr")
        assert isinstance(param, FloatParameter)
        assert param.low == pytest.approx(0.001)
        assert param.high == pytest.approx(0.3)
        assert param.log is True
        assert param.step == pytest.approx(0.001)

    def test_float_parameter_defaults_preserved(self):
        """FloatParameter with default log=False, step=None survives round-trip."""
        space = SearchSpace().add_float("x", -5.0, 5.0)

        restored = SearchSpace.from_dict(space.to_dict())

        param = restored.get_parameter("x")
        assert param.log is False
        assert param.step is None

    def test_int_parameter_values_preserved(self):
        """IntParameter low, high, log, step all survive round-trip."""
        space = SearchSpace().add_int("n", 10, 1000, log=True, step=5)

        restored = SearchSpace.from_dict(space.to_dict())

        param = restored.get_parameter("n")
        assert isinstance(param, IntParameter)
        assert param.low == 10
        assert param.high == 1000
        assert param.log is True
        assert param.step == 5

    def test_int_parameter_defaults_preserved(self):
        """IntParameter with default log=False, step=1 survives round-trip."""
        space = SearchSpace().add_int("depth", 3, 10)

        restored = SearchSpace.from_dict(space.to_dict())

        param = restored.get_parameter("depth")
        assert param.log is False
        assert param.step == 1

    def test_categorical_choices_preserved(self):
        """CategoricalParameter choices survive round-trip including order."""
        choices = ["lbfgs", "adam", "sgd", "rmsprop"]
        space = SearchSpace().add_categorical("solver", choices)

        restored = SearchSpace.from_dict(space.to_dict())

        param = restored.get_parameter("solver")
        assert isinstance(param, CategoricalParameter)
        assert param.choices == choices

    def test_categorical_mixed_types_preserved(self):
        """CategoricalParameter with mixed types survives round-trip."""
        choices = [1, "two", 3.0, True]
        space = SearchSpace().add_categorical("mixed", choices)

        restored = SearchSpace.from_dict(space.to_dict())

        param = restored.get_parameter("mixed")
        assert param.choices == choices

    def test_conditional_parameter_preserved(self):
        """ConditionalParameter parent_name, parent_value, inner param all survive round-trip."""
        space = SearchSpace()
        space.add_categorical("optimizer", ["adam", "sgd"])
        inner = FloatParameter(name="momentum", low=0.5, high=0.99, log=False, step=0.01)
        space.add_conditional(
            name="momentum",
            parent_name="optimizer",
            parent_value="sgd",
            parameter=inner,
        )

        restored = SearchSpace.from_dict(space.to_dict())

        param = restored.get_parameter("momentum")
        assert isinstance(param, ConditionalParameter)
        assert param.parent_name == "optimizer"
        assert param.parent_value == "sgd"
        assert isinstance(param.parameter, FloatParameter)
        assert param.parameter.low == pytest.approx(0.5)
        assert param.parameter.high == pytest.approx(0.99)
        assert param.parameter.step == pytest.approx(0.01)


class TestSearchSpaceDuplicateParams:
    """Tests that adding a param with the same name overwrites the previous one."""

    def test_same_type_overwrite(self):
        """Adding float with same name overwrites previous float."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)
        space.add_float("x", 0.0, 10.0)

        assert len(space) == 1
        param = space.get_parameter("x")
        assert isinstance(param, FloatParameter)
        assert param.high == pytest.approx(10.0)

    def test_different_type_overwrite(self):
        """Adding int with same name as existing float replaces with IntParameter."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)
        space.add_int("x", 0, 10)

        assert len(space) == 1
        param = space.get_parameter("x")
        assert isinstance(param, IntParameter)
        assert param.high == 10

    def test_categorical_overwrites_numeric(self):
        """Adding categorical with same name as existing numeric replaces it."""
        space = SearchSpace()
        space.add_int("x", 1, 10)
        space.add_categorical("x", ["a", "b"])

        assert len(space) == 1
        param = space.get_parameter("x")
        assert isinstance(param, CategoricalParameter)
        assert param.choices == ["a", "b"]

    def test_overwrite_via_add_parameter(self):
        """add_parameter also overwrites by name."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)
        space.add_parameter(IntParameter(name="x", low=5, high=50))

        assert len(space) == 1
        param = space.get_parameter("x")
        assert isinstance(param, IntParameter)
        assert param.low == 5

    def test_overwrite_preserves_other_params(self):
        """Overwriting one param does not affect other params."""
        space = SearchSpace()
        space.add_float("x", 0.0, 1.0)
        space.add_float("y", 0.0, 5.0)
        space.add_float("x", 0.0, 99.0)

        assert len(space) == 2
        assert space.get_parameter("y").high == pytest.approx(5.0)
        assert space.get_parameter("x").high == pytest.approx(99.0)


class TestSearchSpaceBoundaryValues:
    """Tests for parameter construction at boundary values."""

    def test_float_low_equals_high_raises(self):
        """FloatParameter with low==high should raise ValueError."""
        with pytest.raises(ValueError, match="low.*must be less than.*high"):
            FloatParameter(name="x", low=1.0, high=1.0)

    def test_float_low_greater_than_high_raises(self):
        """FloatParameter with low>high should raise ValueError."""
        with pytest.raises(ValueError):
            FloatParameter(name="x", low=5.0, high=1.0)

    def test_int_low_equals_high_raises(self):
        """IntParameter with low==high should raise ValueError."""
        with pytest.raises(ValueError, match="low.*must be less than.*high"):
            IntParameter(name="n", low=5, high=5)

    def test_int_low_greater_than_high_raises(self):
        """IntParameter with low>high should raise ValueError."""
        with pytest.raises(ValueError):
            IntParameter(name="n", low=10, high=1)

    def test_float_log_nonpositive_low_raises(self):
        """Float log scale with low<=0 should raise ValueError."""
        with pytest.raises(ValueError, match="log scale requires positive"):
            FloatParameter(name="lr", low=0.0, high=1.0, log=True)

    def test_float_log_negative_low_raises(self):
        """Float log scale with negative low should raise ValueError."""
        with pytest.raises(ValueError, match="log scale requires positive"):
            FloatParameter(name="lr", low=-0.1, high=1.0, log=True)

    def test_int_log_nonpositive_low_raises(self):
        """Int log scale with low<=0 should raise ValueError."""
        with pytest.raises(ValueError, match="log scale requires positive"):
            IntParameter(name="n", low=0, high=100, log=True)

    def test_categorical_empty_choices_raises(self):
        """CategoricalParameter with empty choices should raise ValueError."""
        with pytest.raises(ValueError, match="choices cannot be empty"):
            CategoricalParameter(name="x", choices=[])

    def test_add_float_with_invalid_bounds_raises(self):
        """SearchSpace.add_float with low>=high should raise ValueError."""
        space = SearchSpace()
        with pytest.raises(ValueError):
            space.add_float("x", 5.0, 5.0)

    def test_add_int_with_invalid_bounds_raises(self):
        """SearchSpace.add_int with low>=high should raise ValueError."""
        space = SearchSpace()
        with pytest.raises(ValueError):
            space.add_int("n", 10, 10)
