"""Tests for hyperparameter reparameterization."""

import pytest
import numpy as np

from auto_sklearn.meta.reparameterization import (
    LinearReparameterization,
    LogProductReparameterization,
    RatioReparameterization,
    ReparameterizationResult,
    ReparameterizedSpace,
)
from auto_sklearn.search.space import SearchSpace


class TestLogProductReparameterization:
    """Tests for LogProductReparameterization."""

    def test_forward_computes_log_product(self):
        """Verify forward computes log(p1) + log(p2)."""
        reparam = LogProductReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        original = {"a": 0.1, "b": 100}
        transformed = reparam.forward(original)

        expected_product = np.log(0.1) + np.log(100)  # log(10) ≈ 2.303
        assert transformed[reparam._product_name] == pytest.approx(expected_product, rel=1e-5)

    def test_forward_computes_log_ratio(self):
        """Verify forward computes log(p1) - log(p2)."""
        reparam = LogProductReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        original = {"a": 0.1, "b": 100}
        transformed = reparam.forward(original)

        expected_ratio = np.log(0.1) - np.log(100)  # log(0.001) ≈ -6.908
        assert transformed[reparam._ratio_name] == pytest.approx(expected_ratio, rel=1e-5)

    def test_inverse_recovers_original(self):
        """Verify inverse(forward(x)) ≈ x."""
        reparam = LogProductReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        original = {"a": 0.05, "b": 200}
        transformed = reparam.forward(original)
        recovered = reparam.inverse(transformed)

        assert recovered["a"] == pytest.approx(original["a"], rel=1e-5)
        assert recovered["b"] == pytest.approx(original["b"], rel=1e-5)

    def test_forward_inverse_roundtrip(self):
        """Verify forward(inverse(x)) ≈ x."""
        reparam = LogProductReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        # Start with transformed values
        transformed = {reparam._product_name: 5.0, reparam._ratio_name: -2.0}
        original = reparam.inverse(transformed)
        recovered = reparam.forward(original)

        assert recovered[reparam._product_name] == pytest.approx(transformed[reparam._product_name], rel=1e-5)
        assert recovered[reparam._ratio_name] == pytest.approx(transformed[reparam._ratio_name], rel=1e-5)

    def test_preserves_product(self):
        """Verify product a*b is preserved through roundtrip."""
        reparam = LogProductReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        original = {"a": 0.1, "b": 100}
        original_product = original["a"] * original["b"]

        transformed = reparam.forward(original)
        recovered = reparam.inverse(transformed)
        recovered_product = recovered["a"] * recovered["b"]

        assert recovered_product == pytest.approx(original_product, rel=1e-5)

    def test_custom_names(self):
        """Verify custom product and ratio names."""
        reparam = LogProductReparameterization(
            name="test",
            param1="learning_rate",
            param2="n_estimators",
            product_name="learning_budget",
            ratio_name="lr_intensity",
        )

        assert reparam._product_name == "learning_budget"
        assert reparam._ratio_name == "lr_intensity"
        assert reparam.transformed_params == ["learning_budget", "lr_intensity"]

    def test_get_transformed_bounds(self):
        """Verify transformed bounds are computed correctly."""
        reparam = LogProductReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        original_bounds = {"a": (0.01, 0.3), "b": (50, 500)}
        transformed_bounds = reparam.get_transformed_bounds(original_bounds)

        # Product bounds
        product_name = reparam._product_name
        assert product_name in transformed_bounds

        # Low: log(0.01) + log(50) ≈ -0.69
        # High: log(0.3) + log(500) ≈ 5.01
        low, high, log_scale = transformed_bounds[product_name]
        assert low < high

    def test_extreme_values_handled(self):
        """Verify extreme values don't cause overflow."""
        reparam = LogProductReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        # Very small values
        original = {"a": 1e-6, "b": 1e-6}
        transformed = reparam.forward(original)
        recovered = reparam.inverse(transformed)

        assert np.isfinite(transformed[reparam._product_name])
        assert np.isfinite(transformed[reparam._ratio_name])
        assert recovered["a"] == pytest.approx(original["a"], rel=1e-3)
        assert recovered["b"] == pytest.approx(original["b"], rel=1e-3)

    def test_zero_handling(self):
        """Verify near-zero values are handled with epsilon."""
        reparam = LogProductReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        # Very close to zero
        original = {"a": 1e-12, "b": 1.0}
        transformed = reparam.forward(original)

        # Should not produce NaN or Inf
        assert np.isfinite(transformed[reparam._product_name])
        assert np.isfinite(transformed[reparam._ratio_name])


class TestLinearReparameterization:
    """Tests for LinearReparameterization."""

    def test_two_param_forward(self):
        """Verify forward with two parameters."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b"],
        )

        original = {"a": 0.3, "b": 0.7}
        transformed = reparam.forward(original)

        assert transformed[reparam._total_name] == pytest.approx(1.0, rel=1e-5)

    def test_two_param_inverse_roundtrip(self):
        """Verify inverse(forward(x)) ≈ x for two params."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b"],
        )

        original = {"a": 0.3, "b": 0.7}
        transformed = reparam.forward(original)
        recovered = reparam.inverse(transformed)

        assert recovered["a"] == pytest.approx(original["a"], rel=1e-5)
        assert recovered["b"] == pytest.approx(original["b"], rel=1e-5)

    def test_preserves_sum(self):
        """Verify sum a+b is preserved through roundtrip."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b"],
        )

        original = {"a": 0.3, "b": 0.5}
        original_sum = original["a"] + original["b"]

        transformed = reparam.forward(original)
        recovered = reparam.inverse(transformed)
        recovered_sum = recovered["a"] + recovered["b"]

        assert recovered_sum == pytest.approx(original_sum, rel=1e-5)

    def test_three_param_roundtrip(self):
        """Verify roundtrip with three parameters."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b", "c"],
        )

        original = {"a": 0.2, "b": 0.3, "c": 0.5}
        transformed = reparam.forward(original)
        recovered = reparam.inverse(transformed)

        assert recovered["a"] == pytest.approx(original["a"], rel=1e-5)
        assert recovered["b"] == pytest.approx(original["b"], rel=1e-5)
        assert recovered["c"] == pytest.approx(original["c"], rel=1e-5)

    def test_with_weights(self):
        """Verify weighted linear reparameterization."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b"],
            weights=[2.0, 1.0],  # 2*a + b
        )

        original = {"a": 0.3, "b": 0.4}
        expected_total = 2.0 * 0.3 + 1.0 * 0.4  # 1.0

        transformed = reparam.forward(original)

        assert transformed[reparam._total_name] == pytest.approx(expected_total, rel=1e-5)

    def test_transformed_params(self):
        """Verify transformed_params returns correct names."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b", "c"],
            total_name="total",
            ratio_prefix="r",
        )

        params = reparam.transformed_params

        assert "total" in params
        assert "r_a" in params
        assert "r_b" in params
        assert len(params) == 3  # total + (n-1) ratios


class TestRatioReparameterization:
    """Tests for RatioReparameterization."""

    def test_forward(self):
        """Verify forward computes total and ratio."""
        reparam = RatioReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        original = {"a": 0.3, "b": 0.7}
        transformed = reparam.forward(original)

        assert transformed[reparam._total_name] == pytest.approx(1.0, rel=1e-5)
        assert transformed[reparam._ratio_name] == pytest.approx(0.3, rel=1e-5)

    def test_inverse(self):
        """Verify inverse recovers original."""
        reparam = RatioReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        transformed = {reparam._total_name: 1.0, reparam._ratio_name: 0.4}
        original = reparam.inverse(transformed)

        assert original["a"] == pytest.approx(0.4, rel=1e-5)
        assert original["b"] == pytest.approx(0.6, rel=1e-5)

    def test_roundtrip(self):
        """Verify roundtrip preserves values."""
        reparam = RatioReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        original = {"a": 0.25, "b": 0.75}
        transformed = reparam.forward(original)
        recovered = reparam.inverse(transformed)

        assert recovered["a"] == pytest.approx(original["a"], rel=1e-5)
        assert recovered["b"] == pytest.approx(original["b"], rel=1e-5)

    def test_preserves_sum(self):
        """Verify sum is preserved."""
        reparam = RatioReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        original = {"a": 0.3, "b": 0.5}
        original_sum = original["a"] + original["b"]

        transformed = reparam.forward(original)
        recovered = reparam.inverse(transformed)
        recovered_sum = recovered["a"] + recovered["b"]

        assert recovered_sum == pytest.approx(original_sum, rel=1e-5)

    def test_get_transformed_bounds(self):
        """Verify transformed bounds."""
        reparam = RatioReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        original_bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        transformed_bounds = reparam.get_transformed_bounds(original_bounds)

        # Total: 0+0 to 1+1
        total_bounds = transformed_bounds[reparam._total_name]
        assert total_bounds[0] == 0.0
        assert total_bounds[1] == 2.0

        # Ratio: always 0 to 1
        ratio_bounds = transformed_bounds[reparam._ratio_name]
        assert ratio_bounds[0] == 0.0
        assert ratio_bounds[1] == 1.0


class TestReparameterizedSpace:
    """Tests for ReparameterizedSpace."""

    def test_build_transformed_space(self):
        """Verify transformed space is built correctly."""
        original = SearchSpace()
        original.add_float("learning_rate", 0.01, 0.3, log=True)
        original.add_int("n_estimators", 50, 500)
        original.add_int("max_depth", 3, 10)

        reparam = LogProductReparameterization(
            name="lr_budget",
            param1="learning_rate",
            param2="n_estimators",
        )

        rspace = ReparameterizedSpace(original, [reparam])
        transformed = rspace.build_transformed_space()

        # Should have: max_depth (untransformed) + 2 transformed params
        assert len(transformed) == 3
        assert "max_depth" in transformed
        assert reparam._product_name in transformed
        assert reparam._ratio_name in transformed
        # Original params should NOT be in transformed space
        assert "learning_rate" not in transformed
        assert "n_estimators" not in transformed

    def test_inverse_transform(self):
        """Verify inverse transform converts back to original."""
        original = SearchSpace()
        original.add_float("learning_rate", 0.01, 0.3, log=True)
        original.add_int("n_estimators", 50, 500)

        reparam = LogProductReparameterization(
            name="lr_budget",
            param1="learning_rate",
            param2="n_estimators",
        )

        rspace = ReparameterizedSpace(original, [reparam])

        # Simulate transformed params
        transformed_params = {
            reparam._product_name: np.log(0.1) + np.log(100),
            reparam._ratio_name: np.log(0.1) - np.log(100),
        }

        original_params = rspace.inverse_transform(transformed_params)

        assert "learning_rate" in original_params
        assert "n_estimators" in original_params
        assert original_params["learning_rate"] == pytest.approx(0.1, rel=1e-3)
        assert original_params["n_estimators"] == pytest.approx(100, rel=1e-3)

    def test_forward_transform(self):
        """Verify forward transform to reparameterized space."""
        original = SearchSpace()
        original.add_float("learning_rate", 0.01, 0.3, log=True)
        original.add_int("n_estimators", 50, 500)

        reparam = LogProductReparameterization(
            name="lr_budget",
            param1="learning_rate",
            param2="n_estimators",
        )

        rspace = ReparameterizedSpace(original, [reparam])

        original_params = {"learning_rate": 0.1, "n_estimators": 100}
        transformed = rspace.forward_transform(original_params)

        assert reparam._product_name in transformed
        assert reparam._ratio_name in transformed

    def test_untransformed_params_pass_through(self):
        """Verify untransformed params pass through unchanged."""
        original = SearchSpace()
        original.add_float("learning_rate", 0.01, 0.3)
        original.add_int("n_estimators", 50, 500)
        original.add_int("max_depth", 3, 10)

        reparam = LogProductReparameterization(
            name="lr_budget",
            param1="learning_rate",
            param2="n_estimators",
        )

        rspace = ReparameterizedSpace(original, [reparam])

        transformed = {
            reparam._product_name: 5.0,
            reparam._ratio_name: -2.0,
            "max_depth": 7,
        }

        original_params = rspace.inverse_transform(transformed)

        assert original_params["max_depth"] == 7

    def test_multiple_reparameterizations(self):
        """Verify multiple reparameterizations work together."""
        original = SearchSpace()
        original.add_float("learning_rate", 0.01, 0.3)
        original.add_int("n_estimators", 50, 500)
        original.add_float("reg_alpha", 0.0, 1.0)
        original.add_float("reg_lambda", 0.0, 1.0)

        lr_reparam = LogProductReparameterization(
            name="lr_budget",
            param1="learning_rate",
            param2="n_estimators",
        )
        reg_reparam = RatioReparameterization(
            name="regularization",
            param1="reg_alpha",
            param2="reg_lambda",
        )

        rspace = ReparameterizedSpace(original, [lr_reparam, reg_reparam])
        transformed = rspace.build_transformed_space()

        # Should have 4 transformed params
        assert len(transformed) == 4


class TestReparameterizationRepr:
    """Tests for repr methods."""

    def test_log_product_repr(self):
        """Verify LogProductReparameterization repr."""
        reparam = LogProductReparameterization(
            name="test",
            param1="a",
            param2="b",
        )

        repr_str = repr(reparam)

        assert "LogProductReparameterization" in repr_str
        assert "a" in repr_str
        assert "b" in repr_str

    def test_linear_repr(self):
        """Verify LinearReparameterization repr."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b", "c"],
        )

        repr_str = repr(reparam)

        assert "LinearReparameterization" in repr_str

    def test_reparameterized_space_repr(self):
        """Verify ReparameterizedSpace repr."""
        original = SearchSpace()
        original.add_float("a", 0.0, 1.0)

        reparam = LogProductReparameterization(name="test", param1="a", param2="b")
        rspace = ReparameterizedSpace(original, [reparam])

        repr_str = repr(rspace)

        assert "ReparameterizedSpace" in repr_str
