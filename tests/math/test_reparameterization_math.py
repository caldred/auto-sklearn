"""Mathematical tests for reparameterization correctness."""

import pytest
import numpy as np
from scipy import stats

from auto_sklearn.meta.reparameterization import (
    LinearReparameterization,
    LogProductReparameterization,
    RatioReparameterization,
)


class TestLogProductMathematicalProperties:
    """Mathematical verification of LogProductReparameterization."""

    def test_log_product_identity(self):
        """Verify forward(inverse(x)) = x within tolerance."""
        reparam = LogProductReparameterization(
            name="test", param1="a", param2="b"
        )

        # Test multiple random points
        np.random.seed(42)
        for _ in range(100):
            # Generate random transformed params
            product = np.random.uniform(-5, 10)
            ratio = np.random.uniform(-5, 5)

            transformed = {
                reparam._product_name: product,
                reparam._ratio_name: ratio,
            }

            original = reparam.inverse(transformed)
            recovered = reparam.forward(original)

            np.testing.assert_allclose(
                recovered[reparam._product_name], product, rtol=1e-7
            )
            np.testing.assert_allclose(
                recovered[reparam._ratio_name], ratio, rtol=1e-7
            )

    def test_log_product_inverse_identity(self):
        """Verify inverse(forward(x)) = x within tolerance."""
        reparam = LogProductReparameterization(
            name="test", param1="a", param2="b"
        )

        # Test multiple random points
        np.random.seed(42)
        for _ in range(100):
            # Generate random original params (positive)
            a = np.random.uniform(0.001, 100)
            b = np.random.uniform(0.001, 100)

            original = {"a": a, "b": b}

            transformed = reparam.forward(original)
            recovered = reparam.inverse(transformed)

            np.testing.assert_allclose(recovered["a"], a, rtol=1e-7)
            np.testing.assert_allclose(recovered["b"], b, rtol=1e-7)

    def test_log_product_jacobian_determinant(self):
        """Verify the Jacobian determinant is non-zero (invertible)."""
        reparam = LogProductReparameterization(
            name="test", param1="a", param2="b"
        )

        # For log transform: log_a = (log_product + log_ratio) / 2
        #                    log_b = (log_product - log_ratio) / 2
        # The Jacobian of the inverse transform should have non-zero determinant

        np.random.seed(42)
        for _ in range(50):
            product = np.random.uniform(0, 10)
            ratio = np.random.uniform(-3, 3)

            # Compute numerical Jacobian
            eps = 1e-7
            transformed_base = {
                reparam._product_name: product,
                reparam._ratio_name: ratio,
            }

            orig_base = reparam.inverse(transformed_base)

            # Partial derivatives
            transformed_dp = {reparam._product_name: product + eps, reparam._ratio_name: ratio}
            orig_dp = reparam.inverse(transformed_dp)
            da_dp = (orig_dp["a"] - orig_base["a"]) / eps
            db_dp = (orig_dp["b"] - orig_base["b"]) / eps

            transformed_dr = {reparam._product_name: product, reparam._ratio_name: ratio + eps}
            orig_dr = reparam.inverse(transformed_dr)
            da_dr = (orig_dr["a"] - orig_base["a"]) / eps
            db_dr = (orig_dr["b"] - orig_base["b"]) / eps

            # Jacobian determinant
            det = da_dp * db_dr - da_dr * db_dp

            # Should be non-zero for invertibility
            assert abs(det) > 1e-10

    def test_product_preservation(self):
        """Verify a*b is preserved through the transformation."""
        reparam = LogProductReparameterization(
            name="test", param1="a", param2="b"
        )

        np.random.seed(42)
        for _ in range(100):
            a = np.random.uniform(0.01, 10)
            b = np.random.uniform(0.01, 10)

            original_product = a * b

            transformed = reparam.forward({"a": a, "b": b})

            # log(a) + log(b) = log(a*b)
            expected_log_product = np.log(a) + np.log(b)
            np.testing.assert_allclose(
                transformed[reparam._product_name],
                expected_log_product,
                rtol=1e-7
            )

            # Recover and verify product preserved
            recovered = reparam.inverse(transformed)
            recovered_product = recovered["a"] * recovered["b"]

            np.testing.assert_allclose(recovered_product, original_product, rtol=1e-7)

    def test_numerical_stability_extreme_values(self):
        """Verify numerical stability with extreme values."""
        reparam = LogProductReparameterization(
            name="test", param1="a", param2="b"
        )

        # Test very small values (use 1e-6 to avoid numerical precision limits)
        small = {"a": 1e-6, "b": 1e-6}
        transformed = reparam.forward(small)
        recovered = reparam.inverse(transformed)

        assert np.isfinite(transformed[reparam._product_name])
        assert np.isfinite(transformed[reparam._ratio_name])
        np.testing.assert_allclose(recovered["a"], small["a"], rtol=1e-3)
        np.testing.assert_allclose(recovered["b"], small["b"], rtol=1e-3)

        # Test large values
        large = {"a": 1e8, "b": 1e8}
        transformed = reparam.forward(large)
        recovered = reparam.inverse(transformed)

        assert np.isfinite(transformed[reparam._product_name])
        assert np.isfinite(transformed[reparam._ratio_name])
        np.testing.assert_allclose(recovered["a"], large["a"], rtol=1e-5)
        np.testing.assert_allclose(recovered["b"], large["b"], rtol=1e-5)


class TestRatioMathematicalProperties:
    """Mathematical verification of RatioReparameterization."""

    def test_ratio_sum_preserved(self):
        """Verify a + b unchanged through transformation."""
        reparam = RatioReparameterization(
            name="test", param1="a", param2="b"
        )

        np.random.seed(42)
        for _ in range(100):
            a = np.random.uniform(0.01, 10)
            b = np.random.uniform(0.01, 10)
            original_sum = a + b

            transformed = reparam.forward({"a": a, "b": b})
            recovered = reparam.inverse(transformed)

            recovered_sum = recovered["a"] + recovered["b"]
            np.testing.assert_allclose(recovered_sum, original_sum, rtol=1e-10)

    def test_ratio_roundtrip_identity(self):
        """Verify roundtrip is identity within tolerance."""
        reparam = RatioReparameterization(
            name="test", param1="a", param2="b"
        )

        np.random.seed(42)
        for _ in range(100):
            a = np.random.uniform(0.01, 10)
            b = np.random.uniform(0.01, 10)

            original = {"a": a, "b": b}
            transformed = reparam.forward(original)
            recovered = reparam.inverse(transformed)

            np.testing.assert_allclose(recovered["a"], a, rtol=1e-7)
            np.testing.assert_allclose(recovered["b"], b, rtol=1e-7)

    def test_ratio_bounds_correct(self):
        """Verify ratio is always in [0, 1]."""
        reparam = RatioReparameterization(
            name="test", param1="a", param2="b"
        )

        np.random.seed(42)
        for _ in range(100):
            a = np.random.uniform(0.01, 10)
            b = np.random.uniform(0.01, 10)

            transformed = reparam.forward({"a": a, "b": b})
            ratio = transformed[reparam._ratio_name]

            assert 0 <= ratio <= 1


class TestLinearMathematicalProperties:
    """Mathematical verification of LinearReparameterization."""

    def test_linear_sum_preserved(self):
        """Verify weighted sum preserved."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b", "c"],
            weights=[1.0, 2.0, 0.5],
        )

        np.random.seed(42)
        for _ in range(100):
            a = np.random.uniform(0.01, 10)
            b = np.random.uniform(0.01, 10)
            c = np.random.uniform(0.01, 10)

            original_sum = 1.0 * a + 2.0 * b + 0.5 * c

            transformed = reparam.forward({"a": a, "b": b, "c": c})

            np.testing.assert_allclose(
                transformed[reparam._total_name],
                original_sum,
                rtol=1e-10
            )

    def test_linear_roundtrip_identity(self):
        """Verify roundtrip is identity."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b"],
        )

        np.random.seed(42)
        for _ in range(100):
            a = np.random.uniform(0.01, 10)
            b = np.random.uniform(0.01, 10)

            original = {"a": a, "b": b}
            transformed = reparam.forward(original)
            recovered = reparam.inverse(transformed)

            np.testing.assert_allclose(recovered["a"], a, rtol=1e-10)
            np.testing.assert_allclose(recovered["b"], b, rtol=1e-10)

    def test_linear_three_param_roundtrip(self):
        """Verify roundtrip with three parameters."""
        reparam = LinearReparameterization(
            name="test",
            params=["a", "b", "c"],
        )

        np.random.seed(42)
        for _ in range(100):
            a = np.random.uniform(0.01, 5)
            b = np.random.uniform(0.01, 5)
            c = np.random.uniform(0.01, 5)

            original = {"a": a, "b": b, "c": c}
            transformed = reparam.forward(original)
            recovered = reparam.inverse(transformed)

            np.testing.assert_allclose(recovered["a"], a, rtol=1e-8)
            np.testing.assert_allclose(recovered["b"], b, rtol=1e-8)
            np.testing.assert_allclose(recovered["c"], c, rtol=1e-8)


class TestOrthogonalityImprovement:
    """Tests for orthogonality improvement through reparameterization."""

    def test_log_product_reduces_correlation(self):
        """Verify reparameterization reduces parameter correlation."""
        reparam = LogProductReparameterization(
            name="test", param1="a", param2="b"
        )

        # Generate samples from correlated space (constant product)
        np.random.seed(42)
        n_samples = 1000

        # Original space: a and b are correlated (product ≈ constant)
        target_product = 10.0
        noise = np.random.randn(n_samples) * 0.1
        a_values = np.exp(np.random.uniform(np.log(0.1), np.log(10), n_samples))
        b_values = target_product / a_values * np.exp(noise)

        # Transform to reparameterized space
        product_values = []
        ratio_values = []

        for a, b in zip(a_values, b_values):
            transformed = reparam.forward({"a": a, "b": b})
            product_values.append(transformed[reparam._product_name])
            ratio_values.append(transformed[reparam._ratio_name])

        # Compute correlations
        original_corr = np.corrcoef(np.log(a_values), np.log(b_values))[0, 1]
        transformed_corr = np.corrcoef(product_values, ratio_values)[0, 1]

        # Transformed correlation should be lower (closer to 0)
        assert abs(transformed_corr) < abs(original_corr)


class TestBoundaryBehavior:
    """Tests for behavior at parameter boundaries."""

    def test_log_product_at_boundaries(self):
        """Verify correct behavior at boundary values."""
        reparam = LogProductReparameterization(
            name="test", param1="a", param2="b"
        )

        # At equal values, ratio should be 0
        equal = {"a": 1.0, "b": 1.0}
        transformed = reparam.forward(equal)
        assert transformed[reparam._ratio_name] == pytest.approx(0.0, abs=1e-10)

        # When a >> b, ratio should be positive
        a_larger = {"a": 100.0, "b": 0.1}
        transformed = reparam.forward(a_larger)
        assert transformed[reparam._ratio_name] > 0

        # When a << b, ratio should be negative
        b_larger = {"a": 0.1, "b": 100.0}
        transformed = reparam.forward(b_larger)
        assert transformed[reparam._ratio_name] < 0

    def test_ratio_at_extremes(self):
        """Verify ratio behavior at extremes."""
        reparam = RatioReparameterization(
            name="test", param1="a", param2="b"
        )

        # When a = 0, ratio should be 0
        a_zero = {"a": 0.0, "b": 1.0}
        transformed = reparam.forward(a_zero)
        assert transformed[reparam._ratio_name] == pytest.approx(0.0, abs=1e-10)

        # When b = 0, ratio should be 1 (or close to it)
        b_zero = {"a": 1.0, "b": 0.0}
        transformed = reparam.forward(b_zero)
        assert transformed[reparam._ratio_name] == pytest.approx(1.0, rel=1e-5)

    def test_inverse_at_boundary_ratios(self):
        """Verify inverse at boundary ratio values."""
        reparam = RatioReparameterization(
            name="test", param1="a", param2="b"
        )

        # Ratio = 0: all in b
        transformed_0 = {reparam._total_name: 1.0, reparam._ratio_name: 0.0}
        original = reparam.inverse(transformed_0)
        assert original["a"] == pytest.approx(0.0, abs=1e-10)
        assert original["b"] == pytest.approx(1.0, abs=1e-10)

        # Ratio = 1: all in a
        transformed_1 = {reparam._total_name: 1.0, reparam._ratio_name: 1.0}
        original = reparam.inverse(transformed_1)
        assert original["a"] == pytest.approx(1.0, abs=1e-10)
        assert original["b"] == pytest.approx(0.0, abs=1e-10)

        # Ratio = 0.5: split evenly
        transformed_half = {reparam._total_name: 1.0, reparam._ratio_name: 0.5}
        original = reparam.inverse(transformed_half)
        assert original["a"] == pytest.approx(0.5, abs=1e-10)
        assert original["b"] == pytest.approx(0.5, abs=1e-10)
