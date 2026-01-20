"""Mathematical correctness tests for quantile sampling."""

import pytest
import numpy as np
from scipy import stats

from auto_sklearn.core.model.quantile_sampler import (
    LinearInterpolationSampler,
    ParametricSampler,
    AutoSelectSampler,
    QuantileSampler,
    SamplingStrategy,
)


# =============================================================================
# Test: Linear Interpolation Mathematical Properties
# =============================================================================


class TestLinearInterpolationMath:
    """Mathematical correctness tests for linear interpolation sampler."""

    def test_uniform_distribution(self):
        """
        Verify that sampling from uniform quantiles produces uniform distribution.

        For a uniform distribution U(a, b):
        - Q(tau) = a + tau * (b - a)
        - F(x) = (x - a) / (b - a)
        """
        a, b = 10, 100
        n_levels = 11
        quantile_levels = np.linspace(0, 1, n_levels)
        quantile_values = a + quantile_levels * (b - a)

        sampler = LinearInterpolationSampler()
        sampler.fit(quantile_levels, quantile_values)

        # Sample many values
        np.random.seed(42)
        samples = sampler.sample(10000)

        # Should be approximately uniform
        expected_mean = (a + b) / 2
        expected_std = (b - a) / np.sqrt(12)

        assert np.mean(samples) == pytest.approx(expected_mean, rel=0.05)
        assert np.std(samples) == pytest.approx(expected_std, rel=0.1)

    def test_normal_distribution_approximation(self):
        """
        Verify that sampling from normal quantiles approximates normal distribution.
        """
        loc, scale = 50, 10
        quantile_levels = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        quantile_values = stats.norm.ppf(quantile_levels, loc=loc, scale=scale)

        sampler = LinearInterpolationSampler()
        sampler.fit(quantile_levels, quantile_values)

        # Sample many values
        np.random.seed(42)
        samples = sampler.sample(10000)

        # Should approximate normal distribution
        assert np.mean(samples) == pytest.approx(loc, rel=0.05)
        assert np.std(samples) == pytest.approx(scale, rel=0.1)

    def test_quantile_recovery(self):
        """
        Verify that ppf recovers original quantile values exactly.
        """
        quantile_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantile_values = np.array([10, 25, 50, 75, 90])

        sampler = LinearInterpolationSampler()
        sampler.fit(quantile_levels, quantile_values)

        recovered = sampler.ppf(quantile_levels)

        np.testing.assert_array_almost_equal(recovered, quantile_values)

    def test_interpolation_midpoint(self):
        """
        Verify linear interpolation at midpoints.
        """
        quantile_levels = np.array([0.0, 1.0])
        quantile_values = np.array([0, 100])

        sampler = LinearInterpolationSampler()
        sampler.fit(quantile_levels, quantile_values)

        # Midpoint should be 50
        assert sampler.ppf(np.array([0.5]))[0] == pytest.approx(50)

        # Quarter points
        assert sampler.ppf(np.array([0.25]))[0] == pytest.approx(25)
        assert sampler.ppf(np.array([0.75]))[0] == pytest.approx(75)

    def test_monotonicity(self):
        """
        Verify that ppf is monotonically increasing.
        """
        quantile_levels = np.array([0.1, 0.5, 0.9])
        quantile_values = np.array([10, 50, 90])

        sampler = LinearInterpolationSampler()
        sampler.fit(quantile_levels, quantile_values)

        # Test at many points
        test_levels = np.linspace(0.1, 0.9, 100)
        values = sampler.ppf(test_levels)

        # Should be monotonically increasing
        assert np.all(np.diff(values) >= 0)


# =============================================================================
# Test: Parametric Sampler Mathematical Properties
# =============================================================================


class TestParametricSamplerMath:
    """Mathematical correctness tests for parametric sampler."""

    def test_normal_parameter_recovery(self):
        """
        Verify that normal distribution parameters are recovered correctly.
        """
        true_loc, true_scale = 100, 15
        quantile_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantile_values = stats.norm.ppf(quantile_levels, loc=true_loc, scale=true_scale)

        sampler = ParametricSampler("normal")
        sampler.fit(quantile_levels, quantile_values)

        # Should recover parameters
        fitted_loc, fitted_scale = sampler._params

        assert fitted_loc == pytest.approx(true_loc, rel=0.05)
        assert fitted_scale == pytest.approx(true_scale, rel=0.05)

    def test_normal_distribution_samples(self):
        """
        Verify that samples from fitted normal follow expected distribution.
        """
        true_loc, true_scale = 50, 10
        quantile_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantile_values = stats.norm.ppf(quantile_levels, loc=true_loc, scale=true_scale)

        sampler = ParametricSampler("normal")
        sampler.fit(quantile_levels, quantile_values)

        # Generate samples
        np.random.seed(42)
        samples = sampler.sample(10000)

        # Kolmogorov-Smirnov test against expected normal
        ks_stat, p_value = stats.kstest(samples, 'norm', args=(true_loc, true_scale))

        # Should not reject null hypothesis (samples are from expected distribution)
        assert p_value > 0.01

    def test_skew_normal_fit(self):
        """
        Verify that skew-normal distribution is fitted and sampled correctly.
        """
        # Generate quantiles from skew-normal
        a, loc, scale = 2, 50, 10
        quantile_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantile_values = stats.skewnorm.ppf(quantile_levels, a, loc=loc, scale=scale)

        sampler = ParametricSampler("skew_normal")
        sampler.fit(quantile_levels, quantile_values)

        # Should recover approximate ppf values
        fitted_values = sampler.ppf(quantile_levels)

        # Within reasonable tolerance
        np.testing.assert_array_almost_equal(
            fitted_values, quantile_values, decimal=1
        )


# =============================================================================
# Test: Auto-Select Sampler
# =============================================================================


class TestAutoSelectSamplerMath:
    """Mathematical correctness tests for auto-select sampler."""

    def test_selects_good_fit_for_normal(self):
        """
        Verify that auto-select chooses a distribution that fits well.
        """
        loc, scale = 50, 10
        quantile_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantile_values = stats.norm.ppf(quantile_levels, loc=loc, scale=scale)

        sampler = AutoSelectSampler()
        sampler.fit(quantile_levels, quantile_values)

        # The fitted sampler should reproduce quantiles well
        fitted_values = sampler.ppf(quantile_levels)

        # Error should be small
        mse = np.mean((fitted_values - quantile_values) ** 2)
        assert mse < 1.0

    def test_selects_distribution_for_linear_data(self):
        """
        Verify auto-select works for simple linear data.
        """
        quantile_levels = np.array([0.1, 0.5, 0.9])
        quantile_values = np.array([10, 50, 90])

        sampler = AutoSelectSampler()
        sampler.fit(quantile_levels, quantile_values)

        # Should select something reasonable
        assert sampler.selected_distribution is not None

        # Fitted values should match well
        fitted_values = sampler.ppf(quantile_levels)
        np.testing.assert_array_almost_equal(
            fitted_values, quantile_values, decimal=1
        )


# =============================================================================
# Test: Consistent Sampling Paths
# =============================================================================


class TestConsistentSamplingPaths:
    """Tests for consistent sampling across properties."""

    def test_same_uniforms_used(self):
        """
        Verify that the same uniform samples are used across multiple calls.
        """
        sampler = QuantileSampler(
            strategy=SamplingStrategy.LINEAR_INTERPOLATION,
            n_samples=100,
            random_state=42,
        )

        uniforms_original = sampler.uniform_samples.copy()

        # Sample from different quantile distributions
        levels = [0.1, 0.5, 0.9]
        samples1 = sampler.sample_property("prop1", levels, np.array([10, 50, 90]))
        samples2 = sampler.sample_property("prop2", levels, np.array([20, 60, 100]))

        # Uniforms should not have changed
        np.testing.assert_array_equal(sampler._uniform_samples, uniforms_original)

    def test_correlated_sampling_paths(self):
        """
        Verify that sampling paths maintain correlation structure.

        If we sample from two distributions with the same uniforms,
        samples with the same index should be from corresponding quantiles.
        """
        sampler = QuantileSampler(
            strategy=SamplingStrategy.LINEAR_INTERPOLATION,
            n_samples=1000,
            random_state=42,
        )

        # Two uniform distributions with different ranges
        levels = np.array([0.0, 1.0])
        samples1 = sampler.sample_property("prop1", levels, np.array([0, 100]))
        samples2 = sampler.sample_property("prop2", levels, np.array([0, 200]))

        # Samples should be perfectly correlated
        correlation = np.corrcoef(samples1, samples2)[0, 1]

        assert correlation == pytest.approx(1.0, abs=0.01)

    def test_rank_preservation(self):
        """
        Verify that ranks are preserved across properties.

        If sample i is at the 30th percentile of property 1,
        it should also be at the 30th percentile of property 2.
        """
        sampler = QuantileSampler(
            strategy=SamplingStrategy.LINEAR_INTERPOLATION,
            n_samples=100,
            random_state=42,
        )

        levels = np.linspace(0.05, 0.95, 19)
        values1 = np.linspace(0, 100, 19)
        values2 = np.linspace(50, 150, 19)

        samples1 = sampler.sample_property("prop1", levels, values1)
        samples2 = sampler.sample_property("prop2", levels, values2)

        # Ranks should be identical
        ranks1 = np.argsort(np.argsort(samples1))
        ranks2 = np.argsort(np.argsort(samples2))

        np.testing.assert_array_equal(ranks1, ranks2)


# =============================================================================
# Test: Pinball Loss Properties
# =============================================================================


class TestPinballLossMath:
    """Mathematical properties of pinball loss."""

    def test_pinball_at_median(self):
        """
        At tau=0.5, pinball loss equals 0.5 * MAE.
        """
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])

        residual = y_true - y_pred
        pinball = np.mean(np.where(
            residual >= 0,
            0.5 * residual,
            -0.5 * residual,
        ))

        expected = 0.5 * np.mean(np.abs(residual))

        assert pinball == pytest.approx(expected)

    def test_pinball_asymmetry(self):
        """
        Verify asymmetric penalty for under/over prediction.
        """
        y_true = np.array([100])

        # Under-prediction by 10
        y_pred_under = np.array([90])
        residual_under = y_true - y_pred_under

        # Over-prediction by 10
        y_pred_over = np.array([110])
        residual_over = y_true - y_pred_over

        # At tau=0.9, under-prediction penalized more
        tau = 0.9
        loss_under = np.where(residual_under >= 0, tau * residual_under, (tau - 1) * residual_under)[0]
        loss_over = np.where(residual_over >= 0, tau * residual_over, (tau - 1) * residual_over)[0]

        # Under-prediction loss: 0.9 * 10 = 9
        # Over-prediction loss: -(0.9 - 1) * (-10) = 0.1 * 10 = 1
        assert loss_under == pytest.approx(9)
        assert loss_over == pytest.approx(1)
        assert loss_under > loss_over

    def test_pinball_minimized_at_quantile(self):
        """
        Verify that pinball loss is minimized at the true quantile.

        For a distribution, the pinball loss at tau is minimized
        by predicting Q(tau), the tau-th quantile.
        """
        # Generate samples from a known distribution
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randn(n_samples) * 10 + 50

        tau = 0.3
        true_quantile = np.quantile(y_true, tau)

        # Pinball loss function
        def pinball(y_pred_val):
            y_pred = np.full(n_samples, y_pred_val)
            residual = y_true - y_pred
            return np.mean(np.where(
                residual >= 0,
                tau * residual,
                (tau - 1) * residual,
            ))

        # Loss should be minimized near the true quantile
        losses = []
        predictions = np.linspace(30, 70, 100)
        for p in predictions:
            losses.append(pinball(p))

        min_idx = np.argmin(losses)
        best_pred = predictions[min_idx]

        assert best_pred == pytest.approx(true_quantile, rel=0.1)


# =============================================================================
# Test: Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_extreme_quantile_levels(self):
        """
        Verify handling of extreme quantile levels (0.01, 0.99).
        """
        sampler = LinearInterpolationSampler()
        sampler.fit(
            np.array([0.01, 0.5, 0.99]),
            np.array([-100, 0, 100]),
        )

        # Should not produce NaN or Inf
        samples = sampler.sample(1000)

        assert not np.any(np.isnan(samples))
        assert not np.any(np.isinf(samples))

    def test_very_small_scale(self):
        """
        Verify handling of very small scale distributions.
        """
        sampler = ParametricSampler("normal")
        loc, scale = 1000, 0.001

        levels = np.array([0.1, 0.5, 0.9])
        values = stats.norm.ppf(levels, loc=loc, scale=scale)

        sampler.fit(levels, values)

        # Should handle small scale
        samples = sampler.sample(100)
        assert not np.any(np.isnan(samples))

    def test_large_values(self):
        """
        Verify handling of large values.
        """
        sampler = LinearInterpolationSampler()
        sampler.fit(
            np.array([0.1, 0.5, 0.9]),
            np.array([1e6, 1e7, 1e8]),
        )

        samples = sampler.sample(1000)

        assert not np.any(np.isnan(samples))
        assert not np.any(np.isinf(samples))
        assert np.all(samples >= 1e6)
        assert np.all(samples <= 1e8)
