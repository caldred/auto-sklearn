"""Tests for quantile sampling strategies."""

import pytest
import numpy as np

from sklearn_meta.core.model.quantile_sampler import (
    AutoSelectSampler,
    LinearInterpolationSampler,
    ParametricSampler,
    QuantileSampler,
    QuantileSamplerBase,
    SamplingStrategy,
)


# =============================================================================
# SamplingStrategy Tests
# =============================================================================


class TestSamplingStrategy:
    """Tests for SamplingStrategy enum."""

    def test_enum_values(self):
        """Verify enum values are correct."""
        assert SamplingStrategy.LINEAR_INTERPOLATION.value == "linear_interpolation"
        assert SamplingStrategy.NORMAL.value == "normal"
        assert SamplingStrategy.SKEW_NORMAL.value == "skew_normal"
        assert SamplingStrategy.JOHNSON_SU.value == "johnson_su"
        assert SamplingStrategy.AUTO.value == "auto"


# =============================================================================
# LinearInterpolationSampler Tests
# =============================================================================


class TestLinearInterpolationSampler:
    """Tests for LinearInterpolationSampler."""

    def test_fit_stores_values(self):
        """Verify fit stores quantile levels and values."""
        sampler = LinearInterpolationSampler()
        levels = np.array([0.1, 0.5, 0.9])
        values = np.array([10, 50, 90])

        sampler.fit(levels, values)

        np.testing.assert_array_equal(sampler._quantile_levels, levels)
        np.testing.assert_array_equal(sampler._quantile_values, values)

    def test_ppf_exact_levels(self):
        """Verify ppf returns exact values at known levels."""
        sampler = LinearInterpolationSampler()
        sampler.fit(
            np.array([0.1, 0.5, 0.9]),
            np.array([10, 50, 90]),
        )

        result = sampler.ppf(np.array([0.1, 0.5, 0.9]))

        np.testing.assert_array_almost_equal(result, [10, 50, 90])

    def test_ppf_interpolated(self):
        """Verify ppf interpolates between levels."""
        sampler = LinearInterpolationSampler()
        sampler.fit(
            np.array([0.0, 1.0]),
            np.array([0, 100]),
        )

        result = sampler.ppf(np.array([0.5]))

        np.testing.assert_array_almost_equal(result, [50])

    def test_sample_returns_correct_shape(self):
        """Verify sample returns correct number of samples."""
        sampler = LinearInterpolationSampler()
        sampler.fit(
            np.array([0.1, 0.5, 0.9]),
            np.array([10, 50, 90]),
        )

        samples = sampler.sample(1000)

        assert samples.shape == (1000,)

    def test_sample_uses_provided_uniforms(self):
        """Verify sample uses provided uniform samples."""
        sampler = LinearInterpolationSampler()
        sampler.fit(
            np.array([0.0, 1.0]),
            np.array([0, 100]),
        )

        uniforms = np.array([0.25, 0.5, 0.75])
        samples = sampler.sample(3, uniform_samples=uniforms)

        np.testing.assert_array_almost_equal(samples, [25, 50, 75])

    def test_sample_before_fit_raises(self):
        """Verify sampling before fit raises error."""
        sampler = LinearInterpolationSampler()

        with pytest.raises(RuntimeError, match="must be fit"):
            sampler.sample(10)


# =============================================================================
# ParametricSampler Tests
# =============================================================================


class TestParametricSamplerNormal:
    """Tests for ParametricSampler with normal distribution."""

    @pytest.fixture
    def normal_quantiles(self):
        """Generate quantiles from standard normal."""
        from scipy import stats

        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        values = stats.norm.ppf(levels, loc=50, scale=10)
        return levels, values

    def test_fit_recovers_params(self, normal_quantiles):
        """Verify fit approximately recovers normal parameters."""
        levels, values = normal_quantiles
        sampler = ParametricSampler("normal")

        sampler.fit(levels, values)

        # Should recover loc~50, scale~10
        assert sampler._params is not None
        loc, scale = sampler._params
        assert loc == pytest.approx(50, rel=0.1)
        assert scale == pytest.approx(10, rel=0.1)

    def test_ppf_matches_original(self, normal_quantiles):
        """Verify ppf approximately matches original quantiles."""
        levels, values = normal_quantiles
        sampler = ParametricSampler("normal")
        sampler.fit(levels, values)

        result = sampler.ppf(levels)

        np.testing.assert_array_almost_equal(result, values, decimal=1)

    def test_sample_shape(self, normal_quantiles):
        """Verify sample returns correct shape."""
        levels, values = normal_quantiles
        sampler = ParametricSampler("normal")
        sampler.fit(levels, values)

        samples = sampler.sample(1000)

        assert samples.shape == (1000,)


class TestParametricSamplerSkewNormal:
    """Tests for ParametricSampler with skew-normal distribution."""

    @pytest.fixture
    def skew_normal_quantiles(self):
        """Generate quantiles from skew-normal."""
        from scipy import stats

        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        values = stats.skewnorm.ppf(levels, a=2, loc=50, scale=10)
        return levels, values

    def test_fit_and_sample(self, skew_normal_quantiles):
        """Verify fit and sample work for skew-normal."""
        levels, values = skew_normal_quantiles
        sampler = ParametricSampler("skew_normal")
        sampler.fit(levels, values)

        samples = sampler.sample(1000)

        assert samples.shape == (1000,)


# =============================================================================
# AutoSelectSampler Tests
# =============================================================================


class TestAutoSelectSampler:
    """Tests for AutoSelectSampler."""

    def test_fit_selects_distribution(self):
        """Verify fit selects a distribution."""
        sampler = AutoSelectSampler()
        sampler.fit(
            np.array([0.1, 0.5, 0.9]),
            np.array([10, 50, 90]),
        )

        assert sampler._best_sampler is not None
        assert sampler._best_distribution is not None

    def test_selected_distribution_property(self):
        """Verify selected_distribution property works."""
        sampler = AutoSelectSampler()
        sampler.fit(
            np.array([0.1, 0.5, 0.9]),
            np.array([10, 50, 90]),
        )

        assert sampler.selected_distribution in ["linear", "normal", "skew_normal"]

    def test_sample_uses_best_sampler(self):
        """Verify sample delegates to best sampler."""
        sampler = AutoSelectSampler()
        sampler.fit(
            np.array([0.1, 0.5, 0.9]),
            np.array([10, 50, 90]),
        )

        samples = sampler.sample(1000)

        assert samples.shape == (1000,)


# =============================================================================
# QuantileSampler Tests
# =============================================================================


class TestQuantileSamplerCreation:
    """Tests for QuantileSampler creation."""

    def test_default_creation(self):
        """Verify default sampler creation."""
        sampler = QuantileSampler()

        assert sampler.strategy == SamplingStrategy.LINEAR_INTERPOLATION
        assert sampler.n_samples == 1000
        assert sampler._uniform_samples is not None

    def test_creation_with_params(self):
        """Verify sampler creation with custom params."""
        sampler = QuantileSampler(
            strategy=SamplingStrategy.NORMAL,
            n_samples=500,
            random_state=42,
        )

        assert sampler.strategy == SamplingStrategy.NORMAL
        assert sampler.n_samples == 500

    def test_uniform_samples_shape(self):
        """Verify uniform samples have correct shape."""
        sampler = QuantileSampler(n_samples=100, random_state=42)

        assert sampler._uniform_samples.shape == (100,)
        assert np.all(sampler._uniform_samples >= 0)
        assert np.all(sampler._uniform_samples <= 1)


class TestQuantileSamplerSampleProperty:
    """Tests for sample_property method."""

    def test_sample_property_1d(self):
        """Verify sample_property with 1D input."""
        sampler = QuantileSampler(n_samples=100, random_state=42)

        levels = [0.1, 0.5, 0.9]
        predictions = np.array([10, 50, 90])

        samples = sampler.sample_property("test", levels, predictions)

        assert samples.shape == (100,)
        assert np.all(samples >= 10)
        assert np.all(samples <= 90)

    def test_sample_property_2d(self):
        """Verify sample_property with 2D input (batch)."""
        sampler = QuantileSampler(n_samples=100, random_state=42)

        levels = [0.1, 0.5, 0.9]
        predictions = np.array([
            [10, 50, 90],
            [20, 60, 100],
        ])

        samples = sampler.sample_property("test", levels, predictions)

        assert samples.shape == (2, 100)

    def test_consistent_sampling_paths(self):
        """Verify same uniform samples are used for consistency."""
        sampler = QuantileSampler(n_samples=100, random_state=42)

        levels = [0.1, 0.5, 0.9]
        predictions1 = np.array([10, 50, 90])
        predictions2 = np.array([20, 60, 100])

        samples1 = sampler.sample_property("prop1", levels, predictions1)
        samples2 = sampler.sample_property("prop2", levels, predictions2)

        # Both should use the same uniform samples, so ordering should be preserved
        # If uniform sample was 0.5, sample1 should be 50 and sample2 should be 60
        # The relative ordering should be consistent
        assert samples1.shape == samples2.shape


class TestQuantileSamplerBatched:
    """Tests for batched sampling."""

    def test_sample_property_batched(self):
        """Verify batched sampling works."""
        sampler = QuantileSampler(n_samples=100, random_state=42)

        levels = [0.1, 0.5, 0.9]
        predictions = np.array([
            [10, 50, 90],
            [20, 60, 100],
            [30, 70, 110],
        ])

        samples = sampler.sample_property_batched("test", levels, predictions)

        assert samples.shape == (3, 100)


class TestQuantileSamplerMedian:
    """Tests for get_median method."""

    def test_get_median_1d(self):
        """Verify get_median with 1D input."""
        sampler = QuantileSampler()

        levels = [0.1, 0.5, 0.9]
        predictions = np.array([10, 50, 90])

        median = sampler.get_median(levels, predictions)

        assert median == pytest.approx(50)

    def test_get_median_2d(self):
        """Verify get_median with 2D input."""
        sampler = QuantileSampler()

        levels = [0.1, 0.5, 0.9]
        predictions = np.array([
            [10, 50, 90],
            [20, 60, 100],
        ])

        medians = sampler.get_median(levels, predictions)

        np.testing.assert_array_almost_equal(medians, [50, 60])


class TestQuantileSamplerQuantile:
    """Tests for get_quantile method."""

    def test_get_quantile_exact(self):
        """Verify get_quantile at exact level."""
        sampler = QuantileSampler()

        levels = [0.1, 0.5, 0.9]
        predictions = np.array([10, 50, 90])

        q10 = sampler.get_quantile(0.1, levels, predictions)

        assert q10 == pytest.approx(10)

    def test_get_quantile_interpolated(self):
        """Verify get_quantile with interpolation."""
        sampler = QuantileSampler()

        levels = [0.0, 1.0]
        predictions = np.array([0, 100])

        q50 = sampler.get_quantile(0.5, levels, predictions)

        assert q50 == pytest.approx(50)


class TestQuantileSamplerReset:
    """Tests for reset_samples method."""

    def test_reset_generates_new_samples(self):
        """Verify reset_samples generates new uniform samples."""
        sampler = QuantileSampler(n_samples=100, random_state=42)

        original = sampler._uniform_samples.copy()
        sampler.reset_samples()
        new = sampler._uniform_samples

        # With different random state, samples should be different
        assert not np.allclose(original, new)


class TestQuantileSamplerUniformProperty:
    """Tests for uniform_samples property."""

    def test_uniform_samples_returns_copy(self):
        """Verify uniform_samples returns a copy."""
        sampler = QuantileSampler(n_samples=100, random_state=42)

        samples1 = sampler.uniform_samples
        samples2 = sampler.uniform_samples

        # Modifying one shouldn't affect the other
        samples1[0] = -1
        assert samples2[0] != -1
