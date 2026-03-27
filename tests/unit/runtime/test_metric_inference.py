"""Tests for automatic greater_is_better inference from metric names."""

import warnings

from sklearn_meta.runtime.config import TuningConfig, RunConfigBuilder


class TestGreaterIsBetterInference:
    """TuningConfig infers greater_is_better from known sklearn metrics."""

    def test_unknown_metric_warns_and_defaults_false(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = TuningConfig(metric="my_custom_metric")
            assert cfg.greater_is_better is False
            assert len(w) == 1
            assert "my_custom_metric" in str(w[0].message)

    def test_default_metric_infers_true(self):
        """Default metric (neg_mean_squared_error) is a known scorer."""
        cfg = TuningConfig()
        assert cfg.greater_is_better is True


class TestBuilderGreaterIsBetterInference:
    """RunConfigBuilder.tuning() propagates inference."""

    def test_builder_infers_from_metric(self):
        config = RunConfigBuilder().tuning(metric="roc_auc").build()
        assert config.tuning.greater_is_better is True

    def test_builder_explicit_override(self):
        config = (
            RunConfigBuilder()
            .tuning(metric="roc_auc", greater_is_better=False)
            .build()
        )
        assert config.tuning.greater_is_better is False
