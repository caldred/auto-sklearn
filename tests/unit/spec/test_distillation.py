"""Tests for knowledge distillation module."""


import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from sklearn_meta.spec.distillation import (
    DistillationConfig,
    validate_distillation_estimator,
)
from sklearn_meta.engine.distillation import build_distillation_objective


# =============================================================================
# DistillationConfig tests
# =============================================================================


class TestDistillationConfig:
    """Tests for DistillationConfig validation and immutability."""

    def test_default_values(self):
        """Verify default temperature and alpha."""
        config = DistillationConfig()

        assert config.temperature == 3.0
        assert config.alpha == 0.5

    def test_custom_values(self):
        """Verify custom temperature and alpha."""
        config = DistillationConfig(temperature=5.0, alpha=0.7)

        assert config.temperature == 5.0
        assert config.alpha == 0.7

    def test_frozen(self):
        """Verify config is immutable."""
        config = DistillationConfig()

        with pytest.raises(FrozenInstanceError):
            config.temperature = 10.0

    def test_temperature_zero_raises(self):
        """Verify temperature=0 raises error."""
        with pytest.raises(ValueError, match="temperature must be > 0"):
            DistillationConfig(temperature=0.0)

    def test_temperature_negative_raises(self):
        """Verify negative temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be > 0"):
            DistillationConfig(temperature=-1.0)

    def test_alpha_below_zero_raises(self):
        """Verify alpha < 0 raises error."""
        with pytest.raises(ValueError, match="alpha must be in"):
            DistillationConfig(alpha=-0.1)

    def test_alpha_above_one_raises(self):
        """Verify alpha > 1 raises error."""
        with pytest.raises(ValueError, match="alpha must be in"):
            DistillationConfig(alpha=1.1)

    def test_alpha_boundary_zero(self):
        """Verify alpha=0 is valid (pure CE)."""
        config = DistillationConfig(alpha=0.0)
        assert config.alpha == 0.0

    def test_alpha_boundary_one(self):
        """Verify alpha=1 is valid (pure KL)."""
        config = DistillationConfig(alpha=1.0)
        assert config.alpha == 1.0


# =============================================================================
# build_distillation_objective tests
# =============================================================================


class TestBuildDistillationObjective:
    """Tests for the distillation objective function."""

    def test_output_shapes(self):
        """Verify gradient and hessian have correct shapes."""
        n = 100
        soft_targets = np.random.rand(n)
        y_true = np.random.randint(0, 2, n).astype(float)
        y_pred = np.random.randn(n)

        config = DistillationConfig()
        obj = build_distillation_objective(soft_targets, config)
        grad, hess = obj(y_true, y_pred)

        assert grad.shape == (n,)
        assert hess.shape == (n,)

    def test_hessian_positive(self):
        """Verify hessian is always positive."""
        n = 200
        soft_targets = np.random.rand(n)
        y_true = np.random.randint(0, 2, n).astype(float)
        y_pred = np.random.randn(n) * 5

        config = DistillationConfig()
        obj = build_distillation_objective(soft_targets, config)
        _, hess = obj(y_true, y_pred)

        assert np.all(hess > 0)

    def test_gradient_near_zero_when_student_matches_teacher(self):
        """Verify gradient approaches 0 when student matches teacher and hard labels."""
        n = 50
        # Set up scenario where student perfectly matches both teacher and hard labels
        soft_targets = np.full(n, 0.7)
        y_true = np.full(n, 1.0)  # hard labels

        # Student logit that gives sigmoid = 0.7: z = log(0.7/0.3)
        z_match = np.log(0.7 / 0.3)
        y_pred = np.full(n, z_match)

        # With alpha=1 (pure KL), gradient from KL term should be ~0
        config = DistillationConfig(temperature=1.0, alpha=1.0)
        obj = build_distillation_objective(soft_targets, config)
        grad, _ = obj(y_true, y_pred)

        np.testing.assert_array_almost_equal(grad, 0.0, decimal=5)

    def test_alpha_zero_produces_pure_ce(self):
        """Verify alpha=0 produces pure cross-entropy gradient."""
        n = 50
        soft_targets = np.random.rand(n)
        y_true = np.random.randint(0, 2, n).astype(float)
        y_pred = np.random.randn(n)

        config = DistillationConfig(alpha=0.0)
        obj = build_distillation_objective(soft_targets, config)
        grad, hess = obj(y_true, y_pred)

        # Pure CE gradient: sigmoid(z) - y_true
        p_s = 1.0 / (1.0 + np.exp(-y_pred))
        expected_grad = p_s - y_true
        expected_hess = p_s * (1 - p_s)
        expected_hess = np.maximum(expected_hess, 1e-7)

        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=6)
        np.testing.assert_array_almost_equal(hess, expected_hess, decimal=6)

    def test_alpha_one_produces_pure_kl(self):
        """Verify alpha=1 produces pure KL gradient."""
        n = 50
        soft_targets = np.random.rand(n)
        y_true = np.random.randint(0, 2, n).astype(float)
        y_pred = np.random.randn(n)
        T = 3.0

        config = DistillationConfig(temperature=T, alpha=1.0)
        obj = build_distillation_objective(soft_targets, config)
        grad, hess = obj(y_true, y_pred)

        # Pure KL gradient: T * (sigmoid(z/T) - q_t)
        q_s = 1.0 / (1.0 + np.exp(-y_pred / T))
        q_t_raw = np.clip(soft_targets, 1e-7, 1 - 1e-7)
        teacher_logits_scaled = np.log(q_t_raw / (1.0 - q_t_raw))
        q_t = 1.0 / (1.0 + np.exp(-teacher_logits_scaled))
        expected_grad = T * (q_s - q_t)
        expected_hess = q_s * (1 - q_s)
        expected_hess = np.maximum(expected_hess, 1e-7)

        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=6)
        np.testing.assert_array_almost_equal(hess, expected_hess, decimal=6)

    def test_numerical_stability_extreme_logits(self):
        """Verify no NaN/Inf with extreme logit values."""
        n = 20
        soft_targets = np.random.rand(n)
        y_true = np.random.randint(0, 2, n).astype(float)
        y_pred = np.array([100, -100, 500, -500, 0] * 4, dtype=float)

        config = DistillationConfig()
        obj = build_distillation_objective(soft_targets, config)
        grad, hess = obj(y_true, y_pred)

        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))
        assert np.all(hess > 0)

    def test_multiclass_output_shapes(self):
        """Verify gradient and hessian have correct shapes for multiclass."""
        n, n_classes = 100, 5
        soft_targets = np.random.dirichlet(np.ones(n_classes), size=n)
        y_true = np.random.randint(0, n_classes, n).astype(float)
        y_pred = np.random.randn(n * n_classes)

        config = DistillationConfig()
        obj = build_distillation_objective(soft_targets, config)
        grad, hess = obj(y_true, y_pred)

        assert grad.shape == (n * n_classes,)
        assert hess.shape == (n * n_classes,)

    def test_multiclass_hessian_positive(self):
        """Verify multiclass hessian is always positive."""
        n, n_classes = 80, 4
        soft_targets = np.random.dirichlet(np.ones(n_classes), size=n)
        y_true = np.random.randint(0, n_classes, n).astype(float)
        y_pred = np.random.randn(n * n_classes) * 5

        config = DistillationConfig()
        obj = build_distillation_objective(soft_targets, config)
        _, hess = obj(y_true, y_pred)

        assert np.all(hess > 0)

    def test_multiclass_gradient_finite(self):
        """Verify no NaN/Inf in multiclass gradient with extreme logits."""
        n, n_classes = 20, 3
        soft_targets = np.random.dirichlet(np.ones(n_classes), size=n)
        y_true = np.random.randint(0, n_classes, n).astype(float)
        y_pred = np.array([100, -100, 50] * n, dtype=float)

        config = DistillationConfig()
        obj = build_distillation_objective(soft_targets, config)
        grad, hess = obj(y_true, y_pred)

        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))

    def test_numerical_stability_near_boundary_teacher_probs(self):
        """Verify stability with teacher probs near 0 and 1."""
        n = 10
        soft_targets = np.array([0.0, 1.0, 1e-10, 1 - 1e-10, 0.5,
                                 0.0, 1.0, 1e-10, 1 - 1e-10, 0.5])
        y_true = np.random.randint(0, 2, n).astype(float)
        y_pred = np.random.randn(n)

        config = DistillationConfig()
        obj = build_distillation_objective(soft_targets, config)
        grad, hess = obj(y_true, y_pred)

        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))

    def test_gradient_finite_difference_approximation(self):
        """Verify gradient matches finite-difference approximation."""
        n = 10
        np.random.seed(42)
        soft_targets = np.random.rand(n)
        y_true = np.random.randint(0, 2, n).astype(float)
        y_pred = np.random.randn(n)
        T = 2.0
        alpha = 0.6

        config = DistillationConfig(temperature=T, alpha=alpha)
        obj = build_distillation_objective(soft_targets, config)
        grad, _ = obj(y_true, y_pred)

        # Compute loss numerically for finite differences
        def _sigmoid(z):
            return np.where(
                z >= 0,
                1.0 / (1.0 + np.exp(-z)),
                np.exp(z) / (1.0 + np.exp(z)),
            )

        def compute_loss_element(z, y, q_t_raw_val, T, alpha):
            """Compute per-element loss."""
            q_t_clipped = np.clip(q_t_raw_val, 1e-7, 1 - 1e-7)
            teacher_logit = np.log(q_t_clipped / (1 - q_t_clipped))
            q_t = _sigmoid(teacher_logit)
            q_s = _sigmoid(z / T)
            p_s = _sigmoid(z)

            # KL divergence component (from teacher to student at temperature T)
            kl = q_t * np.log(q_t / np.clip(q_s, 1e-15, None)) + \
                 (1 - q_t) * np.log((1 - q_t) / np.clip(1 - q_s, 1e-15, None))

            # CE component
            ce = -(y * np.log(np.clip(p_s, 1e-15, None)) +
                   (1 - y) * np.log(np.clip(1 - p_s, 1e-15, None)))

            return alpha * T * T * kl + (1 - alpha) * ce

        eps = 1e-5
        fd_grad = np.zeros(n)
        for i in range(n):
            z_plus = y_pred.copy()
            z_minus = y_pred.copy()
            z_plus[i] += eps
            z_minus[i] -= eps

            loss_plus = compute_loss_element(
                z_plus[i], y_true[i], soft_targets[i], T, alpha
            )
            loss_minus = compute_loss_element(
                z_minus[i], y_true[i], soft_targets[i], T, alpha
            )
            fd_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_array_almost_equal(grad, fd_grad, decimal=4)


# =============================================================================
# validate_distillation_estimator tests
# =============================================================================


class TestValidateDistillationEstimator:
    """Tests for validate_distillation_estimator."""

    def test_accepts_xgb_like(self):
        """Verify acceptance of estimator with objective parameter."""
        class MockXGB:
            def __init__(self, objective=None, n_estimators=100):
                self.objective = objective
                self.n_estimators = n_estimators

            def fit(self, X, y):
                pass

            def predict(self, X):
                pass

        # Should not raise
        validate_distillation_estimator(MockXGB)

    def test_accepts_lgbm_like(self):
        """Verify acceptance of estimator with objective parameter."""
        class MockLGBM:
            def __init__(self, objective=None, num_leaves=31):
                self.objective = objective
                self.num_leaves = num_leaves

            def fit(self, X, y):
                pass

            def predict(self, X):
                pass

        # Should not raise
        validate_distillation_estimator(MockLGBM)

    def test_rejects_random_forest(self):
        """Verify rejection of RandomForest (no objective param)."""
        from sklearn.ensemble import RandomForestClassifier

        with pytest.raises(ValueError, match="does not support custom objectives"):
            validate_distillation_estimator(RandomForestClassifier)

    def test_rejects_logistic_regression(self):
        """Verify rejection of LogisticRegression (no objective param)."""
        from sklearn.linear_model import LogisticRegression

        with pytest.raises(ValueError, match="does not support custom objectives"):
            validate_distillation_estimator(LogisticRegression)
