"""Runtime distillation logic: builds gradient/hessian closure."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from sklearn_meta.spec.distillation import DistillationConfig


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis=1.

    Args:
        z: Array of shape (n_samples, n_classes).

    Returns:
        Softmax probabilities with the same shape.
    """
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _build_binary_distillation_objective(
    soft_targets: np.ndarray,
    config: DistillationConfig,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Build a binary (sigmoid) distillation objective.

    Args:
        soft_targets: 1-D array of teacher probabilities, shape (n_samples,).
        config: Distillation hyper-parameters.

    Returns:
        Closure ``(y_true, y_pred) -> (grad, hess)`` compatible with
        XGBoost / LightGBM custom objectives.
    """
    T = config.temperature
    alpha = config.alpha

    q_t_raw = np.clip(soft_targets, 1e-7, 1 - 1e-7)
    teacher_logits_scaled = np.log(q_t_raw / (1.0 - q_t_raw))

    def objective(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_s = _sigmoid(y_pred / T)
        q_t = _sigmoid(teacher_logits_scaled)
        p_s = _sigmoid(y_pred)

        grad = alpha * T * (q_s - q_t) + (1.0 - alpha) * (p_s - y_true)
        hess = alpha * q_s * (1.0 - q_s) + (1.0 - alpha) * p_s * (1.0 - p_s)
        hess = np.maximum(hess, 1e-7)

        return grad, hess

    return objective


def _build_multiclass_distillation_objective(
    soft_targets: np.ndarray,
    config: DistillationConfig,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Build a multiclass (softmax) distillation objective.

    The KL-divergence component is computed between teacher and student
    softmax distributions at the given temperature.  The hard-label
    component uses standard softmax cross-entropy.

    Args:
        soft_targets: 2-D array of teacher probabilities,
            shape (n_samples, n_classes).
        config: Distillation hyper-parameters.

    Returns:
        Closure ``(y_true, y_pred) -> (grad, hess)`` where *y_pred* arrives
        in XGBoost multi-output format (flattened to length
        ``n_samples * n_classes``, class-major order) and is reshaped
        internally.  The returned *grad* and *hess* are flattened back to
        the same 1-D layout.
    """
    T = config.temperature
    alpha = config.alpha
    n_classes = soft_targets.shape[1]

    # Pre-compute teacher logits (inverse softmax, up to a constant).
    q_t_clipped = np.clip(soft_targets, 1e-7, 1.0)
    teacher_logits = np.log(q_t_clipped)  # unnormalised; softmax re-normalises

    def objective(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(y_true)
        # XGBoost passes y_pred as flat (n_samples * n_classes,)
        logits = y_pred.reshape(n_samples, n_classes)

        # --- KL component (temperature-scaled) ---
        q_s_T = _softmax(logits / T)
        q_t_T = _softmax(teacher_logits / T)

        kl_grad = alpha * T * (q_s_T - q_t_T)

        # --- Hard-label CE component ---
        p_s = _softmax(logits)
        # One-hot encode y_true
        y_onehot = np.zeros_like(p_s)
        y_onehot[np.arange(n_samples), y_true.astype(int)] = 1.0
        ce_grad = (1.0 - alpha) * (p_s - y_onehot)

        grad = kl_grad + ce_grad

        # Diagonal Hessian approximation (same decomposition)
        kl_hess = alpha * q_s_T * (1.0 - q_s_T)
        ce_hess = (1.0 - alpha) * p_s * (1.0 - p_s)
        hess = kl_hess + ce_hess
        hess = np.maximum(hess, 1e-7)

        return grad.ravel(), hess.ravel()

    return objective


def build_distillation_objective(
    soft_targets: np.ndarray,
    config: DistillationConfig,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Build a custom objective for knowledge distillation.

    Dispatches to a binary (sigmoid) or multiclass (softmax) implementation
    based on the dimensionality of *soft_targets*.

    Args:
        soft_targets: Teacher probability predictions.  1-D for binary /
            single-output, 2-D ``(n_samples, n_classes)`` for multiclass.
        config: Distillation hyper-parameters.

    Returns:
        Closure compatible with XGBoost / LightGBM custom objectives.
    """
    if soft_targets.ndim == 1:
        return _build_binary_distillation_objective(soft_targets, config)
    else:
        return _build_multiclass_distillation_objective(soft_targets, config)
