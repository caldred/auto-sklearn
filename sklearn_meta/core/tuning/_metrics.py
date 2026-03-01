"""Shared scoring and logging functions for tuning."""
import logging
from typing import List, Optional, Tuple, Set

import numpy as np

logger = logging.getLogger(__name__)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Calculate mean pinball loss for quantile regression."""
    residual = y_true - y_pred
    loss = np.where(residual >= 0, tau * residual, (tau - 1) * residual)
    return float(np.mean(loss))


def log_feature_selection(
    logger: logging.Logger,
    node_name: str,
    feature_cols: Tuple[str, ...],
    selected_features: List[str],
) -> None:
    """Log feature selection results."""
    n_original = len(feature_cols)
    n_selected = len(selected_features) if selected_features else n_original
    dropped = set(feature_cols) - set(selected_features) if selected_features else set()
    logger.info(
        "Feature selection for '%s': %d/%d features kept",
        node_name, n_selected, n_original,
    )
    if dropped:
        logger.info("  Dropped: %s", sorted(dropped))
