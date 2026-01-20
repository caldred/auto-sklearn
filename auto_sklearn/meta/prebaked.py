"""
Pre-baked Hyperparameter Reparameterizations.

This module provides pre-defined reparameterizations for common model/
hyperparameter combinations. These encode domain knowledge about which
hyperparameters have functional relationships and how to best transform them.

The pre-baked reparameterizations are designed to:
1. Convert correlated parameter spaces to orthogonal representations
2. Encode known tradeoffs (e.g., learning_rate vs n_estimators)
3. Simplify tuning by reducing effective dimensionality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from auto_sklearn.meta.reparameterization import (
    LinearReparameterization,
    LogProductReparameterization,
    RatioReparameterization,
    Reparameterization,
)


@dataclass
class PrebakedConfig:
    """Configuration for a pre-baked reparameterization."""

    name: str
    description: str
    model_patterns: List[str]  # Model class name patterns that match
    param_patterns: List[str]  # Required params (all must be present)
    create_reparam: Callable[[], Reparameterization]
    priority: int = 0  # Higher = applied first


# Global registry of pre-baked reparameterizations
PREBAKED_REGISTRY: Dict[str, PrebakedConfig] = {}


def register_prebaked(config: PrebakedConfig) -> None:
    """Register a pre-baked reparameterization."""
    PREBAKED_REGISTRY[config.name] = config


def get_prebaked_reparameterization(
    model_class: Type,
    param_names: List[str],
) -> List[Reparameterization]:
    """
    Get applicable pre-baked reparameterizations for a model.

    Args:
        model_class: The model class being tuned.
        param_names: List of hyperparameter names in the search space.

    Returns:
        List of applicable Reparameterization objects.
    """
    model_name = model_class.__name__
    param_set = set(param_names)

    applicable = []

    for config in sorted(
        PREBAKED_REGISTRY.values(),
        key=lambda c: -c.priority
    ):
        # Check model pattern
        model_match = any(
            pattern.lower() in model_name.lower()
            for pattern in config.model_patterns
        )

        if not model_match:
            continue

        # Check all required params are present
        param_match = all(
            any(p in param_set for p in pattern.split("|"))
            for pattern in config.param_patterns
        )

        if param_match:
            applicable.append(config.create_reparam())

    return applicable


def get_all_prebaked_for_model(model_class: Type) -> List[PrebakedConfig]:
    """
    Get all pre-baked configs that could apply to a model.

    Useful for documentation and exploration.
    """
    model_name = model_class.__name__
    matching = []

    for config in PREBAKED_REGISTRY.values():
        if any(p.lower() in model_name.lower() for p in config.model_patterns):
            matching.append(config)

    return matching


# =============================================================================
# Gradient Boosting Reparameterizations
# =============================================================================

register_prebaked(PrebakedConfig(
    name="xgb_learning_budget",
    description=(
        "XGBoost learning rate × n_estimators tradeoff. "
        "Higher learning rate means fewer trees needed for same total learning. "
        "Reparameterized to: total_budget (lr * n_est) and lr_ratio."
    ),
    model_patterns=["XGB", "XGBoost", "LGBM", "LightGBM", "GradientBoosting"],
    param_patterns=["learning_rate|eta", "n_estimators|num_boost_round"],
    create_reparam=lambda: LogProductReparameterization(
        name="learning_budget",
        param1="learning_rate",
        param2="n_estimators",
        product_name="learning_budget",
        ratio_name="lr_intensity",
    ),
    priority=10,
))

register_prebaked(PrebakedConfig(
    name="xgb_regularization",
    description=(
        "XGBoost L1 (alpha) and L2 (lambda) regularization. "
        "Both reduce overfitting; reparameterized to total_reg and l1_ratio."
    ),
    model_patterns=["XGB", "XGBoost"],
    param_patterns=["reg_alpha|alpha", "reg_lambda|lambda"],
    create_reparam=lambda: RatioReparameterization(
        name="xgb_regularization",
        param1="reg_alpha",
        param2="reg_lambda",
        total_name="total_regularization",
        ratio_name="l1_ratio",
    ),
    priority=5,
))

register_prebaked(PrebakedConfig(
    name="lgbm_regularization",
    description=(
        "LightGBM L1 and L2 regularization. "
        "Reparameterized to total regularization strength and L1/L2 ratio."
    ),
    model_patterns=["LGBM", "LightGBM"],
    param_patterns=["reg_alpha|lambda_l1", "reg_lambda|lambda_l2"],
    create_reparam=lambda: RatioReparameterization(
        name="lgbm_regularization",
        param1="reg_alpha",
        param2="reg_lambda",
        total_name="total_regularization",
        ratio_name="l1_ratio",
    ),
    priority=5,
))

register_prebaked(PrebakedConfig(
    name="gbm_depth_leaves",
    description=(
        "Gradient boosting max_depth and num_leaves tradeoff. "
        "Both control tree complexity; reparameterized to complexity and style."
    ),
    model_patterns=["LGBM", "LightGBM", "CatBoost"],
    param_patterns=["max_depth", "num_leaves|max_leaves"],
    create_reparam=lambda: LogProductReparameterization(
        name="tree_complexity",
        param1="max_depth",
        param2="num_leaves",
        product_name="tree_complexity",
        ratio_name="depth_vs_leaves",
    ),
    priority=3,
))


# =============================================================================
# Neural Network Reparameterizations
# =============================================================================

register_prebaked(PrebakedConfig(
    name="nn_learning_epochs",
    description=(
        "Neural network learning rate × epochs tradeoff. "
        "Similar to boosting: higher LR needs fewer epochs."
    ),
    model_patterns=["MLP", "Neural", "NN", "Keras", "Torch"],
    param_patterns=["learning_rate|lr", "epochs|n_epochs|max_epochs"],
    create_reparam=lambda: LogProductReparameterization(
        name="nn_learning_budget",
        param1="learning_rate",
        param2="epochs",
        product_name="learning_budget",
        ratio_name="lr_intensity",
    ),
    priority=10,
))

register_prebaked(PrebakedConfig(
    name="nn_dropout",
    description=(
        "Multiple dropout layers in neural networks. "
        "Reparameterized to total dropout and distribution."
    ),
    model_patterns=["MLP", "Neural", "NN"],
    param_patterns=["dropout1|dropout_1", "dropout2|dropout_2"],
    create_reparam=lambda: LinearReparameterization(
        name="nn_dropout",
        params=["dropout1", "dropout2"],
        total_name="total_dropout",
        ratio_prefix="dropout_alloc",
    ),
    priority=5,
))

register_prebaked(PrebakedConfig(
    name="nn_weight_decay_dropout",
    description=(
        "Weight decay and dropout both provide regularization. "
        "Reparameterized to total regularization and style (implicit vs explicit)."
    ),
    model_patterns=["MLP", "Neural", "NN", "Torch"],
    param_patterns=["weight_decay|l2", "dropout"],
    create_reparam=lambda: RatioReparameterization(
        name="nn_regularization",
        param1="weight_decay",
        param2="dropout",
        total_name="total_regularization",
        ratio_name="weight_decay_ratio",
    ),
    priority=3,
))


# =============================================================================
# Linear Model Reparameterizations
# =============================================================================

register_prebaked(PrebakedConfig(
    name="elastic_net",
    description=(
        "ElasticNet L1 and L2 regularization. "
        "The classic example of reparameterization: alpha (total) and l1_ratio."
    ),
    model_patterns=["ElasticNet", "SGD", "Linear"],
    param_patterns=["alpha|C", "l1_ratio"],
    # This is already parameterized well in sklearn, but we support it
    create_reparam=lambda: RatioReparameterization(
        name="elastic_net",
        param1="l1_penalty",
        param2="l2_penalty",
        total_name="alpha",
        ratio_name="l1_ratio",
    ),
    priority=1,  # Low priority since sklearn does this
))


# =============================================================================
# Random Forest / Tree Ensemble Reparameterizations
# =============================================================================

register_prebaked(PrebakedConfig(
    name="rf_complexity",
    description=(
        "Random Forest max_depth and min_samples_split tradeoff. "
        "Both control tree complexity/overfitting."
    ),
    model_patterns=["RandomForest", "ExtraTrees", "Forest"],
    param_patterns=["max_depth", "min_samples_split|min_samples_leaf"],
    create_reparam=lambda: LogProductReparameterization(
        name="rf_complexity",
        param1="max_depth",
        param2="min_samples_split",
        product_name="tree_complexity",
        ratio_name="depth_vs_samples",
    ),
    priority=5,
))

register_prebaked(PrebakedConfig(
    name="rf_sampling",
    description=(
        "Random Forest max_features and max_samples tradeoff. "
        "Both control the diversity vs accuracy of individual trees."
    ),
    model_patterns=["RandomForest", "ExtraTrees", "Bagging"],
    param_patterns=["max_features", "max_samples"],
    create_reparam=lambda: LogProductReparameterization(
        name="rf_sampling",
        param1="max_features",
        param2="max_samples",
        product_name="sampling_intensity",
        ratio_name="feature_vs_sample",
    ),
    priority=3,
))


# =============================================================================
# SVM Reparameterizations
# =============================================================================

register_prebaked(PrebakedConfig(
    name="svm_kernel",
    description=(
        "SVM C and gamma tradeoff for RBF kernel. "
        "Higher C with lower gamma can give similar boundaries."
    ),
    model_patterns=["SVC", "SVR", "SVM"],
    param_patterns=["C", "gamma"],
    create_reparam=lambda: LogProductReparameterization(
        name="svm_kernel",
        param1="C",
        param2="gamma",
        product_name="kernel_strength",
        ratio_name="c_gamma_ratio",
    ),
    priority=5,
))


# =============================================================================
# CatBoost Specific
# =============================================================================

register_prebaked(PrebakedConfig(
    name="catboost_regularization",
    description=(
        "CatBoost l2_leaf_reg and random_strength tradeoff. "
        "Both provide regularization through different mechanisms."
    ),
    model_patterns=["CatBoost"],
    param_patterns=["l2_leaf_reg", "random_strength"],
    create_reparam=lambda: RatioReparameterization(
        name="catboost_regularization",
        param1="l2_leaf_reg",
        param2="random_strength",
        total_name="total_regularization",
        ratio_name="l2_vs_random",
    ),
    priority=5,
))


# =============================================================================
# Utility Functions
# =============================================================================

def suggest_reparameterizations(
    model_class: Type,
    param_names: List[str],
    include_descriptions: bool = True,
) -> List[Dict[str, Any]]:
    """
    Suggest reparameterizations with explanations.

    Args:
        model_class: The model class.
        param_names: List of hyperparameter names.
        include_descriptions: Whether to include detailed descriptions.

    Returns:
        List of dictionaries with reparameterization details.
    """
    model_name = model_class.__name__
    param_set = set(param_names)
    suggestions = []

    for config in PREBAKED_REGISTRY.values():
        # Check model pattern
        model_match = any(
            pattern.lower() in model_name.lower()
            for pattern in config.model_patterns
        )

        if not model_match:
            continue

        # Find which params would match
        matching_params = []
        for pattern in config.param_patterns:
            alternatives = pattern.split("|")
            for alt in alternatives:
                if alt in param_set:
                    matching_params.append(alt)
                    break

        if len(matching_params) == len(config.param_patterns):
            suggestion = {
                "name": config.name,
                "applies": True,
                "params": matching_params,
            }
            if include_descriptions:
                suggestion["description"] = config.description

            suggestions.append(suggestion)
        elif matching_params:
            # Partial match - mention what's missing
            suggestion = {
                "name": config.name,
                "applies": False,
                "params": matching_params,
                "missing": [
                    p for p in config.param_patterns
                    if not any(alt in param_set for alt in p.split("|"))
                ],
            }
            if include_descriptions:
                suggestion["description"] = config.description

            suggestions.append(suggestion)

    return suggestions


def create_reparameterized_space(
    original_space,
    model_class: Type,
    custom_reparams: Optional[List[Reparameterization]] = None,
) -> "ReparameterizedSpace":
    """
    Create a reparameterized search space for a model.

    Combines pre-baked and custom reparameterizations.

    Args:
        original_space: The original SearchSpace.
        model_class: The model class being tuned.
        custom_reparams: Additional custom reparameterizations.

    Returns:
        ReparameterizedSpace ready for use in optimization.
    """
    from auto_sklearn.meta.reparameterization import ReparameterizedSpace

    param_names = original_space.parameter_names

    # Get pre-baked reparameterizations
    reparams = get_prebaked_reparameterization(model_class, param_names)

    # Add custom reparameterizations
    if custom_reparams:
        reparams.extend(custom_reparams)

    return ReparameterizedSpace(original_space, reparams)
