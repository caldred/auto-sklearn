"""Estimator creation and output routing — runtime concerns extracted from NodeSpec."""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn_meta.spec.node import NodeSpec


def create_estimator(node: NodeSpec, params: Optional[Dict[str, Any]] = None) -> Any:
    """Create an instance of the estimator with given parameters."""
    all_params = dict(node.fixed_params)
    if params:
        all_params.update(params)
    return node.estimator_class(**all_params)


def get_output(node: NodeSpec, model: Any, X) -> Any:
    """Get the output from a fitted model based on output_type."""
    from sklearn_meta.spec.node import OutputType

    if node.output_type == OutputType.PREDICTION:
        return model.predict(X)
    elif node.output_type == OutputType.PROBA:
        return model.predict_proba(X)
    elif node.output_type == OutputType.TRANSFORM:
        return model.transform(X)
    elif node.output_type == OutputType.QUANTILES:
        return model.predict(X)
    else:
        raise ValueError(f"Unknown output type: {node.output_type}")
