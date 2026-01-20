"""ModelNode: Definition of a single model in the graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from auto_sklearn.core.data.context import DataContext
    from auto_sklearn.search.space import SearchSpace


class OutputType:
    """Output types for model nodes."""

    PREDICTION = "prediction"
    """Raw predictions (for regression or binary classification)."""

    PROBA = "proba"
    """Class probabilities (for classification)."""

    TRANSFORM = "transform"
    """Transformed features (for preprocessors/transformers)."""

    QUANTILES = "quantiles"
    """Quantile predictions (for quantile regression)."""


@dataclass
class ModelNode:
    """
    Definition of a single model in the graph.

    This class defines what model to use and how to configure it,
    but contains no training logic. Training is handled by the
    TuningOrchestrator.

    Attributes:
        name: Unique identifier for this node.
        estimator_class: The sklearn-compatible estimator class.
        search_space: Hyperparameter search space (optional).
        output_type: Type of output this model produces.
        condition: Optional callable that determines if this node should run.
        plugins: List of plugin names to apply to this node.
        fixed_params: Parameters that are fixed (not tuned).
        fit_params: Additional parameters passed to fit().
        feature_cols: Optional list of feature columns to use.
        description: Human-readable description of this node.
    """

    name: str
    estimator_class: Type
    search_space: Optional[SearchSpace] = None
    output_type: str = OutputType.PREDICTION
    condition: Optional[Callable[[DataContext], bool]] = None
    plugins: List[str] = field(default_factory=list)
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    fit_params: Dict[str, Any] = field(default_factory=dict)
    feature_cols: Optional[List[str]] = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate node configuration."""
        if not self.name:
            raise ValueError("Node name cannot be empty")
        if not self.estimator_class:
            raise ValueError("Estimator class is required")
        # Validate estimator has required methods
        if not hasattr(self.estimator_class, "fit"):
            raise ValueError(
                f"Estimator {self.estimator_class} must have a 'fit' method"
            )
        if self.output_type == OutputType.PREDICTION:
            if not hasattr(self.estimator_class, "predict"):
                raise ValueError(
                    f"Estimator {self.estimator_class} must have a 'predict' method "
                    "for output_type='prediction'"
                )
        elif self.output_type == OutputType.PROBA:
            if not hasattr(self.estimator_class, "predict_proba"):
                raise ValueError(
                    f"Estimator {self.estimator_class} must have a 'predict_proba' method "
                    "for output_type='proba'"
                )
        elif self.output_type == OutputType.TRANSFORM:
            if not hasattr(self.estimator_class, "transform"):
                raise ValueError(
                    f"Estimator {self.estimator_class} must have a 'transform' method "
                    "for output_type='transform'"
                )

    @property
    def has_search_space(self) -> bool:
        """Whether this node has hyperparameters to tune."""
        return self.search_space is not None and len(self.search_space) > 0

    @property
    def is_conditional(self) -> bool:
        """Whether this node has a condition for execution."""
        return self.condition is not None

    def should_run(self, ctx: DataContext) -> bool:
        """Check if this node should run given the current context."""
        if self.condition is None:
            return True
        return self.condition(ctx)

    def create_estimator(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create an instance of the estimator with given parameters.

        Args:
            params: Hyperparameters to use. If None, uses fixed_params only.

        Returns:
            Configured estimator instance.
        """
        all_params = dict(self.fixed_params)
        if params:
            all_params.update(params)
        return self.estimator_class(**all_params)

    def get_output(self, model: Any, X) -> Any:
        """
        Get the output from a fitted model based on output_type.

        Args:
            model: Fitted estimator.
            X: Input features.

        Returns:
            Model output (predictions, probabilities, or transformed features).
        """
        if self.output_type == OutputType.PREDICTION:
            return model.predict(X)
        elif self.output_type == OutputType.PROBA:
            return model.predict_proba(X)
        elif self.output_type == OutputType.TRANSFORM:
            return model.transform(X)
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")

    def __repr__(self) -> str:
        class_name = self.estimator_class.__name__
        return f"ModelNode(name={self.name!r}, estimator={class_name})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelNode):
            return NotImplemented
        return self.name == other.name
