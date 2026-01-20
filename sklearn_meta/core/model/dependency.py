"""Dependency definitions for the model graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DependencyType(Enum):
    """Types of dependencies between model nodes."""

    PREDICTION = "prediction"
    """Stacking: predictions from source become features for target."""

    TRANSFORM = "transform"
    """Pipeline: transformed X flows from source to target."""

    FEATURE = "feature"
    """Single feature engineering: source output becomes one feature."""

    PROBA = "proba"
    """Stacking with probabilities: class probabilities become features."""

    BASE_MARGIN = "base_margin"
    """XGBoost-style stacking: predictions used as base margin."""

    CONDITIONAL_SAMPLE = "conditional_sample"
    """Joint quantile: condition on sampled values from upstream property."""


@dataclass
class ConditionalSampleConfig:
    """
    Configuration for conditional sample dependencies.

    Used in joint quantile regression to specify how upstream
    property values should be used as conditioning features.

    Attributes:
        property_name: Name of the upstream property to condition on.
        use_actual_during_training: If True, use actual Y values during training
                                   instead of predictions (default: True).
    """

    property_name: str
    use_actual_during_training: bool = True


@dataclass
class DependencyEdge:
    """
    Represents a directed edge in the model graph.

    An edge from source to target indicates that target depends on source.
    The dependency type determines how the source's output is used.

    Attributes:
        source: Name of the source node.
        target: Name of the target node.
        dep_type: How the dependency is used.
        column_name: Optional name for the feature in target's input.
                    If None, defaults to "pred_{source}" or similar.
        conditional_config: Configuration for CONDITIONAL_SAMPLE dependencies.
    """

    source: str
    target: str
    dep_type: DependencyType = DependencyType.PREDICTION
    column_name: Optional[str] = None
    conditional_config: Optional[ConditionalSampleConfig] = None

    def __post_init__(self) -> None:
        """Validate edge configuration."""
        if not self.source:
            raise ValueError("Source node name cannot be empty")
        if not self.target:
            raise ValueError("Target node name cannot be empty")
        if self.source == self.target:
            raise ValueError("Self-loops are not allowed")
        if isinstance(self.dep_type, str):
            self.dep_type = DependencyType(self.dep_type)
        # Validate conditional_config is provided for CONDITIONAL_SAMPLE
        if (
            self.dep_type == DependencyType.CONDITIONAL_SAMPLE
            and self.conditional_config is None
        ):
            raise ValueError(
                "conditional_config is required for CONDITIONAL_SAMPLE dependency type"
            )

    @property
    def feature_name(self) -> str:
        """Get the feature name for this dependency."""
        if self.column_name:
            return self.column_name
        prefix_map = {
            DependencyType.PREDICTION: "pred",
            DependencyType.PROBA: "proba",
            DependencyType.TRANSFORM: "trans",
            DependencyType.FEATURE: "feat",
            DependencyType.BASE_MARGIN: "margin",
            DependencyType.CONDITIONAL_SAMPLE: "cond",
        }
        prefix = prefix_map.get(self.dep_type, "out")
        return f"{prefix}_{self.source}"

    def __repr__(self) -> str:
        return f"Edge({self.source} -> {self.target}, type={self.dep_type.value})"

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.dep_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DependencyEdge):
            return NotImplemented
        return (
            self.source == other.source
            and self.target == other.target
            and self.dep_type == other.dep_type
        )
