"""NodeSpec: Definition of a single model in the graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn_meta.spec.distillation import DistillationConfig
    from sklearn_meta.search.space import SearchSpace


class OutputType(str, Enum):
    PREDICTION = "prediction"
    PROBA = "proba"
    TRANSFORM = "transform"
    QUANTILES = "quantiles"


@dataclass
class NodeSpec:
    """
    Definition of a single model in the graph.

    This class defines what model to use and how to configure it,
    but contains no training logic. Training is handled by the engine.
    """

    name: str
    estimator_class: Type
    search_space: Optional[SearchSpace] = None
    output_type: OutputType = OutputType.PREDICTION
    condition: Optional[Callable[..., bool]] = None
    plugins: List[str] = field(default_factory=list)
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    fit_params: Dict[str, Any] = field(default_factory=dict)
    feature_cols: Optional[List[str]] = None
    description: str = ""
    distillation_config: Optional[DistillationConfig] = None

    def __post_init__(self) -> None:
        """Validate node configuration."""
        if not self.name:
            raise ValueError("Node name cannot be empty")
        if not self.estimator_class:
            raise ValueError("Estimator class is required")
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

        if self.distillation_config is not None:
            from sklearn_meta.spec.distillation import validate_distillation_estimator
            validate_distillation_estimator(self.estimator_class)

    @property
    def is_distilled(self) -> bool:
        """Whether this node uses knowledge distillation."""
        return self.distillation_config is not None

    @property
    def has_search_space(self) -> bool:
        """Whether this node has hyperparameters to tune."""
        return self.search_space is not None and len(self.search_space) > 0

    @property
    def is_conditional(self) -> bool:
        """Whether this node has a condition for execution."""
        return self.condition is not None

    def should_run(self, data) -> bool:
        """Check if this node should run given the current data."""
        if self.condition is None:
            return True
        return self.condition(data)

    def create_estimator(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Create an instance of the estimator with given parameters."""
        from sklearn_meta.engine.estimator_factory import create_estimator as _create
        return _create(self, params)

    def get_output(self, model: Any, X) -> Any:
        """Get the output from a fitted model based on output_type."""
        from sklearn_meta.engine.estimator_factory import get_output as _get
        return _get(self, model, X)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this NodeSpec to a JSON-safe dictionary."""
        cls = self.estimator_class
        estimator_class_str = f"{cls.__module__}.{cls.__qualname__}"

        result: Dict[str, Any] = {
            "name": self.name,
            "estimator_class": estimator_class_str,
            "output_type": self.output_type.value,
            "fixed_params": self.fixed_params,
            "fit_params": self.fit_params,
            "feature_cols": self.feature_cols,
            "description": self.description,
            "plugins": self.plugins,
        }

        if self.distillation_config is not None:
            result["distillation_config"] = {
                "alpha": self.distillation_config.alpha,
                "temperature": self.distillation_config.temperature,
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeSpec":
        """Reconstruct a NodeSpec from a dictionary produced by to_dict()."""
        import importlib

        class_path = data["estimator_class"]
        path_parts = class_path.split(".")
        estimator_class = None
        for split_idx in range(len(path_parts) - 1, 0, -1):
            module_path = ".".join(path_parts[:split_idx])
            attr_parts = path_parts[split_idx:]
            try:
                obj = importlib.import_module(module_path)
            except ModuleNotFoundError as exc:
                if exc.name != module_path:
                    raise
                continue

            try:
                for attr_name in attr_parts:
                    obj = getattr(obj, attr_name)
            except AttributeError:
                continue

            estimator_class = obj
            break

        if estimator_class is None:
            raise ImportError(
                f"Could not resolve estimator class '{class_path}'"
            )

        distillation_config = None
        if "distillation_config" in data and data["distillation_config"] is not None:
            from sklearn_meta.spec.distillation import DistillationConfig
            distillation_config = DistillationConfig(**data["distillation_config"])

        return cls(
            name=data["name"],
            estimator_class=estimator_class,
            output_type=OutputType(data["output_type"]),
            fixed_params=data.get("fixed_params", {}),
            fit_params=data.get("fit_params", {}),
            feature_cols=data.get("feature_cols"),
            description=data.get("description", ""),
            plugins=data.get("plugins", []),
            distillation_config=distillation_config,
        )

    def __repr__(self) -> str:
        class_name = self.estimator_class.__name__
        return f"NodeSpec(name={self.name!r}, estimator={class_name})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeSpec):
            return NotImplemented
        return self.name == other.name
