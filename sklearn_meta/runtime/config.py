"""RunConfig and supporting configuration types."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from sklearn_meta.engine.estimator_scaling import EstimatorScalingConfig
from sklearn_meta.engine.strategy import OptimizationStrategy


def _serialize_custom_reparameterization(reparameterization: Any) -> Dict[str, Any]:
    """Serialize a custom reparameterization to JSON-safe state."""
    from sklearn_meta.persistence.manifest import to_json_safe

    from sklearn_meta.spec._resolve import get_class_path

    cls = reparameterization.__class__
    return {
        "kind": "reparameterization",
        "class_path": get_class_path(cls),
        "state": to_json_safe(
            dict(reparameterization.__dict__),
            path=f"{cls.__name__}.state",
        ),
    }


def _deserialize_custom_reparameterization(data: Any) -> Any:
    """Reconstruct a custom reparameterization from serialized state."""
    if not isinstance(data, dict) or data.get("kind") != "reparameterization":
        return data

    from sklearn_meta.spec._resolve import resolve_class_path

    reparameterization_cls = resolve_class_path(data["class_path"])
    reparameterization = reparameterization_cls.__new__(reparameterization_cls)
    reparameterization.__dict__.update(dict(data.get("state", {})))
    return reparameterization


def _infer_greater_is_better(metric: str) -> bool:
    """Infer score direction from a sklearn metric name.

    All standard sklearn scorers follow the higher-is-better convention
    (loss metrics are already negated, e.g. ``neg_mean_squared_error``).
    For unknown metrics, warn and default to ``False``.
    """
    try:
        from sklearn.metrics import get_scorer_names
        known = set(get_scorer_names())
    except ImportError:
        from sklearn.metrics import SCORERS  # type: ignore[attr-defined]
        known = set(SCORERS.keys())

    if metric in known:
        return True

    warnings.warn(
        f"Unknown metric '{metric}': defaulting to greater_is_better=False. "
        f"Pass greater_is_better explicitly to suppress this warning.",
        UserWarning,
        stacklevel=4,
    )
    return False


# ---------------------------------------------------------------------------
# CV types (moved from core/data/cv.py)
# ---------------------------------------------------------------------------

class CVStrategy(Enum):
    """Cross-validation splitting strategies."""
    GROUP = "group"
    STRATIFIED_GROUP = "stratified_group"
    STRATIFIED = "stratified"
    RANDOM = "random"
    TIME_SERIES = "time_series"


@dataclass
class CVFold:
    """Represents a single cross-validation fold."""
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    repeat_idx: int = 0

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_val(self) -> int:
        return len(self.val_indices)

    def __repr__(self) -> str:
        return (
            f"CVFold(fold={self.fold_idx}, repeat={self.repeat_idx}, "
            f"n_train={self.n_train}, n_val={self.n_val})"
        )


@dataclass
class NestedCVFold:
    """Nested cross-validation fold structure."""
    outer_fold: CVFold
    inner_folds: List[CVFold]

    @property
    def fold_idx(self) -> int:
        return self.outer_fold.fold_idx

    @property
    def n_inner_folds(self) -> int:
        return len(self.inner_folds)

    def __repr__(self) -> str:
        return (
            f"NestedCVFold(outer={self.outer_fold}, "
            f"n_inner_folds={self.n_inner_folds})"
        )


@dataclass
class CVConfig:
    """Cross-validation configuration."""
    n_splits: int = 5
    n_repeats: int = 1
    strategy: CVStrategy = CVStrategy.GROUP
    shuffle: bool = True
    random_state: int = 42
    inner_cv: Optional[CVConfig] = None

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {self.n_repeats}")
        if isinstance(self.strategy, str):
            self.strategy = CVStrategy(self.strategy)

    @property
    def is_nested(self) -> bool:
        return self.inner_cv is not None

    @property
    def total_folds(self) -> int:
        return self.n_splits * self.n_repeats

    def with_inner_cv(
        self,
        n_splits: int = 3,
        strategy: Optional[CVStrategy] = None,
    ) -> CVConfig:
        inner_strategy = strategy if strategy is not None else self.strategy
        return CVConfig(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            strategy=self.strategy,
            shuffle=self.shuffle,
            random_state=self.random_state,
            inner_cv=CVConfig(
                n_splits=n_splits,
                n_repeats=1,
                strategy=inner_strategy,
                shuffle=self.shuffle,
                random_state=self.random_state + 1,
            ),
        )

    def __repr__(self) -> str:
        nested_str = f", inner_cv={self.inner_cv}" if self.is_nested else ""
        return (
            f"CVConfig(n_splits={self.n_splits}, n_repeats={self.n_repeats}, "
            f"strategy={self.strategy.value}{nested_str})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
            "strategy": self.strategy.value,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "inner_cv": self.inner_cv.to_dict() if self.inner_cv is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CVConfig":
        inner_cv = data.get("inner_cv")
        return cls(
            n_splits=data.get("n_splits", 5),
            n_repeats=data.get("n_repeats", 1),
            strategy=data.get("strategy", CVStrategy.GROUP),
            shuffle=data.get("shuffle", True),
            random_state=data.get("random_state", 42),
            inner_cv=cls.from_dict(inner_cv) if inner_cv is not None else None,
        )


@dataclass
class FoldResult:
    """Result from fitting a model on a single fold."""
    fold: CVFold
    model: object
    val_predictions: np.ndarray
    val_score: float
    train_score: Optional[float] = None
    fit_time: float = 0.0
    predict_time: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def fold_idx(self) -> int:
        return self.fold.fold_idx

    def __repr__(self) -> str:
        return (
            f"FoldResult(fold={self.fold_idx}, val_score={self.val_score:.4f}, "
            f"fit_time={self.fit_time:.2f}s)"
        )


@dataclass
class CVResult:
    """Aggregated results from cross-validation."""
    fold_results: List[FoldResult]
    oof_predictions: np.ndarray
    node_name: str
    repeat_oof: Optional[np.ndarray] = None  # (n_repeats, n_samples, ...) when n_repeats > 1

    @property
    def n_folds(self) -> int:
        return len(self.fold_results)

    @property
    def val_scores(self) -> np.ndarray:
        return np.array([r.val_score for r in self.fold_results])

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.val_scores))

    @property
    def std_score(self) -> float:
        return float(np.std(self.val_scores))

    @property
    def total_fit_time(self) -> float:
        return sum(r.fit_time for r in self.fold_results)

    @property
    def models(self) -> List[object]:
        return [r.model for r in self.fold_results]

    def __repr__(self) -> str:
        return (
            f"CVResult(node={self.node_name}, n_folds={self.n_folds}, "
            f"mean_score={self.mean_score:.4f} +/- {self.std_score:.4f})"
        )


# ---------------------------------------------------------------------------
# Feature selection config (moved from selection/selector.py)
# ---------------------------------------------------------------------------

class FeatureSelectionMethod(str, Enum):
    SHADOW = "shadow"
    PERMUTATION = "permutation"
    THRESHOLD = "threshold"


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""
    enabled: bool = True
    method: FeatureSelectionMethod = FeatureSelectionMethod.SHADOW
    n_shadows: int = 5
    threshold_mult: float = 1.414
    threshold_percentile: float = 10.0
    retune_after_pruning: bool = True
    min_features: int = 1
    max_features: Optional[int] = None
    random_state: int = 42
    feature_groups: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method.value,
            "n_shadows": self.n_shadows,
            "threshold_mult": self.threshold_mult,
            "threshold_percentile": self.threshold_percentile,
            "retune_after_pruning": self.retune_after_pruning,
            "min_features": self.min_features,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "feature_groups": self.feature_groups,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSelectionConfig":
        method = data.get("method", FeatureSelectionMethod.SHADOW)
        if isinstance(method, str):
            method = FeatureSelectionMethod(method)
        return cls(
            enabled=data.get("enabled", True),
            method=method,
            n_shadows=data.get("n_shadows", 5),
            threshold_mult=data.get("threshold_mult", 1.414),
            threshold_percentile=data.get("threshold_percentile", 10.0),
            retune_after_pruning=data.get("retune_after_pruning", True),
            min_features=data.get("min_features", 1),
            max_features=data.get("max_features"),
            random_state=data.get("random_state", 42),
            feature_groups=data.get("feature_groups"),
        )


# ---------------------------------------------------------------------------
# Tuning config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TuningConfig:
    """Configuration for hyperparameter tuning.

    If ``greater_is_better`` is not specified, it is inferred from the metric
    name.  All standard sklearn scorer names (e.g. ``"roc_auc"``,
    ``"neg_mean_squared_error"``) follow the higher-is-better convention.
    For unknown metrics a warning is issued and ``False`` is used.
    """
    n_trials: int = 100
    timeout: Optional[float] = None
    early_stopping_rounds: Optional[int] = None
    metric: str = "neg_mean_squared_error"
    greater_is_better: Optional[bool] = None
    strategy: OptimizationStrategy = OptimizationStrategy.LAYER_BY_LAYER
    show_progress: bool = False

    def __post_init__(self) -> None:
        if self.greater_is_better is None:
            object.__setattr__(
                self, "greater_is_better", _infer_greater_is_better(self.metric)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "early_stopping_rounds": self.early_stopping_rounds,
            "metric": self.metric,
            "greater_is_better": self.greater_is_better,
            "strategy": self.strategy.value,
            "show_progress": self.show_progress,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TuningConfig":
        strategy = data.get("strategy", OptimizationStrategy.LAYER_BY_LAYER)
        if isinstance(strategy, str):
            strategy = OptimizationStrategy(strategy)
        return cls(
            n_trials=data.get("n_trials", 100),
            timeout=data.get("timeout"),
            early_stopping_rounds=data.get("early_stopping_rounds"),
            metric=data.get("metric", "neg_mean_squared_error"),
            greater_is_better=data.get("greater_is_better"),
            strategy=strategy,
            show_progress=data.get("show_progress", False),
        )


# ---------------------------------------------------------------------------
# Reparameterization config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReparameterizationConfig:
    """Configuration for hyperparameter reparameterization."""
    enabled: bool = True
    use_prebaked: bool = True
    custom_reparameterizations: tuple = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "use_prebaked": self.use_prebaked,
            "custom_reparameterizations": [
                _serialize_custom_reparameterization(reparameterization)
                for reparameterization in self.custom_reparameterizations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReparameterizationConfig":
        return cls(
            enabled=data.get("enabled", True),
            use_prebaked=data.get("use_prebaked", True),
            custom_reparameterizations=tuple(
                _deserialize_custom_reparameterization(item)
                for item in data.get("custom_reparameterizations", ())
            ),
        )


# ---------------------------------------------------------------------------
# Unified RunConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """Unified configuration for a training run."""
    cv: CVConfig = field(default_factory=CVConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    feature_selection: Optional[FeatureSelectionConfig] = None
    reparameterization: Optional[ReparameterizationConfig] = None
    estimator_scaling: Optional[EstimatorScalingConfig] = None
    verbosity: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cv": self.cv.to_dict(),
            "tuning": self.tuning.to_dict(),
            "feature_selection": (
                self.feature_selection.to_dict()
                if self.feature_selection is not None else None
            ),
            "reparameterization": (
                self.reparameterization.to_dict()
                if self.reparameterization is not None else None
            ),
            "estimator_scaling": (
                self.estimator_scaling.to_dict()
                if self.estimator_scaling is not None else None
            ),
            "verbosity": self.verbosity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        return cls(
            cv=CVConfig.from_dict(data.get("cv", {})),
            tuning=TuningConfig.from_dict(data.get("tuning", {})),
            feature_selection=(
                FeatureSelectionConfig.from_dict(data["feature_selection"])
                if data.get("feature_selection") is not None else None
            ),
            reparameterization=(
                ReparameterizationConfig.from_dict(data["reparameterization"])
                if data.get("reparameterization") is not None else None
            ),
            estimator_scaling=(
                EstimatorScalingConfig.from_dict(data["estimator_scaling"])
                if data.get("estimator_scaling") is not None else None
            ),
            verbosity=data.get("verbosity", 1),
        )


# ---------------------------------------------------------------------------
# RunConfigBuilder
# ---------------------------------------------------------------------------

class RunConfigBuilder:
    """Fluent builder for RunConfig."""

    def __init__(self) -> None:
        self._cv_kwargs: Dict[str, Any] = {}
        self._tuning_kwargs: Dict[str, Any] = {}
        self._feature_selection_config: Optional[FeatureSelectionConfig] = None
        self._reparam_config: Optional[ReparameterizationConfig] = None
        self._scaling_config: Optional[EstimatorScalingConfig] = None
        self._verbosity: int = 1

    def cv(
        self,
        n_splits: int = 5,
        n_repeats: int = 1,
        strategy: CVStrategy | str = CVStrategy.GROUP,
        shuffle: bool = True,
        random_state: int = 42,
        inner_cv: int | CVConfig | None = None,
        inner_strategy: CVStrategy | str | None = None,
    ) -> RunConfigBuilder:
        if isinstance(strategy, str):
            strategy = CVStrategy(strategy)

        resolved_inner_cv: Optional[CVConfig] = None
        if inner_cv is not None:
            if isinstance(inner_strategy, str):
                inner_strategy = CVStrategy(inner_strategy)

            if isinstance(inner_cv, CVConfig):
                resolved_inner_cv = inner_cv
            else:
                resolved_inner_cv = CVConfig(
                    n_splits=inner_cv,
                    n_repeats=1,
                    strategy=inner_strategy if inner_strategy is not None else strategy,
                    shuffle=shuffle,
                    random_state=random_state + 1,
                )

        self._cv_kwargs = dict(
            n_splits=n_splits,
            n_repeats=n_repeats,
            strategy=strategy,
            shuffle=shuffle,
            random_state=random_state,
            inner_cv=resolved_inner_cv,
        )
        return self

    def tuning(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        early_stopping_rounds: Optional[int] = None,
        metric: str = "neg_mean_squared_error",
        greater_is_better: Optional[bool] = None,
        strategy: OptimizationStrategy | str = OptimizationStrategy.LAYER_BY_LAYER,
        show_progress: bool = False,
    ) -> RunConfigBuilder:
        if isinstance(strategy, str):
            strategy = OptimizationStrategy(strategy)
        self._tuning_kwargs = dict(
            n_trials=n_trials,
            timeout=timeout,
            early_stopping_rounds=early_stopping_rounds,
            metric=metric,
            greater_is_better=greater_is_better,
            strategy=strategy,
            show_progress=show_progress,
        )
        return self

    def feature_selection(
        self,
        method: str = "shadow",
        n_shadows: int = 5,
        threshold_mult: float = 1.414,
        retune_after_pruning: bool = True,
        min_features: int = 1,
        max_features: Optional[int] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ) -> RunConfigBuilder:
        self._feature_selection_config = FeatureSelectionConfig(
            enabled=True,
            method=FeatureSelectionMethod(method),
            n_shadows=n_shadows,
            threshold_mult=threshold_mult,
            retune_after_pruning=retune_after_pruning,
            min_features=min_features,
            max_features=max_features,
            feature_groups=feature_groups,
        )
        return self

    def reparameterization(
        self,
        enabled: bool = True,
        use_prebaked: bool = True,
        custom_reparameterizations: list | None = None,
    ) -> RunConfigBuilder:
        self._reparam_config = ReparameterizationConfig(
            enabled=enabled,
            use_prebaked=use_prebaked,
            custom_reparameterizations=tuple(custom_reparameterizations or []),
        )
        return self

    def estimator_scaling(
        self,
        tuning_n_estimators: Optional[int] = None,
        final_n_estimators: Optional[int] = None,
        scaling_search: bool = False,
        scaling_factors: Optional[List[float]] = None,
        scaling_estimators: Optional[List[int]] = None,
    ) -> RunConfigBuilder:
        self._scaling_config = EstimatorScalingConfig(
            tuning_n_estimators=tuning_n_estimators,
            final_n_estimators=final_n_estimators,
            scaling_search=scaling_search,
            scaling_factors=scaling_factors,
            scaling_estimators=scaling_estimators,
        )
        return self

    def verbosity(self, level: int) -> RunConfigBuilder:
        self._verbosity = level
        return self

    def build(self) -> RunConfig:
        return RunConfig(
            cv=CVConfig(**self._cv_kwargs) if self._cv_kwargs else CVConfig(),
            tuning=TuningConfig(**self._tuning_kwargs) if self._tuning_kwargs else TuningConfig(),
            feature_selection=self._feature_selection_config,
            reparameterization=self._reparam_config,
            estimator_scaling=self._scaling_config,
            verbosity=self._verbosity,
        )
