"""RunConfig and supporting configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from sklearn_meta.engine.estimator_scaling import EstimatorScalingConfig
from sklearn_meta.engine.strategy import OptimizationStrategy


# ---------------------------------------------------------------------------
# CV types (moved from core/data/cv.py)
# ---------------------------------------------------------------------------

class CVStrategy(Enum):
    """Cross-validation splitting strategies."""
    GROUP = "group"
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


# ---------------------------------------------------------------------------
# Tuning config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    n_trials: int = 100
    timeout: Optional[float] = None
    early_stopping_rounds: Optional[int] = None
    metric: str = "neg_mean_squared_error"
    greater_is_better: bool = False
    strategy: OptimizationStrategy = OptimizationStrategy.LAYER_BY_LAYER
    show_progress: bool = False


# ---------------------------------------------------------------------------
# Reparameterization config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReparameterizationConfig:
    """Configuration for hyperparameter reparameterization."""
    enabled: bool = True
    use_prebaked: bool = True
    custom_reparameterizations: tuple = ()


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
    ) -> RunConfigBuilder:
        if isinstance(strategy, str):
            strategy = CVStrategy(strategy)
        self._cv_kwargs = dict(
            n_splits=n_splits,
            n_repeats=n_repeats,
            strategy=strategy,
            shuffle=shuffle,
            random_state=random_state,
        )
        return self

    def tuning(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        early_stopping_rounds: Optional[int] = None,
        metric: str = "neg_mean_squared_error",
        greater_is_better: bool = False,
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
        scaling_factors: Optional[List[int]] = None,
    ) -> RunConfigBuilder:
        self._scaling_config = EstimatorScalingConfig(
            tuning_n_estimators=tuning_n_estimators,
            final_n_estimators=final_n_estimators,
            scaling_search=scaling_search,
            scaling_factors=scaling_factors,
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
