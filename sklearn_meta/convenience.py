"""Convenience helpers for common sklearn-meta workflows.

These are thin wrappers over GraphBuilder, RunConfigBuilder, and fit().
They reduce boilerplate for the most common use cases while returning
the same ``TrainingRun`` objects as the full API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type, Union, TYPE_CHECKING

import pandas as pd
from sklearn.base import ClassifierMixin

from sklearn_meta.artifacts.training import TrainingRun

if TYPE_CHECKING:
    from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.runtime.config import CVConfig, CVStrategy, RunConfig, TuningConfig
from sklearn_meta.spec.dependency import DependencyType
from sklearn_meta.spec.node import OutputType
from sklearn_meta.spec.builder import GraphBuilder, NodeBuilder


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _apply_params(node: NodeBuilder, params: Dict[str, Any]) -> None:
    """Apply a params dict to a NodeBuilder.

    Accepted value formats (mirrors NodeBuilder.param):
        ``(low, high)``                 -- numeric range
        ``(low, high, {"log": True})``  -- numeric range with options
        ``[choice1, choice2, ...]``     -- categorical
    """
    for name, spec in params.items():
        if isinstance(spec, list):
            node.param(name, spec)
        elif isinstance(spec, tuple):
            if len(spec) == 2:
                node.param(name, spec[0], spec[1])
            elif len(spec) == 3 and isinstance(spec[2], dict):
                node.param(name, spec[0], spec[1], **spec[2])
            else:
                raise ValueError(
                    f"Invalid param spec for '{name}': expected "
                    f"(low, high) or (low, high, {{options}}), got {spec!r}"
                )
        else:
            raise TypeError(
                f"Invalid param spec for '{name}': expected a tuple "
                f"(low, high), list [choices], or (low, high, {{options}}), "
                f"got {type(spec).__name__}"
            )


def _resolve_cv(
    cv: Union[int, CVConfig],
    strategy: Optional[Union[str, CVStrategy]],
    estimator_class: Type,
) -> CVConfig:
    """Build a CVConfig from the ``cv`` and ``strategy`` arguments."""
    if isinstance(cv, CVConfig):
        return cv

    resolved_strategy = _resolve_strategy(strategy, estimator_class)
    return CVConfig(n_splits=cv, strategy=resolved_strategy)


def _resolve_strategy(
    strategy: Optional[Union[str, CVStrategy]],
    estimator_class: Type,
) -> CVStrategy:
    """Resolve an explicit or inferred CV strategy for an estimator."""
    if strategy in (None, "auto"):
        if _is_classifier_estimator(estimator_class):
            return CVStrategy.STRATIFIED
        return CVStrategy.RANDOM

    if isinstance(strategy, str):
        return CVStrategy(strategy)

    return strategy


def _build_run_config(
    cv_config: CVConfig,
    *,
    n_trials: int,
    metric: str,
    tuning_strategy: OptimizationStrategy,
    verbosity: int,
) -> RunConfig:
    """Build a RunConfig while preserving a caller-supplied CVConfig intact."""
    return RunConfig(
        cv=cv_config,
        tuning=TuningConfig(
            n_trials=n_trials,
            metric=metric,
            strategy=tuning_strategy,
        ),
        verbosity=verbosity,
    )


def _normalize_estimator(
    estimator: Any,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> tuple[Type, Dict[str, Any]]:
    """Normalize an estimator class or instance into class + fixed params."""
    resolved_fixed = dict(fixed_params or {})

    if isinstance(estimator, type):
        return estimator, resolved_fixed

    if hasattr(estimator, "fit"):
        if not hasattr(estimator, "get_params"):
            raise TypeError(
                "Estimator instances passed to convenience helpers must "
                "implement get_params()."
            )

        instance_params = estimator.get_params(deep=False)
        instance_params.update(resolved_fixed)
        return estimator.__class__, instance_params

    raise TypeError(
        "Expected an estimator class or fitted-config estimator instance, "
        f"got {type(estimator).__name__}."
    )


def _normalize_model_spec(
    name: str,
    model_spec: Union[Type, tuple, Any],
) -> tuple[Type, Optional[Dict[str, Any]], Dict[str, Any]]:
    """Normalize a flexible model spec into class, params, and fixed params."""
    if isinstance(model_spec, tuple):
        if len(model_spec) == 2:
            estimator, params = model_spec
            estimator_class, resolved_fixed = _normalize_estimator(estimator)
            return estimator_class, params, resolved_fixed
        if len(model_spec) == 3:
            estimator, params, fixed = model_spec
            estimator_class, resolved_fixed = _normalize_estimator(estimator, fixed)
            return estimator_class, params, resolved_fixed
        raise ValueError(
            f"Invalid model spec for '{name}': tuple must have "
            f"2 or 3 elements, got {len(model_spec)}"
        )

    estimator_class, resolved_fixed = _normalize_estimator(model_spec)
    return estimator_class, None, resolved_fixed


def _model_class_from_spec(model_spec: Union[Type, tuple, Any]) -> Type:
    """Extract the estimator class from a convenience model spec."""
    estimator_class, _, _ = _normalize_model_spec("model", model_spec)
    return estimator_class


def _is_classifier_estimator(estimator_class: Type) -> bool:
    """Whether an estimator class should default to classifier behavior."""
    return isinstance(estimator_class, type) and issubclass(
        estimator_class,
        ClassifierMixin,
    )


def _resolve_stack_base_output_type(
    name: str,
    estimator_class: Type,
    fixed_params: Dict[str, Any],
    stack_output: Literal["auto", "prediction", "proba"],
) -> OutputType:
    """Resolve the output type a base stack node should expose."""
    supports_proba = _supports_probability_output(estimator_class, fixed_params)

    if stack_output == "prediction":
        return OutputType.PREDICTION

    if stack_output == "proba":
        if not supports_proba:
            raise ValueError(
                f"Base model '{name}' does not support predict_proba(), so "
                "stack_output='proba' is invalid for this stack."
            )
        return OutputType.PROBA

    if stack_output == "auto":
        if _is_classifier_estimator(estimator_class) and supports_proba:
            return OutputType.PROBA
        return OutputType.PREDICTION

    raise ValueError(
        "stack_output must be one of 'auto', 'prediction', or 'proba', "
        f"got {stack_output!r}."
    )


def _supports_probability_output(
    estimator_class: Type,
    fixed_params: Dict[str, Any],
) -> bool:
    """Whether a configured estimator instance exposes predict_proba()."""
    try:
        estimator = estimator_class(**fixed_params)
    except Exception:
        return hasattr(estimator_class, "predict_proba")
    return hasattr(estimator, "predict_proba")


def _fit_graph(
    graph,
    X: pd.DataFrame,
    y: Any,
    config: RunConfig,
    *,
    groups: Any = None,
    services: Optional[RuntimeServices] = None,
) -> TrainingRun:
    """Route convenience helpers through the public fit() entry point."""
    from sklearn_meta import fit as fit_graph

    return fit_graph(graph, X, y, config, groups=groups, services=services)


class ComparisonResult:
    """Result bundle returned by :func:`compare`.

    Supports dict-style access: ``result["rf"]`` returns the
    ``TrainingRun`` for the model named ``"rf"``.
    """

    def __init__(self, runs: Dict[str, TrainingRun], metric: str) -> None:
        self.runs = runs
        self.metric = metric

        # Pre-compute ranking once
        first_run = next(iter(runs.values()))
        ascending = not bool(first_run.config.tuning.greater_is_better)
        scored = sorted(
            ((name, run.best_score_) for name, run in runs.items()),
            key=lambda x: x[1],
            reverse=not ascending,
        )
        self._rankings: List[tuple[str, float]] = scored

    # -- Quick accessors ---------------------------------------------------

    @property
    def rankings(self) -> List[tuple[str, float]]:
        """Ordered list of ``(model_name, score)`` tuples, best first."""
        return list(self._rankings)

    @property
    def best_name(self) -> str:
        """Name of the best-scoring model."""
        return self._rankings[0][0]

    @property
    def best_run(self) -> TrainingRun:
        """``TrainingRun`` for the best-scoring model."""
        return self.runs[self.best_name]

    # -- DataFrame view ----------------------------------------------------

    @property
    def leaderboard(self) -> pd.DataFrame:
        """Sorted comparison table (model, score, time)."""
        rows = [
            {"model": name, "score": score, "time": self.runs[name].total_time}
            for name, score in self._rankings
        ]
        return pd.DataFrame(rows)

    def to_frame(self) -> pd.DataFrame:
        """Alias for :attr:`leaderboard`."""
        return self.leaderboard.copy()

    # -- Dict-like access --------------------------------------------------

    def __getitem__(self, name: str) -> TrainingRun:
        try:
            return self.runs[name]
        except KeyError:
            raise KeyError(
                f"No model named '{name}'. Available: {list(self.runs)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.runs

    def __len__(self) -> int:
        return len(self.runs)

    def __repr__(self) -> str:
        header = f"ComparisonResult(metric={self.metric!r}, n_models={len(self)})"
        ranking_lines = [
            f"  {i+1}. {name}: {score:.4f}"
            for i, (name, score) in enumerate(self._rankings)
        ]
        return header + "\n" + "\n".join(ranking_lines)


def _build_single_model_run(
    estimator: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    params: Optional[Dict[str, Any]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    n_trials: int = 100,
    metric: str = "neg_mean_squared_error",
    cv: Union[int, CVConfig] = 5,
    strategy: Optional[Union[str, CVStrategy]] = None,
    groups: Any = None,
    verbosity: int = 1,
    services: Optional[RuntimeServices] = None,
) -> TrainingRun:
    """Shared implementation for tune() and cross_validate()."""
    estimator_class, resolved_fixed_params = _normalize_estimator(
        estimator,
        fixed_params,
    )
    builder = GraphBuilder().add_model("model", estimator_class)

    if params:
        _apply_params(builder, params)
    if resolved_fixed_params:
        builder.fixed_params(**resolved_fixed_params)

    graph = builder.build()

    tuning_strategy = (
        OptimizationStrategy.NONE if n_trials == 0
        else OptimizationStrategy.LAYER_BY_LAYER
    )

    cv_config = _resolve_cv(cv, strategy, estimator_class)
    config = _build_run_config(
        cv_config,
        n_trials=n_trials,
        metric=metric,
        tuning_strategy=tuning_strategy,
        verbosity=verbosity,
    )

    return _fit_graph(graph, X, y, config, groups=groups, services=services)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def tune(
    estimator_class: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    params: Optional[Dict[str, Any]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    n_trials: int = 100,
    metric: str = "neg_mean_squared_error",
    cv: Union[int, CVConfig] = 5,
    strategy: Optional[Union[str, CVStrategy]] = None,
    groups: Any = None,
    verbosity: int = 1,
    services: Optional[RuntimeServices] = None,
) -> TrainingRun:
    """Tune a single model with cross-validated hyperparameter search.

    This is a convenience wrapper around ``GraphBuilder`` +
    ``RunConfigBuilder`` + ``fit()`` for the most common workflow:
    tuning one estimator.

    Args:
        estimator_class: sklearn-compatible estimator class or instance.
        X: Feature DataFrame.
        y: Target array.
        params: Search space as a dict.  Values can be:
            - ``(low, high)`` for a numeric range
            - ``(low, high, {"log": True})`` for range with options
            - ``[choice1, choice2, ...]`` for categorical
        fixed_params: Non-tuned parameters passed to the estimator.
        n_trials: Number of Optuna trials.
        metric: sklearn scorer name (e.g. ``"roc_auc"``, ``"accuracy"``).
        cv: Number of CV folds or a ``CVConfig``.
        strategy: CV strategy (``"stratified"``, ``"random"``, etc.).
            If omitted, defaults to ``"stratified"`` for classifiers and
            ``"random"`` otherwise. Ignored when ``cv`` is a ``CVConfig``.
        groups: Group labels for group CV.
        verbosity: Logging verbosity (0 = silent).

    Returns:
        A ``TrainingRun`` with ``.predict()``, ``.best_params_``,
        ``.best_score_``, etc.

    Example::

        from sklearn_meta import tune

        result = tune(
            RandomForestClassifier,
            X_train, y_train,
            params={"n_estimators": (50, 500), "max_depth": (3, 20)},
            fixed_params={"random_state": 42},
            n_trials=100,
            metric="accuracy",
        )
        result.best_params_
        result.predict(X_test)
    """
    if params is None:
        raise ValueError(
            "tune() requires a params dict defining the search space. "
            "Use cross_validate() for CV without tuning."
        )
    return _build_single_model_run(
        estimator_class, X, y,
        params=params,
        fixed_params=fixed_params,
        n_trials=n_trials,
        metric=metric,
        cv=cv,
        strategy=strategy,
        groups=groups,
        verbosity=verbosity,
        services=services,
    )


def cross_validate(
    estimator_class: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    fixed_params: Optional[Dict[str, Any]] = None,
    metric: str = "neg_mean_squared_error",
    cv: Union[int, CVConfig] = 5,
    strategy: Optional[Union[str, CVStrategy]] = None,
    groups: Any = None,
    verbosity: int = 1,
    services: Optional[RuntimeServices] = None,
) -> TrainingRun:
    """Cross-validate a single model with fixed hyperparameters.

    Like :func:`tune` but without hyperparameter search.  Useful for
    getting OOF predictions, fold models, and an inference graph from
    a fixed configuration.

    Args:
        estimator_class: sklearn-compatible estimator class or instance.
        X: Feature DataFrame.
        y: Target array.
        fixed_params: Parameters passed to the estimator.
        metric: sklearn scorer name for evaluation.
        cv: Number of CV folds or a ``CVConfig``.
        strategy: CV strategy. If omitted, defaults to ``"stratified"``
            for classifiers and ``"random"`` otherwise. Ignored when
            ``cv`` is a ``CVConfig``.
        groups: Group labels for group CV.
        verbosity: Logging verbosity (0 = silent).

    Returns:
        A ``TrainingRun`` with ``.predict()``, ``.best_score_``,
        ``.oof_predictions_``, etc.

    Example::

        from sklearn_meta import cross_validate

        result = cross_validate(
            RandomForestClassifier,
            X_train, y_train,
            fixed_params={"n_estimators": 100, "random_state": 42},
            metric="accuracy",
        )
        result.best_score_
        result.oof_predictions_
    """
    return _build_single_model_run(
        estimator_class, X, y,
        params=None,
        fixed_params=fixed_params,
        n_trials=0,
        metric=metric,
        cv=cv,
        strategy=strategy,
        groups=groups,
        verbosity=verbosity,
        services=services,
    )


def stack(
    base_models: Dict[str, Union[Type, tuple, Any]],
    meta_model: Union[Type, tuple, Any],
    X: pd.DataFrame,
    y: Any,
    *,
    metric: str = "neg_mean_squared_error",
    n_trials: int = 50,
    cv: Union[int, CVConfig] = 5,
    stack_output: Literal["auto", "prediction", "proba"] = "auto",
    strategy: Optional[Union[str, CVStrategy]] = None,
    groups: Any = None,
    verbosity: int = 1,
    services: Optional[RuntimeServices] = None,
) -> TrainingRun:
    """Build and fit a stacking ensemble in one call.

    Base model predictions are stacked as features for the meta-learner,
    with automatic out-of-fold handling to prevent data leakage.

    Args:
        base_models: Dict mapping model names to either:
            - An estimator class (no tuning, uses defaults)
            - An estimator instance (uses its configured params as fixed params)
            - A tuple of ``(estimator_class, params_dict)`` for tuning.
              See :func:`tune` for the ``params_dict`` format.
            - A tuple of ``(estimator_or_instance, params_dict, fixed_params_dict)``.
        meta_model: Meta-learner, same format as base_models values.
        X: Feature DataFrame.
        y: Target array.
        metric: sklearn scorer name.
        n_trials: Number of Optuna trials per model.
        cv: Number of CV folds or a ``CVConfig``.
        stack_output: Output exposed by base models to the meta-learner.
            ``"auto"`` uses probabilities for classifiers when available and
            predictions otherwise. ``"prediction"`` always stacks labels or
            regression outputs. ``"proba"`` requires all base models to
            support ``predict_proba()``.
        strategy: CV strategy. If omitted, defaults to ``"stratified"``
            for classifier meta-learners and ``"random"`` otherwise.
            Ignored when ``cv`` is a ``CVConfig``.
        groups: Group labels for group CV.
        verbosity: Logging verbosity (0 = silent).

    Returns:
        A ``TrainingRun``.  The meta-learner is the leaf node, so
        ``.predict()`` returns the ensemble prediction and
        ``.best_score_`` returns the meta-learner's CV score.

    Example::

        from sklearn_meta import stack

        result = stack(
            base_models={
                "rf": (RandomForestClassifier, {"n_estimators": (50, 500)}),
                "xgb": (XGBClassifier, {"learning_rate": (0.01, 0.3, {"log": True})}),
            },
            meta_model=LogisticRegression,
            X=X_train, y=y_train,
            metric="accuracy",
            n_trials=50,
        )
        result.predict(X_test)
    """
    builder = GraphBuilder()
    base_names: List[str] = []
    base_output_types: Dict[str, OutputType] = {}

    for name, model_spec in base_models.items():
        base_names.append(name)
        estimator_class, params, fixed_params = _normalize_model_spec(name, model_spec)
        output_type = _resolve_stack_base_output_type(
            name, estimator_class, fixed_params, stack_output,
        )
        base_output_types[name] = output_type
        _add_model_from_normalized(
            builder, name, estimator_class, params, fixed_params,
            output_type=output_type,
        )

    # Meta-learner
    meta_node = _add_model_from_spec(builder, "meta", meta_model)
    for base_name in base_names:
        dep_type = (
            DependencyType.PROBA
            if base_output_types[base_name] == OutputType.PROBA
            else DependencyType.PREDICTION
        )
        meta_node.depends_on(base_name, dep_type=dep_type)

    graph = builder.build()

    cv_config = _resolve_cv(
        cv,
        strategy,
        _model_class_from_spec(meta_model),
    )
    config = _build_run_config(
        cv_config,
        n_trials=n_trials,
        metric=metric,
        tuning_strategy=OptimizationStrategy.LAYER_BY_LAYER,
        verbosity=verbosity,
    )

    return _fit_graph(graph, X, y, config, groups=groups, services=services)


def compare(
    models: Dict[str, Union[Type, tuple, Any]],
    X: pd.DataFrame,
    y: Any,
    *,
    metric: str = "neg_mean_squared_error",
    n_trials: int = 100,
    cv: Union[int, CVConfig] = 5,
    strategy: Optional[Union[str, CVStrategy]] = None,
    groups: Any = None,
    verbosity: int = 1,
    services: Optional[RuntimeServices] = None,
) -> ComparisonResult:
    """Fit several single-model specs and return a sortable leaderboard.

    Accepted model formats mirror :func:`tune`/:func:`cross_validate`:
        ``EstimatorClass``
        ``EstimatorInstance``
        ``(EstimatorOrInstance, params_dict)``
        ``(EstimatorOrInstance, params_dict, fixed_params_dict)``
    """
    if not models:
        raise ValueError("compare() requires at least one model.")

    runs: Dict[str, TrainingRun] = {}
    for name, model_spec in models.items():
        estimator_class, params, fixed_params = _normalize_model_spec(name, model_spec)
        runs[name] = _build_single_model_run(
            estimator_class,
            X,
            y,
            params=params,
            fixed_params=fixed_params,
            n_trials=n_trials if params else 0,
            metric=metric,
            cv=cv,
            strategy=strategy,
            groups=groups,
            verbosity=verbosity,
            services=services,
        )

    return ComparisonResult(runs=runs, metric=metric)


def _add_model_from_normalized(
    builder: GraphBuilder,
    name: str,
    estimator_class: Type,
    params: Optional[Dict[str, Any]],
    fixed_params: Dict[str, Any],
    *,
    output_type: Optional[OutputType] = None,
) -> NodeBuilder:
    """Add a model to a GraphBuilder from already-normalized components."""
    node = builder.add_model(name, estimator_class)
    if output_type is not None:
        node.output_type(output_type)
    if params:
        _apply_params(node, params)
    if fixed_params:
        node.fixed_params(**fixed_params)
    return node


def _add_model_from_spec(
    builder: GraphBuilder,
    name: str,
    model_spec: Union[Type, tuple, Any],
    *,
    output_type: Optional[OutputType] = None,
) -> NodeBuilder:
    """Add a model to a GraphBuilder from a flexible spec format."""
    estimator_class, params, fixed_params = _normalize_model_spec(name, model_spec)
    return _add_model_from_normalized(
        builder, name, estimator_class, params, fixed_params,
        output_type=output_type,
    )
