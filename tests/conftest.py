"""Shared test fixtures for the auto-sklearn test suite."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def classification_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Binary classification dataset with known structure."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def multiclass_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Multiclass classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def regression_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Regression dataset with known noise level."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def grouped_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Dataset with group structure for group CV."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y_series = pd.Series(y, name="target")
    # 100 groups of 10 samples each
    groups = pd.Series(np.repeat(np.arange(100), 10), name="group")
    return X_df, y_series, groups


@pytest.fixture
def small_classification_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Small binary classification dataset for quick tests."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def small_regression_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Small regression dataset for quick tests."""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def time_series_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Time series dataset with temporal ordering."""
    np.random.seed(42)
    n_samples = 500

    # Create time-dependent features
    time = np.arange(n_samples)
    trend = time * 0.01
    seasonality = np.sin(2 * np.pi * time / 50)
    noise = np.random.randn(n_samples) * 0.5

    X = pd.DataFrame({
        "time": time,
        "feature_1": np.random.randn(n_samples) + trend,
        "feature_2": np.random.randn(n_samples) + seasonality,
        "feature_3": np.random.randn(n_samples),
    })
    y = pd.Series(trend + seasonality + noise, name="target")

    return X, y


# =============================================================================
# DataContext Fixtures
# =============================================================================


@pytest.fixture
def data_context(classification_data):
    """DataContext from classification data."""
    from sklearn_meta.core.data.context import DataContext

    X, y = classification_data
    return DataContext(X=X, y=y)


@pytest.fixture
def data_context_with_groups(grouped_data):
    """DataContext with groups for group CV."""
    from sklearn_meta.core.data.context import DataContext

    X, y, groups = grouped_data
    return DataContext(X=X, y=y, groups=groups)


@pytest.fixture
def regression_context(regression_data):
    """DataContext from regression data."""
    from sklearn_meta.core.data.context import DataContext

    X, y = regression_data
    return DataContext(X=X, y=y)


@pytest.fixture
def small_context(small_classification_data):
    """Small DataContext for quick tests."""
    from sklearn_meta.core.data.context import DataContext

    X, y = small_classification_data
    return DataContext(X=X, y=y)


# =============================================================================
# CV Configuration Fixtures
# =============================================================================


@pytest.fixture
def cv_config_stratified():
    """Stratified CV configuration."""
    from sklearn_meta.core.data.cv import CVConfig, CVStrategy

    return CVConfig(
        n_splits=5,
        n_repeats=1,
        strategy=CVStrategy.STRATIFIED,
        shuffle=True,
        random_state=42,
    )


@pytest.fixture
def cv_config_group():
    """Group CV configuration."""
    from sklearn_meta.core.data.cv import CVConfig, CVStrategy

    return CVConfig(
        n_splits=5,
        n_repeats=1,
        strategy=CVStrategy.GROUP,
        shuffle=False,
        random_state=42,
    )


@pytest.fixture
def cv_config_repeated():
    """Repeated CV configuration."""
    from sklearn_meta.core.data.cv import CVConfig, CVStrategy

    return CVConfig(
        n_splits=5,
        n_repeats=3,
        strategy=CVStrategy.STRATIFIED,
        shuffle=True,
        random_state=42,
    )


@pytest.fixture
def cv_config_nested():
    """Nested CV configuration."""
    from sklearn_meta.core.data.cv import CVConfig, CVStrategy

    return CVConfig(
        n_splits=5,
        n_repeats=1,
        strategy=CVStrategy.STRATIFIED,
        shuffle=True,
        random_state=42,
    ).with_inner_cv(n_splits=3)


# =============================================================================
# Model Node Fixtures
# =============================================================================


@pytest.fixture
def rf_classifier_node():
    """ModelNode for Random Forest classifier."""
    from sklearn_meta.core.model.node import ModelNode
    from sklearn_meta.search.space import SearchSpace

    space = SearchSpace()
    space.add_int("n_estimators", 10, 100)
    space.add_int("max_depth", 3, 10)

    return ModelNode(
        name="rf",
        estimator_class=RandomForestClassifier,
        search_space=space,
        fixed_params={"random_state": 42},
    )


@pytest.fixture
def lr_classifier_node():
    """ModelNode for Logistic Regression classifier."""
    from sklearn_meta.core.model.node import ModelNode
    from sklearn_meta.search.space import SearchSpace

    space = SearchSpace()
    space.add_float("C", 0.01, 10.0, log=True)

    return ModelNode(
        name="lr",
        estimator_class=LogisticRegression,
        search_space=space,
        fixed_params={"random_state": 42, "max_iter": 1000},
    )


@pytest.fixture
def rf_regressor_node():
    """ModelNode for Random Forest regressor."""
    from sklearn_meta.core.model.node import ModelNode
    from sklearn_meta.search.space import SearchSpace

    space = SearchSpace()
    space.add_int("n_estimators", 10, 100)
    space.add_int("max_depth", 3, 10)

    return ModelNode(
        name="rf_reg",
        estimator_class=RandomForestRegressor,
        search_space=space,
        fixed_params={"random_state": 42},
    )


@pytest.fixture
def scaler_node():
    """ModelNode for StandardScaler transformer."""
    from sklearn_meta.core.model.node import ModelNode, OutputType

    return ModelNode(
        name="scaler",
        estimator_class=StandardScaler,
        output_type=OutputType.TRANSFORM,
    )


# =============================================================================
# Graph Fixtures
# =============================================================================


@pytest.fixture
def simple_graph(rf_classifier_node):
    """Graph with a single node."""
    from sklearn_meta.core.model.graph import ModelGraph

    graph = ModelGraph()
    graph.add_node(rf_classifier_node)
    return graph


@pytest.fixture
def two_model_graph(rf_classifier_node, lr_classifier_node):
    """Graph with two independent nodes."""
    from sklearn_meta.core.model.graph import ModelGraph

    graph = ModelGraph()
    graph.add_node(rf_classifier_node)
    graph.add_node(lr_classifier_node)
    return graph


@pytest.fixture
def stacking_graph():
    """Graph with base models stacked into meta-learner."""
    from sklearn_meta.core.model.graph import ModelGraph
    from sklearn_meta.core.model.node import ModelNode, OutputType
    from sklearn_meta.core.model.dependency import DependencyEdge, DependencyType
    from sklearn_meta.search.space import SearchSpace

    # Base models
    rf_space = SearchSpace()
    rf_space.add_int("n_estimators", 10, 50)

    rf_node = ModelNode(
        name="rf_base",
        estimator_class=RandomForestClassifier,
        search_space=rf_space,
        output_type=OutputType.PROBA,
        fixed_params={"random_state": 42},
    )

    lr_node = ModelNode(
        name="lr_base",
        estimator_class=LogisticRegression,
        fixed_params={"random_state": 42, "max_iter": 1000},
        output_type=OutputType.PROBA,
    )

    # Meta model
    meta_node = ModelNode(
        name="meta",
        estimator_class=LogisticRegression,
        fixed_params={"random_state": 42, "max_iter": 1000},
    )

    graph = ModelGraph()
    graph.add_node(rf_node)
    graph.add_node(lr_node)
    graph.add_node(meta_node)

    # Add stacking edges
    graph.add_edge(DependencyEdge(
        source="rf_base",
        target="meta",
        dep_type=DependencyType.PROBA,
    ))
    graph.add_edge(DependencyEdge(
        source="lr_base",
        target="meta",
        dep_type=DependencyType.PROBA,
    ))

    return graph


@pytest.fixture
def linear_graph():
    """Linear graph A -> B -> C."""
    from sklearn_meta.core.model.graph import ModelGraph
    from sklearn_meta.core.model.node import ModelNode
    from sklearn_meta.core.model.dependency import DependencyEdge, DependencyType

    node_a = ModelNode(name="A", estimator_class=LogisticRegression)
    node_b = ModelNode(name="B", estimator_class=LogisticRegression)
    node_c = ModelNode(name="C", estimator_class=LogisticRegression)

    graph = ModelGraph()
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)

    graph.add_edge(DependencyEdge(source="A", target="B", dep_type=DependencyType.PREDICTION))
    graph.add_edge(DependencyEdge(source="B", target="C", dep_type=DependencyType.PREDICTION))

    return graph


@pytest.fixture
def diamond_graph():
    """Diamond-shaped graph: A -> B, A -> C, B -> D, C -> D."""
    from sklearn_meta.core.model.graph import ModelGraph
    from sklearn_meta.core.model.node import ModelNode
    from sklearn_meta.core.model.dependency import DependencyEdge, DependencyType

    nodes = {name: ModelNode(name=name, estimator_class=LogisticRegression)
             for name in ["A", "B", "C", "D"]}

    graph = ModelGraph()
    for node in nodes.values():
        graph.add_node(node)

    graph.add_edge(DependencyEdge(source="A", target="B", dep_type=DependencyType.PREDICTION))
    graph.add_edge(DependencyEdge(source="A", target="C", dep_type=DependencyType.PREDICTION))
    graph.add_edge(DependencyEdge(source="B", target="D", dep_type=DependencyType.PREDICTION))
    graph.add_edge(DependencyEdge(source="C", target="D", dep_type=DependencyType.PREDICTION))

    return graph


# =============================================================================
# Search Space Fixtures
# =============================================================================


@pytest.fixture
def simple_search_space():
    """Simple search space with various parameter types."""
    from sklearn_meta.search.space import SearchSpace

    space = SearchSpace()
    space.add_int("n_estimators", 10, 100)
    space.add_float("learning_rate", 0.01, 0.3, log=True)
    space.add_categorical("booster", ["gbtree", "dart"])
    return space


@pytest.fixture
def xgb_search_space():
    """XGBoost-like search space for reparameterization tests."""
    from sklearn_meta.search.space import SearchSpace

    space = SearchSpace()
    space.add_float("learning_rate", 0.01, 0.3, log=True)
    space.add_int("n_estimators", 50, 500)
    space.add_int("max_depth", 3, 10)
    space.add_float("reg_alpha", 0.0, 1.0)
    space.add_float("reg_lambda", 0.0, 1.0)
    return space


# =============================================================================
# Tuning Configuration Fixtures
# =============================================================================


@pytest.fixture
def tuning_config():
    """Basic tuning configuration."""
    from sklearn_meta.core.tuning.orchestrator import TuningConfig
    from sklearn_meta.core.tuning.strategy import OptimizationStrategy

    return TuningConfig(
        strategy=OptimizationStrategy.LAYER_BY_LAYER,
        n_trials=10,
        metric="accuracy",
        greater_is_better=True,
        verbose=0,
    )


@pytest.fixture
def tuning_config_regression():
    """Tuning configuration for regression."""
    from sklearn_meta.core.tuning.orchestrator import TuningConfig
    from sklearn_meta.core.tuning.strategy import OptimizationStrategy

    return TuningConfig(
        strategy=OptimizationStrategy.LAYER_BY_LAYER,
        n_trials=10,
        metric="neg_mean_squared_error",
        greater_is_better=False,
        verbose=0,
    )


# =============================================================================
# Mock Classes
# =============================================================================


class MockOptunaTrial:
    """Mock Optuna trial for testing."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, seed: int = 42):
        self._params = params or {}
        self._suggested = {}
        np.random.seed(seed)

    def suggest_float(
        self, name: str, low: float, high: float,
        log: bool = False, step: Optional[float] = None
    ) -> float:
        if name in self._params:
            return self._params[name]
        if log:
            value = np.exp(np.random.uniform(np.log(low), np.log(high)))
        else:
            value = np.random.uniform(low, high)
        self._suggested[name] = value
        return value

    def suggest_int(
        self, name: str, low: int, high: int,
        log: bool = False, step: int = 1
    ) -> int:
        if name in self._params:
            return self._params[name]
        if log:
            value = int(np.exp(np.random.uniform(np.log(low), np.log(high))))
        else:
            value = np.random.randint(low, high + 1)
        self._suggested[name] = value
        return value

    def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
        if name in self._params:
            return self._params[name]
        value = np.random.choice(choices)
        self._suggested[name] = value
        return value


@pytest.fixture
def mock_trial():
    """Mock Optuna trial fixture."""
    return MockOptunaTrial()


class MockSearchBackend:
    """Mock search backend for testing."""

    def __init__(self, best_params: Optional[Dict[str, Any]] = None):
        from sklearn_meta.search.backends.base import OptimizationResult, TrialResult

        self._best_params = best_params or {}
        self._trials: List[TrialResult] = []
        self.OptimizationResult = OptimizationResult
        self.TrialResult = TrialResult

    def optimize(
        self,
        objective,
        search_space,
        n_trials: int = 10,
        timeout: Optional[float] = None,
        study_name: str = "test",
    ):
        """Run mock optimization."""
        from sklearn_meta.search.backends.base import OptimizationResult, TrialResult

        best_value = float('inf')
        best_params = self._best_params
        trials = []

        for i in range(n_trials):
            trial = MockOptunaTrial(seed=42 + i)
            params = search_space.sample_optuna(trial)

            if self._best_params:
                params.update(self._best_params)

            value = objective(params)

            if value < best_value:
                best_value = value
                best_params = params.copy()

            trials.append(TrialResult(
                params=params,
                value=value,
                trial_id=i,
                duration=0.1,
                state="COMPLETE",
            ))

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            trials=trials,
            n_trials=n_trials,
            study_name=study_name,
        )


@pytest.fixture
def mock_search_backend():
    """Mock search backend fixture."""
    return MockSearchBackend()


# =============================================================================
# Utility Functions
# =============================================================================


def create_fitted_model(estimator_class, X, y, **params):
    """Helper to create a fitted model."""
    model = estimator_class(**params)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_rf(classification_data):
    """Fitted RandomForest classifier."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_lr(classification_data):
    """Fitted LogisticRegression classifier."""
    X, y = classification_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


# =============================================================================
# Markers and Skip Conditions
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_arrays_equal(a, b, msg="Arrays not equal"):
    """Assert two arrays are equal."""
    np.testing.assert_array_equal(a, b, err_msg=msg)


def assert_arrays_almost_equal(a, b, decimal=7, msg="Arrays not almost equal"):
    """Assert two arrays are almost equal."""
    np.testing.assert_array_almost_equal(a, b, decimal=decimal, err_msg=msg)


def assert_sets_disjoint(set_a, set_b, msg="Sets should be disjoint"):
    """Assert two sets have no overlap."""
    intersection = set(set_a) & set(set_b)
    assert len(intersection) == 0, f"{msg}: intersection={intersection}"


def assert_complete_coverage(indices, n_samples, msg="Indices should cover all samples"):
    """Assert indices cover all samples exactly once."""
    all_indices = np.sort(np.concatenate(indices))
    expected = np.arange(n_samples)
    np.testing.assert_array_equal(all_indices, expected, err_msg=msg)
