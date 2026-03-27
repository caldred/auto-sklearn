"""Tests for OrderSearchPlugin."""

import pytest
import numpy as np
import pandas as pd

from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import CVConfig, CVStrategy, RunConfig, TuningConfig
from sklearn_meta.spec.quantile import (
    JointQuantileConfig,
    JointQuantileGraphSpec,
    OrderConstraint,
)
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.engine.runner import GraphRunner
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.plugins.joint_quantile.order_search import (
    OrderSearchConfig,
    OrderSearchPlugin,
    OrderSearchResult,
)


# =============================================================================
# Mock classes
# =============================================================================


class MockQuantileRegressor:
    """Mock quantile regressor for testing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._fitted = False

    def fit(self, X, y, **fit_params):
        self._fitted = True
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)
        return self

    def predict(self, X):
        tau = getattr(self, "quantile_alpha", 0.5)
        offset = (tau - 0.5) * self._y_std * 2
        return np.full(len(X), self._y_mean + offset)


class MockSearchBackend:
    """Mock search backend for testing."""

    def optimize(self, objective, search_space, n_trials=10, timeout=None, callbacks=None, study_name="test", early_stopping_rounds=None):
        from sklearn_meta.search.backends.base import OptimizationResult, TrialResult

        params = {}
        value = objective(params)

        return OptimizationResult(
            best_params=params,
            best_value=value,
            trials=[TrialResult(
                params=params,
                value=value,
                trial_id=0,
                duration=0.1,
                state="COMPLETE",
            )],
            n_trials=1,
            study_name=study_name,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def joint_data():
    """Create synthetic joint regression data."""
    np.random.seed(42)
    n_samples = 100

    X = np.random.randn(n_samples, 3)
    X_df = pd.DataFrame(X, columns=["f1", "f2", "f3"])

    # Y1 depends on X
    y1 = X[:, 0] + np.random.randn(n_samples) * 0.5

    # Y2 depends on Y1
    y2 = 0.8 * y1 + np.random.randn(n_samples) * 0.3

    # Y3 depends on both
    y3 = 0.3 * y1 + 0.5 * y2 + np.random.randn(n_samples) * 0.2

    targets = {
        "y1": pd.Series(y1),
        "y2": pd.Series(y2),
        "y3": pd.Series(y3),
    }

    return X_df, targets


@pytest.fixture
def joint_graph():
    """Create a JointQuantileGraphSpec."""
    config = JointQuantileConfig(
        property_names=["y1", "y2", "y3"],
        estimator_class=MockQuantileRegressor,
        quantile_levels=[0.25, 0.5, 0.75],
        n_inference_samples=50,
    )
    return JointQuantileGraphSpec(config)


@pytest.fixture
def orchestrator(joint_graph):
    """Create a GraphRunner with RuntimeServices."""
    cv_config = CVConfig(
        n_splits=2,
        strategy=CVStrategy.RANDOM,
        shuffle=True,
        random_state=42,
    )
    tuning_config = TuningConfig(
        strategy=OptimizationStrategy.NONE,
        n_trials=1,
    )
    config = RunConfig(cv=cv_config, tuning=tuning_config, verbosity=0)
    services = RuntimeServices(search_backend=MockSearchBackend())
    runner = GraphRunner(services)
    # Expose config and runner as a simple namespace for tests
    class _Orchestrator:
        def __init__(self):
            self.graph = joint_graph
            self.runner = runner
            self.config = config
            self.services = services
        def fit(self, ctx, targets=None):
            return runner.fit(self.graph, ctx, self.config)
    return _Orchestrator()


# =============================================================================
# OrderSearchConfig Tests
# =============================================================================


class TestOrderSearchConfigCreation:
    """Tests for OrderSearchConfig creation."""

    def test_creation_with_params(self):
        """Verify config creation with custom params."""
        config = OrderSearchConfig(
            max_iterations=20,
            n_random_restarts=5,
            verbose=0,
        )

        assert config.max_iterations == 20
        assert config.n_random_restarts == 5


# =============================================================================
# OrderSearchResult Tests
# =============================================================================


class TestOrderSearchResultProperties:
    """Tests for OrderSearchResult properties."""


# =============================================================================
# OrderSearchPlugin Tests
# =============================================================================


class TestOrderSearchPluginCreation:
    """Tests for OrderSearchPlugin creation."""

    def test_creation_with_config(self):
        """Verify plugin creation with custom config."""
        config = OrderSearchConfig(max_iterations=5)
        plugin = OrderSearchPlugin(config=config)

        assert plugin.config.max_iterations == 5


class TestOrderSearchPluginSearchOrder:
    """Tests for search_order method."""

    @pytest.mark.slow
    def test_search_order_returns_result(self, joint_data, joint_graph, orchestrator):
        """Verify search_order returns OrderSearchResult."""
        X, targets = joint_data
        ctx = DataView.from_Xy(X, targets["y1"])

        plugin = OrderSearchPlugin(config=OrderSearchConfig(
            max_iterations=2,
            verbose=0,
        ))

        result = plugin.search_order(
            graph=joint_graph,
            data=ctx,
            targets=targets,
            orchestrator=orchestrator,
            random_state=42,
        )

        assert isinstance(result, OrderSearchResult)
        assert result.best_order is not None
        assert result.best_score is not None

    @pytest.mark.slow
    def test_search_order_preserves_constraints(self, joint_data, orchestrator):
        """Verify search preserves order constraints."""
        X, targets = joint_data
        ctx = DataView.from_Xy(X, targets["y1"])

        # Create graph with constraint: y1 must be first
        config = JointQuantileConfig(
            property_names=["y1", "y2", "y3"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.5],
            order_constraints=OrderConstraint(
                fixed_positions={"y1": 0},
            ),
        )
        graph = JointQuantileGraphSpec(config)

        # Update orchestrator's graph
        orchestrator.graph = graph

        plugin = OrderSearchPlugin(config=OrderSearchConfig(
            max_iterations=2,
            verbose=0,
        ))

        result = plugin.search_order(
            graph=graph,
            data=ctx,
            targets=targets,
            orchestrator=orchestrator,
            random_state=42,
        )

        # y1 should still be first
        assert result.best_order[0] == "y1"

    @pytest.mark.slow
    def test_search_order_with_random_restarts(self, joint_data, joint_graph, orchestrator):
        """Verify random restarts are performed."""
        X, targets = joint_data
        ctx = DataView.from_Xy(X, targets["y1"])

        plugin = OrderSearchPlugin(config=OrderSearchConfig(
            max_iterations=1,
            n_random_restarts=2,
            verbose=0,
        ))

        result = plugin.search_order(
            graph=joint_graph,
            data=ctx,
            targets=targets,
            orchestrator=orchestrator,
            random_state=42,
        )

        # Search should complete with restarts
        assert result.best_order is not None


class TestOrderSearchPluginScoring:
    """Tests for scoring functions."""

    def test_pinball_loss_calculation(self):
        """Verify pinball loss calculation."""
        plugin = OrderSearchPlugin()

        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 32])
        tau = 0.5

        loss = plugin._pinball_loss(y_true, y_pred, tau)

        # Manual calculation
        residual = y_true - y_pred
        expected = np.mean(np.where(
            residual >= 0,
            tau * residual,
            (tau - 1) * residual,
        ))

        assert loss == pytest.approx(expected)


class TestOrderSearchPluginRandomOrder:
    """Tests for random order generation."""

    def test_generate_random_order_no_constraints(self, joint_graph):
        """Verify random order generation without constraints."""
        plugin = OrderSearchPlugin()
        plugin._rng = np.random.RandomState(42)

        order = plugin._generate_random_order(joint_graph)

        assert set(order) == {"y1", "y2", "y3"}

    def test_generate_random_order_with_fixed_positions(self):
        """Verify random order respects fixed positions."""
        config = JointQuantileConfig(
            property_names=["y1", "y2", "y3"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.5],
            order_constraints=OrderConstraint(
                fixed_positions={"y1": 0},
            ),
        )
        graph = JointQuantileGraphSpec(config)

        plugin = OrderSearchPlugin()
        plugin._rng = np.random.RandomState(42)

        order = plugin._generate_random_order(graph)

        assert order[0] == "y1"


class TestOrderSearchPluginRepr:
    """Tests for plugin representation."""
