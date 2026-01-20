"""End-to-end integration tests for joint quantile regression."""

import pytest
import numpy as np
import pandas as pd

from auto_sklearn.core.data.context import DataContext
from auto_sklearn.core.data.cv import CVConfig, CVStrategy
from auto_sklearn.core.data.manager import DataManager
from auto_sklearn.core.model.joint_quantile_graph import (
    JointQuantileConfig,
    JointQuantileGraph,
    OrderConstraint,
)
from auto_sklearn.core.model.joint_quantile_fitted import JointQuantileFittedGraph
from auto_sklearn.core.model.quantile_node import QuantileScalingConfig
from auto_sklearn.core.model.quantile_sampler import SamplingStrategy
from auto_sklearn.core.tuning.joint_quantile_orchestrator import JointQuantileOrchestrator
from auto_sklearn.core.tuning.orchestrator import TuningConfig
from auto_sklearn.core.tuning.strategy import OptimizationStrategy


# =============================================================================
# Mock XGBoost-like estimator
# =============================================================================


class MockQuantileRegressor:
    """
    Mock quantile regressor that mimics XGBoost quantile regression.

    For testing purposes, this generates predictions based on linear
    regression with adjustments for the quantile level.
    """

    def __init__(
        self,
        objective="reg:squarederror",
        quantile_alpha=0.5,
        n_estimators=100,
        max_depth=6,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=None,
        **kwargs,
    ):
        self.objective = objective
        self.quantile_alpha = quantile_alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self._fitted = False

    def fit(self, X, y, **fit_params):
        self._fitted = True
        # Store training statistics
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)
        self._y_min = np.min(y)
        self._y_max = np.max(y)

        # Simple linear coefficients
        X_arr = X.values if hasattr(X, "values") else X
        self._coef = np.zeros(X_arr.shape[1])
        if X_arr.shape[1] > 0:
            # Use simple correlation as weights
            for i in range(X_arr.shape[1]):
                corr = np.corrcoef(X_arr[:, i], y)[0, 1]
                self._coef[i] = 0 if np.isnan(corr) else corr * self._y_std

        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, "values") else X

        # Base prediction from linear model
        base_pred = self._y_mean + X_arr @ self._coef * 0.1

        # Adjust for quantile level
        # At tau=0.5, no adjustment
        # At tau < 0.5, shift predictions lower
        # At tau > 0.5, shift predictions higher
        from scipy import stats

        z_score = stats.norm.ppf(self.quantile_alpha)
        adjustment = z_score * self._y_std * 0.5

        return base_pred + adjustment


class MockSearchBackend:
    """Mock search backend for testing."""

    def optimize(self, objective, search_space, n_trials=10, timeout=None, study_name="test"):
        from auto_sklearn.search.backends.base import OptimizationResult, TrialResult

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
# Test Data Generation
# =============================================================================


def generate_correlated_data(n_samples=500, random_state=42):
    """
    Generate synthetic data with correlated targets.

    Creates three targets where each depends on the previous:
    - Y1 = f(X) + noise
    - Y2 = f(X, Y1) + noise
    - Y3 = f(X, Y1, Y2) + noise
    """
    np.random.seed(random_state)

    # Features
    X = np.random.randn(n_samples, 5)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

    # Target 1: depends on X only
    y1 = 2 * X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

    # Target 2: depends on X and Y1
    y2 = X[:, 2] + 0.5 * y1 + np.random.randn(n_samples) * 0.5

    # Target 3: depends on X, Y1, and Y2
    y3 = X[:, 3] - X[:, 4] + 0.3 * y1 + 0.4 * y2 + np.random.randn(n_samples) * 0.5

    targets = {
        "price": pd.Series(y1, name="price"),
        "volume": pd.Series(y2, name="volume"),
        "volatility": pd.Series(y3, name="volatility"),
    }

    return X_df, targets


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.fixture
def synthetic_data():
    """Create synthetic correlated data."""
    return generate_correlated_data(n_samples=200, random_state=42)


@pytest.fixture
def cv_config():
    """Create CV configuration."""
    return CVConfig(
        n_splits=3,
        strategy=CVStrategy.RANDOM,
        shuffle=True,
        random_state=42,
    )


@pytest.fixture
def tuning_config(cv_config):
    """Create tuning configuration."""
    return TuningConfig(
        strategy=OptimizationStrategy.NONE,
        n_trials=1,
        metric="neg_mean_squared_error",
        greater_is_better=False,
        verbose=0,
        cv_config=cv_config,
    )


class TestJointQuantileE2E:
    """End-to-end tests for joint quantile regression."""

    @pytest.mark.integration
    def test_full_pipeline(self, synthetic_data, cv_config, tuning_config):
        """Test complete pipeline from config to inference."""
        X, targets = synthetic_data

        # 1. Configure
        config = JointQuantileConfig(
            property_names=["price", "volume", "volatility"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            n_inference_samples=100,
            random_state=42,
        )

        # 2. Build graph
        graph = JointQuantileGraph(config)

        # 3. Create orchestrator
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        # 4. Fit
        ctx = DataContext(X=X, y=targets["price"])
        fit_result = orchestrator.fit(ctx, targets)

        # 5. Create fitted graph for inference
        fitted_graph = JointQuantileFittedGraph.from_fit_result(fit_result)

        # 6. Inference: sample from joint distribution
        X_test = X.iloc[:10]
        samples = fitted_graph.sample_joint_efficient(X_test, n_samples=100)

        assert samples.shape == (10, 100, 3)

        # 7. Point predictions
        medians = fitted_graph.predict_median(X_test)
        assert medians.shape == (10, 3)

    @pytest.mark.integration
    def test_order_change_and_refit(self, synthetic_data, cv_config, tuning_config):
        """Test changing property order and refitting."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume", "volatility"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.25, 0.5, 0.75],
            n_inference_samples=50,
        )

        graph = JointQuantileGraph(config)

        # Initial order
        assert graph.property_order == ["price", "volume", "volatility"]

        # Change order
        graph.set_order(["volume", "price", "volatility"])
        assert graph.property_order == ["volume", "price", "volatility"]

        # Should still be fittable
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        ctx = DataContext(X=X, y=targets["price"])
        fit_result = orchestrator.fit(ctx, targets)

        assert len(fit_result.fitted_nodes) == 3

    @pytest.mark.integration
    def test_with_order_constraints(self, synthetic_data, cv_config, tuning_config):
        """Test with order constraints."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume", "volatility"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.25, 0.5, 0.75],
            order_constraints=OrderConstraint(
                fixed_positions={"price": 0},
                must_precede=[("volume", "volatility")],
            ),
        )

        graph = JointQuantileGraph(config)

        # Price should be first due to fixed position
        assert graph.property_order[0] == "price"

        # Volume should precede volatility
        vol_idx = graph.property_order.index("volume")
        volat_idx = graph.property_order.index("volatility")
        assert vol_idx < volat_idx

        # Fit should work
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        ctx = DataContext(X=X, y=targets["price"])
        fit_result = orchestrator.fit(ctx, targets)

        assert len(fit_result.fitted_nodes) == 3

    @pytest.mark.integration
    def test_with_quantile_scaling(self, synthetic_data, cv_config, tuning_config):
        """Test with quantile-dependent parameter scaling."""
        X, targets = synthetic_data

        scaling = QuantileScalingConfig(
            base_params={"n_estimators": 100, "max_depth": 6},
            scaling_rules={
                "reg_lambda": {"base": 1.0, "tail_multiplier": 2.0},
            },
        )

        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            quantile_scaling=scaling,
        )

        graph = JointQuantileGraph(config)
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        ctx = DataContext(X=X, y=targets["price"])
        targets_subset = {k: v for k, v in targets.items() if k in ["price", "volume"]}
        fit_result = orchestrator.fit(ctx, targets_subset)

        # Verify nodes have quantile-scaled parameters
        price_node = fit_result.get_node("price")
        models_low = price_node.quantile_models[0.1]
        models_med = price_node.quantile_models[0.5]

        # Low quantile should have higher regularization
        assert models_low[0].reg_lambda >= models_med[0].reg_lambda

    @pytest.mark.integration
    def test_quantile_predictions_ordering(self, synthetic_data, cv_config, tuning_config):
        """Test that quantile predictions are properly ordered."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        )

        graph = JointQuantileGraph(config)
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        ctx = DataContext(X=X, y=targets["price"])
        targets_subset = {"price": targets["price"]}
        fit_result = orchestrator.fit(ctx, targets_subset)

        fitted_graph = JointQuantileFittedGraph.from_fit_result(fit_result)

        # Get quantile predictions
        X_test = X.iloc[:10]
        q10 = fitted_graph.predict_quantile(X_test, 0.1)
        q50 = fitted_graph.predict_quantile(X_test, 0.5)
        q90 = fitted_graph.predict_quantile(X_test, 0.9)

        # Quantile ordering: q10 <= q50 <= q90
        assert np.all(q10[:, 0] <= q50[:, 0] + 0.1)  # Small tolerance
        assert np.all(q50[:, 0] <= q90[:, 0] + 0.1)


class TestJointQuantileSampling:
    """Tests for joint sampling behavior."""

    @pytest.mark.integration
    def test_sample_shape(self, synthetic_data, cv_config, tuning_config):
        """Test that samples have correct shape."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume", "volatility"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            n_inference_samples=200,
        )

        graph = JointQuantileGraph(config)
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        ctx = DataContext(X=X, y=targets["price"])
        fit_result = orchestrator.fit(ctx, targets)
        fitted_graph = JointQuantileFittedGraph.from_fit_result(fit_result)

        X_test = X.iloc[:5]
        samples = fitted_graph.sample_joint_efficient(X_test, n_samples=200)

        assert samples.shape == (5, 200, 3)

    @pytest.mark.integration
    def test_sample_statistics(self, synthetic_data, cv_config, tuning_config):
        """Test that sample statistics are reasonable."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            n_inference_samples=500,
        )

        graph = JointQuantileGraph(config)
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        ctx = DataContext(X=X, y=targets["price"])
        targets_subset = {"price": targets["price"], "volume": targets["volume"]}
        fit_result = orchestrator.fit(ctx, targets_subset)
        fitted_graph = JointQuantileFittedGraph.from_fit_result(fit_result)

        X_test = X.iloc[:10]
        samples = fitted_graph.sample_joint_efficient(X_test, n_samples=500)

        # Check that sample median is close to predicted median
        medians = fitted_graph.predict_median(X_test)
        sample_medians = np.median(samples, axis=1)

        # Within reasonable tolerance
        np.testing.assert_array_almost_equal(
            medians, sample_medians, decimal=0
        )


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.integration
    def test_single_property(self, synthetic_data, cv_config, tuning_config):
        """Test with single property (degenerates to standard quantile regression)."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        graph = JointQuantileGraph(config)
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        ctx = DataContext(X=X, y=targets["price"])
        fit_result = orchestrator.fit(ctx, {"price": targets["price"]})

        assert len(fit_result.fitted_nodes) == 1

    @pytest.mark.integration
    def test_two_properties(self, synthetic_data, cv_config, tuning_config):
        """Test with two properties."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.25, 0.5, 0.75],
        )

        graph = JointQuantileGraph(config)
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        ctx = DataContext(X=X, y=targets["price"])
        fit_result = orchestrator.fit(ctx, {
            "price": targets["price"],
            "volume": targets["volume"],
        })

        assert len(fit_result.fitted_nodes) == 2

    @pytest.mark.integration
    def test_minimal_quantile_levels(self, synthetic_data, cv_config, tuning_config):
        """Test with minimal quantile levels."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.5],  # Only median
        )

        graph = JointQuantileGraph(config)
        orchestrator = JointQuantileOrchestrator(
            graph=graph,
            data_manager=DataManager(cv_config),
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        ctx = DataContext(X=X, y=targets["price"])
        fit_result = orchestrator.fit(ctx, {
            "price": targets["price"],
            "volume": targets["volume"],
        })

        assert fit_result.get_node("price").n_quantiles == 1
