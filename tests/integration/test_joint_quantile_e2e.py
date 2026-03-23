"""End-to-end integration tests for joint quantile regression."""

import pytest
import numpy as np
import pandas as pd

from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import CVConfig, CVStrategy, RunConfig, TuningConfig, FeatureSelectionConfig
from sklearn_meta.spec.quantile import (
    JointQuantileConfig,
    JointQuantileGraphSpec,
    OrderConstraint,
    QuantileScalingConfig,
)
from sklearn_meta.artifacts.inference import JointQuantileInferenceGraph, QuantileFittedNode
from sklearn_meta.artifacts.training import QuantileNodeRunResult
from sklearn_meta.spec.quantile_sampler import QuantileSampler
from sklearn_meta.engine.runner import GraphRunner
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.search.space import SearchSpace


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
        self.feature_importances_ = np.abs(self._coef)

        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, "values") else X

        # Base prediction from linear model -- handle extra columns at inference
        # (e.g., conditioning columns added by joint quantile sampling)
        n_coef = len(self._coef)
        n_feat = X_arr.shape[1]
        if n_feat >= n_coef:
            base_pred = self._y_mean + X_arr[:, :n_coef] @ self._coef * 0.1
        else:
            base_pred = self._y_mean + X_arr @ self._coef[:n_feat] * 0.1

        # Adjust for quantile level
        from scipy import stats

        z_score = stats.norm.ppf(self.quantile_alpha)
        adjustment = z_score * self._y_std * 0.5

        return base_pred + adjustment

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        return {
            "objective": self.objective,
            "quantile_alpha": self.quantile_alpha,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "random_state": self.random_state,
        }


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
# Helper to build JointQuantileInferenceGraph from TrainingRun
# =============================================================================


def _build_jq_inference(fit_result, graph):
    """Build a JointQuantileInferenceGraph from a TrainingRun and its JointQuantileGraphSpec."""
    fitted_nodes = {}
    for prop_name in graph.property_order:
        node_name = f"quantile_{prop_name}"
        result = fit_result.node_results[node_name]
        if isinstance(result, QuantileNodeRunResult):
            fitted_nodes[prop_name] = QuantileFittedNode(
                quantile_models=result.quantile_models,
                quantile_levels=list(result.quantile_models.keys()),
                selected_features=result.selected_features,
            )
        else:
            # Fallback: wrap regular models as a single-quantile node
            fitted_nodes[prop_name] = QuantileFittedNode(
                quantile_models={0.5: result.models},
                quantile_levels=[0.5],
                selected_features=result.selected_features,
            )

    sampler = QuantileSampler(
        strategy=graph.config.sampling_strategy,
        n_samples=graph.config.n_inference_samples,
        random_state=graph.config.random_state,
    )

    return JointQuantileInferenceGraph(
        graph=graph,
        fitted_nodes=fitted_nodes,
        quantile_sampler=sampler,
    )


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
    )


@pytest.fixture
def run_config(cv_config, tuning_config):
    """Create run configuration."""
    return RunConfig(cv=cv_config, tuning=tuning_config, verbosity=0)


@pytest.fixture
def services():
    """Create runtime services."""
    return RuntimeServices(search_backend=MockSearchBackend())


class TestJointQuantileE2E:
    """End-to-end tests for joint quantile regression."""

    @pytest.mark.integration
    def test_full_pipeline(self, synthetic_data, run_config, services):
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
        graph = JointQuantileGraphSpec(config)

        # 3. Fit -- bind all named targets so each quantile node resolves
        #    its own property_name from data.targets.
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        runner = GraphRunner(services)
        fit_result = runner.fit(graph, ctx, run_config)

        # 4. Create fitted graph for inference
        fitted_graph = _build_jq_inference(fit_result, graph)

        # 5. Inference: sample from joint distribution
        X_test = X.iloc[:10]
        samples = fitted_graph.sample_joint(X_test, n_samples=100)

        # sample_joint returns Dict[str, ndarray] with shape (n_data, n_samples)
        assert len(samples) == 3
        for prop_name in ["price", "volume", "volatility"]:
            assert samples[prop_name].shape == (10, 100)

        # 6. Point predictions
        medians = fitted_graph.predict_median(X_test)
        assert len(medians) == 3
        for prop_name in ["price", "volume", "volatility"]:
            assert medians[prop_name].shape == (10,)

    @pytest.mark.integration
    def test_pipeline_with_feature_selection(self, synthetic_data, cv_config):
        """Joint quantile pipeline should fit and infer with feature selection enabled."""
        X, targets = synthetic_data

        search_space = SearchSpace().add_from_shorthand(max_depth=(3, 8))
        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            search_space=search_space,
            n_inference_samples=100,
        )
        graph = JointQuantileGraphSpec(config)

        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="neg_mean_squared_error",
            greater_is_better=False,
        )
        run_cfg = RunConfig(
            cv=cv_config,
            tuning=tuning_config,
            feature_selection=FeatureSelectionConfig(
                enabled=True,
                method="shadow",
                n_shadows=3,
                retune_after_pruning=True,
                min_features=1,
            ),
            verbosity=0,
        )

        svc = RuntimeServices(search_backend=MockSearchBackend())
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(svc).fit(graph, ctx, run_cfg)

        # Node names in the graph are "quantile_price", "quantile_volume"
        assert fit_result.node_results["quantile_price"].selected_features is not None
        assert fit_result.node_results["quantile_volume"].selected_features is not None

        fitted_graph = _build_jq_inference(fit_result, graph)
        medians = fitted_graph.predict_median(X.iloc[:5])
        assert len(medians) == 2
        for prop_name in ["price", "volume"]:
            assert medians[prop_name].shape == (5,)

    @pytest.mark.integration
    def test_order_change_and_refit(self, synthetic_data, run_config, services):
        """Test changing property order and refitting."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume", "volatility"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.25, 0.5, 0.75],
            n_inference_samples=50,
        )

        graph = JointQuantileGraphSpec(config)

        # Initial order
        assert graph.property_order == ["price", "volume", "volatility"]

        # Change order
        graph.set_order(["volume", "price", "volatility"])
        assert graph.property_order == ["volume", "price", "volatility"]

        # Should still be fittable
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)

        assert len(fit_result.node_results) == 3

    @pytest.mark.integration
    def test_with_order_constraints(self, synthetic_data, run_config, services):
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

        graph = JointQuantileGraphSpec(config)

        # Price should be first due to fixed position
        assert graph.property_order[0] == "price"

        # Volume should precede volatility
        vol_idx = graph.property_order.index("volume")
        volat_idx = graph.property_order.index("volatility")
        assert vol_idx < volat_idx

        # Fit should work
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)

        assert len(fit_result.node_results) == 3

    @pytest.mark.integration
    def test_with_quantile_scaling(self, synthetic_data, run_config, services):
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

        graph = JointQuantileGraphSpec(config)
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)

        # Verify nodes have quantile-scaled parameters
        price_result = fit_result.node_results["quantile_price"]
        models_low = price_result.quantile_models[0.1]
        models_med = price_result.quantile_models[0.5]

        # Low quantile should have higher regularization
        assert models_low[0].reg_lambda >= models_med[0].reg_lambda

    @pytest.mark.integration
    def test_quantile_predictions_ordering(self, synthetic_data, run_config, services):
        """Test that quantile predictions are properly ordered."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        )

        graph = JointQuantileGraphSpec(config)
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)

        fitted_graph = _build_jq_inference(fit_result, graph)

        # Get quantile predictions
        X_test = X.iloc[:10]
        q10 = fitted_graph.predict_quantile(X_test, 0.1)
        q50 = fitted_graph.predict_quantile(X_test, 0.5)
        q90 = fitted_graph.predict_quantile(X_test, 0.9)

        # Quantile ordering: q10 <= q50 <= q90
        assert np.all(q10["price"] <= q50["price"] + 0.1)  # Small tolerance
        assert np.all(q50["price"] <= q90["price"] + 0.1)


class TestJointQuantileSampling:
    """Tests for joint sampling behavior."""

    @pytest.mark.integration
    def test_sample_shape(self, synthetic_data, run_config, services):
        """Test that samples have correct shape."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume", "volatility"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
            n_inference_samples=200,
        )

        graph = JointQuantileGraphSpec(config)
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)
        fitted_graph = _build_jq_inference(fit_result, graph)

        X_test = X.iloc[:5]
        samples = fitted_graph.sample_joint(X_test, n_samples=200)

        # sample_joint returns Dict[str, ndarray] with shape (n_data, n_samples)
        assert len(samples) == 3
        for prop_name in ["price", "volume", "volatility"]:
            assert samples[prop_name].shape == (5, 200)

    @pytest.mark.integration
    def test_sample_statistics(self, synthetic_data, run_config, services):
        """Test that sample statistics are reasonable."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            n_inference_samples=500,
        )

        graph = JointQuantileGraphSpec(config)
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)
        fitted_graph = _build_jq_inference(fit_result, graph)

        X_test = X.iloc[:10]
        samples = fitted_graph.sample_joint(X_test, n_samples=500)

        # Check that sample median is close to predicted median
        medians = fitted_graph.predict_median(X_test)

        for prop_name in ["price", "volume"]:
            sample_medians = np.median(samples[prop_name], axis=1)
            # Within reasonable tolerance
            np.testing.assert_array_almost_equal(
                medians[prop_name], sample_medians, decimal=0
            )


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.integration
    def test_single_property(self, synthetic_data, run_config, services):
        """Test with single property (degenerates to standard quantile regression)."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        graph = JointQuantileGraphSpec(config)
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)

        assert len(fit_result.node_results) == 1

    @pytest.mark.integration
    def test_two_properties(self, synthetic_data, run_config, services):
        """Test with two properties."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.25, 0.5, 0.75],
        )

        graph = JointQuantileGraphSpec(config)
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)

        assert len(fit_result.node_results) == 2

    @pytest.mark.integration
    def test_minimal_quantile_levels(self, synthetic_data, run_config, services):
        """Test with minimal quantile levels."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.5],  # Only median
        )

        graph = JointQuantileGraphSpec(config)
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)

        # Check the quantile node result has 1 quantile level
        price_result = fit_result.node_results["quantile_price"]
        assert len(price_result.quantile_models) == 1

    @pytest.mark.integration
    def test_named_target_binding_contract(self, synthetic_data, run_config, services):
        """Each quantile node should resolve its own named target, not __default__."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.5],
        )
        graph = JointQuantileGraphSpec(config)

        # Bind *only* named targets -- no __default__ at all.
        ctx = DataView.from_X(X)
        for name, y in targets.items():
            ctx = ctx.bind_target(y, name=name)

        # __default__ should NOT be present
        assert "__default__" not in ctx.targets

        # Fitting must still succeed because the trainer resolves
        # node.property_name from data.targets.
        fit_result = GraphRunner(services).fit(graph, ctx, run_config)

        assert "quantile_price" in fit_result.node_results
        assert "quantile_volume" in fit_result.node_results

    @pytest.mark.integration
    def test_missing_named_target_raises(self, synthetic_data, run_config, services):
        """Fitting should fail when the required named target is absent."""
        X, targets = synthetic_data

        config = JointQuantileConfig(
            property_names=["price", "volume"],
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.5],
        )
        graph = JointQuantileGraphSpec(config)

        # Only bind "price" -- "volume" is missing
        ctx = DataView.from_X(X)
        ctx = ctx.bind_target(targets["price"], name="price")

        with pytest.raises((ValueError, KeyError)):
            GraphRunner(services).fit(graph, ctx, run_config)
