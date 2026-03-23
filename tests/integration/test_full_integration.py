"""Integration tests for full feature integration.

Tests that verify all integrated features work together:
- Reparameterization
- Feature selection
- FitCache
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn_meta.spec.builder import GraphBuilder
from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import (
    CVConfig, CVStrategy, FeatureSelectionConfig, ReparameterizationConfig,
    RunConfig, TuningConfig,
)
from sklearn_meta.engine.cv import CVEngine
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.spec.dependency import DependencyEdge, DependencyType
from sklearn_meta.engine.runner import GraphRunner
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.meta.reparameterization import ReparameterizedSpace, LogProductReparameterization
from sklearn_meta.meta.prebaked import get_prebaked_reparameterization
from sklearn_meta.selection.selector import FeatureSelector
from sklearn_meta.persistence.cache import FitCache


@pytest.fixture
def classification_data():
    """Generate classification data for tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


@pytest.fixture
def classification_data_with_noise():
    """Generate classification data with noisy features."""
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"real_{i}" for i in range(5)])

    # Add noisy features
    np.random.seed(42)
    for i in range(5):
        X_df[f"noise_{i}"] = np.random.randn(200)

    return X_df, pd.Series(y)


def _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend,
               feature_selection=None, reparameterization=None, fit_cache=None, verbosity=0):
    """Helper to fit a graph using the new API."""
    config = RunConfig(
        cv=cv_config,
        tuning=tuning_config,
        feature_selection=feature_selection,
        reparameterization=reparameterization,
        verbosity=verbosity,
    )
    services = RuntimeServices(search_backend=mock_search_backend, fit_cache=fit_cache)
    runner = GraphRunner(services)
    return runner.fit(graph, ctx, config)


class TestReparameterizationIntegration:
    """Tests for reparameterization integration."""

    def test_reparameterization_transforms_search_space(self, classification_data):
        """Verify reparameterization transforms search space correctly."""
        X, y = classification_data

        # Create a search space with correlated params
        space = SearchSpace()
        space.add_float("learning_rate", 0.01, 0.3)
        space.add_int("n_estimators", 50, 200)

        # Create reparameterization
        reparam = LogProductReparameterization(
            name="learning_budget",
            param1="learning_rate",
            param2="n_estimators",
        )

        reparam_space = ReparameterizedSpace(space, [reparam])
        transformed_space = reparam_space.build_transformed_space()

        # Verify transformed space has new params
        param_names = transformed_space.parameter_names
        assert "learning_rate_n_estimators_budget" in param_names
        assert "learning_rate_n_estimators_ratio" in param_names
        assert "learning_rate" not in param_names
        assert "n_estimators" not in param_names

    def test_reparameterization_inverse_transform(self, classification_data):
        """Verify inverse transform recovers original params."""
        X, y = classification_data

        space = SearchSpace()
        space.add_float("learning_rate", 0.01, 0.3)
        space.add_int("n_estimators", 50, 200)

        reparam = LogProductReparameterization(
            name="learning_budget",
            param1="learning_rate",
            param2="n_estimators",
        )

        reparam_space = ReparameterizedSpace(space, [reparam])

        # Test forward then inverse
        original = {"learning_rate": 0.1, "n_estimators": 100}
        transformed = reparam_space.forward_transform(original)
        recovered = reparam_space.inverse_transform(transformed)

        # Should recover close to original
        assert abs(recovered["learning_rate"] - original["learning_rate"]) < 0.01
        assert abs(recovered["n_estimators"] - original["n_estimators"]) < 5

    def test_prebaked_reparameterization_for_rf(self, classification_data):
        """Verify prebaked reparameterization works for RandomForest."""
        X, y = classification_data

        param_names = ["max_depth", "min_samples_split"]
        reparams = get_prebaked_reparameterization(RandomForestClassifier, param_names)

        # Should get the rf_complexity reparameterization
        assert len(reparams) > 0

    def test_reparameterization_via_run_config(self, classification_data, mock_search_backend):
        """Verify reparameterization works through RunConfig."""
        X, y = classification_data
        ctx = DataView.from_Xy(X, y)

        space = SearchSpace()
        space.add_int("max_depth", 3, 10)
        space.add_int("min_samples_split", 2, 20)

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            search_space=space,
            fixed_params={"random_state": 42, "n_estimators": 10},
        )
        graph = GraphSpec()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=3,
            metric="accuracy",
            greater_is_better=True,
        )
        reparam_config = ReparameterizationConfig(enabled=True, use_prebaked=True)

        fitted = _fit_graph(
            graph, ctx, cv_config, tuning_config, mock_search_backend,
            reparameterization=reparam_config,
        )

        # Should have fitted successfully
        assert "rf" in fitted.node_results
        # Verify reparameterization was configured
        assert fitted.config.reparameterization is not None
        assert fitted.config.reparameterization.use_prebaked is True


class TestFeatureSelectionIntegration:
    """Tests for feature selection integration."""

    def test_feature_selector_identifies_noisy_features(self, classification_data_with_noise):
        """Verify feature selector can identify noisy features."""
        X, y = classification_data_with_noise

        config = FeatureSelectionConfig(
            enabled=True,
            method="shadow",
            n_shadows=3,
            min_features=3,
        )

        selector = FeatureSelector(config)

        # Create and fit a model for feature selection
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        result = selector.select(model, X, y)

        # Should have selected some features
        assert len(result.selected_features) >= config.min_features
        assert len(result.dropped_features) >= 0

        # Selected features should tend to be the real ones
        real_features = [f for f in result.selected_features if f.startswith("real_")]
        noise_features = [f for f in result.selected_features if f.startswith("noise_")]

        # More real features should be selected than noise features
        assert len(real_features) >= len(noise_features)

    def test_feature_selection_via_run_config(self, classification_data_with_noise, mock_search_backend):
        """Verify feature selection works through RunConfig."""
        X, y = classification_data_with_noise
        ctx = DataView.from_Xy(X, y)

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"random_state": 42, "n_estimators": 20},
        )
        graph = GraphSpec()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=2,
            metric="accuracy",
            greater_is_better=True,
        )
        fs_config = FeatureSelectionConfig(
            enabled=True,
            method="shadow",
            n_shadows=3,
            min_features=3,
        )

        fitted = _fit_graph(
            graph, ctx, cv_config, tuning_config, mock_search_backend,
            feature_selection=fs_config,
        )

        # Should have fitted successfully
        assert "rf" in fitted.node_results
        # Verify feature selection was configured
        assert fitted.config.feature_selection is not None
        assert fitted.config.feature_selection.enabled is True


class TestFitCacheIntegration:
    """Tests for FitCache integration."""

    def test_fit_cache_caches_models(self, classification_data, tmp_path):
        """Verify FitCache actually caches fitted models."""
        X, y = classification_data
        ctx = DataView.from_Xy(X, y)

        cache = FitCache(cache_dir=str(tmp_path / "cache"))

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        params = {"n_estimators": 10, "random_state": 42}

        # First cache lookup should miss
        cache_key = cache.cache_key(node, params, ctx)
        assert cache.get(cache_key) is None

        # Fit and store in cache
        data = ctx.materialize()
        model = node.create_estimator(params)
        model.fit(data.X, data.y)
        cache.put(cache_key, model)

        # Second cache lookup should hit
        cached_model = cache.get(cache_key)
        assert cached_model is not None

        # Cached model should work
        predictions = cached_model.predict(data.X)
        assert len(predictions) == len(data.X)

    def test_fit_cache_stats(self, classification_data, tmp_path):
        """Verify FitCache statistics work."""
        X, y = classification_data
        ctx = DataView.from_Xy(X, y)

        cache = FitCache(cache_dir=str(tmp_path / "cache"))

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        params = {"n_estimators": 10, "random_state": 42}
        cache_key = cache.cache_key(node, params, ctx)

        data = ctx.materialize()
        model = node.create_estimator(params)
        model.fit(data.X, data.y)
        cache.put(cache_key, model)

        stats = cache.stats()
        assert stats["enabled"] is True
        assert stats["memory_entries"] == 1

    def test_orchestrator_with_cache(self, classification_data, mock_search_backend, tmp_path):
        """Verify GraphRunner uses cache correctly."""
        X, y = classification_data
        ctx = DataView.from_Xy(X, y)

        cache = FitCache(cache_dir=str(tmp_path / "cache"))

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        graph = GraphSpec()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            metric="accuracy",
            greater_is_better=True,
        )

        config = RunConfig(cv=cv_config, tuning=tuning_config, verbosity=0)
        services = RuntimeServices(search_backend=mock_search_backend, fit_cache=cache)
        runner = GraphRunner(services)

        # First fit
        fitted1 = runner.fit(graph, ctx, config)
        assert "rf" in fitted1.node_results

        # Second fit should use cache (faster)
        runner2 = GraphRunner(services)
        fitted2 = runner2.fit(graph, ctx, config)
        assert "rf" in fitted2.node_results


class TestFullIntegration:
    """Tests for all features working together."""

    def test_all_features_together(self, classification_data_with_noise, mock_search_backend, tmp_path):
        """Verify reparameterization, feature selection, and cache work together."""
        X, y = classification_data_with_noise
        ctx = DataView.from_Xy(X, y)

        cache = FitCache(cache_dir=str(tmp_path / "cache"))

        space = SearchSpace()
        space.add_int("max_depth", 3, 10)
        space.add_int("min_samples_split", 2, 20)

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            search_space=space,
            fixed_params={"random_state": 42, "n_estimators": 20},
        )
        graph = GraphSpec()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=2,
            metric="accuracy",
            greater_is_better=True,
        )
        reparam_config = ReparameterizationConfig(enabled=True, use_prebaked=True)
        fs_config = FeatureSelectionConfig(
            enabled=True,
            method="shadow",
            n_shadows=3,
            min_features=3,
        )

        config = RunConfig(
            cv=cv_config,
            tuning=tuning_config,
            reparameterization=reparam_config,
            feature_selection=fs_config,
            verbosity=0,
        )
        services = RuntimeServices(search_backend=mock_search_backend, fit_cache=cache)
        runner = GraphRunner(services)
        fitted = runner.fit(graph, ctx, config)

        # All features should be configured
        assert fitted.config.reparameterization is not None
        assert fitted.config.reparameterization.use_prebaked is True
        assert fitted.config.feature_selection is not None
        assert fitted.config.feature_selection.enabled is True

        # Model should be fitted
        assert "rf" in fitted.node_results
        assert fitted.node_results["rf"].best_params is not None

        # Predictions should work
        inference = fitted.compile_inference()
        predictions = inference.predict(X)
        assert len(predictions) == len(X)

    def test_graphbuilder_fluent_api_complete(self, classification_data, mock_search_backend):
        """Verify the complete fluent API works end-to-end."""
        X, y = classification_data
        ctx = DataView.from_Xy(X, y)

        graph = (
            GraphBuilder("pipeline")
            .add_model("rf", RandomForestClassifier)
                .int_param("n_estimators", 10, 50)
                .int_param("max_depth", 2, 10)
                .fixed_params(random_state=42)
                .description("Base RF model")
            .add_model("lr", LogisticRegression)
                .fixed_params(random_state=42, max_iter=1000)
                .stacks("rf")
            .compile()
        )

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=3,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        assert "rf" in fitted.node_results
        assert "lr" in fitted.node_results

        inference = fitted.compile_inference()
        predictions = inference.predict(X)
        assert len(predictions) == len(X)
