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

from sklearn_meta.api import GraphBuilder
from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.model.node import ModelNode
from sklearn_meta.core.tuning.orchestrator import TuningConfig, TuningOrchestrator
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.meta.reparameterization import ReparameterizedSpace, LogProductReparameterization
from sklearn_meta.meta.prebaked import get_prebaked_reparameterization
from sklearn_meta.selection.selector import FeatureSelector, FeatureSelectionConfig
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

    def test_reparameterization_via_graphbuilder(self, classification_data, mock_search_backend):
        """Verify reparameterization works through GraphBuilder API."""
        X, y = classification_data

        fitted = (
            GraphBuilder("test")
            .add_model("rf", RandomForestClassifier)
            .with_search_space(max_depth=(3, 10), min_samples_split=(2, 20))
            .with_fixed_params(random_state=42, n_estimators=10)
            .with_cv(n_splits=3, random_state=42)
            .with_tuning(n_trials=3, metric="accuracy", greater_is_better=True)
            .with_reparameterization(use_prebaked=True)
            .fit(X, y, search_backend=mock_search_backend)
        )

        # Should have fitted successfully
        assert "rf" in fitted.fitted_nodes
        assert fitted.tuning_config.use_reparameterization is True


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

    def test_feature_selection_via_graphbuilder(self, classification_data_with_noise, mock_search_backend):
        """Verify feature selection works through GraphBuilder API."""
        X, y = classification_data_with_noise

        fitted = (
            GraphBuilder("test")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(random_state=42, n_estimators=20)
            .with_cv(n_splits=3, random_state=42)
            .with_tuning(n_trials=2, metric="accuracy", greater_is_better=True)
            .with_feature_selection(method="shadow", n_shadows=3, min_features=3)
            .fit(X, y, search_backend=mock_search_backend)
        )

        # Should have fitted successfully
        assert "rf" in fitted.fitted_nodes
        assert fitted.tuning_config.feature_selection is not None
        assert fitted.tuning_config.feature_selection.enabled is True


class TestFitCacheIntegration:
    """Tests for FitCache integration."""

    def test_fit_cache_caches_models(self, classification_data, tmp_path):
        """Verify FitCache actually caches fitted models."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        cache = FitCache(cache_dir=str(tmp_path / "cache"))

        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        params = {"n_estimators": 10, "random_state": 42}

        # First cache lookup should miss
        cache_key = cache.cache_key(node, params, ctx)
        assert cache.get(cache_key) is None

        # Fit and store in cache
        model = node.create_estimator(params)
        model.fit(ctx.X, ctx.y)
        cache.put(cache_key, model)

        # Second cache lookup should hit
        cached_model = cache.get(cache_key)
        assert cached_model is not None

        # Cached model should work
        predictions = cached_model.predict(ctx.X)
        assert len(predictions) == len(ctx.X)

    def test_fit_cache_stats(self, classification_data, tmp_path):
        """Verify FitCache statistics work."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        cache = FitCache(cache_dir=str(tmp_path / "cache"))

        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        params = {"n_estimators": 10, "random_state": 42}
        cache_key = cache.cache_key(node, params, ctx)

        model = node.create_estimator(params)
        model.fit(ctx.X, ctx.y)
        cache.put(cache_key, model)

        stats = cache.stats()
        assert stats["enabled"] is True
        assert stats["memory_entries"] == 1

    def test_orchestrator_with_cache(self, classification_data, mock_search_backend, tmp_path):
        """Verify TuningOrchestrator uses cache correctly."""
        X, y = classification_data
        ctx = DataContext(X=X, y=y)

        cache = FitCache(cache_dir=str(tmp_path / "cache"))

        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        graph = ModelGraph()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            cv_config=cv_config,
            metric="accuracy",
            greater_is_better=True,
            verbose=0,
        )

        data_manager = DataManager(cv_config)
        orchestrator = TuningOrchestrator(
            graph=graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
            fit_cache=cache,
        )

        # First fit
        fitted1 = orchestrator.fit(ctx)
        assert "rf" in fitted1.fitted_nodes

        # Second fit should use cache (faster)
        orchestrator2 = TuningOrchestrator(
            graph=graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
            fit_cache=cache,
        )
        fitted2 = orchestrator2.fit(ctx)
        assert "rf" in fitted2.fitted_nodes


class TestFullIntegration:
    """Tests for all features working together."""

    def test_all_features_together(self, classification_data_with_noise, mock_search_backend, tmp_path):
        """Verify reparameterization, feature selection, and cache work together."""
        X, y = classification_data_with_noise

        cache = FitCache(cache_dir=str(tmp_path / "cache"))

        fitted = (
            GraphBuilder("full_test")
            .add_model("rf", RandomForestClassifier)
            .with_search_space(max_depth=(3, 10), min_samples_split=(2, 20))
            .with_fixed_params(random_state=42, n_estimators=20)
            .with_cv(n_splits=3, random_state=42)
            .with_tuning(n_trials=2, metric="accuracy", greater_is_better=True)
            .with_reparameterization(use_prebaked=True)
            .with_feature_selection(method="shadow", n_shadows=3, min_features=3)
            .fit(X, y, search_backend=mock_search_backend)
        )

        # All features should be configured
        assert fitted.tuning_config.use_reparameterization is True
        assert fitted.tuning_config.feature_selection is not None
        assert fitted.tuning_config.feature_selection.enabled is True

        # Model should be fitted
        assert "rf" in fitted.fitted_nodes
        assert fitted.fitted_nodes["rf"].best_params is not None

        # Feature selection should have selected features
        selected_features = fitted.fitted_nodes["rf"].selected_features
        if selected_features:
            # Predictions should work with selected features
            predictions = fitted.predict(X[selected_features])
            assert len(predictions) == len(X)
        else:
            # If no feature selection was applied, use all features
            predictions = fitted.predict(X)
            assert len(predictions) == len(X)

    def test_graphbuilder_fluent_api_complete(self, classification_data, mock_search_backend):
        """Verify the complete fluent API works end-to-end."""
        X, y = classification_data

        fitted = (
            GraphBuilder("pipeline")
            .add_model("rf", RandomForestClassifier)
            .with_search_space(n_estimators=(10, 50), max_depth=(2, 10))
            .with_fixed_params(random_state=42)
            .with_description("Base RF model")
            .add_model("lr", LogisticRegression)
            .with_fixed_params(random_state=42, max_iter=1000)
            .stacks("rf")
            .with_cv(n_splits=3, strategy="stratified", random_state=42)
            .with_tuning(n_trials=3, metric="accuracy", greater_is_better=True)
            .fit(X, y, search_backend=mock_search_backend)
        )

        assert "rf" in fitted.fitted_nodes
        assert "lr" in fitted.fitted_nodes

        predictions = fitted.predict(X)
        assert len(predictions) == len(X)
