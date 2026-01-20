"""Integration tests for full pipeline execution."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error

from auto_sklearn.core.data.context import DataContext
from auto_sklearn.core.data.cv import CVConfig, CVStrategy
from auto_sklearn.core.data.manager import DataManager
from auto_sklearn.core.model.graph import ModelGraph
from auto_sklearn.core.model.node import ModelNode, OutputType
from auto_sklearn.core.model.dependency import DependencyEdge, DependencyType
from auto_sklearn.core.tuning.orchestrator import TuningConfig, TuningOrchestrator
from auto_sklearn.core.tuning.strategy import OptimizationStrategy
from auto_sklearn.search.space import SearchSpace


@pytest.fixture
def classification_pipeline_data():
    """Generate classification data for pipeline tests."""
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        n_classes=2,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(15)]), pd.Series(y)


@pytest.fixture
def regression_pipeline_data():
    """Generate regression data for pipeline tests."""
    X, y = make_regression(
        n_samples=500,
        n_features=15,
        n_informative=8,
        noise=0.5,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(15)]), pd.Series(y)


class TestSimplePipeline:
    """Tests for simple single-model pipelines."""

    def test_simple_rf_pipeline(self, classification_pipeline_data, mock_search_backend):
        """Verify single RF model tunes and predicts correctly."""
        X, y = classification_pipeline_data
        ctx = DataContext(X=X, y=y)

        # Create simple graph
        space = SearchSpace()
        space.add_int("n_estimators", 10, 50)
        space.add_int("max_depth", 2, 5)

        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            search_space=space,
            fixed_params={"random_state": 42},
        )

        graph = ModelGraph()
        graph.add_node(node)

        # Configure tuning
        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=5,
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
        )

        # Fit and verify
        fitted = orchestrator.fit(ctx)

        assert "rf" in fitted.fitted_nodes
        assert fitted.fitted_nodes["rf"].best_params is not None
        assert fitted.fitted_nodes["rf"].cv_result is not None

        # Predictions should work
        predictions = fitted.predict(X)
        assert len(predictions) == len(X)

    def test_lr_pipeline_classification(self, classification_pipeline_data, mock_search_backend):
        """Verify Logistic Regression pipeline works."""
        X, y = classification_pipeline_data
        ctx = DataContext(X=X, y=y)

        space = SearchSpace()
        space.add_float("C", 0.1, 10.0, log=True)

        node = ModelNode(
            name="lr",
            estimator_class=LogisticRegression,
            search_space=space,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        graph = ModelGraph()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=3,
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
        )

        fitted = orchestrator.fit(ctx)

        assert fitted.fitted_nodes["lr"].mean_score > 0.5  # Better than random


class TestTwoModelEnsemble:
    """Tests for two-model ensembles."""

    def test_two_model_ensemble_fits(self, classification_pipeline_data, mock_search_backend):
        """Verify two independent models can be fitted."""
        X, y = classification_pipeline_data
        ctx = DataContext(X=X, y=y)

        # Create two independent nodes
        rf_space = SearchSpace().add_int("n_estimators", 5, 20)
        lr_space = SearchSpace().add_float("C", 0.1, 10.0)

        rf_node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            search_space=rf_space,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        lr_node = ModelNode(
            name="lr",
            estimator_class=LogisticRegression,
            search_space=lr_space,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        graph = ModelGraph()
        graph.add_node(rf_node)
        graph.add_node(lr_node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=3,
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
        )

        fitted = orchestrator.fit(ctx)

        # Both models should be fitted
        assert "rf" in fitted.fitted_nodes
        assert "lr" in fitted.fitted_nodes
        assert fitted.fitted_nodes["rf"].mean_score > 0
        assert fitted.fitted_nodes["lr"].mean_score > 0


class TestStackingPipeline:
    """Tests for stacking pipelines."""

    def test_stacking_fits_layers(self, classification_pipeline_data, mock_search_backend):
        """Verify stacking fits base and meta layers correctly."""
        X, y = classification_pipeline_data
        ctx = DataContext(X=X, y=y)

        # Base models
        rf_node = ModelNode(
            name="rf_base",
            estimator_class=RandomForestClassifier,
            output_type=OutputType.PROBA,
            fixed_params={"n_estimators": 10, "random_state": 42, "max_depth": 3},
        )
        lr_node = ModelNode(
            name="lr_base",
            estimator_class=LogisticRegression,
            output_type=OutputType.PROBA,
            fixed_params={"random_state": 42, "max_iter": 1000},
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

        # Stacking edges
        graph.add_edge(DependencyEdge(source="rf_base", target="meta", dep_type=DependencyType.PROBA))
        graph.add_edge(DependencyEdge(source="lr_base", target="meta", dep_type=DependencyType.PROBA))

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=1,
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
        )

        fitted = orchestrator.fit(ctx)

        # All nodes should be fitted
        assert "rf_base" in fitted.fitted_nodes
        assert "lr_base" in fitted.fitted_nodes
        assert "meta" in fitted.fitted_nodes

    def test_stacking_oof_not_from_train(self, classification_pipeline_data, mock_search_backend):
        """Verify OOF predictions are from validation, not training."""
        X, y = classification_pipeline_data
        ctx = DataContext(X=X, y=y)

        rf_node = ModelNode(
            name="rf_base",
            estimator_class=RandomForestClassifier,
            output_type=OutputType.PROBA,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        graph = ModelGraph()
        graph.add_node(rf_node)

        cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=1,
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
        )

        fitted = orchestrator.fit(ctx)

        # Get OOF predictions
        oof = fitted.get_oof_predictions("rf_base")

        # OOF should have shape matching data
        assert oof.shape[0] == len(X)

        # OOF should not be perfect (would indicate data leakage)
        if oof.ndim > 1:
            oof_preds = np.argmax(oof, axis=1)
        else:
            oof_preds = (oof > 0.5).astype(int)

        accuracy = (oof_preds == y.values).mean()
        # Should be good but not perfect
        assert 0.6 < accuracy < 0.99


class TestRegressionPipeline:
    """Tests for regression pipelines."""

    def test_rf_regression_pipeline(self, regression_pipeline_data, mock_search_backend):
        """Verify RF regression pipeline works."""
        X, y = regression_pipeline_data
        ctx = DataContext(X=X, y=y)

        space = SearchSpace()
        space.add_int("n_estimators", 10, 50)

        node = ModelNode(
            name="rf_reg",
            estimator_class=RandomForestRegressor,
            search_space=space,
            fixed_params={"random_state": 42, "max_depth": 5},
        )

        graph = ModelGraph()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=3,
            cv_config=cv_config,
            metric="neg_mean_squared_error",
            greater_is_better=False,
            verbose=0,
        )

        data_manager = DataManager(cv_config)
        orchestrator = TuningOrchestrator(
            graph=graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(ctx)

        assert "rf_reg" in fitted.fitted_nodes

        # Predictions should work
        predictions = fitted.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).all()


class TestNoTuningPipeline:
    """Tests for pipeline without hyperparameter tuning."""

    def test_no_tuning_uses_fixed_params(self, classification_pipeline_data, mock_search_backend):
        """Verify no tuning strategy uses fixed params only."""
        X, y = classification_pipeline_data
        ctx = DataContext(X=X, y=y)

        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 20, "max_depth": 3, "random_state": 42},
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
        )

        fitted = orchestrator.fit(ctx)

        # Best params should be the fixed params
        assert fitted.fitted_nodes["rf"].best_params["n_estimators"] == 20
        assert fitted.fitted_nodes["rf"].best_params["max_depth"] == 3


class TestGreedyOptimization:
    """Tests for greedy optimization strategy."""

    def test_greedy_fits_nodes_sequentially(self, classification_pipeline_data, mock_search_backend):
        """Verify greedy strategy fits nodes one by one."""
        X, y = classification_pipeline_data
        ctx = DataContext(X=X, y=y)

        rf_node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )
        lr_node = ModelNode(
            name="lr",
            estimator_class=LogisticRegression,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        graph = ModelGraph()
        graph.add_node(rf_node)
        graph.add_node(lr_node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.GREEDY,
            n_trials=1,
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
        )

        fitted = orchestrator.fit(ctx)

        assert "rf" in fitted.fitted_nodes
        assert "lr" in fitted.fitted_nodes


class TestFittedGraphPrediction:
    """Tests for FittedGraph prediction."""

    def test_predict_uses_ensemble(self, classification_pipeline_data, mock_search_backend):
        """Verify prediction uses ensemble of CV models."""
        X, y = classification_pipeline_data
        ctx = DataContext(X=X, y=y)

        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        graph = ModelGraph()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
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
        )

        fitted = orchestrator.fit(ctx)

        # Should have 5 models (one per fold)
        assert len(fitted.fitted_nodes["rf"].models) == 5

        # Prediction should work and return averaged predictions
        predictions = fitted.predict(X)
        assert predictions.shape == (len(X),)
