"""Integration tests for full pipeline execution."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import CVConfig, CVStrategy, RunConfig, TuningConfig
from sklearn_meta.engine.cv import CVEngine
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.spec.dependency import DependencyEdge, DependencyType
from sklearn_meta.engine.runner import GraphRunner
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.search.space import SearchSpace


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


def _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend, verbosity=0):
    """Helper to fit a graph using the new API."""
    config = RunConfig(cv=cv_config, tuning=tuning_config, verbosity=verbosity)
    services = RuntimeServices(search_backend=mock_search_backend)
    runner = GraphRunner(services)
    return runner.fit(graph, ctx, config)


class TestSimplePipeline:
    """Tests for simple single-model pipelines."""

    def test_simple_rf_pipeline(self, classification_pipeline_data, mock_search_backend):
        """Verify single RF model tunes and predicts correctly."""
        X, y = classification_pipeline_data
        ctx = DataView.from_Xy(X, y)

        # Create simple graph
        space = SearchSpace()
        space.add_int("n_estimators", 10, 50)
        space.add_int("max_depth", 2, 5)

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            search_space=space,
            fixed_params={"random_state": 42},
        )

        graph = GraphSpec()
        graph.add_node(node)

        # Configure tuning
        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=5,
            metric="accuracy",
            greater_is_better=True,
        )

        # Fit and verify
        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        assert "rf" in fitted.node_results
        assert fitted.node_results["rf"].best_params is not None
        assert fitted.node_results["rf"].cv_result is not None

        # Predictions should work
        inference = fitted.compile_inference()
        predictions = inference.predict(X)
        assert len(predictions) == len(X)

    def test_lr_pipeline_classification(self, classification_pipeline_data, mock_search_backend):
        """Verify Logistic Regression pipeline works."""
        X, y = classification_pipeline_data
        ctx = DataView.from_Xy(X, y)

        space = SearchSpace()
        space.add_float("C", 0.1, 10.0, log=True)

        node = NodeSpec(
            name="lr",
            estimator_class=LogisticRegression,
            search_space=space,
            fixed_params={"random_state": 42, "max_iter": 1000},
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

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        assert fitted.node_results["lr"].mean_score > 0.5  # Better than random


class TestTwoModelEnsemble:
    """Tests for two-model ensembles."""

    def test_two_model_ensemble_fits(self, classification_pipeline_data, mock_search_backend):
        """Verify two independent models can be fitted."""
        X, y = classification_pipeline_data
        ctx = DataView.from_Xy(X, y)

        # Create two independent nodes
        rf_space = SearchSpace().add_int("n_estimators", 5, 20)
        lr_space = SearchSpace().add_float("C", 0.1, 10.0)

        rf_node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            search_space=rf_space,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        lr_node = NodeSpec(
            name="lr",
            estimator_class=LogisticRegression,
            search_space=lr_space,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        graph = GraphSpec()
        graph.add_node(rf_node)
        graph.add_node(lr_node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=3,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        # Both models should be fitted
        assert "rf" in fitted.node_results
        assert "lr" in fitted.node_results
        assert fitted.node_results["rf"].mean_score > 0
        assert fitted.node_results["lr"].mean_score > 0


class TestStackingPipeline:
    """Tests for stacking pipelines."""

    def test_stacking_fits_layers(self, classification_pipeline_data, mock_search_backend):
        """Verify stacking fits base and meta layers correctly."""
        X, y = classification_pipeline_data
        ctx = DataView.from_Xy(X, y)

        # Base models
        rf_node = NodeSpec(
            name="rf_base",
            estimator_class=RandomForestClassifier,
            output_type=OutputType.PROBA,
            fixed_params={"n_estimators": 10, "random_state": 42, "max_depth": 3},
        )
        lr_node = NodeSpec(
            name="lr_base",
            estimator_class=LogisticRegression,
            output_type=OutputType.PROBA,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        # Meta model
        meta_node = NodeSpec(
            name="meta",
            estimator_class=LogisticRegression,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        graph = GraphSpec()
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
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        # All nodes should be fitted
        assert "rf_base" in fitted.node_results
        assert "lr_base" in fitted.node_results
        assert "meta" in fitted.node_results

    def test_stacking_oof_not_from_train(self, classification_pipeline_data, mock_search_backend):
        """Verify OOF predictions are from validation, not training."""
        X, y = classification_pipeline_data
        ctx = DataView.from_Xy(X, y)

        rf_node = NodeSpec(
            name="rf_base",
            estimator_class=RandomForestClassifier,
            output_type=OutputType.PROBA,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        graph = GraphSpec()
        graph.add_node(rf_node)

        cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=1,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        # Get OOF predictions
        oof = fitted.node_results["rf_base"].oof_predictions

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

    def test_stacking_allows_parallel_edges_with_distinct_column_names(
        self,
        classification_pipeline_data,
        mock_search_backend,
    ):
        """Verify a meta node can consume the same base model output twice via custom names."""
        X, y = classification_pipeline_data
        ctx = DataView.from_Xy(X, y)

        base_node = NodeSpec(
            name="rf_base",
            estimator_class=RandomForestClassifier,
            output_type=OutputType.PROBA,
            fixed_params={"n_estimators": 10, "random_state": 42, "max_depth": 3},
        )
        meta_node = NodeSpec(
            name="meta",
            estimator_class=LogisticRegression,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        graph = GraphSpec()
        graph.add_node(base_node)
        graph.add_node(meta_node)
        graph.add_edge(
            DependencyEdge(
                source="rf_base",
                target="meta",
                dep_type=DependencyType.PROBA,
                column_name="rf_stack_a",
            )
        )
        graph.add_edge(
            DependencyEdge(
                source="rf_base",
                target="meta",
                dep_type=DependencyType.PROBA,
                column_name="rf_stack_b",
            )
        )

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=1,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)
        inference = fitted.compile_inference()
        predictions = inference.predict(X, node_name="meta")

        assert "meta" in fitted.node_results
        assert len(predictions) == len(X)


class TestRegressionPipeline:
    """Tests for regression pipelines."""

    def test_rf_regression_pipeline(self, regression_pipeline_data, mock_search_backend):
        """Verify RF regression pipeline works."""
        X, y = regression_pipeline_data
        ctx = DataView.from_Xy(X, y)

        space = SearchSpace()
        space.add_int("n_estimators", 10, 50)

        node = NodeSpec(
            name="rf_reg",
            estimator_class=RandomForestRegressor,
            search_space=space,
            fixed_params={"random_state": 42, "max_depth": 5},
        )

        graph = GraphSpec()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=3,
            metric="neg_mean_squared_error",
            greater_is_better=False,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        assert "rf_reg" in fitted.node_results

        # Predictions should work
        inference = fitted.compile_inference()
        predictions = inference.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).all()


class TestNoTuningPipeline:
    """Tests for pipeline without hyperparameter tuning."""

    def test_no_tuning_uses_fixed_params(self, classification_pipeline_data, mock_search_backend):
        """Verify no tuning strategy uses fixed params only."""
        X, y = classification_pipeline_data
        ctx = DataView.from_Xy(X, y)

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 20, "max_depth": 3, "random_state": 42},
        )

        graph = GraphSpec()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        # Best params should be the fixed params
        assert fitted.node_results["rf"].best_params["n_estimators"] == 20
        assert fitted.node_results["rf"].best_params["max_depth"] == 3


class TestGreedyOptimization:
    """Tests for greedy optimization strategy."""

    def test_greedy_fits_nodes_sequentially(self, classification_pipeline_data, mock_search_backend):
        """Verify greedy strategy fits nodes one by one."""
        X, y = classification_pipeline_data
        ctx = DataView.from_Xy(X, y)

        rf_node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )
        lr_node = NodeSpec(
            name="lr",
            estimator_class=LogisticRegression,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        graph = GraphSpec()
        graph.add_node(rf_node)
        graph.add_node(lr_node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.GREEDY,
            n_trials=1,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        assert "rf" in fitted.node_results
        assert "lr" in fitted.node_results


class TestInferenceGraphPrediction:
    """Tests for InferenceGraph prediction."""

    def test_predict_uses_ensemble(self, classification_pipeline_data, mock_search_backend):
        """Verify prediction uses ensemble of CV models."""
        X, y = classification_pipeline_data
        ctx = DataView.from_Xy(X, y)

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        graph = GraphSpec()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        # Should have 5 models (one per fold)
        assert len(fitted.node_results["rf"].models) == 5

        # Prediction should work and return averaged predictions
        inference = fitted.compile_inference()
        predictions = inference.predict(X)
        assert predictions.shape == (len(X),)
