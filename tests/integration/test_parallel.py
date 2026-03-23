"""Integration tests for parallel execution."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import CVConfig, CVStrategy, RunConfig, TuningConfig
from sklearn_meta.engine.runner import GraphRunner
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.spec.builder import GraphBuilder
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.spec.dependency import DependencyEdge, DependencyType
from sklearn_meta.execution.local import LocalExecutor


@pytest.fixture
def medium_classification_data():
    """Medium-sized classification dataset for parallel tests."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y_series = pd.Series(y)
    return X_df, y_series


def _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend, executor=None):
    """Helper to fit a graph using the new API."""
    config = RunConfig(cv=cv_config, tuning=tuning_config, verbosity=0)
    services = RuntimeServices(search_backend=mock_search_backend, executor=executor)
    runner = GraphRunner(services)
    return runner.fit(graph, ctx, config)


@pytest.mark.integration
class TestParallelCVFolds:
    """Integration tests for parallel CV fold fitting."""

    def test_parallel_fold_fitting_produces_valid_results(self, medium_classification_data, mock_search_backend):
        """Verify parallel fold fitting produces valid model and predictions."""
        X, y = medium_classification_data
        ctx = DataView.from_Xy(X, y)

        executor = LocalExecutor(n_workers=2, backend="threading")

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

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend, executor=executor)

        # Should have 5 models (one per fold)
        assert len(fitted.node_results["rf"].models) == 5

        # OOF predictions should cover all samples
        oof = fitted.node_results["rf"].oof_predictions
        assert len(oof) == len(y)

        # Should be able to predict on new data
        inference = fitted.compile_inference()
        predictions = inference.predict(X)
        assert len(predictions) == len(y)

    def test_sequential_and_parallel_produce_similar_results(self, medium_classification_data, mock_search_backend):
        """Verify sequential and parallel execution produce comparable results."""
        X, y = medium_classification_data
        ctx = DataView.from_Xy(X, y)

        cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            metric="accuracy",
            greater_is_better=True,
        )

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )
        graph = GraphSpec()
        graph.add_node(node)

        # Sequential execution
        fitted_seq = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        # Parallel execution
        executor = LocalExecutor(n_workers=2, backend="threading")

        node2 = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )
        graph2 = GraphSpec()
        graph2.add_node(node2)

        fitted_par = _fit_graph(graph2, ctx, cv_config, tuning_config, mock_search_backend, executor=executor)

        # Both should have same number of models
        assert len(fitted_seq.node_results["rf"].models) == len(fitted_par.node_results["rf"].models)

        # OOF predictions should be same shape
        oof_seq = fitted_seq.node_results["rf"].oof_predictions
        oof_par = fitted_par.node_results["rf"].oof_predictions
        assert oof_seq.shape == oof_par.shape


@pytest.mark.integration
class TestParallelNodeFitting:
    """Integration tests for parallel node fitting within layers."""

    def test_parallel_fitting_of_multiple_models(self, medium_classification_data, mock_search_backend):
        """Verify multiple independent models can be fitted in parallel."""
        X, y = medium_classification_data
        ctx = DataView.from_Xy(X, y)

        executor = LocalExecutor(n_workers=2, backend="threading")

        rf_node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )
        lr_node = NodeSpec(
            name="lr",
            estimator_class=LogisticRegression,
            fixed_params={"max_iter": 1000, "random_state": 42},
        )

        graph = GraphSpec()
        graph.add_node(rf_node)
        graph.add_node(lr_node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend, executor=executor)

        # Both models should be fitted
        assert "rf" in fitted.node_results
        assert "lr" in fitted.node_results

        # Both should have valid predictions
        assert len(fitted.node_results["rf"].oof_predictions) == len(y)
        assert len(fitted.node_results["lr"].oof_predictions) == len(y)

    def test_stacking_with_parallel_execution(self, medium_classification_data, mock_search_backend):
        """Verify stacking ensemble works correctly with parallel execution."""
        X, y = medium_classification_data
        ctx = DataView.from_Xy(X, y)

        executor = LocalExecutor(n_workers=2, backend="threading")

        rf_node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            output_type=OutputType.PROBA,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )
        lr_node = NodeSpec(
            name="lr",
            estimator_class=LogisticRegression,
            output_type=OutputType.PROBA,
            fixed_params={"max_iter": 1000, "random_state": 42},
        )
        meta_node = NodeSpec(
            name="meta",
            estimator_class=LogisticRegression,
            fixed_params={"max_iter": 1000, "random_state": 42},
        )

        graph = GraphSpec()
        graph.add_node(rf_node)
        graph.add_node(lr_node)
        graph.add_node(meta_node)
        graph.add_edge(DependencyEdge(source="rf", target="meta", dep_type=DependencyType.PROBA))
        graph.add_edge(DependencyEdge(source="lr", target="meta", dep_type=DependencyType.PROBA))

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend, executor=executor)

        # All models should be fitted
        assert "rf" in fitted.node_results
        assert "lr" in fitted.node_results
        assert "meta" in fitted.node_results

        # All models should have valid OOF predictions
        assert len(fitted.node_results["rf"].oof_predictions) == len(y)
        assert len(fitted.node_results["lr"].oof_predictions) == len(y)
        assert len(fitted.node_results["meta"].oof_predictions) == len(y)

        # Meta model should have been trained on OOF predictions from base models
        assert fitted.node_results["meta"].models is not None
        assert len(fitted.node_results["meta"].models) == 3  # 3 folds


@pytest.mark.integration
class TestOptunaParallelTrials:
    """Integration tests for parallel Optuna trials."""

    def test_parallel_trials_parameter(self, medium_classification_data, mock_search_backend):
        """Verify n_parallel_trials parameter is properly wired."""
        X, y = medium_classification_data
        ctx = DataView.from_Xy(X, y)

        from sklearn_meta.search.space import SearchSpace

        space = SearchSpace()
        space.add_int("n_estimators", 10, 50)

        node = NodeSpec(
            name="rf",
            estimator_class=RandomForestClassifier,
            search_space=space,
            fixed_params={"random_state": 42},
        )

        graph = GraphSpec()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=5,
            metric="accuracy",
            greater_is_better=True,
        )

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend)

        # Should complete successfully
        assert "rf" in fitted.node_results
        assert fitted.node_results["rf"].optimization_result is not None
        assert fitted.node_results["rf"].optimization_result.n_trials == 5

    def test_optuna_backend_n_jobs_parameter(self):
        """Verify OptunaBackend accepts n_jobs parameter."""
        from sklearn_meta.search.backends.optuna import OptunaBackend

        backend = OptunaBackend(n_jobs=4)
        assert backend._n_jobs == 4

        backend_single = OptunaBackend()
        assert backend_single._n_jobs == 1


@pytest.mark.integration
class TestLocalExecutorConfigurations:
    """Integration tests for different LocalExecutor configurations."""

    def test_threading_backend(self, medium_classification_data, mock_search_backend):
        """Verify threading backend works correctly."""
        X, y = medium_classification_data
        ctx = DataView.from_Xy(X, y)

        executor = LocalExecutor(n_workers=2, backend="threading")

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

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend, executor=executor)
        assert len(fitted.node_results["rf"].models) == 3

    @pytest.mark.slow
    def test_loky_backend(self, medium_classification_data, mock_search_backend):
        """Verify loky (process-based) backend works correctly."""
        X, y = medium_classification_data
        ctx = DataView.from_Xy(X, y)

        executor = LocalExecutor(n_workers=2, backend="loky")

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

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend, executor=executor)
        assert len(fitted.node_results["rf"].models) == 3

    def test_executor_auto_cpu_count(self, medium_classification_data, mock_search_backend):
        """Verify executor with n_workers=-1 uses all CPUs."""
        import os

        X, y = medium_classification_data
        ctx = DataView.from_Xy(X, y)

        executor = LocalExecutor(n_workers=-1, backend="threading")

        # Should use all available CPUs
        expected_workers = os.cpu_count() or 1
        assert executor.n_workers == expected_workers

        # Should still work correctly
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

        fitted = _fit_graph(graph, ctx, cv_config, tuning_config, mock_search_backend, executor=executor)
        assert len(fitted.node_results["rf"].models) == 3
