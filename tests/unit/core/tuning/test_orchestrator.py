"""Tests for TuningOrchestrator."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from auto_sklearn.core.data.context import DataContext
from auto_sklearn.core.data.cv import CVConfig, CVStrategy
from auto_sklearn.core.data.manager import DataManager
from auto_sklearn.core.model.graph import ModelGraph
from auto_sklearn.core.model.node import ModelNode, OutputType
from auto_sklearn.core.model.dependency import DependencyEdge, DependencyType
from auto_sklearn.core.tuning.orchestrator import (
    FittedGraph,
    FittedNode,
    TuningConfig,
    TuningOrchestrator,
)
from auto_sklearn.core.tuning.strategy import OptimizationStrategy
from auto_sklearn.search.space import SearchSpace


class TestTuningConfig:
    """Tests for TuningConfig dataclass."""

    def test_default_values(self):
        """Verify default values are set correctly."""
        config = TuningConfig()

        assert config.strategy == OptimizationStrategy.LAYER_BY_LAYER
        assert config.n_trials == 100
        assert config.timeout is None
        assert config.metric == "neg_mean_squared_error"
        assert config.greater_is_better is False
        assert config.verbose == 1

    def test_custom_values(self):
        """Verify custom values are set correctly."""
        config = TuningConfig(
            strategy=OptimizationStrategy.GREEDY,
            n_trials=50,
            timeout=3600.0,
            metric="accuracy",
            greater_is_better=True,
            verbose=2,
        )

        assert config.strategy == OptimizationStrategy.GREEDY
        assert config.n_trials == 50
        assert config.timeout == 3600.0
        assert config.metric == "accuracy"
        assert config.greater_is_better is True
        assert config.verbose == 2


class TestFittedNode:
    """Tests for FittedNode dataclass."""

    def test_oof_predictions_property(self, small_context, cv_config_stratified):
        """Verify oof_predictions property works."""
        from auto_sklearn.core.data.cv import CVResult, FoldResult

        node = ModelNode(name="test", estimator_class=LogisticRegression)

        fold_results = [
            FoldResult(
                fold=None,
                model=None,
                val_predictions=np.array([0.5]),
                val_score=0.8,
            )
        ]

        cv_result = CVResult(
            fold_results=fold_results,
            oof_predictions=np.array([0.1, 0.2, 0.3]),
            node_name="test",
        )

        fitted = FittedNode(
            node=node,
            cv_result=cv_result,
            best_params={"C": 1.0},
        )

        np.testing.assert_array_equal(fitted.oof_predictions, np.array([0.1, 0.2, 0.3]))

    def test_mean_score_property(self, small_context, cv_config_stratified):
        """Verify mean_score property works."""
        from auto_sklearn.core.data.cv import CVResult, FoldResult, CVFold

        node = ModelNode(name="test", estimator_class=LogisticRegression)

        fold_results = [
            FoldResult(
                fold=CVFold(fold_idx=0, train_indices=np.array([0]), val_indices=np.array([1])),
                model=None,
                val_predictions=np.array([0.5]),
                val_score=0.8,
            ),
            FoldResult(
                fold=CVFold(fold_idx=1, train_indices=np.array([0]), val_indices=np.array([1])),
                model=None,
                val_predictions=np.array([0.5]),
                val_score=0.9,
            ),
        ]

        cv_result = CVResult(
            fold_results=fold_results,
            oof_predictions=np.array([0.1]),
            node_name="test",
        )

        fitted = FittedNode(
            node=node,
            cv_result=cv_result,
            best_params={},
        )

        assert fitted.mean_score == pytest.approx(0.85)


class TestFittedGraph:
    """Tests for FittedGraph dataclass."""

    def test_get_node(self, simple_graph, small_context, mock_search_backend):
        """Verify get_node returns correct fitted node."""
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
            graph=simple_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)
        fitted_node = fitted.get_node("rf")

        assert fitted_node.node.name == "rf"

    def test_get_oof_predictions(self, simple_graph, small_context, mock_search_backend):
        """Verify get_oof_predictions returns correct array."""
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
            graph=simple_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)
        oof = fitted.get_oof_predictions("rf")

        assert len(oof) == small_context.n_samples


class TestTuningOrchestratorFit:
    """Tests for TuningOrchestrator.fit()."""

    def test_fit_returns_fitted_graph(self, simple_graph, small_context, mock_search_backend):
        """Verify fit returns FittedGraph."""
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
            graph=simple_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)

        assert isinstance(fitted, FittedGraph)
        assert len(fitted.fitted_nodes) > 0

    def test_fit_records_time(self, simple_graph, small_context, mock_search_backend):
        """Verify fit records total time."""
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
            graph=simple_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)

        assert fitted.total_time > 0


class TestLayerByLayerStrategy:
    """Tests for layer-by-layer optimization strategy."""

    def test_layer_by_layer_fits_all_nodes(self, stacking_graph, small_context, mock_search_backend):
        """Verify all nodes are fitted with layer-by-layer strategy."""
        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=2,
            cv_config=cv_config,
            metric="accuracy",
            greater_is_better=True,
            verbose=0,
        )

        data_manager = DataManager(cv_config)
        orchestrator = TuningOrchestrator(
            graph=stacking_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)

        # All nodes should be fitted
        assert "rf_base" in fitted.fitted_nodes
        assert "lr_base" in fitted.fitted_nodes
        assert "meta" in fitted.fitted_nodes

    def test_layer_by_layer_dependencies_first(self, stacking_graph, small_context, mock_search_backend):
        """Verify dependencies are fitted before dependents."""
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
            graph=stacking_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)

        # Meta should see OOF from base models
        assert fitted.fitted_nodes["meta"] is not None


class TestOOFPredictions:
    """Tests for OOF prediction routing."""

    def test_oof_predictions_routed(self, simple_graph, small_context, mock_search_backend):
        """Verify OOF predictions are routed correctly."""
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
            graph=simple_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)
        oof = fitted.get_oof_predictions("rf")

        # OOF should have one prediction per sample
        assert len(oof) == small_context.n_samples

    def test_oof_no_leakage(self, small_context, mock_search_backend):
        """Verify OOF predictions don't leak training data."""
        # Create a graph with a model that overfits
        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 100, "max_depth": None, "random_state": 42},
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

        fitted = orchestrator.fit(small_context)
        oof = fitted.get_oof_predictions("rf")

        # OOF accuracy should be less than perfect (indicating no leakage)
        oof_accuracy = (oof == small_context.y.values).mean()
        assert oof_accuracy < 0.99, "Perfect OOF accuracy suggests data leakage"


class TestNoTuningStrategy:
    """Tests for no-tuning strategy."""

    def test_no_tuning_uses_fixed_params(self, simple_graph, small_context, mock_search_backend):
        """Verify no tuning uses only fixed params."""
        # Modify the node to have a search space but use NONE strategy
        graph = ModelGraph()
        space = SearchSpace().add_int("n_estimators", 10, 100)
        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            search_space=space,
            fixed_params={"n_estimators": 25, "random_state": 42},
        )
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

        fitted = orchestrator.fit(small_context)

        # Should use fixed params
        assert fitted.fitted_nodes["rf"].best_params["n_estimators"] == 25


class TestGreedyStrategy:
    """Tests for greedy optimization strategy."""

    def test_greedy_fits_all_nodes(self, two_model_graph, small_context, mock_search_backend):
        """Verify greedy strategy fits all nodes."""
        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.GREEDY,
            n_trials=2,
            cv_config=cv_config,
            metric="accuracy",
            greater_is_better=True,
            verbose=0,
        )

        data_manager = DataManager(cv_config)
        orchestrator = TuningOrchestrator(
            graph=two_model_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)

        assert "rf" in fitted.fitted_nodes
        assert "lr" in fitted.fitted_nodes


class TestConditionalNodes:
    """Tests for conditional node execution."""

    def test_conditional_node_skipped(self, small_context, mock_search_backend):
        """Verify conditional node is skipped when condition is False."""
        # Create node that should not run
        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
            condition=lambda ctx: ctx.n_samples > 10000,  # Will be False
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

        fitted = orchestrator.fit(small_context)

        # Node should not be fitted
        assert "rf" not in fitted.fitted_nodes

    def test_conditional_node_runs(self, small_context, mock_search_backend):
        """Verify conditional node runs when condition is True."""
        # Create node that should run
        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
            condition=lambda ctx: ctx.n_samples > 10,  # Will be True
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

        fitted = orchestrator.fit(small_context)

        # Node should be fitted
        assert "rf" in fitted.fitted_nodes


class TestFittedGraphPredict:
    """Tests for FittedGraph.predict()."""

    def test_predict_returns_array(self, simple_graph, small_context, mock_search_backend):
        """Verify predict returns prediction array."""
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
            graph=simple_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)
        predictions = fitted.predict(small_context.X)

        assert len(predictions) == small_context.n_samples

    def test_predict_uses_ensemble(self, simple_graph, small_context, mock_search_backend):
        """Verify predict uses ensemble of CV models."""
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
            graph=simple_graph,
            data_manager=data_manager,
            search_backend=mock_search_backend,
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(small_context)

        # Should have 5 models (one per fold)
        assert len(fitted.fitted_nodes["rf"].models) == 5

        # Predictions should work
        predictions = fitted.predict(small_context.X)
        assert len(predictions) == small_context.n_samples
