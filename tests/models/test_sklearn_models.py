"""Tests for compatibility with various sklearn models."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    SGDClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.node import ModelNode, OutputType
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.tuning.orchestrator import TuningConfig, TuningOrchestrator
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.search.space import SearchSpace


@pytest.fixture
def classification_data_small():
    """Small classification dataset for model tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


@pytest.fixture
def regression_data_small():
    """Small regression dataset for model tests."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


class TestRandomForestClassifier:
    """Tests for RandomForestClassifier compatibility."""

    def test_rf_classifier_node_creation(self):
        """Verify RF classifier node can be created."""
        space = SearchSpace()
        space.add_int("n_estimators", 10, 100)
        space.add_int("max_depth", 2, 10)

        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            search_space=space,
            fixed_params={"random_state": 42},
        )

        assert node.name == "rf"
        assert node.has_search_space

    def test_rf_classifier_fits_and_predicts(self, classification_data_small):
        """Verify RF classifier fits and predicts correctly."""
        X, y = classification_data_small

        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_rf_classifier_proba_output(self, classification_data_small):
        """Verify RF classifier produces probability output."""
        X, y = classification_data_small

        node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            output_type=OutputType.PROBA,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        model = node.create_estimator()
        model.fit(X, y)

        proba = node.get_output(model, X)

        assert proba.shape == (len(X), 2)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(len(X)))


class TestGradientBoostingClassifier:
    """Tests for GradientBoostingClassifier compatibility."""

    def test_gbc_node_creation(self):
        """Verify GBC node can be created."""
        space = SearchSpace()
        space.add_float("learning_rate", 0.01, 0.3)
        space.add_int("n_estimators", 10, 100)

        node = ModelNode(
            name="gbc",
            estimator_class=GradientBoostingClassifier,
            search_space=space,
            fixed_params={"random_state": 42},
        )

        assert node.name == "gbc"

    def test_gbc_fits_and_predicts(self, classification_data_small):
        """Verify GBC fits and predicts correctly."""
        X, y = classification_data_small

        node = ModelNode(
            name="gbc",
            estimator_class=GradientBoostingClassifier,
            fixed_params={"n_estimators": 10, "max_depth": 3, "random_state": 42},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)


class TestLogisticRegression:
    """Tests for LogisticRegression compatibility."""

    def test_lr_node_creation(self):
        """Verify LR node can be created."""
        space = SearchSpace()
        space.add_float("C", 0.01, 10.0, log=True)

        node = ModelNode(
            name="lr",
            estimator_class=LogisticRegression,
            search_space=space,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        assert node.name == "lr"

    def test_lr_fits_and_predicts(self, classification_data_small):
        """Verify LR fits and predicts correctly."""
        X, y = classification_data_small

        node = ModelNode(
            name="lr",
            estimator_class=LogisticRegression,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)

    def test_lr_proba_output(self, classification_data_small):
        """Verify LR produces probability output."""
        X, y = classification_data_small

        node = ModelNode(
            name="lr",
            estimator_class=LogisticRegression,
            output_type=OutputType.PROBA,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        model = node.create_estimator()
        model.fit(X, y)

        proba = node.get_output(model, X)

        assert proba.shape == (len(X), 2)


class TestSVMClassifier:
    """Tests for SVC compatibility."""

    def test_svc_node_creation(self):
        """Verify SVC node can be created."""
        space = SearchSpace()
        space.add_float("C", 0.1, 10.0)
        space.add_float("gamma", 0.001, 1.0, log=True)

        node = ModelNode(
            name="svc",
            estimator_class=SVC,
            search_space=space,
            fixed_params={"random_state": 42, "probability": True},
        )

        assert node.name == "svc"

    def test_svc_fits_and_predicts(self, classification_data_small):
        """Verify SVC fits and predicts correctly."""
        X, y = classification_data_small

        node = ModelNode(
            name="svc",
            estimator_class=SVC,
            fixed_params={"random_state": 42},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)

    def test_svc_proba_requires_probability(self, classification_data_small):
        """Verify SVC with probability=True works for proba output."""
        X, y = classification_data_small

        node = ModelNode(
            name="svc",
            estimator_class=SVC,
            output_type=OutputType.PROBA,
            fixed_params={"random_state": 42, "probability": True},
        )

        model = node.create_estimator()
        model.fit(X, y)

        proba = node.get_output(model, X)

        assert proba.shape == (len(X), 2)


class TestMLPClassifier:
    """Tests for MLPClassifier compatibility."""

    def test_mlp_node_creation(self):
        """Verify MLP node can be created."""
        space = SearchSpace()
        space.add_float("alpha", 0.0001, 0.1, log=True)
        space.add_categorical("activation", ["relu", "tanh"])

        node = ModelNode(
            name="mlp",
            estimator_class=MLPClassifier,
            search_space=space,
            fixed_params={"random_state": 42, "max_iter": 500},
        )

        assert node.name == "mlp"

    def test_mlp_fits_and_predicts(self, classification_data_small):
        """Verify MLP fits and predicts correctly."""
        X, y = classification_data_small

        node = ModelNode(
            name="mlp",
            estimator_class=MLPClassifier,
            fixed_params={
                "random_state": 42,
                "max_iter": 200,
                "hidden_layer_sizes": (10,),
            },
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)


class TestDecisionTreeClassifier:
    """Tests for DecisionTreeClassifier compatibility."""

    def test_dt_node_creation(self):
        """Verify DT node can be created."""
        space = SearchSpace()
        space.add_int("max_depth", 2, 20)
        space.add_int("min_samples_split", 2, 20)

        node = ModelNode(
            name="dt",
            estimator_class=DecisionTreeClassifier,
            search_space=space,
            fixed_params={"random_state": 42},
        )

        assert node.name == "dt"

    def test_dt_fits_and_predicts(self, classification_data_small):
        """Verify DT fits and predicts correctly."""
        X, y = classification_data_small

        node = ModelNode(
            name="dt",
            estimator_class=DecisionTreeClassifier,
            fixed_params={"random_state": 42, "max_depth": 5},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)


class TestExtraTreesClassifier:
    """Tests for ExtraTreesClassifier compatibility."""

    def test_etc_fits_and_predicts(self, classification_data_small):
        """Verify ExtraTrees fits and predicts correctly."""
        X, y = classification_data_small

        node = ModelNode(
            name="etc",
            estimator_class=ExtraTreesClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)


class TestAdaBoostClassifier:
    """Tests for AdaBoostClassifier compatibility."""

    def test_adaboost_fits_and_predicts(self, classification_data_small):
        """Verify AdaBoost fits and predicts correctly."""
        X, y = classification_data_small

        node = ModelNode(
            name="adaboost",
            estimator_class=AdaBoostClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)


class TestSGDClassifier:
    """Tests for SGDClassifier compatibility."""

    def test_sgd_fits_and_predicts(self, classification_data_small):
        """Verify SGD fits and predicts correctly."""
        X, y = classification_data_small

        node = ModelNode(
            name="sgd",
            estimator_class=SGDClassifier,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)


class TestRegressionModels:
    """Tests for regression model compatibility."""

    def test_rf_regressor_fits_and_predicts(self, regression_data_small):
        """Verify RF regressor fits and predicts correctly."""
        X, y = regression_data_small

        node = ModelNode(
            name="rf_reg",
            estimator_class=RandomForestRegressor,
            fixed_params={"n_estimators": 10, "random_state": 42},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)
        assert np.isfinite(predictions).all()

    def test_ridge_fits_and_predicts(self, regression_data_small):
        """Verify Ridge regressor fits and predicts correctly."""
        X, y = regression_data_small

        node = ModelNode(
            name="ridge",
            estimator_class=Ridge,
            fixed_params={"alpha": 1.0, "random_state": 42},
        )

        model = node.create_estimator()
        model.fit(X, y)

        predictions = node.get_output(model, X)

        assert predictions.shape == (len(X),)
        assert np.isfinite(predictions).all()


class TestModelInPipeline:
    """Tests for models working within the full pipeline."""

    def test_multiple_models_in_pipeline(self, classification_data_small, mock_search_backend):
        """Verify multiple different model types work in pipeline."""
        X, y = classification_data_small
        ctx = DataContext(X=X, y=y)

        # Create nodes with different model types
        rf_node = ModelNode(
            name="rf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 5, "random_state": 42},
        )
        lr_node = ModelNode(
            name="lr",
            estimator_class=LogisticRegression,
            fixed_params={"random_state": 42, "max_iter": 1000},
        )
        dt_node = ModelNode(
            name="dt",
            estimator_class=DecisionTreeClassifier,
            fixed_params={"max_depth": 3, "random_state": 42},
        )

        graph = ModelGraph()
        graph.add_node(rf_node)
        graph.add_node(lr_node)
        graph.add_node(dt_node)

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

        # All models should be fitted
        assert "rf" in fitted.fitted_nodes
        assert "lr" in fitted.fitted_nodes
        assert "dt" in fitted.fitted_nodes

        # All should have reasonable scores
        for name in ["rf", "lr", "dt"]:
            assert fitted.fitted_nodes[name].mean_score > 0.5
