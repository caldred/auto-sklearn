"""Tests for JointQuantileOrchestrator."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.joint_quantile_graph import JointQuantileConfig, JointQuantileGraph
from sklearn_meta.core.tuning.joint_quantile_orchestrator import (
    FittedQuantileNode,
    JointQuantileFitResult,
    JointQuantileOrchestrator,
)
from sklearn_meta.core.tuning.orchestrator import TuningConfig
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.selection.selector import (
    FeatureSelectionConfig,
    FeatureSelectionResult,
    FeatureSelector,
)


# =============================================================================
# Mock classes
# =============================================================================


class MockQuantileRegressor:
    """Mock quantile regressor for testing."""

    def __init__(
        self,
        objective="reg:squarederror",
        quantile_alpha=0.5,
        n_estimators=100,
        max_depth=6,
        **kwargs,
    ):
        self.objective = objective
        self.quantile_alpha = quantile_alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._fitted = False
        self._X_train = None

    def fit(self, X, y, **fit_params):
        self._fitted = True
        self._X_train = X
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)
        return self

    def predict(self, X):
        # Return median-adjusted predictions
        offset = (self.quantile_alpha - 0.5) * self._y_std * 2
        return np.full(len(X), self._y_mean + offset)


class MockSearchBackend:
    """Mock search backend for testing."""

    def __init__(self, best_params=None):
        self._best_params = best_params or {}

    def optimize(
        self,
        objective,
        search_space,
        n_trials=10,
        timeout=None,
        callbacks=None,
        study_name="test",
        early_stopping_rounds=None,
    ):
        from sklearn_meta.search.backends.base import OptimizationResult, TrialResult

        # Just run objective once with default params
        params = self._best_params or {}
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
def joint_regression_data():
    """Create synthetic data with correlated targets."""
    np.random.seed(42)
    n_samples = 200

    # Generate features
    X = np.random.randn(n_samples, 5)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

    # Generate correlated targets
    # Y1 depends on X
    y1 = X[:, 0] * 2 + X[:, 1] + np.random.randn(n_samples) * 0.5

    # Y2 depends on X and Y1
    y2 = X[:, 2] + 0.5 * y1 + np.random.randn(n_samples) * 0.5

    # Y3 depends on X, Y1, and Y2
    y3 = X[:, 3] - X[:, 4] + 0.3 * y1 + 0.4 * y2 + np.random.randn(n_samples) * 0.5

    targets = {
        "y1": pd.Series(y1),
        "y2": pd.Series(y2),
        "y3": pd.Series(y3),
    }

    return X_df, targets


@pytest.fixture
def joint_quantile_graph():
    """Create a JointQuantileGraph for testing."""
    config = JointQuantileConfig(
        property_names=["y1", "y2", "y3"],
        estimator_class=MockQuantileRegressor,
        quantile_levels=[0.1, 0.5, 0.9],
        n_inference_samples=100,
    )
    return JointQuantileGraph(config)


@pytest.fixture
def single_property_graph_with_search_space():
    """Create a single-property graph with a tunable search space."""
    search_space = SearchSpace().add_from_shorthand(max_depth=(3, 6))
    config = JointQuantileConfig(
        property_names=["y1"],
        estimator_class=MockQuantileRegressor,
        quantile_levels=[0.1, 0.5, 0.9],
        search_space=search_space,
        n_inference_samples=100,
    )
    return JointQuantileGraph(config)


@pytest.fixture
def data_manager():
    """Create a DataManager for testing."""
    cv_config = CVConfig(
        n_splits=3,
        strategy=CVStrategy.RANDOM,
        shuffle=True,
        random_state=42,
    )
    return DataManager(cv_config)


@pytest.fixture
def tuning_config():
    """Create a TuningConfig for testing."""
    return TuningConfig(
        strategy=OptimizationStrategy.NONE,
        n_trials=1,
        metric="neg_mean_squared_error",
        greater_is_better=False,
        verbose=0,
        cv_config=CVConfig(
            n_splits=3,
            strategy=CVStrategy.RANDOM,
            shuffle=True,
            random_state=42,
        ),
    )


@pytest.fixture
def mock_backend():
    """Create a mock search backend."""
    return MockSearchBackend()


# =============================================================================
# FittedQuantileNode Tests
# =============================================================================


class TestFittedQuantileNodeProperties:
    """Tests for FittedQuantileNode properties."""

    def test_quantile_levels(self, joint_quantile_graph):
        """Verify quantile_levels property."""
        node = joint_quantile_graph.get_quantile_node("y1")

        fitted = FittedQuantileNode(
            node=node,
            quantile_models={0.1: [None], 0.5: [None], 0.9: [None]},
            oof_quantile_predictions=np.zeros((100, 3)),
            best_params={},
        )

        assert fitted.quantile_levels == [0.1, 0.5, 0.9]

    def test_n_quantiles(self, joint_quantile_graph):
        """Verify n_quantiles property."""
        node = joint_quantile_graph.get_quantile_node("y1")

        fitted = FittedQuantileNode(
            node=node,
            quantile_models={0.1: [None], 0.5: [None], 0.9: [None]},
            oof_quantile_predictions=np.zeros((100, 3)),
            best_params={},
        )

        assert fitted.n_quantiles == 3

    def test_n_folds(self, joint_quantile_graph):
        """Verify n_folds property."""
        node = joint_quantile_graph.get_quantile_node("y1")

        fitted = FittedQuantileNode(
            node=node,
            quantile_models={
                0.1: [None, None, None],  # 3 folds
                0.5: [None, None, None],
                0.9: [None, None, None],
            },
            oof_quantile_predictions=np.zeros((100, 3)),
            best_params={},
        )

        assert fitted.n_folds == 3

    def test_predict_quantiles_uses_selected_features(self, joint_quantile_graph):
        """Inference should use the selected feature subset when present."""

        class SpyModel:
            def __init__(self):
                self.seen_columns = None

            def predict(self, X):
                self.seen_columns = list(X.columns)
                return np.zeros(len(X))

        node = joint_quantile_graph.get_quantile_node("y1")
        model = SpyModel()
        fitted = FittedQuantileNode(
            node=node,
            quantile_models={0.5: [model]},
            oof_quantile_predictions=np.zeros((5, 1)),
            best_params={},
            selected_features=["feature_1", "feature_3"],
        )

        X = pd.DataFrame(
            np.random.randn(5, 5),
            columns=[f"feature_{i}" for i in range(5)],
        )
        _ = fitted.predict_quantiles(X)

        assert model.seen_columns == ["feature_1", "feature_3"]

    def test_predict_quantiles_missing_selected_feature_raises(self, joint_quantile_graph):
        """Inference should fail fast when a required selected feature is missing."""
        node = joint_quantile_graph.get_quantile_node("y1")
        fitted = FittedQuantileNode(
            node=node,
            quantile_models={0.5: [MockQuantileRegressor()]},
            oof_quantile_predictions=np.zeros((5, 1)),
            best_params={},
            selected_features=["missing_feature"],
        )

        X = pd.DataFrame(np.random.randn(5, 2), columns=["feature_0", "feature_1"])
        with pytest.raises(ValueError, match="Missing required selected feature columns"):
            fitted.predict_quantiles(X)

    def test_predict_quantiles_backward_compat_without_selected_features(
        self, joint_quantile_graph
    ):
        """Older serialized objects without selected_features should still infer."""
        class ConstantModel:
            def predict(self, X):
                return np.zeros(len(X))

        node = joint_quantile_graph.get_quantile_node("y1")
        fitted = FittedQuantileNode(
            node=node,
            quantile_models={0.5: [ConstantModel()]},
            oof_quantile_predictions=np.zeros((5, 1)),
            best_params={},
        )
        # Simulate legacy payloads saved before selected_features existed.
        delattr(fitted, "selected_features")

        X = pd.DataFrame(np.random.randn(5, 2), columns=["feature_0", "feature_1"])
        preds = fitted.predict_quantiles(X)
        assert preds.shape == (5, 1)


# =============================================================================
# JointQuantileOrchestrator Tests
# =============================================================================


class TestJointQuantileOrchestratorCreation:
    """Tests for JointQuantileOrchestrator creation."""

    def test_basic_creation(
        self, joint_quantile_graph, data_manager, mock_backend, tuning_config
    ):
        """Verify basic orchestrator creation."""
        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        assert orchestrator.graph is joint_quantile_graph
        assert orchestrator.tuning_config is tuning_config


class TestJointQuantileOrchestratorFit:
    """Tests for fit method."""

    def test_fit_returns_result(
        self,
        joint_regression_data,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify fit returns JointQuantileFitResult."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])

        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        result = orchestrator.fit(ctx, targets)

        assert isinstance(result, JointQuantileFitResult)
        assert len(result.fitted_nodes) == 3

    def test_fit_all_properties_fitted(
        self,
        joint_regression_data,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify all properties are fitted."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])

        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        result = orchestrator.fit(ctx, targets)

        assert "y1" in result.fitted_nodes
        assert "y2" in result.fitted_nodes
        assert "y3" in result.fitted_nodes

    def test_fit_oof_predictions_shape(
        self,
        joint_regression_data,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify OOF predictions have correct shape."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])

        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        result = orchestrator.fit(ctx, targets)

        for prop_name, fitted in result.fitted_nodes.items():
            assert fitted.oof_quantile_predictions.shape == (len(X), 3)

    def test_fit_missing_target_raises(
        self,
        joint_regression_data,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify missing target raises error."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])

        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        # Remove one target
        incomplete_targets = {"y1": targets["y1"], "y2": targets["y2"]}

        with pytest.raises(ValueError, match="Missing target"):
            orchestrator.fit(ctx, incomplete_targets)


class TestJointQuantileOrchestratorFeatureSelection:
    """Tests for joint-quantile feature selection integration."""

    def test_feature_selection_runs_and_retunes_with_narrowed_space(
        self,
        joint_regression_data,
        single_property_graph_with_search_space,
        data_manager,
        mock_backend,
        monkeypatch,
    ):
        """Selection should run and trigger a narrowed-space median re-tune."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])
        fs_config = FeatureSelectionConfig(
            enabled=True,
            method="shadow",
            retune_after_pruning=True,
            min_features=1,
        )
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="neg_mean_squared_error",
            greater_is_better=False,
            verbose=0,
            cv_config=CVConfig(
                n_splits=3,
                strategy=CVStrategy.RANDOM,
                shuffle=True,
                random_state=42,
            ),
            feature_selection=fs_config,
        )

        optimize_calls = []

        def fake_optimize(self, node, ctx, tau, search_space_override=None):
            optimize_calls.append(search_space_override)
            return {"max_depth": 4}, None

        def fake_select(self, node, ctx, params):
            selected = [ctx.feature_cols[0], ctx.feature_cols[1]]
            return FeatureSelectionResult(
                selected_features=selected,
                dropped_features=[f for f in ctx.feature_cols if f not in selected],
                importances={f: 1.0 for f in ctx.feature_cols},
                method_used="shadow",
            )

        monkeypatch.setattr(JointQuantileOrchestrator, "_optimize_at_quantile", fake_optimize)
        monkeypatch.setattr(FeatureSelector, "select_for_node", fake_select)

        orchestrator = JointQuantileOrchestrator(
            graph=single_property_graph_with_search_space,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )
        result = orchestrator.fit(ctx, {"y1": targets["y1"]})

        fitted_node = result.get_node("y1")
        assert fitted_node.selected_features == [ctx.feature_cols[0], ctx.feature_cols[1]]
        assert len(optimize_calls) == 2
        assert optimize_calls[0] is None
        assert optimize_calls[1] is not None

    def test_feature_selection_skipped_when_node_has_no_search_space(
        self,
        joint_regression_data,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        monkeypatch,
    ):
        """No-search-space mode should skip feature selection for joint quantile nodes."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])
        fs_config = FeatureSelectionConfig(enabled=True, method="shadow")
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            n_trials=1,
            metric="neg_mean_squared_error",
            greater_is_better=False,
            verbose=0,
            cv_config=CVConfig(
                n_splits=3,
                strategy=CVStrategy.RANDOM,
                shuffle=True,
                random_state=42,
            ),
            feature_selection=fs_config,
        )

        called = {"select": False}

        def fail_if_called(self, node, ctx, params):
            called["select"] = True
            raise AssertionError("Feature selector should not run without search space")

        monkeypatch.setattr(FeatureSelector, "select_for_node", fail_if_called)

        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )
        result = orchestrator.fit(ctx, targets)

        assert called["select"] is False
        assert result.get_node("y1").selected_features is None


class TestJointQuantileOrchestratorConditionalContext:
    """Tests for conditional context preparation."""

    def test_prepare_conditional_context_first_property(
        self,
        joint_regression_data,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify first property has no conditioning features."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])

        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        prop_ctx = orchestrator._prepare_conditional_context(
            ctx, "y1", targets, {}
        )

        # First property should have no extra features
        assert list(prop_ctx.X.columns) == list(X.columns)

    def test_prepare_conditional_context_second_property(
        self,
        joint_regression_data,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify second property has conditioning on first."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])

        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        prop_ctx = orchestrator._prepare_conditional_context(
            ctx, "y2", targets, {}
        )

        # Second property should have cond_y1 feature
        assert "cond_y1" in prop_ctx.X.columns

    def test_prepare_conditional_context_third_property(
        self,
        joint_regression_data,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify third property has conditioning on first two."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])

        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        prop_ctx = orchestrator._prepare_conditional_context(
            ctx, "y3", targets, {}
        )

        # Third property should have both cond_y1 and cond_y2
        assert "cond_y1" in prop_ctx.X.columns
        assert "cond_y2" in prop_ctx.X.columns


class TestJointQuantileOrchestratorPinballLoss:
    """Tests for pinball loss calculation."""

    def test_pinball_loss_at_median(
        self,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify pinball loss at median (tau=0.5)."""
        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 32])

        loss = orchestrator._pinball_loss(y_true, y_pred, 0.5)

        # At tau=0.5, pinball loss = 0.5 * mean(|y - y_pred|)
        expected = 0.5 * np.mean(np.abs(y_true - y_pred))
        assert loss == pytest.approx(expected)

    def test_pinball_loss_asymmetric(
        self,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify pinball loss is asymmetric for non-median quantiles."""
        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        y_true = np.array([10])
        y_pred_under = np.array([5])  # Under-prediction
        y_pred_over = np.array([15])  # Over-prediction

        # For tau=0.9, under-prediction is penalized more
        loss_under = orchestrator._pinball_loss(y_true, y_pred_under, 0.9)
        loss_over = orchestrator._pinball_loss(y_true, y_pred_over, 0.9)

        assert loss_under > loss_over


class TestJointQuantileFitResultMethods:
    """Tests for JointQuantileFitResult methods."""

    def test_get_node(
        self,
        joint_regression_data,
        joint_quantile_graph,
        data_manager,
        mock_backend,
        tuning_config,
    ):
        """Verify get_node returns correct fitted node."""
        X, targets = joint_regression_data
        ctx = DataContext.from_Xy(X, targets["y1"])

        orchestrator = JointQuantileOrchestrator(
            graph=joint_quantile_graph,
            data_manager=data_manager,
            search_backend=mock_backend,
            tuning_config=tuning_config,
        )

        result = orchestrator.fit(ctx, targets)
        fitted = result.get_node("y1")

        assert fitted.node.property_name == "y1"
