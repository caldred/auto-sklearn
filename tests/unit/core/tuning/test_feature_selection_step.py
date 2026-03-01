"""Tests for the extracted _apply_feature_selection step."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from sklearn.linear_model import LogisticRegression

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.model.node import ModelNode, OutputType
from sklearn_meta.core.tuning.orchestrator import TuningConfig, TuningOrchestrator
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.selection.selector import (
    FeatureSelectionConfig,
    FeatureSelectionMethod,
    FeatureSelectionResult,
)
from sklearn_meta.search.space import SearchSpace


@pytest.fixture
def simple_ctx():
    """Create a simple DataContext with named features."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    })
    y = pd.Series(np.random.randint(0, 2, size=n), name="target")
    return DataContext.from_Xy(X, y)


@pytest.fixture
def simple_node():
    """Create a simple model node."""
    return ModelNode(
        name="test_lr",
        estimator_class=LogisticRegression,
        fixed_params={"max_iter": 200},
    )


@pytest.fixture
def orchestrator(simple_ctx):
    """Create a minimal TuningOrchestrator with strategy=NONE."""
    graph = ModelGraph()
    graph.add_node(ModelNode(
        name="test_lr",
        estimator_class=LogisticRegression,
        fixed_params={"max_iter": 200},
    ))

    cv_config = CVConfig(
        n_splits=2,
        strategy=CVStrategy.STRATIFIED,
        shuffle=True,
        random_state=42,
    )
    dm = DataManager(cv_config)

    tuning_config = TuningConfig(
        strategy=OptimizationStrategy.NONE,
        n_trials=1,
        metric="accuracy",
        greater_is_better=True,
    )

    backend = MagicMock()
    return TuningOrchestrator(
        graph=graph,
        data_manager=dm,
        search_backend=backend,
        tuning_config=tuning_config,
    )


class TestApplyFeatureSelectionDisabled:
    """Test _apply_feature_selection when feature selection is disabled."""

    def test_returns_original_when_no_config(self, orchestrator, simple_node, simple_ctx):
        """When feature_selection is None, returns inputs unchanged."""
        orchestrator.tuning_config.feature_selection = None
        best_params = {"C": 1.0}

        ctx_out, params_out, sel_features, opt_result = (
            orchestrator._apply_feature_selection(
                simple_node, simple_ctx, best_params,
                search_space=None, reparam_space=None,
            )
        )

        assert ctx_out is simple_ctx
        assert params_out is best_params
        assert sel_features is None
        assert opt_result is None

    def test_returns_original_when_disabled(self, orchestrator, simple_node, simple_ctx):
        """When feature_selection.enabled is False, returns inputs unchanged."""
        orchestrator.tuning_config.feature_selection = FeatureSelectionConfig(
            enabled=False,
        )
        best_params = {"C": 1.0}

        ctx_out, params_out, sel_features, opt_result = (
            orchestrator._apply_feature_selection(
                simple_node, simple_ctx, best_params,
                search_space=None, reparam_space=None,
            )
        )

        assert ctx_out is simple_ctx
        assert params_out is best_params
        assert sel_features is None
        assert opt_result is None


class TestApplyFeatureSelectionEnabled:
    """Test _apply_feature_selection when feature selection is enabled."""

    def test_selects_features_and_filters_context(
        self, orchestrator, simple_node, simple_ctx
    ):
        """When enabled, selected features are applied to context."""
        fs_config = FeatureSelectionConfig(
            enabled=True,
            method=FeatureSelectionMethod.THRESHOLD,
            retune_after_pruning=False,
        )
        orchestrator.tuning_config.feature_selection = fs_config
        best_params = {"C": 1.0}

        mock_result = FeatureSelectionResult(
            selected_features=["f1", "f3"],
            dropped_features=["f2"],
            importances={"f1": 0.5, "f2": 0.01, "f3": 0.4},
            method_used="threshold",
        )

        with patch.object(
            FeatureSelectionResult, "__init__", return_value=None
        ):
            pass  # not needed

        from sklearn_meta.selection.selector import FeatureSelector

        with patch.object(
            FeatureSelector, "select_for_node", return_value=mock_result
        ):
            ctx_out, params_out, sel_features, opt_result = (
                orchestrator._apply_feature_selection(
                    simple_node, simple_ctx, best_params,
                    search_space=None, reparam_space=None,
                )
            )

        assert sel_features == ["f1", "f3"]
        assert list(ctx_out.feature_cols) == ["f1", "f3"]
        assert params_out is best_params
        assert opt_result is None

    def test_retune_after_pruning(self, orchestrator, simple_node, simple_ctx):
        """When retune_after_pruning=True, _optimize_node is called."""
        fs_config = FeatureSelectionConfig(
            enabled=True,
            method=FeatureSelectionMethod.THRESHOLD,
            retune_after_pruning=True,
        )
        orchestrator.tuning_config.feature_selection = fs_config
        best_params = {"C": 1.0}

        mock_result = FeatureSelectionResult(
            selected_features=["f1", "f3"],
            dropped_features=["f2"],
            importances={"f1": 0.5, "f2": 0.01, "f3": 0.4},
            method_used="threshold",
        )

        # Create a search space that supports narrow_around
        search_space = MagicMock()
        search_space.__len__ = MagicMock(return_value=1)
        narrowed_space = MagicMock()
        search_space.narrow_around.return_value = narrowed_space

        retuned_params = {"C": 2.0}
        mock_opt_result = MagicMock()

        from sklearn_meta.selection.selector import FeatureSelector

        with patch.object(
            FeatureSelector, "select_for_node", return_value=mock_result
        ), patch.object(
            orchestrator, "_optimize_node",
            return_value=(retuned_params, mock_opt_result),
        ) as mock_optimize:
            ctx_out, params_out, sel_features, opt_result = (
                orchestrator._apply_feature_selection(
                    simple_node, simple_ctx, best_params,
                    search_space=search_space, reparam_space=None,
                )
            )

        assert sel_features == ["f1", "f3"]
        assert params_out == {"C": 2.0}
        assert opt_result is mock_opt_result
        mock_optimize.assert_called_once()

        # Verify narrow_around was called with correct params
        search_space.narrow_around.assert_called_once_with(
            center=best_params,
            factor=0.5,
            regularization_bias=0.25,
        )

    def test_retune_uses_original_space_from_reparam(
        self, orchestrator, simple_node, simple_ctx
    ):
        """When reparam_space is provided, retune uses its original_space."""
        fs_config = FeatureSelectionConfig(
            enabled=True,
            method=FeatureSelectionMethod.THRESHOLD,
            retune_after_pruning=True,
        )
        orchestrator.tuning_config.feature_selection = fs_config
        best_params = {"C": 1.0}

        mock_result = FeatureSelectionResult(
            selected_features=["f1", "f3"],
            dropped_features=["f2"],
            importances={"f1": 0.5, "f2": 0.01, "f3": 0.4},
            method_used="threshold",
        )

        # Set up reparam_space with an original_space
        original_space = MagicMock()
        original_space.__len__ = MagicMock(return_value=1)
        narrowed_space = MagicMock()
        original_space.narrow_around.return_value = narrowed_space

        reparam_space = MagicMock()
        reparam_space.original_space = original_space

        # The transformed search_space should NOT be used for narrowing
        search_space = MagicMock()

        from sklearn_meta.selection.selector import FeatureSelector

        with patch.object(
            FeatureSelector, "select_for_node", return_value=mock_result
        ), patch.object(
            orchestrator, "_optimize_node",
            return_value=({"C": 3.0}, MagicMock()),
        ):
            ctx_out, params_out, sel_features, opt_result = (
                orchestrator._apply_feature_selection(
                    simple_node, simple_ctx, best_params,
                    search_space=search_space, reparam_space=reparam_space,
                )
            )

        # narrow_around should be called on the original space, not search_space
        original_space.narrow_around.assert_called_once()
        search_space.narrow_around.assert_not_called()
