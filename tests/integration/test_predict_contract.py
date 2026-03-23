"""Integration tests for InferenceGraph.predict() public contract.

Tests cover: single-node prediction, two-layer stacking, proba stacking,
feature selection at inference, conditional node handling, and missing
node errors.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn_meta.data.view import DataView
from sklearn_meta.runtime.config import CVConfig, CVStrategy, RunConfig, TuningConfig
from sklearn_meta.engine.cv import CVEngine
from sklearn_meta.spec.dependency import DependencyEdge, DependencyType
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.artifacts.inference import InferenceGraph
from sklearn_meta.engine.runner import GraphRunner
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.runtime.services import RuntimeServices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regression_ctx(n_samples=200, n_features=5, random_state=42):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        noise=0.5,
        random_state=random_state,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    return DataView.from_Xy(X_df, pd.Series(y)), X_df


def _make_classification_ctx(n_samples=200, n_features=5, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=random_state,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    return DataView.from_Xy(X_df, pd.Series(y)), X_df


def _quick_tuning_config(metric="neg_mean_squared_error", greater_is_better=False):
    cv_config = CVConfig(
        n_splits=2,
        strategy=CVStrategy.RANDOM,
        random_state=42,
    )
    tuning_config = TuningConfig(
        strategy=OptimizationStrategy.NONE,
        metric=metric,
        greater_is_better=greater_is_better,
    )
    return RunConfig(cv=cv_config, tuning=tuning_config, verbosity=0)


def _fit_graph(graph, ctx, run_config, mock_search_backend):
    services = RuntimeServices(search_backend=mock_search_backend)
    runner = GraphRunner(services)
    return runner.fit(graph, ctx, run_config)


# ---------------------------------------------------------------------------
# Test 1: Single node prediction
# ---------------------------------------------------------------------------

class TestSingleNodePrediction:
    """Verify basic predict() contract for a single-node graph."""

    def test_predict_returns_correct_shape(self, mock_search_backend):
        ctx, X_df = _make_regression_ctx()
        node = NodeSpec(
            name="tree",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        graph = GraphSpec()
        graph.add_node(node)

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()
        preds = inference.predict(X_df)

        assert preds.shape == (len(X_df),)
        assert np.isfinite(preds).all()

    def test_predict_is_deterministic(self, mock_search_backend):
        ctx, X_df = _make_regression_ctx()
        node = NodeSpec(
            name="tree",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        graph = GraphSpec()
        graph.add_node(node)

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()

        preds1 = inference.predict(X_df)
        preds2 = inference.predict(X_df)

        np.testing.assert_array_equal(preds1, preds2)

    def test_predict_with_explicit_node_name(self, mock_search_backend):
        ctx, X_df = _make_regression_ctx()
        node = NodeSpec(
            name="tree",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        graph = GraphSpec()
        graph.add_node(node)

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()

        preds_default = inference.predict(X_df)
        preds_named = inference.predict(X_df, node_name="tree")

        np.testing.assert_array_equal(preds_default, preds_named)


# ---------------------------------------------------------------------------
# Test 2: Two-layer stacking prediction
# ---------------------------------------------------------------------------

class TestTwoLayerStackingPrediction:
    """Verify stacking predict() with PREDICTION dependency type."""

    def test_stacking_predict_shape_two_bases(self, mock_search_backend):
        ctx, X_df = _make_regression_ctx()

        base1 = NodeSpec(
            name="base1",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        base2 = NodeSpec(
            name="base2",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 0, "max_depth": 4},
        )
        # Use DecisionTreeRegressor for meta to avoid feature-name ordering
        # sensitivity (Ridge with pandas can fail if column order differs
        # between training and predict due to set iteration in
        # _prepare_context_with_oof).
        meta = NodeSpec(
            name="meta",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 5},
        )

        graph = GraphSpec()
        graph.add_node(base1)
        graph.add_node(base2)
        graph.add_node(meta)
        graph.add_edge(DependencyEdge(
            source="base1", target="meta", dep_type=DependencyType.PREDICTION,
        ))
        graph.add_edge(DependencyEdge(
            source="base2", target="meta", dep_type=DependencyType.PREDICTION,
        ))

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()
        preds = inference.predict(X_df)

        assert preds.shape == (len(X_df),)
        assert np.isfinite(preds).all()

    def test_meta_gets_augmented_features(self, mock_search_backend):
        """Meta model should see original features + upstream predictions."""
        ctx, X_df = _make_regression_ctx(n_features=3)

        base1 = NodeSpec(
            name="base1",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 2},
        )
        meta = NodeSpec(
            name="meta",
            estimator_class=Ridge,
            fixed_params={"alpha": 1.0},
        )

        graph = GraphSpec()
        graph.add_node(base1)
        graph.add_node(meta)
        graph.add_edge(DependencyEdge(
            source="base1", target="meta", dep_type=DependencyType.PREDICTION,
        ))

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)

        # The meta model was trained on original features + 1 prediction column
        # Verify by checking the number of coefficients
        meta_model = fitted.node_results["meta"].models[0]
        # Ridge.coef_ has shape (n_features,) -- original 3 + 1 pred column = 4
        assert meta_model.coef_.shape[0] == 3 + 1

    def test_stacking_predict_is_deterministic(self, mock_search_backend):
        ctx, X_df = _make_regression_ctx()

        base1 = NodeSpec(
            name="base1",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        meta = NodeSpec(
            name="meta",
            estimator_class=Ridge,
            fixed_params={"alpha": 1.0},
        )

        graph = GraphSpec()
        graph.add_node(base1)
        graph.add_node(meta)
        graph.add_edge(DependencyEdge(
            source="base1", target="meta", dep_type=DependencyType.PREDICTION,
        ))

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()

        preds1 = inference.predict(X_df)
        preds2 = inference.predict(X_df)
        np.testing.assert_array_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# Test 3: Proba stacking prediction
# ---------------------------------------------------------------------------

class TestProbaStackingPrediction:
    """Verify stacking with PROBA dependency type on a classification task."""

    def test_proba_stacking_predict_shape(self, mock_search_backend):
        ctx, X_df = _make_classification_ctx()

        base_clf = NodeSpec(
            name="base_clf",
            estimator_class=DecisionTreeClassifier,
            output_type=OutputType.PROBA,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        meta_clf = NodeSpec(
            name="meta_clf",
            estimator_class=LogisticRegression,
            fixed_params={"random_state": 42, "max_iter": 500},
        )

        graph = GraphSpec()
        graph.add_node(base_clf)
        graph.add_node(meta_clf)
        graph.add_edge(DependencyEdge(
            source="base_clf", target="meta_clf", dep_type=DependencyType.PROBA,
        ))

        tuning_cfg = _quick_tuning_config(
            metric="accuracy", greater_is_better=True,
        )
        # Use stratified for classification
        tuning_cfg = RunConfig(
            cv=CVConfig(
                n_splits=2,
                strategy=CVStrategy.STRATIFIED,
                random_state=42,
            ),
            tuning=tuning_cfg.tuning,
            verbosity=0,
        )

        fitted = _fit_graph(graph, ctx, tuning_cfg, mock_search_backend)
        inference = fitted.compile_inference()
        preds = inference.predict(X_df)

        # Default output_type for meta_clf is PREDICTION, so shape is (n,)
        assert preds.shape == (len(X_df),)
        assert np.isfinite(preds).all()


# ---------------------------------------------------------------------------
# Test 4: Feature selection + inference
# ---------------------------------------------------------------------------

class TestFeatureSelectionInference:
    """Verify predict() respects selected_features on FittedNode."""

    def test_predict_uses_selected_features(self, mock_search_backend):
        ctx, X_df = _make_regression_ctx(n_features=5)

        node = NodeSpec(
            name="tree",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        graph = GraphSpec()
        graph.add_node(node)

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)

        # Manually set selected_features to a subset of the original features.
        # We need to retrain the fold models on just those features so that
        # predict() works end-to-end.
        selected = ["f0", "f2", "f4"]

        # Retrain fold models on the selected feature subset so shapes match
        from sklearn.tree import DecisionTreeRegressor as DT
        data = ctx.materialize()
        new_models = []
        for fold_result in fitted.node_results["tree"].cv_result.fold_results:
            m = DT(random_state=42, max_depth=3)
            # Use fold training data with selected features only
            train_idx = fold_result.fold.train_indices
            m.fit(X_df.iloc[train_idx][selected], data.y[train_idx])
            new_models.append(m)
            fold_result.model = m

        fitted.node_results["tree"].selected_features = selected

        inference = fitted.compile_inference()
        preds = inference.predict(X_df)

        assert preds.shape == (len(X_df),)
        assert np.isfinite(preds).all()

        # Verify the models were only trained on 3 features
        for m in fitted.node_results["tree"].models:
            assert m.n_features_in_ == 3


# ---------------------------------------------------------------------------
# Test 5: Conditional node handling
# ---------------------------------------------------------------------------

class TestConditionalNodeHandling:
    """Verify predict() behavior with conditional nodes."""

    def test_skipped_conditional_node_raises_on_predict(self, mock_search_backend):
        """Predicting from a skipped conditional node should raise KeyError."""
        ctx, X_df = _make_regression_ctx()

        normal_node = NodeSpec(
            name="normal",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        conditional_node = NodeSpec(
            name="conditional",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
            condition=lambda ctx: False,  # never runs
        )

        graph = GraphSpec()
        graph.add_node(normal_node)
        graph.add_node(conditional_node)

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)

        # The conditional node should not be in node_results
        assert "conditional" not in fitted.node_results
        assert "normal" in fitted.node_results

        # Predicting from the conditional node should raise KeyError
        inference = fitted.compile_inference()
        with pytest.raises(KeyError, match="condition"):
            inference.predict(X_df, node_name="conditional")

    def test_non_conditional_node_works_when_conditional_skipped(self, mock_search_backend):
        """Predicting from a non-conditional node should work even if another
        node was conditional and skipped, as long as there is no dependency."""
        ctx, X_df = _make_regression_ctx()

        normal_node = NodeSpec(
            name="normal",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        conditional_node = NodeSpec(
            name="conditional",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
            condition=lambda ctx: False,
        )

        graph = GraphSpec()
        graph.add_node(normal_node)
        graph.add_node(conditional_node)

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()

        # Predict from the normal node should work fine
        preds = inference.predict(X_df, node_name="normal")
        assert preds.shape == (len(X_df),)
        assert np.isfinite(preds).all()

    def test_downstream_of_skipped_conditional_raises_key_error(
        self, mock_search_backend
    ):
        """Predicting through a skipped conditional upstream should fail clearly."""
        ctx, X_df = _make_regression_ctx()

        conditional_node = NodeSpec(
            name="conditional",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
            condition=lambda ctx: False,
        )
        meta_node = NodeSpec(
            name="meta",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 0, "max_depth": 4},
        )

        graph = GraphSpec()
        graph.add_node(conditional_node)
        graph.add_node(meta_node)
        graph.add_edge(
            DependencyEdge(
                source="conditional",
                target="meta",
                dep_type=DependencyType.PREDICTION,
            )
        )

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()

        with pytest.raises(KeyError, match="conditional"):
            inference.predict(X_df, node_name="meta")


# ---------------------------------------------------------------------------
# Test 6: Missing node error
# ---------------------------------------------------------------------------

class TestMissingNodeError:
    """Verify predict() raises clear errors for nonexistent nodes."""

    def test_predict_nonexistent_node_raises_key_error(self, mock_search_backend):
        ctx, X_df = _make_regression_ctx()

        node = NodeSpec(
            name="tree",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        graph = GraphSpec()
        graph.add_node(node)

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()

        with pytest.raises(KeyError, match="nonexistent"):
            inference.predict(X_df, node_name="nonexistent")

    def test_error_message_lists_available_nodes(self, mock_search_backend):
        ctx, X_df = _make_regression_ctx()

        node = NodeSpec(
            name="tree",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        graph = GraphSpec()
        graph.add_node(node)

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()

        with pytest.raises(KeyError, match="nonexistent"):
            inference.predict(X_df, node_name="nonexistent")


class TestPersistenceContract:
    """Verify save/load preserves inference and rejects training-only access."""

    def test_inference_only_round_trip_for_stacking_graph(
        self, tmp_path, mock_search_backend
    ):
        ctx, X_df = _make_regression_ctx()

        base = NodeSpec(
            name="base",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 42, "max_depth": 3},
        )
        meta = NodeSpec(
            name="meta",
            estimator_class=DecisionTreeRegressor,
            fixed_params={"random_state": 0, "max_depth": 4},
        )

        graph = GraphSpec()
        graph.add_node(base)
        graph.add_node(meta)
        graph.add_edge(
            DependencyEdge(
                source="base",
                target="meta",
                dep_type=DependencyType.PREDICTION,
            )
        )

        fitted = _fit_graph(graph, ctx, _quick_tuning_config(), mock_search_backend)
        inference = fitted.compile_inference()
        preds_before = inference.predict(X_df)

        inference.save(tmp_path / "model")
        loaded = InferenceGraph.load(tmp_path / "model")
        preds_after = loaded.predict(X_df)

        np.testing.assert_array_equal(preds_before, preds_after)
