"""Tests for convenience helpers and TrainingRun shortcuts."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn_meta import compare, cross_validate, stack, tune
from sklearn_meta.artifacts.training import NodeRunResult, RunMetadata, TrainingRun
from sklearn_meta.runtime.config import (
    CVConfig,
    CVFold,
    CVResult,
    CVStrategy,
    FoldResult,
    RunConfig,
    TuningConfig,
)
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec


def _make_node_result(name: str, best_params: dict | None = None, score: float = 0.85):
    fold = CVFold(fold_idx=0, train_indices=np.array([0, 1]), val_indices=np.array([2, 3]))
    model = MagicMock()
    fold_result = FoldResult(
        fold=fold,
        model=model,
        val_predictions=np.array([0.1, 0.2]),
        val_score=score,
    )
    cv_result = CVResult(
        fold_results=[fold_result],
        oof_predictions=np.array([0.1, 0.2, 0.3, 0.4]),
        node_name=name,
    )
    return NodeRunResult(
        node_name=name,
        cv_result=cv_result,
        best_params=best_params or {"x": 1},
    )


def _make_graph(node_names: list[str]) -> GraphSpec:
    graph = GraphSpec()
    for name in node_names:
        graph.add_node(NodeSpec(name=name, estimator_class=RandomForestClassifier))
    return graph


def _make_training_run(
    node_names: list[str] | None = None,
    edges: list | None = None,
) -> TrainingRun:
    if node_names is None:
        node_names = ["model"]

    graph = _make_graph(node_names)
    if edges:
        for edge in edges:
            graph.add_edge(edge)

    node_results = {
        name: _make_node_result(name, score=0.85 if name != "meta" else 0.90)
        for name in node_names
    }

    metadata = RunMetadata(
        timestamp="2024-01-01T00:00:00",
        sklearn_meta_version="0.2.0",
        data_shape=(4, 2),
        feature_names=["a", "b"],
        cv_config=None,
        tuning_config_summary={},
        total_trials=1,
        data_hash=None,
        random_state=42,
    )
    config = RunConfig(
        cv=CVConfig(n_splits=2),
        tuning=TuningConfig(metric="accuracy", greater_is_better=True),
    )
    return TrainingRun(graph=graph, config=config, node_results=node_results, metadata=metadata)


def _patch_fit():
    """Patch the public fit() path used by convenience helpers."""
    return patch("sklearn_meta.fit")


class TestTrainingRunShortcuts:
    def test_best_params_single_node(self):
        run = _make_training_run(["model"])
        assert run.best_params_ == {"x": 1}

    def test_best_score_single_node(self):
        run = _make_training_run(["model"])
        assert run.best_score_ == 0.85

    def test_oof_predictions_single_node(self):
        run = _make_training_run(["model"])
        np.testing.assert_array_equal(run.oof_predictions_, [0.1, 0.2, 0.3, 0.4])

    def test_shortcuts_resolve_to_leaf_node(self):
        from sklearn_meta.spec.dependency import DependencyEdge, DependencyType

        edge = DependencyEdge(source="rf", target="meta", dep_type=DependencyType.PREDICTION)
        run = _make_training_run(["rf", "meta"], edges=[edge])
        assert run.best_score_ == 0.90

    def test_shortcuts_raise_on_ambiguous_leaves(self):
        run = _make_training_run(["rf", "xgb"])
        with pytest.raises(ValueError, match="Ambiguous"):
            _ = run.best_params_


class _FakeTreeModel:
    """Minimal fake with only feature_importances_ (no get_booster, etc.)."""
    def __init__(self, importances):
        self.feature_importances_ = np.asarray(importances)


class _FakeLinearModel:
    """Minimal fake with only coef_."""
    def __init__(self, coef):
        self.coef_ = np.asarray(coef)


class TestFeatureImportances:
    def test_feature_importances_with_tree_model(self):
        """feature_importances_ extracts and averages across fold models."""
        run = _make_training_run(["model"])

        for fr in run.node_results["model"].cv_result.fold_results:
            fr.model = _FakeTreeModel([0.7, 0.3])

        result = run.feature_importances_
        assert isinstance(result, pd.Series)
        assert result.name == "importance"
        assert list(result.index) == ["a", "b"]  # sorted descending
        np.testing.assert_allclose(result.values, [0.7, 0.3])

    def test_feature_importances_averages_across_folds(self):
        """Multiple fold models should be averaged."""
        fold0 = CVFold(fold_idx=0, train_indices=np.array([0, 1]), val_indices=np.array([2, 3]))
        fold1 = CVFold(fold_idx=1, train_indices=np.array([2, 3]), val_indices=np.array([0, 1]))

        fr0 = FoldResult(fold=fold0, model=_FakeTreeModel([0.8, 0.2]), val_predictions=np.array([0.1, 0.2]), val_score=0.9)
        fr1 = FoldResult(fold=fold1, model=_FakeTreeModel([0.6, 0.4]), val_predictions=np.array([0.3, 0.4]), val_score=0.8)
        cv_result = CVResult(fold_results=[fr0, fr1], oof_predictions=np.array([0.3, 0.4, 0.1, 0.2]), node_name="model")

        graph = _make_graph(["model"])
        node_result = NodeRunResult(node_name="model", cv_result=cv_result, best_params={})
        metadata = RunMetadata(
            timestamp="2024-01-01T00:00:00", sklearn_meta_version="0.2.0",
            data_shape=(4, 2), feature_names=["a", "b"], cv_config=None,
            tuning_config_summary={}, total_trials=0, data_hash=None, random_state=42,
        )
        config = RunConfig(cv=CVConfig(n_splits=2), tuning=TuningConfig(metric="accuracy", greater_is_better=True))
        run = TrainingRun(graph=graph, config=config, node_results={"model": node_result}, metadata=metadata)

        result = run.feature_importances_
        np.testing.assert_allclose(result["a"], 0.7)
        np.testing.assert_allclose(result["b"], 0.3)

    def test_feature_importances_uses_selected_features(self):
        """When feature selection was applied, use selected_features."""
        run = _make_training_run(["model"])
        run.node_results["model"].selected_features = ["b"]

        for fr in run.node_results["model"].cv_result.fold_results:
            fr.model = _FakeTreeModel([1.0])

        result = run.feature_importances_
        assert list(result.index) == ["b"]

    def test_feature_importances_with_linear_model(self):
        """Linear models use coefficient magnitude."""
        run = _make_training_run(["model"])

        for fr in run.node_results["model"].cv_result.fold_results:
            fr.model = _FakeLinearModel([[-0.5, 0.8]])

        result = run.feature_importances_
        np.testing.assert_allclose(result["b"], 0.8)
        np.testing.assert_allclose(result["a"], 0.5)


class TestSummary:
    def test_summary_single_model(self):
        run = _make_training_run(["model"])
        text = run.summary()

        assert "TrainingRun Summary" in text
        assert "RandomForestClassifier" in text
        assert "accuracy" in text
        assert "0.8500" in text
        assert "1 folds" in text
        assert "x: 1" in text
        assert "Features: 2" in text

    def test_summary_multi_model(self):
        from sklearn_meta.spec.dependency import DependencyEdge, DependencyType

        edge = DependencyEdge(source="rf", target="meta", dep_type=DependencyType.PREDICTION)
        run = _make_training_run(["rf", "meta"], edges=[edge])
        text = run.summary()

        assert "--- rf ---" in text
        assert "--- meta ---" in text

    def test_summary_specific_node(self):
        from sklearn_meta.spec.dependency import DependencyEdge, DependencyType

        edge = DependencyEdge(source="rf", target="meta", dep_type=DependencyType.PREDICTION)
        run = _make_training_run(["rf", "meta"], edges=[edge])
        text = run.summary(node_name="rf")

        assert "RandomForestClassifier" in text
        assert "--- meta ---" not in text

    def test_summary_with_feature_selection(self):
        run = _make_training_run(["model"])
        run.node_results["model"].selected_features = ["a"]
        text = run.summary()

        assert "1 kept, 1 dropped" in text

    def test_summary_with_fit_time(self):
        run = _make_training_run(["model"])
        run.node_results["model"].cv_result.fold_results[0].fit_time = 3.7
        text = run.summary()

        assert "Fit time: 3.7s" in text


class TestTune:
    def test_basic_tune(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
            y = np.array([0, 1, 0, 1])

            result = tune(
                RandomForestClassifier,
                X,
                y,
                params={"n_estimators": (50, 500), "max_depth": (3, 20)},
                n_trials=10,
                metric="accuracy",
                verbosity=0,
            )

            mock_fit.assert_called_once()
            assert isinstance(result, TrainingRun)

    def test_tune_builds_correct_graph(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(4)})
            y = np.array([0, 1, 0, 1])

            tune(
                RandomForestClassifier,
                X,
                y,
                params={"n_estimators": (50, 500)},
                fixed_params={"random_state": 42},
                n_trials=10,
                metric="accuracy",
                verbosity=0,
            )

            graph = mock_fit.call_args[0][0]
            node = graph.get_node("model")
            assert node.estimator_class is RandomForestClassifier
            assert node.fixed_params == {"random_state": 42}
            assert node.search_space is not None

    def test_tune_with_log_param(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(4)})
            y = np.array([0, 1, 0, 1])

            tune(
                GradientBoostingClassifier,
                X,
                y,
                params={"learning_rate": (0.01, 0.3, {"log": True})},
                n_trials=5,
                metric="accuracy",
                verbosity=0,
            )

            mock_fit.assert_called_once()

    def test_tune_with_categorical_param(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(4)})
            y = np.array([0, 1, 0, 1])

            tune(
                RandomForestClassifier,
                X,
                y,
                params={"criterion": ["gini", "entropy"]},
                n_trials=5,
                metric="accuracy",
                verbosity=0,
            )

            mock_fit.assert_called_once()

    def test_tune_sets_tuning_config(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(4)})
            y = np.array([0, 1, 0, 1])

            tune(
                RandomForestClassifier,
                X,
                y,
                params={"n_estimators": (50, 500)},
                n_trials=42,
                metric="roc_auc",
                verbosity=0,
            )

            config = mock_fit.call_args[0][3]
            assert config.tuning.n_trials == 42
            assert config.tuning.metric == "roc_auc"
            assert config.cv.strategy == CVStrategy.STRATIFIED

    def test_tune_defaults_to_random_for_regressors(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(6)})
            y = np.linspace(0.0, 1.0, 6)

            tune(
                RandomForestRegressor,
                X,
                y,
                params={"n_estimators": (10, 50)},
                verbosity=0,
            )

            config = mock_fit.call_args[0][3]
            assert config.cv.strategy == CVStrategy.RANDOM

    def test_tune_accepts_estimator_instance(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(4)})
            y = np.array([0, 1, 0, 1])

            tune(
                RandomForestClassifier(n_estimators=25, random_state=42),
                X,
                y,
                params={"max_depth": (2, 5)},
                verbosity=0,
            )

            node = mock_fit.call_args[0][0].get_node("model")
            assert node.estimator_class is RandomForestClassifier
            assert node.fixed_params["n_estimators"] == 25
            assert node.fixed_params["random_state"] == 42

    def test_tune_requires_params(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="params dict"):
            tune(RandomForestClassifier, X, y, metric="accuracy")

    def test_tune_rejects_invalid_param_spec(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(TypeError, match="Invalid param spec"):
            tune(RandomForestClassifier, X, y, params={"n_estimators": 100})

    def test_tune_rejects_bad_tuple_length(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="Invalid param spec"):
            tune(RandomForestClassifier, X, y, params={"n_estimators": (1, 2, 3, 4)})


class TestCrossValidate:
    def test_basic_cross_validate(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
            y = np.array([0, 1, 0, 1])

            result = cross_validate(
                RandomForestClassifier,
                X,
                y,
                fixed_params={"n_estimators": 100},
                metric="accuracy",
                verbosity=0,
            )

            mock_fit.assert_called_once()
            config = mock_fit.call_args[0][3]
            assert config.tuning.n_trials == 0
            assert isinstance(result, TrainingRun)

    def test_cross_validate_no_search_space(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(4)})
            y = np.array([0, 1, 0, 1])

            cross_validate(RandomForestClassifier, X, y, metric="accuracy", verbosity=0)

            graph = mock_fit.call_args[0][0]
            node = graph.get_node("model")
            assert node.search_space is None

    def test_cross_validate_preserves_supplied_cv_config(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(6)})
            y = np.array([0, 1, 0, 1, 0, 1])
            cv = CVConfig(
                n_splits=3,
                n_repeats=2,
                strategy=CVStrategy.RANDOM,
                shuffle=False,
                random_state=7,
            )

            cross_validate(LogisticRegression, X, y, cv=cv, verbosity=0)

            config = mock_fit.call_args[0][3]
            assert config.cv is cv
            assert config.cv.n_repeats == 2
            assert config.cv.shuffle is False

    def test_cross_validate_defaults_to_stratified_for_classifiers(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(6)})
            y = np.array([0, 1, 0, 1, 0, 1])

            cross_validate(LogisticRegression, X, y, verbosity=0)

            config = mock_fit.call_args[0][3]
            assert config.cv.strategy == CVStrategy.STRATIFIED

    def test_cross_validate_defaults_to_random_for_regressors(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(6)})
            y = np.linspace(0.0, 1.0, 6)

            cross_validate(RandomForestRegressor, X, y, verbosity=0)

            config = mock_fit.call_args[0][3]
            assert config.cv.strategy == CVStrategy.RANDOM

    def test_cross_validate_accepts_estimator_instance(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run()
            X = pd.DataFrame({"a": range(6)})
            y = np.linspace(0.0, 1.0, 6)

            cross_validate(
                RandomForestRegressor(n_estimators=17, random_state=5),
                X,
                y,
                verbosity=0,
            )

            node = mock_fit.call_args[0][0].get_node("model")
            assert node.estimator_class is RandomForestRegressor
            assert node.fixed_params["n_estimators"] == 17
            assert node.fixed_params["random_state"] == 5


class TestStack:
    def test_basic_stack(self):
        with _patch_fit() as mock_fit:
            from sklearn_meta.spec.dependency import DependencyEdge, DependencyType

            edge = DependencyEdge(source="rf", target="meta", dep_type=DependencyType.PREDICTION)
            mock_fit.return_value = _make_training_run(["rf", "meta"], edges=[edge])

            X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
            y = np.array([0, 1, 0, 1])

            result = stack(
                base_models={"rf": (RandomForestClassifier, {"n_estimators": (50, 300)})},
                meta_model=LogisticRegression,
                X=X,
                y=y,
                metric="accuracy",
                n_trials=10,
                verbosity=0,
            )

            mock_fit.assert_called_once()
            graph = mock_fit.call_args[0][0]
            assert "rf" in graph
            assert "meta" in graph
            assert len(graph.get_upstream("meta")) == 1
            assert isinstance(result, TrainingRun)

    def test_stack_with_class_only_base(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run(["rf", "meta"])
            X = pd.DataFrame({"a": range(4)})
            y = np.array([0, 1, 0, 1])

            stack(
                base_models={"rf": RandomForestClassifier},
                meta_model=LogisticRegression,
                X=X,
                y=y,
                metric="accuracy",
                verbosity=0,
            )

            graph = mock_fit.call_args[0][0]
            node = graph.get_node("rf")
            assert node.search_space is None

    def test_stack_with_fixed_params(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run(["rf", "meta"])
            X = pd.DataFrame({"a": range(4)})
            y = np.array([0, 1, 0, 1])

            stack(
                base_models={
                    "rf": (
                        RandomForestClassifier,
                        {"n_estimators": (50, 300)},
                        {"random_state": 42},
                    ),
                },
                meta_model=LogisticRegression,
                X=X,
                y=y,
                metric="accuracy",
                verbosity=0,
            )

            graph = mock_fit.call_args[0][0]
            node = graph.get_node("rf")
            assert node.fixed_params == {"random_state": 42}

    def test_stack_multiple_base_models(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run(["rf", "gb", "meta"])
            X = pd.DataFrame({"a": range(4)})
            y = np.array([0, 1, 0, 1])

            stack(
                base_models={
                    "rf": RandomForestClassifier,
                    "gb": (GradientBoostingClassifier, {"n_estimators": (50, 300)}),
                },
                meta_model=LogisticRegression,
                X=X,
                y=y,
                metric="accuracy",
                verbosity=0,
            )

            graph = mock_fit.call_args[0][0]
            upstream = graph.get_upstream("meta")
            source_names = {e.source for e in upstream}
            assert source_names == {"rf", "gb"}

    def test_stack_preserves_supplied_cv_config(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run(["rf", "meta"])
            X = pd.DataFrame({"a": range(6)})
            y = np.array([0, 1, 0, 1, 0, 1])
            cv = CVConfig(
                n_splits=3,
                n_repeats=2,
                strategy=CVStrategy.RANDOM,
                shuffle=False,
                random_state=11,
            )

            stack(
                base_models={"rf": RandomForestClassifier},
                meta_model=LogisticRegression,
                X=X,
                y=y,
                cv=cv,
                verbosity=0,
            )

            config = mock_fit.call_args[0][3]
            assert config.cv is cv
            assert config.cv.n_repeats == 2
            assert config.cv.shuffle is False

    def test_stack_defaults_to_random_for_regression_meta_model(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run(["rf", "meta"])
            X = pd.DataFrame({"a": range(6)})
            y = np.linspace(0.0, 1.0, 6)

            stack(
                base_models={"rf": RandomForestRegressor},
                meta_model=LinearRegression,
                X=X,
                y=y,
                verbosity=0,
            )

            config = mock_fit.call_args[0][3]
            assert config.cv.strategy == CVStrategy.RANDOM

    def test_stack_accepts_estimator_instances(self):
        with _patch_fit() as mock_fit:
            mock_fit.return_value = _make_training_run(["rf", "meta"])
            X = pd.DataFrame({"a": range(6)})
            y = np.array([0, 1, 0, 1, 0, 1])

            stack(
                base_models={"rf": RandomForestClassifier(n_estimators=13, random_state=2)},
                meta_model=LogisticRegression(C=0.5, max_iter=200),
                X=X,
                y=y,
                verbosity=0,
            )

            graph = mock_fit.call_args[0][0]
            assert graph.get_node("rf").fixed_params["n_estimators"] == 13
            assert graph.get_node("meta").fixed_params["C"] == 0.5

    def test_stack_rejects_invalid_model_spec(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(TypeError, match="Expected an estimator class"):
            stack(
                base_models={"rf": "not a class"},
                meta_model=LogisticRegression,
                X=X,
                y=y,
            )


class TestCompare:
    def test_compare_runs_all_models_and_sorts(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])
        runs = [_make_training_run(["model"]), _make_training_run(["model"])]
        runs[0].node_results["model"] = _make_node_result("model", score=0.81)
        runs[1].node_results["model"] = _make_node_result("model", score=0.92)

        with patch("sklearn_meta.convenience._build_single_model_run", side_effect=runs) as mock_build:
            result = compare(
                {
                    "rf": RandomForestClassifier,
                    "lr": LogisticRegression(max_iter=200),
                },
                X,
                y,
                metric="accuracy",
                verbosity=0,
            )

            assert mock_build.call_count == 2
            assert result.best_name == "lr"
            assert result.best_run is runs[1]
            assert list(result.leaderboard["model"]) == ["lr", "rf"]

    def test_compare_rankings(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])
        runs = [_make_training_run(["model"]), _make_training_run(["model"])]
        runs[0].node_results["model"] = _make_node_result("model", score=0.81)
        runs[1].node_results["model"] = _make_node_result("model", score=0.92)

        with patch("sklearn_meta.convenience._build_single_model_run", side_effect=runs):
            result = compare(
                {"rf": RandomForestClassifier, "lr": LogisticRegression},
                X, y, metric="accuracy", verbosity=0,
            )

        assert result.rankings == [("lr", 0.92), ("rf", 0.81)]

    def test_compare_dict_access(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])
        run = _make_training_run(["model"])

        with patch("sklearn_meta.convenience._build_single_model_run", return_value=run):
            result = compare(
                {"rf": RandomForestClassifier},
                X, y, metric="accuracy", verbosity=0,
            )

        assert result["rf"] is run
        assert "rf" in result
        assert len(result) == 1
        with pytest.raises(KeyError, match="No model named"):
            _ = result["xgb"]

    def test_compare_repr(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])
        run = _make_training_run(["model"])

        with patch("sklearn_meta.convenience._build_single_model_run", return_value=run):
            result = compare(
                {"rf": RandomForestClassifier},
                X, y, metric="accuracy", verbosity=0,
            )

        text = repr(result)
        assert "accuracy" in text
        assert "rf" in text

    def test_compare_uses_tuning_only_for_specs_with_search_space(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])

        with patch(
            "sklearn_meta.convenience._build_single_model_run",
            return_value=_make_training_run(),
        ) as mock_build:
            compare(
                {
                    "rf": RandomForestClassifier,
                    "gb": (GradientBoostingClassifier, {"n_estimators": (10, 30)}),
                },
                X,
                y,
                n_trials=25,
                metric="accuracy",
                verbosity=0,
            )

            rf_call = mock_build.call_args_list[0].kwargs
            gb_call = mock_build.call_args_list[1].kwargs
            assert rf_call["n_trials"] == 0
            assert gb_call["n_trials"] == 25

    def test_compare_requires_at_least_one_model(self):
        X = pd.DataFrame({"a": range(4)})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="at least one model"):
            compare({}, X, y)
