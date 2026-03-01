"""Tests for sklearn API compatibility fixes.

Fix 1: _ClassifierPredictionWrapper for classifier scoring
Fix 2: Shadow selection with permutation importance fallback (X_val/y_val)
Fix 3: n_estimators guard for models that don't support it
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import Ridge
from sklearn.metrics import get_scorer

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.model.node import ModelNode, OutputType
from sklearn_meta.core.tuning.orchestrator import (
    TuningConfig,
    TuningOrchestrator,
    _ClassifierPredictionWrapper,
    _PredictionWrapper,
    _supports_param,
)
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.selection.shadow import ShadowFeatureSelector
from sklearn_meta.selection.importance import PermutationImportanceExtractor


# =============================================================================
# Helpers
# =============================================================================


class _SimpleClassifier(ClassifierMixin, BaseEstimator):
    """Minimal sklearn-API classifier without built-in feature importances."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Store mean per class for a trivial decision rule
        self._class_means = {}
        X_arr = np.asarray(X)
        for c in self.classes_:
            self._class_means[c] = X_arr[np.asarray(y) == c].mean(axis=0)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        dists = np.column_stack([
            np.linalg.norm(X_arr - self._class_means[c], axis=1)
            for c in self.classes_
        ])
        inv = 1.0 / (dists + 1e-8)
        return inv / inv.sum(axis=1, keepdims=True)


class _NoNEstimatorsModel(BaseEstimator, RegressorMixin):
    """Regressor that does NOT accept n_estimators."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


# =============================================================================
# Fix 1: _ClassifierPredictionWrapper
# =============================================================================


class TestClassifierPredictionWrapper:
    """Test that sklearn scorers correctly identify _ClassifierPredictionWrapper as a classifier."""

    def test_wrapper_has_classifier_tag(self):
        """_ClassifierPredictionWrapper should be recognized as a classifier by sklearn tags."""
        proba = np.array([[0.3, 0.7], [0.8, 0.2]])
        wrapper = _ClassifierPredictionWrapper(proba, np.array([0, 1]))
        # sklearn 1.6+ uses __sklearn_tags__(); ClassifierMixin provides it
        assert hasattr(wrapper, "__sklearn_tags__") or getattr(wrapper, "_estimator_type", None) == "classifier"

    def test_predict_returns_class_labels(self):
        proba = np.array([[0.3, 0.7], [0.8, 0.2]])
        wrapper = _ClassifierPredictionWrapper(proba, np.array([0, 1]))
        preds = wrapper.predict(None)
        np.testing.assert_array_equal(preds, [1, 0])

    def test_predict_proba_returns_probabilities(self):
        proba = np.array([[0.3, 0.7], [0.8, 0.2]])
        wrapper = _ClassifierPredictionWrapper(proba, np.array([0, 1]))
        np.testing.assert_array_equal(wrapper.predict_proba(None), proba)

    def test_decision_function_binary(self):
        proba = np.array([[0.3, 0.7], [0.8, 0.2]])
        wrapper = _ClassifierPredictionWrapper(proba, np.array([0, 1]))
        np.testing.assert_array_equal(wrapper.decision_function(None), [0.7, 0.2])

    def test_neg_log_loss_scorer_works_with_classifier_wrapper(self):
        """neg_log_loss scorer should work without error when wrapper is a classifier."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
        wrapper = _ClassifierPredictionWrapper(y_proba, np.array([0, 1]))

        scorer = get_scorer("neg_log_loss")
        score = scorer(wrapper, y_true, y_true)
        assert np.isfinite(score)
        assert score < 0  # neg_log_loss is always negative

    def test_roc_auc_scorer_works_with_classifier_wrapper(self):
        """roc_auc scorer should work without error when wrapper is a classifier."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
        wrapper = _ClassifierPredictionWrapper(y_proba, np.array([0, 1]))

        scorer = get_scorer("roc_auc")
        score = scorer(wrapper, y_true, y_true)
        assert np.isfinite(score)


class TestCalculateScoreClassifier:
    """Test _calculate_score with classifier nodes."""

    def test_calculate_score_uses_classifier_wrapper_for_classifier_node(self):
        """_calculate_score should use _ClassifierPredictionWrapper for classifier nodes."""
        X, y = make_classification(n_samples=60, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_s = pd.Series(y)
        ctx = DataContext.from_Xy(X_df, y_s)

        from sklearn.ensemble import RandomForestClassifier

        node = ModelNode(
            name="clf",
            estimator_class=RandomForestClassifier,
            fixed_params={"n_estimators": 10, "random_state": 42},
            output_type=OutputType.PROBA,
        )
        graph = ModelGraph()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            cv_config=cv_config,
            metric="neg_log_loss",
            greater_is_better=False,
            verbose=0,
        )
        data_manager = DataManager(cv_config)

        from tests.conftest import MockSearchBackend

        orchestrator = TuningOrchestrator(
            graph=graph,
            data_manager=data_manager,
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        # This should not raise — previously failed because _estimator_type
        # was set as instance attribute and sklearn 1.6+ ignored it.
        fitted = orchestrator.fit(ctx)
        assert np.isfinite(fitted.fitted_nodes["clf"].mean_score)

    def test_calculate_score_regressor_uses_plain_wrapper(self):
        """_calculate_score should use _PredictionWrapper for non-classifier nodes."""
        X, y = make_classification(n_samples=60, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_s = pd.Series(y)
        ctx = DataContext.from_Xy(X_df, y_s)

        node = ModelNode(
            name="model",
            estimator_class=_NoNEstimatorsModel,
            fixed_params={"alpha": 1.0},
        )
        graph = ModelGraph()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            cv_config=cv_config,
            metric="neg_mean_squared_error",
            greater_is_better=False,
            verbose=0,
        )
        data_manager = DataManager(cv_config)

        from tests.conftest import MockSearchBackend

        orchestrator = TuningOrchestrator(
            graph=graph,
            data_manager=data_manager,
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        fitted = orchestrator.fit(ctx)
        assert np.isfinite(fitted.fitted_nodes["model"].mean_score)


# =============================================================================
# Fix 2: Shadow selection with permutation importance fallback
# =============================================================================


class TestShadowPermutationFallback:
    """Test shadow selection with models lacking built-in importances."""

    def test_shadow_select_with_permutation_extractor(self):
        """Shadow selection should work when extractor is PermutationImportanceExtractor."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        y_s = pd.Series(y)

        # Split into train/val
        X_train, X_val = X_df.iloc[:70], X_df.iloc[70:]
        y_train, y_val = y_s.iloc[:70], y_s.iloc[70:]

        extractor = PermutationImportanceExtractor(random_state=42)
        selector = ShadowFeatureSelector(
            importance_extractor=extractor,
            n_shadows=2,
            random_state=42,
        )

        # This should not raise ValueError about missing X_val/y_val
        result = selector.fit_select(
            model=_SimpleClassifier(random_state=42),
            X=X_train,
            y=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        assert len(result.features_to_keep) + len(result.features_to_drop) == 10
        assert len(result.feature_importances) == 10

    def test_shadow_select_without_val_raises_for_permutation(self):
        """Shadow selection with PermutationImportanceExtractor and no X_val should raise."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_s = pd.Series(y)

        extractor = PermutationImportanceExtractor(random_state=42)
        selector = ShadowFeatureSelector(
            importance_extractor=extractor,
            n_shadows=2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="X_val and y_val"):
            selector.fit_select(
                model=_SimpleClassifier(random_state=42),
                X=X_df,
                y=y_s,
            )

    def test_select_features_convenience_passes_val(self):
        """select_features convenience method should pass X_val/y_val through."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_s = pd.Series(y)

        X_train, X_val = X_df.iloc[:70], X_df.iloc[70:]
        y_train, y_val = y_s.iloc[:70], y_s.iloc[70:]

        extractor = PermutationImportanceExtractor(random_state=42)
        selector = ShadowFeatureSelector(
            importance_extractor=extractor,
            n_shadows=2,
            random_state=42,
        )

        features = selector.select_features(
            model=_SimpleClassifier(random_state=42),
            X=X_train,
            y=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        assert isinstance(features, list)
        assert all(f in X_df.columns for f in features)


# =============================================================================
# Fix 3: n_estimators guard
# =============================================================================


class TestSupportsParam:
    """Test _supports_param helper."""

    def test_supports_param_present(self):
        from sklearn.ensemble import RandomForestClassifier

        assert _supports_param(RandomForestClassifier, "n_estimators") is True

    def test_supports_param_absent(self):
        assert _supports_param(Ridge, "n_estimators") is False

    def test_supports_param_via_kwargs(self):
        class _ModelWithKwargs(BaseEstimator):
            def __init__(self, **kwargs):
                pass

        assert _supports_param(_ModelWithKwargs, "n_estimators") is True

    def test_supports_param_alpha(self):
        assert _supports_param(Ridge, "alpha") is True


class TestNEstimatorsGuard:
    """Test that n_estimators injection is skipped for unsupported models."""

    def test_tuning_n_estimators_skipped_for_ridge(self):
        """tuning_n_estimators should be silently skipped for Ridge (no n_estimators param)."""
        X, y = make_classification(n_samples=60, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_s = pd.Series(y)
        ctx = DataContext.from_Xy(X_df, y_s)

        node = ModelNode(
            name="ridge",
            estimator_class=Ridge,
            fixed_params={"alpha": 1.0},
        )
        graph = ModelGraph()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            cv_config=cv_config,
            metric="neg_mean_squared_error",
            greater_is_better=False,
            verbose=0,
            tuning_n_estimators=50,
            final_n_estimators=200,
        )
        data_manager = DataManager(cv_config)

        from tests.conftest import MockSearchBackend

        orchestrator = TuningOrchestrator(
            graph=graph,
            data_manager=data_manager,
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        # This should not raise TypeError about n_estimators
        fitted = orchestrator.fit(ctx)
        assert "n_estimators" not in fitted.fitted_nodes["ridge"].best_params
        assert np.isfinite(fitted.fitted_nodes["ridge"].mean_score)

    def test_estimator_scaling_search_skipped_for_ridge(self):
        """estimator_scaling_search should be skipped for models without n_estimators."""
        X, y = make_classification(n_samples=60, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        y_s = pd.Series(y)
        ctx = DataContext.from_Xy(X_df, y_s)

        node = ModelNode(
            name="ridge",
            estimator_class=Ridge,
            fixed_params={"alpha": 1.0},
        )
        graph = ModelGraph()
        graph.add_node(node)

        cv_config = CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42)
        tuning_config = TuningConfig(
            strategy=OptimizationStrategy.NONE,
            cv_config=cv_config,
            metric="neg_mean_squared_error",
            greater_is_better=False,
            verbose=0,
            estimator_scaling_search=True,
        )
        data_manager = DataManager(cv_config)

        from tests.conftest import MockSearchBackend

        orchestrator = TuningOrchestrator(
            graph=graph,
            data_manager=data_manager,
            search_backend=MockSearchBackend(),
            tuning_config=tuning_config,
        )

        # This should not raise TypeError
        fitted = orchestrator.fit(ctx)
        assert "n_estimators" not in fitted.fitted_nodes["ridge"].best_params
        assert np.isfinite(fitted.fitted_nodes["ridge"].mean_score)
