"""Integration tests for parallel execution."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn_meta import GraphBuilder
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


@pytest.mark.integration
class TestParallelCVFolds:
    """Integration tests for parallel CV fold fitting."""

    def test_parallel_fold_fitting_produces_valid_results(self, medium_classification_data):
        """Verify parallel fold fitting produces valid model and predictions."""
        X, y = medium_classification_data

        executor = LocalExecutor(n_workers=2, backend="threading")

        fitted = (
            GraphBuilder("parallel_test")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(n_estimators=10, random_state=42)
            .with_cv(n_splits=5)
            .with_tuning(n_trials=1, metric="accuracy", greater_is_better=True)
            .fit(X, y, executor=executor)
        )

        # Should have 5 models (one per fold)
        assert len(fitted.fitted_nodes["rf"].models) == 5

        # OOF predictions should cover all samples
        oof = fitted.get_oof_predictions("rf")
        assert len(oof) == len(y)

        # Should be able to predict on new data
        predictions = fitted.predict(X)
        assert len(predictions) == len(y)

    def test_sequential_and_parallel_produce_similar_results(self, medium_classification_data):
        """Verify sequential and parallel execution produce comparable results."""
        X, y = medium_classification_data

        # Sequential execution
        fitted_seq = (
            GraphBuilder("seq_test")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(n_estimators=10, random_state=42)
            .with_cv(n_splits=5, random_state=42)
            .with_tuning(n_trials=1, metric="accuracy", greater_is_better=True)
            .fit(X, y)
        )

        # Parallel execution
        executor = LocalExecutor(n_workers=2, backend="threading")
        fitted_par = (
            GraphBuilder("par_test")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(n_estimators=10, random_state=42)
            .with_cv(n_splits=5, random_state=42)
            .with_tuning(n_trials=1, metric="accuracy", greater_is_better=True)
            .fit(X, y, executor=executor)
        )

        # Both should have same number of models
        assert len(fitted_seq.fitted_nodes["rf"].models) == len(fitted_par.fitted_nodes["rf"].models)

        # OOF predictions should be same shape
        oof_seq = fitted_seq.get_oof_predictions("rf")
        oof_par = fitted_par.get_oof_predictions("rf")
        assert oof_seq.shape == oof_par.shape


@pytest.mark.integration
class TestParallelNodeFitting:
    """Integration tests for parallel node fitting within layers."""

    def test_parallel_fitting_of_multiple_models(self, medium_classification_data):
        """Verify multiple independent models can be fitted in parallel."""
        X, y = medium_classification_data

        executor = LocalExecutor(n_workers=2, backend="threading")

        fitted = (
            GraphBuilder("multi_model_parallel")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(n_estimators=10, random_state=42)
            .add_model("lr", LogisticRegression)
            .with_fixed_params(max_iter=1000, random_state=42)
            .with_cv(n_splits=3)
            .with_tuning(n_trials=1, metric="accuracy", greater_is_better=True)
            .fit(X, y, executor=executor)
        )

        # Both models should be fitted
        assert "rf" in fitted.fitted_nodes
        assert "lr" in fitted.fitted_nodes

        # Both should have valid predictions
        assert len(fitted.fitted_nodes["rf"].oof_predictions) == len(y)
        assert len(fitted.fitted_nodes["lr"].oof_predictions) == len(y)

    def test_stacking_with_parallel_execution(self, medium_classification_data):
        """Verify stacking ensemble works correctly with parallel execution."""
        X, y = medium_classification_data

        executor = LocalExecutor(n_workers=2, backend="threading")

        fitted = (
            GraphBuilder("stacking_parallel")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(n_estimators=10, random_state=42)
            .with_output_type("proba")
            .add_model("lr", LogisticRegression)
            .with_fixed_params(max_iter=1000, random_state=42)
            .with_output_type("proba")
            .add_model("meta", LogisticRegression)
            .with_fixed_params(max_iter=1000, random_state=42)
            .stacks_proba("rf", "lr")
            .with_cv(n_splits=3)
            .with_tuning(n_trials=1, metric="accuracy", greater_is_better=True)
            .fit(X, y, executor=executor)
        )

        # All models should be fitted
        assert "rf" in fitted.fitted_nodes
        assert "lr" in fitted.fitted_nodes
        assert "meta" in fitted.fitted_nodes

        # All models should have valid OOF predictions
        assert len(fitted.fitted_nodes["rf"].oof_predictions) == len(y)
        assert len(fitted.fitted_nodes["lr"].oof_predictions) == len(y)
        assert len(fitted.fitted_nodes["meta"].oof_predictions) == len(y)

        # Meta model should have been trained on OOF predictions from base models
        assert fitted.fitted_nodes["meta"].models is not None
        assert len(fitted.fitted_nodes["meta"].models) == 3  # 3 folds


@pytest.mark.integration
class TestOptunaParallelTrials:
    """Integration tests for parallel Optuna trials."""

    def test_parallel_trials_parameter(self, medium_classification_data):
        """Verify n_parallel_trials parameter is properly wired."""
        X, y = medium_classification_data

        from sklearn_meta.search.space import SearchSpace

        space = SearchSpace()
        space.add_int("n_estimators", 10, 50)

        fitted = (
            GraphBuilder("parallel_trials_test")
            .add_model("rf", RandomForestClassifier)
            .with_search_space(space)
            .with_fixed_params(random_state=42)
            .with_cv(n_splits=3)
            .with_tuning(
                n_trials=5,
                metric="accuracy",
                greater_is_better=True,
                n_parallel_trials=2,
            )
            .fit(X, y)
        )

        # Should complete successfully with parallel trials
        assert "rf" in fitted.fitted_nodes
        assert fitted.fitted_nodes["rf"].optimization_result is not None
        assert fitted.fitted_nodes["rf"].optimization_result.n_trials == 5

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

    def test_threading_backend(self, medium_classification_data):
        """Verify threading backend works correctly."""
        X, y = medium_classification_data

        executor = LocalExecutor(n_workers=2, backend="threading")

        fitted = (
            GraphBuilder("threading_test")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(n_estimators=10, random_state=42)
            .with_cv(n_splits=3)
            .with_tuning(n_trials=1, metric="accuracy", greater_is_better=True)
            .fit(X, y, executor=executor)
        )

        assert len(fitted.fitted_nodes["rf"].models) == 3

    @pytest.mark.slow
    def test_loky_backend(self, medium_classification_data):
        """Verify loky (process-based) backend works correctly."""
        X, y = medium_classification_data

        executor = LocalExecutor(n_workers=2, backend="loky")

        fitted = (
            GraphBuilder("loky_test")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(n_estimators=10, random_state=42)
            .with_cv(n_splits=3)
            .with_tuning(n_trials=1, metric="accuracy", greater_is_better=True)
            .fit(X, y, executor=executor)
        )

        assert len(fitted.fitted_nodes["rf"].models) == 3

    def test_executor_auto_cpu_count(self, medium_classification_data):
        """Verify executor with n_workers=-1 uses all CPUs."""
        import os

        X, y = medium_classification_data

        executor = LocalExecutor(n_workers=-1, backend="threading")

        # Should use all available CPUs
        expected_workers = os.cpu_count() or 1
        assert executor.n_workers == expected_workers

        # Should still work correctly
        fitted = (
            GraphBuilder("auto_cpu_test")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(n_estimators=10, random_state=42)
            .with_cv(n_splits=3)
            .with_tuning(n_trials=1, metric="accuracy", greater_is_better=True)
            .fit(X, y, executor=executor)
        )

        assert len(fitted.fitted_nodes["rf"].models) == 3
