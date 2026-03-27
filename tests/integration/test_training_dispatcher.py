"""Integration tests for node-level training dispatch."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from sklearn_meta.data.view import DataView
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.execution.training import LocalTrainingDispatcher
from sklearn_meta.runtime.config import CVConfig, CVStrategy, RunConfig, TuningConfig
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.search.backends.base import OptimizationResult, SearchBackend, TrialResult
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec
from sklearn_meta.spec.quantile import JointQuantileConfig, JointQuantileGraphSpec


class _CountingDispatcher(LocalTrainingDispatcher):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []
        self.job_names = []

    def dispatch(self, jobs, services):
        self.batch_sizes.append(len(jobs))
        self.job_names.append([job.node_spec["name"] for job in jobs])
        return super().dispatch(jobs, services)


class _MockSearchBackend(SearchBackend):
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
        value = objective({})
        return OptimizationResult(
            best_params={},
            best_value=value,
            trials=[TrialResult(params={}, value=value, trial_id=0, duration=0.0)],
            n_trials=1,
            study_name=study_name,
        )

    def suggest_params(self, search_space):
        return {}

    def tell(self, params, value):
        pass

    def get_state(self):
        return {}

    def load_state(self, state):
        pass


class _FixedSearchBackend(SearchBackend):
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
        return OptimizationResult(
            best_params={"alpha": 7.0},
            best_value=0.0,
            trials=[TrialResult(params={"alpha": 7.0}, value=0.0, trial_id=0)],
            n_trials=1,
            study_name=study_name,
        )

    def suggest_params(self, search_space):
        return {"alpha": 7.0}

    def tell(self, params, value):
        pass

    def get_state(self):
        return {}

    def load_state(self, state):
        pass


class _MockQuantileRegressor:
    def __init__(self, objective="reg:quantileerror", quantile_alpha=0.5, **kwargs):
        self.objective = objective
        self.quantile_alpha = quantile_alpha

    def fit(self, X, y, **kwargs):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean + self.quantile_alpha)


def test_single_node_dispatch_matches_inline(small_classification_data):
    X, y = small_classification_data
    data = DataView.from_Xy(X, y)

    graph = GraphSpec()
    graph.add_node(
        NodeSpec(
            name="lr",
            estimator_class=LogisticRegression,
            fixed_params={"max_iter": 1000, "random_state": 42},
        )
    )

    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            metric="accuracy",
            greater_is_better=True,
        ),
        verbosity=0,
    )

    inline_services = RuntimeServices(search_backend=_MockSearchBackend())
    dispatched_services = RuntimeServices(
        search_backend=_MockSearchBackend(),
        training_dispatcher=_CountingDispatcher(),
    )

    from sklearn_meta.engine.runner import GraphRunner

    inline = GraphRunner(inline_services).fit(graph, data, config)
    dispatched = GraphRunner(dispatched_services).fit(graph, data, config)

    assert dispatched_services.training_dispatcher.batch_sizes == [1]
    assert inline.node_results["lr"].oof_predictions.shape == dispatched.node_results["lr"].oof_predictions.shape
    assert len(dispatched.node_results["lr"].models) == 3


def test_callable_condition_falls_back_to_inline(small_classification_data):
    X, y = small_classification_data
    data = DataView.from_Xy(X, y)

    graph = GraphSpec()
    graph.add_node(
        NodeSpec(
            name="conditional",
            estimator_class=LogisticRegression,
            fixed_params={"max_iter": 1000, "random_state": 42},
            condition=lambda view: view.n_rows > 0,
        )
    )

    dispatcher = _CountingDispatcher()
    services = RuntimeServices(
        search_backend=_MockSearchBackend(),
        training_dispatcher=dispatcher,
    )
    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.STRATIFIED, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.NONE,
            metric="accuracy",
            greater_is_better=True,
        ),
        verbosity=0,
    )

    from sklearn_meta.engine.runner import GraphRunner

    run = GraphRunner(services).fit(graph, data, config)

    assert dispatcher.batch_sizes == []
    assert "conditional" in run.node_results


def test_joint_quantile_dispatches_entire_training_layer():
    X = pd.DataFrame({"x1": np.linspace(0.0, 1.0, 12), "x2": np.tile([0.0, 1.0], 6)})
    price = pd.Series(np.linspace(1.0, 2.0, 12))
    volume = pd.Series(np.linspace(2.0, 3.0, 12))
    volatility = pd.Series(np.linspace(3.0, 4.0, 12))
    data = (
        DataView.from_Xy(X, y=price)
        .bind_target(price, name="price")
        .bind_target(volume, name="volume")
        .bind_target(volatility, name="volatility")
    )

    graph = JointQuantileGraphSpec(
        JointQuantileConfig(
            property_names=["price", "volume", "volatility"],
            estimator_class=_MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )
    )

    dispatcher = _CountingDispatcher()
    services = RuntimeServices(
        search_backend=_MockSearchBackend(),
        training_dispatcher=dispatcher,
    )
    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=1,
            metric="neg_mean_squared_error",
            greater_is_better=False,
        ),
        verbosity=0,
    )

    from sklearn_meta.engine.runner import GraphRunner

    run = GraphRunner(services).fit(graph, data, config)

    assert dispatcher.batch_sizes == [3]
    assert set(dispatcher.job_names[0]) == {
        "quantile_price",
        "quantile_volume",
        "quantile_volatility",
    }
    assert len(run.node_results) == 3


def test_parallel_dispatch_preserves_configured_search_backend():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    data = DataView.from_Xy(X, y)

    search_space = SearchSpace()
    search_space.add_float("alpha", 0.1, 10.0)

    graph = GraphSpec()
    graph.add_node(
        NodeSpec(
            name="ridge",
            estimator_class=Ridge,
            search_space=search_space,
        )
    )

    config = RunConfig(
        cv=CVConfig(n_splits=3, strategy=CVStrategy.RANDOM, random_state=42),
        tuning=TuningConfig(
            strategy=OptimizationStrategy.LAYER_BY_LAYER,
            n_trials=1,
            metric="neg_mean_squared_error",
            greater_is_better=False,
        ),
        verbosity=0,
    )

    services = RuntimeServices(
        search_backend=_FixedSearchBackend(),
        training_dispatcher=LocalTrainingDispatcher(n_workers=2, backend="threading", prefer="threads"),
    )

    from sklearn_meta.engine.runner import GraphRunner

    run = GraphRunner(services).fit(graph, data, config)

    assert run.node_results["ridge"].best_params == {"alpha": 7.0}
