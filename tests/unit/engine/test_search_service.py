from __future__ import annotations

from types import SimpleNamespace

from sklearn_meta.engine.search import SearchService
from sklearn_meta.search.backends.base import OptimizationResult, TrialResult
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.spec.node import NodeSpec


class _FixedBackend:
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
        params = {"n_estimators": 999, "learning_rate": 0.2}
        objective(params)
        return OptimizationResult(
            best_params=params,
            best_value=0.0,
            trials=[TrialResult(params=params, value=0.0, trial_id=0)],
            n_trials=1,
            study_name=study_name,
        )


class DummyBoostingEstimator:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X


def test_optimize_node_persists_tuning_n_estimators_in_best_params():
    service = SearchService(_FixedBackend())
    search_space = (
        SearchSpace()
        .add_int("n_estimators", 50, 500, step=50)
        .add_float("learning_rate", 0.01, 0.3, log=True)
    )
    node = NodeSpec(
        name="xgb",
        estimator_class=DummyBoostingEstimator,
        search_space=search_space,
    )
    seen = []

    def cross_validate_fn(params):
        seen.append(dict(params))
        return SimpleNamespace(mean_score=0.5)

    best_params, _ = service.optimize_node(
        node,
        search_space,
        reparam_space=None,
        cross_validate_fn=cross_validate_fn,
        n_trials=1,
        timeout=None,
        early_stopping_rounds=None,
        greater_is_better=True,
        tuning_n_estimators=200,
    )

    assert seen == [{"n_estimators": 200, "learning_rate": 0.2}]
    assert best_params["n_estimators"] == 200
    assert best_params["learning_rate"] == 0.2
