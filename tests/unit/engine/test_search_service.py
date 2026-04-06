from __future__ import annotations

from types import SimpleNamespace

from sklearn_meta.engine.search import SearchService
from sklearn_meta.meta.reparameterization import LogProductReparameterization
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
        seed_params=None,
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


class _CaptureSeedBackend:
    def __init__(self) -> None:
        self.seed_params = None

    def optimize(
        self,
        objective,
        search_space,
        n_trials=10,
        timeout=None,
        callbacks=None,
        study_name="test",
        early_stopping_rounds=None,
        seed_params=None,
    ):
        self.seed_params = seed_params
        params = {"learning_rate": 0.05, "max_depth": 6}
        objective(params)
        return OptimizationResult(
            best_params=params,
            best_value=0.0,
            trials=[TrialResult(params=params, value=0.0, trial_id=0)],
            n_trials=1,
            study_name=study_name,
        )


def test_optimize_node_filters_warm_start_params_to_the_search_space():
    backend = _CaptureSeedBackend()
    service = SearchService(backend)
    search_space = (
        SearchSpace()
        .add_float("learning_rate", 0.01, 0.3, log=True)
        .add_int("max_depth", 3, 10)
    )
    node = NodeSpec(
        name="xgb",
        estimator_class=DummyBoostingEstimator,
        search_space=search_space,
        fixed_params={"booster": "gbtree"},
    )

    def cross_validate_fn(params):
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
        seed_params=[
            {
                "learning_rate": 0.04,
                "max_depth": 5,
                "booster": "gbtree",
                "n_estimators": 200,
            }
        ],
    )

    assert backend.seed_params == [{"learning_rate": 0.04, "max_depth": 5}]
    assert best_params["booster"] == "gbtree"


def test_optimize_node_transforms_warm_start_params_when_reparameterized():
    backend = _CaptureSeedBackend()
    service = SearchService(backend)
    search_space = (
        SearchSpace()
        .add_float("learning_rate", 0.01, 0.3, log=True)
        .add_int("n_estimators", 50, 500, step=50)
    )
    reparam_space = service.build_reparameterized_space(
        NodeSpec(
            name="xgb",
            estimator_class=DummyBoostingEstimator,
            search_space=search_space,
        ),
        search_space,
        SimpleNamespace(
            enabled=True,
            use_prebaked=False,
            custom_reparameterizations=(
                LogProductReparameterization(
                    name="learning_budget",
                    param1="learning_rate",
                    param2="n_estimators",
                ),
            ),
        ),
    )
    assert reparam_space is not None

    def cross_validate_fn(params):
        return SimpleNamespace(mean_score=0.5)

    service.optimize_node(
        NodeSpec(
            name="xgb",
            estimator_class=DummyBoostingEstimator,
            search_space=search_space,
        ),
        search_space,
        reparam_space=reparam_space,
        cross_validate_fn=cross_validate_fn,
        n_trials=1,
        timeout=None,
        early_stopping_rounds=None,
        greater_is_better=True,
        seed_params=[{"learning_rate": 0.05, "n_estimators": 200}],
    )

    assert backend.seed_params is not None
    assert set(backend.seed_params[0]) == {
        "learning_rate_n_estimators_budget",
        "learning_rate_n_estimators_ratio",
    }
