from __future__ import annotations

from types import SimpleNamespace

from sklearn_meta.engine.estimator_scaling import EstimatorScaler, EstimatorScalingConfig


class DummyBoostingEstimator:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X


def test_search_scaling_uses_tuning_n_estimators_when_stage_one_params_omit_it():
    scaler = EstimatorScaler(
        EstimatorScalingConfig(
            tuning_n_estimators=200,
            scaling_search=True,
            scaling_estimators=[200, 300, 500, 750],
        ),
        greater_is_better=True,
    )
    node = SimpleNamespace(name="xgb", estimator_class=DummyBoostingEstimator)
    calls = []

    def cross_validate_fn(params):
        calls.append((params["n_estimators"], params["learning_rate"]))
        score = {
            200: 0.80,
            300: 0.82,
            500: 0.84,
            750: 0.83,
        }[params["n_estimators"]]
        return SimpleNamespace(mean_score=score)

    best_params, cv_result = scaler.search_scaling(
        node,
        None,
        {"learning_rate": 0.2, "max_depth": 5},
        cross_validate_fn,
    )

    assert calls == [
        (200, 0.2),
        (300, 0.2 * 200 / 300),
        (500, 0.2 * 200 / 500),
        (750, 0.2 * 200 / 750),
    ]
    assert best_params["n_estimators"] == 500
    assert best_params["learning_rate"] == 0.2 * 200 / 500
    assert cv_result.mean_score == 0.84


def test_search_scaling_combines_factor_and_estimator_ladders():
    scaler = EstimatorScaler(
        EstimatorScalingConfig(
            tuning_n_estimators=200,
            scaling_search=True,
            scaling_factors=[1.5, 2.5],
            scaling_estimators=[750, 1000],
        ),
        greater_is_better=True,
    )
    node = SimpleNamespace(name="xgb", estimator_class=DummyBoostingEstimator)
    calls = []

    def cross_validate_fn(params):
        calls.append(params["n_estimators"])
        score = {
            200: 0.80,
            300: 0.81,
            500: 0.82,
            750: 0.79,
        }[params["n_estimators"]]
        return SimpleNamespace(mean_score=score)

    best_params, cv_result = scaler.search_scaling(
        node,
        None,
        {"learning_rate": 0.2},
        cross_validate_fn,
    )

    assert calls == [200, 300, 500, 750]
    assert best_params["n_estimators"] == 500
    assert cv_result.mean_score == 0.82
