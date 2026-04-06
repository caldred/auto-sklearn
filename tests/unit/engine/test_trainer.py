from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn_meta.artifacts.training import NodeRunResult
from sklearn_meta.data.view import DataView
from sklearn_meta.engine.trainer import StandardNodeTrainer
from sklearn_meta.runtime.config import (
    CVFold,
    CVResult,
    FeatureSelectionConfig,
    RunConfig,
    TuningConfig,
)
from sklearn_meta.search.backends.base import OptimizationResult, TrialResult
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.spec.node import NodeSpec


class _RecordingSearchService:
    def __init__(self) -> None:
        self.optimize_calls = []
        self.reparam_calls = []

    def build_reparameterized_space(self, node, search_space, reparam_config):
        self.reparam_calls.append(search_space)
        return None

    def optimize_node(
        self,
        node,
        search_space,
        reparam_space,
        cross_validate_fn,
        n_trials,
        timeout,
        early_stopping_rounds,
        greater_is_better,
        tuning_n_estimators=None,
        seed_params=None,
    ):
        self.optimize_calls.append(
            {
                "search_space": search_space,
                "seed_params": seed_params,
            }
        )
        if len(self.optimize_calls) == 1:
            params = {"learning_rate": 0.05, "max_depth": 6}
        else:
            params = {"learning_rate": 0.045, "max_depth": 6}

        return params, OptimizationResult(
            best_params=params,
            best_value=0.0,
            trials=[TrialResult(params=params, value=0.0, trial_id=len(self.optimize_calls) - 1)],
            n_trials=1,
            study_name="trainer-test",
        )


class _FeatureSelectionService:
    def apply(self, node, data, best_params, target_key="y"):
        return (
            SimpleNamespace(selected_features=["f1", "f2"]),
            data.select_features(["f1", "f2"]),
        )


def _dummy_cv_result(node_name: str, best_params: dict[str, float]) -> CVResult:
    fold = CVFold(
        fold_idx=0,
        train_indices=np.asarray([0, 1], dtype=int),
        val_indices=np.asarray([2], dtype=int),
    )
    fold_result = SimpleNamespace(
        fold=fold,
        model=SimpleNamespace(),
        val_predictions=np.asarray([0.5]),
        val_score=0.5,
        fit_time=0.0,
        predict_time=0.0,
        params=best_params,
    )
    return CVResult(
        fold_results=[fold_result],
        oof_predictions=np.asarray([0.5, 0.5, 0.5]),
        node_name=node_name,
    )


def test_feature_selection_retune_uses_narrowed_search_space_and_warm_start(monkeypatch):
    trainer = StandardNodeTrainer()
    search_service = _RecordingSearchService()
    selection_service = _FeatureSelectionService()
    node = NodeSpec(
        name="model",
        estimator_class=LogisticRegression,
        search_space=(
            SearchSpace()
            .add_float("learning_rate", 0.01, 0.3, log=True)
            .add_int("max_depth", 3, 10)
        ),
        fixed_params={"solver": "lbfgs"},
    )
    data = DataView.from_Xy(
        pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, 0.0, 1.0], "f3": [5.0, 6.0, 7.0]}),
        pd.Series([0, 1, 0]),
    )
    config = RunConfig(
        tuning=TuningConfig(metric="accuracy", greater_is_better=True, n_trials=2),
        feature_selection=FeatureSelectionConfig(enabled=True, retune_after_pruning=True),
    )
    services = SimpleNamespace(plugin_registry=None, audit_logger=None, fit_cache=None)

    def fake_cross_validate(self, node, data, params, cv_engine, services, config):
        return _dummy_cv_result(node.name, params)

    monkeypatch.setattr(StandardNodeTrainer, "_cross_validate", fake_cross_validate)

    result = trainer.fit_node(
        node=node,
        data=data,
        config=config,
        services=services,
        cv_engine=SimpleNamespace(),
        search_service=search_service,
        selection_service=selection_service,
    )

    assert isinstance(result, NodeRunResult)
    assert result.selected_features == ["f1", "f2"]
    assert len(search_service.optimize_calls) == 2
    assert search_service.optimize_calls[1]["seed_params"] == [
        {"learning_rate": 0.05, "max_depth": 6}
    ]

    original_space = search_service.optimize_calls[0]["search_space"]
    narrowed_space = search_service.optimize_calls[1]["search_space"]
    original_learning_rate = original_space.get_parameter("learning_rate")
    narrowed_learning_rate = narrowed_space.get_parameter("learning_rate")
    original_max_depth = original_space.get_parameter("max_depth")
    narrowed_max_depth = narrowed_space.get_parameter("max_depth")

    assert narrowed_learning_rate.low > original_learning_rate.low
    assert narrowed_learning_rate.high < original_learning_rate.high
    assert narrowed_max_depth.low < 6
    assert narrowed_max_depth.high > 6
