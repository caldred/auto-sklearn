"""Training artifacts: results from a training run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from sklearn_meta.runtime.config import CVResult, RunConfig

if TYPE_CHECKING:
    from sklearn_meta.artifacts.inference import InferenceGraph
    from sklearn_meta.search.backends.base import OptimizationResult
    from sklearn_meta.spec.graph import GraphSpec


@dataclass
class NodeRunResult:
    node_name: str
    cv_result: CVResult
    best_params: Dict[str, Any]
    selected_features: Optional[List[str]] = None
    optimization_result: Optional[OptimizationResult] = None

    @property
    def oof_predictions(self) -> np.ndarray:
        return self.cv_result.oof_predictions

    @property
    def models(self) -> List[Any]:
        return self.cv_result.models

    @property
    def mean_score(self) -> float:
        return self.cv_result.mean_score


@dataclass
class QuantileNodeRunResult(NodeRunResult):
    quantile_models: Dict[float, List[Any]] = field(default_factory=dict)
    oof_quantile_predictions: Optional[np.ndarray] = None  # (n_samples, n_quantiles)


@dataclass
class RunMetadata:
    timestamp: str  # ISO 8601
    sklearn_meta_version: str
    data_shape: Tuple[int, int]
    feature_names: List[str]
    cv_config: Optional[Dict[str, Any]]
    tuning_config_summary: Dict[str, Any]
    total_trials: int
    data_hash: Optional[str]
    random_state: Optional[int]


@dataclass
class TrainingRun:
    graph: GraphSpec
    config: RunConfig
    node_results: Dict[str, NodeRunResult]  # values may be QuantileNodeRunResult
    metadata: RunMetadata
    total_time: float = 0.0
    _inference_graph: Optional[InferenceGraph] = field(
        default=None, init=False, repr=False, compare=False,
    )

    def compile_inference(self) -> InferenceGraph:
        from sklearn_meta.artifacts.compiler import InferenceCompiler
        return InferenceCompiler.compile(self)

    def _get_inference_graph(self) -> InferenceGraph:
        """Return the cached InferenceGraph, compiling on first access."""
        if self._inference_graph is None:
            self._inference_graph = self.compile_inference()
        return self._inference_graph

    def predict(
        self, X: pd.DataFrame, node_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compile (once) and predict in a single call.

        For repeated predictions, prefer :meth:`compile_inference` to avoid
        re-checking the cache on each call.
        """
        from sklearn_meta.spec.quantile import JointQuantileGraphSpec

        if isinstance(self.graph, JointQuantileGraphSpec):
            raise TypeError(
                "predict() is not supported for joint quantile graphs. "
                "Use compile_inference() to get a JointQuantileInferenceGraph, "
                "then call predict_median(), predict_quantile(), or sample_joint()."
            )
        return self._get_inference_graph().predict(X, node_name=node_name)

    def predict_proba(
        self, X: pd.DataFrame, node_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compile (once) and predict probabilities in a single call."""
        from sklearn_meta.spec.quantile import JointQuantileGraphSpec

        if isinstance(self.graph, JointQuantileGraphSpec):
            raise TypeError(
                "predict_proba() is not supported for joint quantile graphs. "
                "Use compile_inference() for quantile-specific methods."
            )
        return self._get_inference_graph().predict_proba(X, node_name=node_name)

    def get_node(self, node_name: str) -> NodeRunResult:
        """Return a fitted node result by node name."""
        try:
            return self.node_results[node_name]
        except KeyError as exc:
            raise KeyError(f"Node '{node_name}' not found in training run") from exc

    def get_oof_predictions(self, node_name: str) -> np.ndarray:
        """Return cached OOF predictions for a fitted node."""
        return self.get_node(node_name).oof_predictions

    @property
    def fitted_nodes(self) -> Dict[str, NodeRunResult]:
        """Convenience mapping of fitted nodes.

        For joint quantile graphs, keys are property names to match downstream
        usage. For all other graphs, this is equivalent to ``node_results``.
        """
        from sklearn_meta.spec.quantile import JointQuantileGraphSpec

        if not isinstance(self.graph, JointQuantileGraphSpec):
            return self.node_results

        return {
            prop_name: self.node_results[f"quantile_{prop_name}"]
            for prop_name in self.graph.property_order
        }

    def save(self, path, include_training_artifacts=True) -> None:
        # Use joblib for models, JSON manifest for metadata
        import joblib
        from pathlib import Path
        from sklearn_meta.persistence.manifest import write_manifest, to_json_safe

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        fitted_node_names = set(self.node_results)
        fitted_nodes_meta: Dict[str, Any] = {}

        for node_name, result in self.node_results.items():
            node_dir = path / "nodes" / node_name
            node_dir.mkdir(parents=True, exist_ok=True)

            models = result.models
            for i, model in enumerate(models):
                joblib.dump(model, node_dir / f"fold_{i}.joblib")

            if include_training_artifacts:
                training_data: Dict[str, Any] = {
                    "oof_predictions": result.oof_predictions,
                    "repeat_oof": result.cv_result.repeat_oof,
                    "fold_results": [
                        {
                            "fold_idx": fr.fold.fold_idx,
                            "repeat_idx": fr.fold.repeat_idx,
                            "train_indices": fr.fold.train_indices,
                            "val_indices": fr.fold.val_indices,
                            "val_predictions": fr.val_predictions,
                            "val_score": fr.val_score,
                            "train_score": fr.train_score,
                            "fit_time": fr.fit_time,
                            "predict_time": fr.predict_time,
                            "params": fr.params,
                        }
                        for fr in result.cv_result.fold_results
                    ],
                    "optimization_result": result.optimization_result,
                }
                if isinstance(result, QuantileNodeRunResult):
                    training_data["quantile_models"] = result.quantile_models
                    training_data["oof_quantile_predictions"] = result.oof_quantile_predictions
                joblib.dump(training_data, node_dir / "training_artifacts.joblib")

            fitted_nodes_meta[node_name] = to_json_safe({
                "best_params": result.best_params,
                "selected_features": result.selected_features,
                "mean_score": result.mean_score,
                "n_folds": len(models),
                "training_artifacts_included": include_training_artifacts,
                "is_quantile": isinstance(result, QuantileNodeRunResult),
            }, path=f"fitted_nodes.{node_name}")

        # Build graph data for manifest
        graph_nodes = [
            to_json_safe(node.to_dict(), path=f"graph.nodes.{node.name}")
            for node in self.graph.nodes.values()
            if node.name in fitted_node_names
        ]
        graph_edges = [
            to_json_safe(edge.to_dict(), path=f"graph.edges.{edge.source}->{edge.target}")
            for edge in self.graph.edges
            if edge.source in fitted_node_names and edge.target in fitted_node_names
        ]

        # Serialize RunConfig
        run_config_data: Dict[str, Any] = {
            "cv": {
                "n_splits": self.config.cv.n_splits,
                "n_repeats": self.config.cv.n_repeats,
                "strategy": self.config.cv.strategy.value,
                "shuffle": self.config.cv.shuffle,
                "random_state": self.config.cv.random_state,
            },
            "tuning": {
                "n_trials": self.config.tuning.n_trials,
                "timeout": self.config.tuning.timeout,
                "early_stopping_rounds": self.config.tuning.early_stopping_rounds,
                "metric": self.config.tuning.metric,
                "greater_is_better": self.config.tuning.greater_is_better,
                "strategy": self.config.tuning.strategy.value,
                "show_progress": self.config.tuning.show_progress,
            },
            "verbosity": self.config.verbosity,
        }
        if self.config.feature_selection is not None:
            fs = self.config.feature_selection
            run_config_data["feature_selection"] = {
                "enabled": fs.enabled,
                "method": fs.method.value if hasattr(fs.method, "value") else str(fs.method),
                "n_shadows": fs.n_shadows,
                "threshold_mult": fs.threshold_mult,
                "threshold_percentile": fs.threshold_percentile,
                "retune_after_pruning": fs.retune_after_pruning,
                "min_features": fs.min_features,
                "max_features": fs.max_features,
                "random_state": fs.random_state,
            }
        if self.config.reparameterization is not None:
            rp = self.config.reparameterization
            run_config_data["reparameterization"] = {
                "enabled": rp.enabled,
                "use_prebaked": rp.use_prebaked,
            }
        if self.config.estimator_scaling is not None:
            es = self.config.estimator_scaling
            run_config_data["estimator_scaling"] = {
                "tuning_n_estimators": es.tuning_n_estimators,
                "final_n_estimators": es.final_n_estimators,
                "scaling_search": es.scaling_search,
                "scaling_factors": es.scaling_factors,
            }

        manifest = {
            "version": 3,
            "training_artifacts_included": include_training_artifacts,
            "graph": {"nodes": graph_nodes, "edges": graph_edges},
            "fitted_nodes": fitted_nodes_meta,
            "run_config": run_config_data,
            "total_time": self.total_time,
            "metadata": to_json_safe({
                "timestamp": self.metadata.timestamp,
                "sklearn_meta_version": self.metadata.sklearn_meta_version,
                "data_shape": self.metadata.data_shape,
                "feature_names": self.metadata.feature_names,
                "cv_config": self.metadata.cv_config,
                "tuning_config_summary": self.metadata.tuning_config_summary,
                "total_trials": self.metadata.total_trials,
                "data_hash": self.metadata.data_hash,
                "random_state": self.metadata.random_state,
            }, path="metadata"),
        }

        write_manifest(path, manifest)

    @classmethod
    def load(cls, path) -> TrainingRun:
        # Load from manifest + joblib files
        import joblib
        from pathlib import Path
        from sklearn_meta.persistence.manifest import read_manifest
        from sklearn_meta.spec.node import NodeSpec
        from sklearn_meta.spec.graph import GraphSpec
        from sklearn_meta.spec.dependency import DependencyEdge
        from sklearn_meta.runtime.config import (
            CVFold, FoldResult, CVResult as _CVResult, RunConfig as _RunConfig,
            CVConfig as _CVConfig, CVStrategy as _CVStrategy,
            TuningConfig as _TuningConfig,
            FeatureSelectionConfig as _FeatureSelectionConfig,
            FeatureSelectionMethod as _FeatureSelectionMethod,
            ReparameterizationConfig as _ReparameterizationConfig,
            EstimatorScalingConfig as _EstimatorScalingConfig,
        )
        from sklearn_meta.engine.strategy import OptimizationStrategy as _OptStrategy

        path = Path(path)
        manifest = read_manifest(path)

        # Reconstruct graph
        graph = GraphSpec()
        for node_data in manifest["graph"]["nodes"]:
            graph.add_node(NodeSpec.from_dict(node_data))
        for edge_data in manifest["graph"]["edges"]:
            graph.add_edge(DependencyEdge.from_dict(edge_data))

        # Load node results
        node_results: Dict[str, NodeRunResult] = {}

        for node_name, node_meta in manifest["fitted_nodes"].items():
            n_folds = node_meta["n_folds"]
            node_dir = path / "nodes" / node_name

            models = []
            for i in range(n_folds):
                models.append(joblib.load(node_dir / f"fold_{i}.joblib"))

            node_has_training = bool(node_meta.get("training_artifacts_included", False))

            if node_has_training:
                training_data = joblib.load(node_dir / "training_artifacts.joblib")
                fold_results = []
                for model, fold_data in zip(models, training_data["fold_results"]):
                    fold = CVFold(
                        fold_idx=fold_data["fold_idx"],
                        train_indices=np.asarray(fold_data["train_indices"]),
                        val_indices=np.asarray(fold_data["val_indices"]),
                        repeat_idx=fold_data.get("repeat_idx", 0),
                    )
                    fold_results.append(FoldResult(
                        fold=fold, model=model,
                        val_predictions=np.asarray(fold_data["val_predictions"]),
                        val_score=fold_data["val_score"],
                        train_score=fold_data.get("train_score"),
                        fit_time=fold_data.get("fit_time", 0.0),
                        predict_time=fold_data.get("predict_time", 0.0),
                        params=fold_data.get("params", {}),
                    ))
                cv_result = _CVResult(
                    fold_results=fold_results,
                    oof_predictions=np.asarray(training_data["oof_predictions"]),
                    node_name=node_name,
                    repeat_oof=(
                        np.asarray(training_data["repeat_oof"])
                        if training_data.get("repeat_oof") is not None
                        else None
                    ),
                )
                opt_result = training_data.get("optimization_result")

                is_quantile = node_meta.get("is_quantile", False)
                if is_quantile:
                    node_results[node_name] = QuantileNodeRunResult(
                        node_name=node_name, cv_result=cv_result,
                        best_params=node_meta["best_params"],
                        selected_features=node_meta.get("selected_features"),
                        optimization_result=opt_result,
                        quantile_models=training_data.get("quantile_models", {}),
                        oof_quantile_predictions=training_data.get("oof_quantile_predictions"),
                    )
                else:
                    node_results[node_name] = NodeRunResult(
                        node_name=node_name, cv_result=cv_result,
                        best_params=node_meta["best_params"],
                        selected_features=node_meta.get("selected_features"),
                        optimization_result=opt_result,
                    )
            else:
                # Build minimal CV result for inference
                fold_results = []
                for i, model in enumerate(models):
                    fold = CVFold(fold_idx=i, train_indices=np.array([], dtype=int),
                                  val_indices=np.array([], dtype=int))
                    fold_results.append(FoldResult(
                        fold=fold, model=model,
                        val_predictions=np.array([]),
                        val_score=node_meta.get("mean_score", 0.0),
                    ))
                cv_result = _CVResult(fold_results=fold_results,
                                      oof_predictions=np.array([]),
                                      node_name=node_name)
                node_results[node_name] = NodeRunResult(
                    node_name=node_name, cv_result=cv_result,
                    best_params=node_meta["best_params"],
                    selected_features=node_meta.get("selected_features"),
                )

        # Reconstruct metadata
        meta_data = manifest.get("metadata", {})
        metadata = RunMetadata(
            timestamp=meta_data.get("timestamp", ""),
            sklearn_meta_version=meta_data.get("sklearn_meta_version", "unknown"),
            data_shape=tuple(meta_data.get("data_shape", (0, 0))),
            feature_names=meta_data.get("feature_names", []),
            cv_config=meta_data.get("cv_config"),
            tuning_config_summary=meta_data.get("tuning_config_summary", {}),
            total_trials=meta_data.get("total_trials", 0),
            data_hash=meta_data.get("data_hash"),
            random_state=meta_data.get("random_state"),
        )

        # Reconstruct RunConfig
        rc_data = manifest.get("run_config", {})
        cv_data = rc_data.get("cv", {})
        tuning_data = rc_data.get("tuning", {})

        cv_config = _CVConfig(
            n_splits=cv_data.get("n_splits", 5),
            n_repeats=cv_data.get("n_repeats", 1),
            strategy=_CVStrategy(cv_data["strategy"]) if "strategy" in cv_data else _CVStrategy.GROUP,
            shuffle=cv_data.get("shuffle", True),
            random_state=cv_data.get("random_state", 42),
        )
        tuning_config = _TuningConfig(
            n_trials=tuning_data.get("n_trials", 100),
            timeout=tuning_data.get("timeout"),
            early_stopping_rounds=tuning_data.get("early_stopping_rounds"),
            metric=tuning_data.get("metric", "neg_mean_squared_error"),
            greater_is_better=tuning_data.get("greater_is_better", False),
            strategy=_OptStrategy(tuning_data["strategy"]) if "strategy" in tuning_data else _OptStrategy.LAYER_BY_LAYER,
            show_progress=tuning_data.get("show_progress", False),
        )

        fs_config = None
        fs_data = rc_data.get("feature_selection")
        if fs_data is not None:
            fs_config = _FeatureSelectionConfig(
                enabled=fs_data.get("enabled", True),
                method=_FeatureSelectionMethod(fs_data["method"]) if "method" in fs_data else _FeatureSelectionMethod.SHADOW,
                n_shadows=fs_data.get("n_shadows", 5),
                threshold_mult=fs_data.get("threshold_mult", 1.414),
                threshold_percentile=fs_data.get("threshold_percentile", 10.0),
                retune_after_pruning=fs_data.get("retune_after_pruning", True),
                min_features=fs_data.get("min_features", 1),
                max_features=fs_data.get("max_features"),
                random_state=fs_data.get("random_state", 42),
            )

        rp_config = None
        rp_data = rc_data.get("reparameterization")
        if rp_data is not None:
            rp_config = _ReparameterizationConfig(
                enabled=rp_data.get("enabled", True),
                use_prebaked=rp_data.get("use_prebaked", True),
            )

        es_config = None
        es_data = rc_data.get("estimator_scaling")
        if es_data is not None:
            es_config = _EstimatorScalingConfig(
                tuning_n_estimators=es_data.get("tuning_n_estimators"),
                final_n_estimators=es_data.get("final_n_estimators"),
                scaling_search=es_data.get("scaling_search", False),
                scaling_factors=es_data.get("scaling_factors"),
            )

        config = _RunConfig(
            cv=cv_config,
            tuning=tuning_config,
            feature_selection=fs_config,
            reparameterization=rp_config,
            estimator_scaling=es_config,
            verbosity=rc_data.get("verbosity", 1),
        )

        return cls(
            graph=graph, config=config,
            node_results=node_results,
            metadata=metadata,
            total_time=manifest.get("total_time", 0.0),
        )
