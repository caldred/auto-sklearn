"""TuningOrchestrator: Main coordinator for tuning and training."""

from __future__ import annotations

import datetime
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import logging

import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVFold, CVResult, CVStrategy, FoldResult
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.dependency import DependencyType
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.model.node import ModelNode, OutputType
from sklearn_meta.core.tuning.estimator_scaling import (
    EstimatorScaler,
    EstimatorScalingConfig,
    supports_param,
)
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.search.backends.base import OptimizationResult, SearchBackend
from sklearn_meta.meta.reparameterization import Reparameterization, ReparameterizedSpace
from sklearn_meta.meta.prebaked import get_prebaked_reparameterization
from sklearn_meta.core.tuning._metrics import log_feature_selection
from sklearn_meta.selection.selector import FeatureSelector, FeatureSelectionResult
from sklearn_meta.persistence.cache import FitCache
from sklearn_meta.persistence.manifest import (
    MANIFEST_FILENAME,
    read_manifest,
    to_json_safe,
    write_manifest,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sklearn_meta.audit.logger import AuditLogger
    from sklearn_meta.execution.base import Executor
    from sklearn_meta.plugins.registry import PluginRegistry
    from sklearn_meta.selection.selector import FeatureSelectionConfig


@dataclass
class TuningConfig:
    """
    Configuration for the tuning process.

    Attributes:
        strategy: How to optimize (layer-by-layer, greedy, or none).
        n_trials: Number of optimization trials per node.
        timeout: Optional timeout in seconds for optimization.
        early_stopping_rounds: Stop if no improvement for this many trials.
        cv_config: Cross-validation configuration.
        metric: Scoring metric name (e.g., "accuracy", "roc_auc").
        greater_is_better: Whether higher metric values are better.
        feature_selection: Optional feature selection configuration.
        use_reparameterization: Whether to apply reparameterization transforms.
        custom_reparameterizations: Custom reparameterization transforms to apply.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        tuning_n_estimators: n_estimators to use during tuning (faster trials).
            If set, learning_rate is scaled proportionally for the final model.
        final_n_estimators: n_estimators to use for final model. If tuning_n_estimators
            is set, learning_rate is scaled by (tuning_n_estimators / final_n_estimators).
        estimator_scaling_search: If True, search for optimal n_estimators/learning_rate
            scaling after tuning. Tests multipliers [2, 5, 10, 20] with early stopping
            if performance degrades. Overrides tuning_n_estimators/final_n_estimators.
        estimator_scaling_factors: Custom scaling factors to search (default: [2, 5, 10, 20]).
        show_progress: If True, surface Optuna progress in terminal output.
    """

    strategy: OptimizationStrategy = OptimizationStrategy.LAYER_BY_LAYER
    n_trials: int = 100
    timeout: Optional[float] = None
    early_stopping_rounds: Optional[int] = None
    cv_config: Optional[CVConfig] = None
    metric: str = "neg_mean_squared_error"
    greater_is_better: bool = False
    feature_selection: Optional[FeatureSelectionConfig] = None
    use_reparameterization: bool = False
    custom_reparameterizations: Optional[List[Reparameterization]] = None
    verbose: int = 1
    tuning_n_estimators: Optional[int] = None
    final_n_estimators: Optional[int] = None
    estimator_scaling_search: bool = False
    estimator_scaling_factors: Optional[List[int]] = None
    show_progress: bool = False


@dataclass
class FittedNode:
    """
    Result of fitting a single node.

    Attributes:
        node: The original node definition.
        cv_result: Cross-validation results.
        best_params: Best hyperparameters found.
        optimization_result: Full optimization results.
        selected_features: Features selected (if feature selection was used).
    """

    node: ModelNode
    cv_result: CVResult
    best_params: Dict[str, Any]
    optimization_result: Optional[OptimizationResult] = None
    selected_features: Optional[List[str]] = None

    @property
    def oof_predictions(self) -> np.ndarray:
        """Out-of-fold predictions."""
        return self.cv_result.oof_predictions

    @property
    def models(self) -> List[Any]:
        """Fitted models from all folds."""
        return self.cv_result.models

    @property
    def mean_score(self) -> float:
        """Mean CV score."""
        return self.cv_result.mean_score


@dataclass
class RunMetadata:
    """Structured metadata captured during a training run."""

    timestamp: str  # ISO 8601 format
    sklearn_meta_version: str
    data_shape: Tuple[int, int]  # (n_samples, n_features)
    feature_names: List[str]
    cv_config: Optional[Dict[str, Any]]  # Serialized CVConfig
    tuning_config_summary: Dict[str, Any]  # Key tuning config fields
    total_trials: int  # Total optimization trials across all nodes
    data_hash: Optional[str]  # SHA256 of X for reproducibility
    random_state: Optional[int]  # CV random state if available


MANIFEST_VERSION = 2
TRAINING_ARTIFACTS_FILENAME = "training_artifacts.joblib"


def _serialize_cv_config(cv_config: Optional[CVConfig]) -> Optional[Dict[str, Any]]:
    """Serialize CVConfig, including nested inner_cv, to JSON-safe data."""
    if cv_config is None:
        return None
    return {
        "n_splits": cv_config.n_splits,
        "n_repeats": cv_config.n_repeats,
        "strategy": cv_config.strategy.value,
        "shuffle": cv_config.shuffle,
        "random_state": cv_config.random_state,
        "inner_cv": _serialize_cv_config(cv_config.inner_cv),
    }


def _deserialize_cv_config(data: Optional[Dict[str, Any]]) -> Optional[CVConfig]:
    """Reconstruct CVConfig from serialized data."""
    if data is None:
        return None
    return CVConfig(
        n_splits=data["n_splits"],
        n_repeats=data.get("n_repeats", 1),
        strategy=CVStrategy(data.get("strategy", CVStrategy.STRATIFIED.value)),
        shuffle=data.get("shuffle", True),
        random_state=data.get("random_state", 42),
        inner_cv=_deserialize_cv_config(data.get("inner_cv")),
    )


def _serialize_feature_selection_config(config) -> Optional[Dict[str, Any]]:
    """Serialize FeatureSelectionConfig to JSON-safe data."""
    if config is None:
        return None
    return {
        "enabled": config.enabled,
        "method": config.method.value,
        "n_shadows": config.n_shadows,
        "threshold_mult": config.threshold_mult,
        "threshold_percentile": config.threshold_percentile,
        "retune_after_pruning": config.retune_after_pruning,
        "min_features": config.min_features,
        "max_features": config.max_features,
        "random_state": config.random_state,
        "feature_groups": config.feature_groups,
    }


def _deserialize_feature_selection_config(data):
    """Reconstruct FeatureSelectionConfig from serialized data."""
    if data is None:
        return None
    from sklearn_meta.selection.selector import (
        FeatureSelectionConfig,
        FeatureSelectionMethod,
    )

    return FeatureSelectionConfig(
        enabled=data.get("enabled", True),
        method=FeatureSelectionMethod(data.get("method", "shadow")),
        n_shadows=data.get("n_shadows", 5),
        threshold_mult=data.get("threshold_mult", 1.414),
        threshold_percentile=data.get("threshold_percentile", 10.0),
        retune_after_pruning=data.get("retune_after_pruning", True),
        min_features=data.get("min_features", 1),
        max_features=data.get("max_features"),
        random_state=data.get("random_state", 42),
        feature_groups=data.get("feature_groups"),
    )


def _serialize_tuning_config(config: TuningConfig) -> Dict[str, Any]:
    """Serialize TuningConfig for persistence.

    custom_reparameterizations are intentionally not persisted because they are
    training-time objects and may contain arbitrary Python behavior.
    """
    return {
        "strategy": config.strategy.value,
        "n_trials": config.n_trials,
        "timeout": config.timeout,
        "early_stopping_rounds": config.early_stopping_rounds,
        "cv_config": _serialize_cv_config(config.cv_config),
        "metric": config.metric,
        "greater_is_better": config.greater_is_better,
        "feature_selection": _serialize_feature_selection_config(
            config.feature_selection
        ),
        "use_reparameterization": config.use_reparameterization,
        "verbose": config.verbose,
        "tuning_n_estimators": config.tuning_n_estimators,
        "final_n_estimators": config.final_n_estimators,
        "estimator_scaling_search": config.estimator_scaling_search,
        "estimator_scaling_factors": config.estimator_scaling_factors,
        "show_progress": config.show_progress,
    }


def _deserialize_tuning_config(data: Dict[str, Any]) -> TuningConfig:
    """Reconstruct TuningConfig from serialized data.

    custom_reparameterizations are always restored as None.
    """
    defaults = TuningConfig()
    return TuningConfig(
        strategy=OptimizationStrategy(
            data.get("strategy", defaults.strategy.value)
        ),
        n_trials=data.get("n_trials", defaults.n_trials),
        timeout=data.get("timeout", defaults.timeout),
        early_stopping_rounds=data.get(
            "early_stopping_rounds", defaults.early_stopping_rounds
        ),
        cv_config=_deserialize_cv_config(data.get("cv_config")),
        metric=data.get("metric", defaults.metric),
        greater_is_better=data.get(
            "greater_is_better", defaults.greater_is_better
        ),
        feature_selection=_deserialize_feature_selection_config(
            data.get("feature_selection")
        ),
        use_reparameterization=data.get(
            "use_reparameterization", defaults.use_reparameterization
        ),
        custom_reparameterizations=None,
        verbose=data.get("verbose", defaults.verbose),
        tuning_n_estimators=data.get(
            "tuning_n_estimators", defaults.tuning_n_estimators
        ),
        final_n_estimators=data.get(
            "final_n_estimators", defaults.final_n_estimators
        ),
        estimator_scaling_search=data.get(
            "estimator_scaling_search", defaults.estimator_scaling_search
        ),
        estimator_scaling_factors=data.get(
            "estimator_scaling_factors", defaults.estimator_scaling_factors
        ),
        show_progress=data.get("show_progress", defaults.show_progress),
    )


def _serialize_run_metadata(metadata: RunMetadata) -> Dict[str, Any]:
    """Serialize RunMetadata to JSON-safe data."""
    return {
        "timestamp": metadata.timestamp,
        "sklearn_meta_version": metadata.sklearn_meta_version,
        "data_shape": metadata.data_shape,
        "feature_names": metadata.feature_names,
        "cv_config": metadata.cv_config,
        "tuning_config_summary": metadata.tuning_config_summary,
        "total_trials": metadata.total_trials,
        "data_hash": metadata.data_hash,
        "random_state": metadata.random_state,
    }


def _deserialize_run_metadata(data: Dict[str, Any]) -> RunMetadata:
    """Reconstruct RunMetadata from serialized data."""
    return RunMetadata(
        timestamp=data["timestamp"],
        sklearn_meta_version=data["sklearn_meta_version"],
        data_shape=tuple(data["data_shape"]),
        feature_names=data["feature_names"],
        cv_config=data["cv_config"],
        tuning_config_summary=data["tuning_config_summary"],
        total_trials=data["total_trials"],
        data_hash=data["data_hash"],
        random_state=data["random_state"],
    )


def _build_inference_only_cv_result(
    node_name: str,
    models: List[Any],
    mean_score: float,
) -> CVResult:
    """Build a minimal CVResult for inference-only artifacts."""
    fold_results = []
    for index, model in enumerate(models):
        fold = CVFold(
            fold_idx=index,
            train_indices=np.array([], dtype=int),
            val_indices=np.array([], dtype=int),
        )
        fold_results.append(
            FoldResult(
                fold=fold,
                model=model,
                val_predictions=np.array([]),
                val_score=mean_score,
            )
        )
    return CVResult(
        fold_results=fold_results,
        oof_predictions=np.array([]),
        node_name=node_name,
    )


@dataclass
class FittedGraph:
    """
    Result of fitting the entire model graph.

    Attributes:
        graph: The original model graph.
        fitted_nodes: Dictionary mapping node names to fitted results.
        tuning_config: Configuration used for tuning.
        total_time: Total time taken in seconds.
        metadata: Structured metadata captured during the training run.
    """

    graph: ModelGraph
    fitted_nodes: Dict[str, FittedNode]
    tuning_config: TuningConfig
    total_time: float = 0.0
    metadata: Optional[RunMetadata] = None
    _training_artifacts_available: bool = field(default=True, repr=False)

    def get_node(self, name: str) -> FittedNode:
        """Get a fitted node by name."""
        return self.fitted_nodes[name]

    @property
    def training_artifacts_available(self) -> bool:
        """Whether training-only artifacts are available on this fitted graph."""
        return self._training_artifacts_available

    def get_oof_predictions(self, name: str) -> np.ndarray:
        """Get OOF predictions for a node."""
        if not self.training_artifacts_available:
            raise RuntimeError(
                "This artifact was loaded without training artifacts. "
                "OOF predictions are unavailable. Save with "
                "include_training_artifacts=True to preserve them."
            )
        return self.fitted_nodes[name].oof_predictions

    def predict(self, X, node_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using the fitted graph.

        Recursively resolves upstream dependencies to produce predictions for the
        requested node. If no node_name is specified, the first leaf node (a node
        with no downstream dependents) is used.

        **Output types:**
        - PREDICTION: Each fold model calls ``predict(X)``; results are averaged.
        - PROBA: Each fold model calls ``predict_proba(X)``; probability arrays
          are averaged across folds.
        - TRANSFORM: Not typically used as a final output, but the transformed
          features are passed downstream when used as an intermediate node.

        **Stacking at inference time:**
        For stacking graphs, upstream node predictions are computed first
        (recursively) and injected as additional columns into the feature
        matrix before predicting the current node. A prediction cache ensures
        each upstream node is computed at most once per ``predict()`` call.

        **Feature selection:**
        If feature selection was applied during fitting, the same selected
        features are used at inference time. The augmented feature matrix
        (original features + upstream predictions) is filtered to only the
        columns in ``fitted_node.selected_features``.

        **Fold ensembling:**
        Predictions from all CV fold models are averaged (``np.mean``) to
        produce the final output. This applies to both ``predict`` and
        ``predict_proba`` outputs.

        **Conditional nodes:**
        Nodes with a ``condition`` that evaluated to False during fitting are
        not present in ``fitted_nodes``. Attempting to predict from such a
        node, or from a node that depends on a skipped conditional node,
        raises a ``KeyError``.

        Args:
            X: Input features as a pandas DataFrame. Must contain the same
                columns used during fitting (plus any upstream prediction
                columns will be added automatically).
            node_name: Name of the node to predict from. If None, defaults
                to the first leaf node in the graph.

        Returns:
            Predictions as a numpy array. Shape is ``(n_samples,)`` for
            PREDICTION output type, or ``(n_samples, n_classes)`` for PROBA.

        Raises:
            KeyError: If ``node_name`` does not exist in ``fitted_nodes``, or
                if a required upstream node was skipped during fitting (e.g.,
                a conditional node whose condition was False).
            ValueError: If the graph has no leaf nodes.
        """
        if node_name is None:
            leaves = self.graph.get_leaf_nodes()
            if not leaves:
                raise ValueError("Graph has no leaf nodes")
            node_name = leaves[0]

        if node_name not in self.fitted_nodes:
            if node_name in self.graph.nodes:
                node = self.graph.get_node(node_name)
                if node.is_conditional:
                    raise KeyError(
                        f"Node '{node_name}' was not fitted because its condition "
                        f"evaluated to False during training. Cannot predict from "
                        f"a skipped conditional node."
                    )
                raise KeyError(
                    f"Node '{node_name}' exists in the graph but was not fitted. "
                    f"Fitted nodes: {list(self.fitted_nodes.keys())}"
                )
            raise KeyError(
                f"Node '{node_name}' not found in fitted graph. "
                f"Available nodes: {list(self.fitted_nodes.keys())}"
            )

        return self._predict_node(X, node_name)

    def _predict_node(
        self, X, node_name: str, cache: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Recursively predict through the graph."""
        if cache is None:
            cache = {}

        if node_name in cache:
            return cache[node_name]

        if node_name not in self.fitted_nodes:
            raise KeyError(
                f"Upstream node '{node_name}' was not fitted (it may be a "
                f"conditional node whose condition was False during training). "
                f"Available nodes: {list(self.fitted_nodes.keys())}"
            )

        fitted = self.fitted_nodes[node_name]

        # Get upstream predictions and augment features
        upstream_edges = self.graph.get_upstream(node_name)
        X_augmented = X.copy() if hasattr(X, "copy") else X

        for edge in upstream_edges:
            upstream_preds = self._predict_node(X, edge.source, cache)

            # Handle all dependency types that inject features
            if edge.dep_type in (
                DependencyType.PREDICTION,
                DependencyType.PROBA,
                DependencyType.FEATURE,
            ):
                col_name = edge.feature_name
                if upstream_preds.ndim == 1:
                    X_augmented[col_name] = upstream_preds
                else:
                    for i in range(upstream_preds.shape[1]):
                        X_augmented[f"{col_name}_{i}"] = upstream_preds[:, i]
            elif edge.dep_type == DependencyType.TRANSFORM:
                # TRANSFORM replaces features entirely
                X_augmented = upstream_preds
            elif edge.dep_type in (DependencyType.BASE_MARGIN, DependencyType.DISTILL):
                # BASE_MARGIN is handled via fit_params at training time;
                # DISTILL only affects training loss, not inference.
                pass

        # Filter to selected features if feature selection was used
        if fitted.selected_features is not None:
            X_augmented = X_augmented[fitted.selected_features]

        # Ensemble predictions from all fold models
        predictions = []
        for model in fitted.models:
            predictions.append(fitted.node.get_output(model, X_augmented))

        # Average predictions
        result = np.mean(predictions, axis=0)
        cache[node_name] = result
        return result

    def save(
        self,
        path: Union[str, Path],
        include_training_artifacts: bool = False,
    ) -> None:
        """
        Save the fitted graph to a directory.

        Creates a directory structure with a manifest.json describing the
        graph structure and tuning config, and joblib files for each fold
        model of each fitted node.

        Directory format::

            {path}/
                manifest.json
                nodes/
                    {node_name}/
                        fold_0.joblib
                        fold_1.joblib
                        ...

        Args:
            path: Directory path to save to (will be created).
            include_training_artifacts: If True, also persist OOF predictions,
                fold metadata, and optimization results needed for training-time
                inspection after load().

        Raises:
            OSError: If the directory cannot be created.
        """
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if include_training_artifacts and not self.training_artifacts_available:
            raise RuntimeError(
                "This artifact does not have in-memory training artifacts. "
                "It cannot be re-saved with include_training_artifacts=True."
            )

        fitted_node_names = set(self.fitted_nodes)

        # Save fold models for each fitted node
        fitted_nodes_meta: Dict[str, Any] = {}
        for node_name, fitted_node in self.fitted_nodes.items():
            node_dir = path / "nodes" / node_name
            node_dir.mkdir(parents=True, exist_ok=True)

            models = fitted_node.models
            for i, model in enumerate(models):
                joblib.dump(model, node_dir / f"fold_{i}.joblib")

            if include_training_artifacts:
                joblib.dump(
                    {
                        "oof_predictions": fitted_node.oof_predictions,
                        "fold_results": [
                            {
                                "fold_idx": fold_result.fold.fold_idx,
                                "repeat_idx": fold_result.fold.repeat_idx,
                                "train_indices": fold_result.fold.train_indices,
                                "val_indices": fold_result.fold.val_indices,
                                "val_predictions": fold_result.val_predictions,
                                "val_score": fold_result.val_score,
                                "train_score": fold_result.train_score,
                                "fit_time": fold_result.fit_time,
                                "predict_time": fold_result.predict_time,
                                "params": fold_result.params,
                            }
                            for fold_result in fitted_node.cv_result.fold_results
                        ],
                        "optimization_result": fitted_node.optimization_result,
                    },
                    node_dir / TRAINING_ARTIFACTS_FILENAME,
                )

            fitted_nodes_meta[node_name] = to_json_safe({
                "best_params": fitted_node.best_params,
                "selected_features": fitted_node.selected_features,
                "mean_score": fitted_node.mean_score,
                "n_folds": len(models),
                "training_artifacts_included": include_training_artifacts,
            }, path=f"fitted_nodes.{node_name}")

        graph_nodes = [
            to_json_safe(node.to_dict(), path=f"graph.nodes.{node.name}")
            for node in self.graph.nodes.values()
            if node.name in fitted_node_names
        ]
        graph_edges = [
            to_json_safe(
                edge.to_dict(),
                path=f"graph.edges.{edge.source}->{edge.target}",
            )
            for edge in self.graph.edges
            if edge.source in fitted_node_names and edge.target in fitted_node_names
        ]

        # Build manifest
        manifest = {
            "version": MANIFEST_VERSION,
            "training_artifacts_included": include_training_artifacts,
            "graph": {
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
            "fitted_nodes": fitted_nodes_meta,
            "tuning_config": to_json_safe(
                _serialize_tuning_config(self.tuning_config),
                path="tuning_config",
            ),
            "total_time": self.total_time,
        }

        if self.metadata is not None:
            manifest["metadata"] = to_json_safe(
                _serialize_run_metadata(self.metadata),
                path="metadata",
            )

        write_manifest(path, manifest)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FittedGraph":
        """
        Load a FittedGraph from a directory created by save().

        Reads the manifest.json, reconstructs the ModelGraph, loads all
        fold models from joblib files, and rebuilds FittedNode objects.
        By default, loaded artifacts are inference-oriented. Training-only
        artifacts such as OOF predictions are available only if the graph was
        saved with include_training_artifacts=True.

        Args:
            path: Directory path containing saved artifacts.

        Returns:
            FittedGraph ready for inference.

        Raises:
            FileNotFoundError: If manifest.json or any expected fold file is missing.
            ValueError: If manifest version is unsupported or JSON is corrupt.
        """
        import joblib
        from sklearn_meta.core.model.dependency import DependencyEdge

        path = Path(path)
        manifest = read_manifest(path)

        # Validate version
        version = manifest.get("version")
        if version not in (1, MANIFEST_VERSION):
            raise ValueError(
                f"Unsupported manifest version: {version}. "
                f"Only versions 1 and {MANIFEST_VERSION} are supported."
            )
        is_legacy_manifest = version == 1

        # Reconstruct ModelGraph
        graph = ModelGraph()
        for node_data in manifest["graph"]["nodes"]:
            graph.add_node(ModelNode.from_dict(node_data))
        for edge_data in manifest["graph"]["edges"]:
            graph.add_edge(DependencyEdge.from_dict(edge_data))

        # Load fitted nodes
        fitted_nodes: Dict[str, FittedNode] = {}
        training_artifacts_available = (
            False if is_legacy_manifest
            else bool(manifest.get("training_artifacts_included", False))
        )
        for node_name, node_meta in manifest["fitted_nodes"].items():
            n_folds = node_meta["n_folds"]
            node_dir = path / "nodes" / node_name

            # Load fold models
            models = []
            for i in range(n_folds):
                fold_path = node_dir / f"fold_{i}.joblib"
                try:
                    models.append(joblib.load(fold_path))
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Missing fold model file: {fold_path}. "
                        f"Expected {n_folds} fold files for node '{node_name}'."
                    ) from None

            node_has_training_artifacts = (
                False if is_legacy_manifest
                else bool(node_meta.get("training_artifacts_included", False))
            )
            training_artifacts_available = (
                training_artifacts_available and node_has_training_artifacts
            )

            optimization_result = None
            if node_has_training_artifacts:
                training_artifacts_path = node_dir / TRAINING_ARTIFACTS_FILENAME
                try:
                    training_data = joblib.load(training_artifacts_path)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Missing training artifacts file: {training_artifacts_path}. "
                        f"Expected training artifacts for node '{node_name}'."
                    ) from None
                fold_meta = training_data["fold_results"]
                if len(fold_meta) != len(models):
                    raise ValueError(
                        f"Training artifacts for node '{node_name}' are inconsistent. "
                        f"Expected {len(models)} fold records, got {len(fold_meta)}."
                    )

                fold_results = []
                for model, fold_data in zip(models, fold_meta):
                    fold = CVFold(
                        fold_idx=fold_data["fold_idx"],
                        train_indices=np.asarray(fold_data["train_indices"]),
                        val_indices=np.asarray(fold_data["val_indices"]),
                        repeat_idx=fold_data.get("repeat_idx", 0),
                    )
                    fold_results.append(
                        FoldResult(
                            fold=fold,
                            model=model,
                            val_predictions=np.asarray(
                                fold_data["val_predictions"]
                            ),
                            val_score=fold_data["val_score"],
                            train_score=fold_data.get("train_score"),
                            fit_time=fold_data.get("fit_time", 0.0),
                            predict_time=fold_data.get("predict_time", 0.0),
                            params=fold_data.get("params", {}),
                        )
                    )
                cv_result = CVResult(
                    fold_results=fold_results,
                    oof_predictions=np.asarray(training_data["oof_predictions"]),
                    node_name=node_name,
                )
                optimization_result = training_data.get("optimization_result")
            else:
                cv_result = _build_inference_only_cv_result(
                    node_name=node_name,
                    models=models,
                    mean_score=node_meta["mean_score"],
                )

            fitted_nodes[node_name] = FittedNode(
                node=graph.get_node(node_name),
                cv_result=cv_result,
                best_params=node_meta["best_params"],
                optimization_result=optimization_result,
                selected_features=node_meta["selected_features"],
            )

        # Reconstruct TuningConfig
        tuning_config = _deserialize_tuning_config(manifest["tuning_config"])

        # Reconstruct RunMetadata if present
        metadata = None
        if "metadata" in manifest:
            metadata = _deserialize_run_metadata(manifest["metadata"])

        return cls(
            graph=graph,
            fitted_nodes=fitted_nodes,
            tuning_config=tuning_config,
            total_time=manifest.get("total_time", 0.0),
            metadata=metadata,
            _training_artifacts_available=training_artifacts_available,
        )

    def __repr__(self) -> str:
        return (
            f"FittedGraph(nodes={len(self.fitted_nodes)}, "
            f"time={self.total_time:.1f}s)"
        )


class _PredictionWrapper(BaseEstimator):
    """Wrapper that returns pre-computed predictions for regressors."""

    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        preds = self._predictions
        if preds.ndim > 1:
            return np.argmax(preds, axis=1)
        return preds

    def predict_proba(self, X):
        return self._predictions

    def decision_function(self, X):
        preds = self._predictions
        if preds.ndim > 1 and preds.shape[1] == 2:
            return preds[:, 1]
        return preds


class _ClassifierPredictionWrapper(ClassifierMixin, _PredictionWrapper):
    """Wrapper for classifier predictions that sklearn scorers recognize."""

    def __init__(self, predictions, classes):
        super().__init__(predictions)
        self.classes_ = classes

    def predict(self, X):
        preds = self._predictions
        if preds.ndim > 1:
            return self.classes_[np.argmax(preds, axis=1)]
        return preds


# Backward-compatible alias for supports_param (moved to estimator_scaling module)
_supports_param = supports_param


class TuningOrchestrator:
    """
    Main coordinator for tuning and training model graphs.

    This class handles:
    - Layer-wise optimization of model hyperparameters
    - Cross-validation with proper data handling
    - Stacking with leakage-free OOF predictions
    - Plugin lifecycle hooks
    """

    def __init__(
        self,
        graph: ModelGraph,
        data_manager: DataManager,
        search_backend: SearchBackend,
        tuning_config: TuningConfig,
        executor: Optional[Executor] = None,
        plugin_registry: Optional[PluginRegistry] = None,
        audit_logger: Optional[AuditLogger] = None,
        fit_cache: Optional[FitCache] = None,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            graph: Model graph to tune and train.
            data_manager: Handles CV splitting and data routing.
            search_backend: Backend for hyperparameter optimization.
            tuning_config: Tuning configuration.
            executor: Optional executor for parallelization.
            plugin_registry: Optional registry of model plugins.
            audit_logger: Optional logger for auditing.
            fit_cache: Optional cache for model fitting results.
        """
        self.graph = graph
        self.data_manager = data_manager
        self.search_backend = search_backend
        self.tuning_config = tuning_config
        self.executor = executor
        self.plugin_registry = plugin_registry
        self.audit_logger = audit_logger
        self.fit_cache = fit_cache
        self._folds_cache: Dict[int, List] = {}
        self._scorer = None

        # Validate graph
        self.graph.validate()

    def _get_scorer(self):
        """Lazily build and cache the sklearn scorer."""
        if self._scorer is None:
            from sklearn.metrics import get_scorer
            self._scorer = get_scorer(self.tuning_config.metric)
        return self._scorer

    def fit(self, ctx: DataContext) -> FittedGraph:
        """
        Fit the entire model graph.

        Args:
            ctx: Data context with features and target.

        Returns:
            FittedGraph with all fitted nodes.
        """
        start_time = time.time()
        fitted_nodes: Dict[str, FittedNode] = {}

        if self.tuning_config.strategy == OptimizationStrategy.LAYER_BY_LAYER:
            fitted_nodes = self._fit_layer_by_layer(ctx)
        elif self.tuning_config.strategy == OptimizationStrategy.GREEDY:
            fitted_nodes = self._fit_greedy(ctx)
        elif self.tuning_config.strategy == OptimizationStrategy.NONE:
            fitted_nodes = self._fit_no_tuning(ctx)
        else:
            raise ValueError(
                f"Unsupported optimization strategy: {self.tuning_config.strategy}"
            )

        total_time = time.time() - start_time

        # Build RunMetadata
        import pandas as pd
        import sklearn_meta

        cv_cfg = self.tuning_config.cv_config
        cv_config_dict = _serialize_cv_config(cv_cfg)
        cv_random_state = cv_cfg.random_state if cv_cfg is not None else None

        total_trials = sum(
            fn.optimization_result.n_trials
            for fn in fitted_nodes.values()
            if fn.optimization_result is not None
        )

        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(ctx.X).values.tobytes()
        ).hexdigest()

        metadata = RunMetadata(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            sklearn_meta_version=sklearn_meta.__version__,
            data_shape=(len(ctx.X), len(ctx.feature_cols)),
            feature_names=list(ctx.feature_cols),
            cv_config=cv_config_dict,
            tuning_config_summary={
                "strategy": self.tuning_config.strategy.value,
                "n_trials": self.tuning_config.n_trials,
                "metric": self.tuning_config.metric,
                "greater_is_better": self.tuning_config.greater_is_better,
            },
            total_trials=total_trials,
            data_hash=data_hash,
            random_state=cv_random_state,
        )

        return FittedGraph(
            graph=self.graph,
            fitted_nodes=fitted_nodes,
            tuning_config=self.tuning_config,
            total_time=total_time,
            metadata=metadata,
        )

    def _fit_nodes(
        self,
        ctx: DataContext,
        ordered_layers: List[List[str]],
        tune: bool = True,
    ) -> Dict[str, FittedNode]:
        """Fit nodes in the given layer order.

        Args:
            ctx: Data context.
            ordered_layers: List of layers, each a list of node names.
            tune: If True, use _fit_node (with optimization); if False, use fixed params.
        """
        fitted_nodes: Dict[str, FittedNode] = {}
        oof_cache: Dict[str, np.ndarray] = {}

        for layer_idx, layer in enumerate(ordered_layers):
            if self.tuning_config.verbose >= 1 and len(ordered_layers) > 1:
                logger.info(f"Fitting layer {layer_idx + 1}/{len(ordered_layers)}: {layer}")

            layer_ctx = self._prepare_context_with_oof(ctx, layer, oof_cache)

            # Filter nodes that should run
            nodes_to_fit = []
            for node_name in layer:
                node = self.graph.get_node(node_name)
                if node.is_conditional and not node.should_run(layer_ctx):
                    continue
                nodes_to_fit.append((node_name, node))

            # Parallel node fitting within layer if executor available
            if (tune and
                self.executor is not None and
                self.executor.n_workers > 1 and
                len(nodes_to_fit) > 1):

                def fit_node_task(item: Tuple[str, ModelNode]) -> Tuple[str, FittedNode]:
                    name, node = item
                    node_ctx = layer_ctx
                    if node.is_distilled:
                        node_ctx = self._inject_soft_targets(node_ctx, node, oof_cache)
                    return (name, self._fit_node(node, node_ctx))

                results = self.executor.map(fit_node_task, nodes_to_fit)
                for name, fitted in results:
                    if self.audit_logger:
                        self.audit_logger.log_node_start(name)
                    fitted_nodes[name] = fitted
                    oof_cache[name] = fitted.oof_predictions
                    if self.audit_logger:
                        self.audit_logger.log_node_complete(name, fitted.mean_score, fitted.best_params, 0.0)
            else:
                # Sequential fallback
                for node_name, node in nodes_to_fit:
                    if self.audit_logger:
                        self.audit_logger.log_node_start(node_name)

                    node_ctx = layer_ctx
                    if node.is_distilled:
                        node_ctx = self._inject_soft_targets(node_ctx, node, oof_cache)

                    if tune:
                        fitted = self._fit_node(node, node_ctx)
                    else:
                        cv_result = self._cross_validate(node, node_ctx, node.fixed_params)
                        fitted = FittedNode(node=node, cv_result=cv_result, best_params=node.fixed_params)

                    fitted_nodes[node_name] = fitted
                    oof_cache[node_name] = fitted.oof_predictions

                    if self.audit_logger:
                        self.audit_logger.log_node_complete(node_name, fitted.mean_score, fitted.best_params, 0.0)

        return fitted_nodes

    def _fit_layer_by_layer(self, ctx: DataContext) -> Dict[str, FittedNode]:
        """Fit nodes layer by layer."""
        return self._fit_nodes(ctx, self.graph.get_layers(), tune=True)

    def _fit_greedy(self, ctx: DataContext) -> Dict[str, FittedNode]:
        """Fit nodes in topological order."""
        ordered = [[name] for name in self.graph.topological_order()]
        return self._fit_nodes(ctx, ordered, tune=True)

    def _fit_no_tuning(self, ctx: DataContext) -> Dict[str, FittedNode]:
        """Fit nodes without tuning using fixed params."""
        ordered = [[name] for name in self.graph.topological_order()]
        return self._fit_nodes(ctx, ordered, tune=False)

    def _fit_node(self, node: ModelNode, ctx: DataContext) -> FittedNode:
        """Fit a single node with hyperparameter optimization."""
        # Apply plugin modifications to search space
        search_space = node.search_space
        if self.plugin_registry and search_space:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                search_space = plugin.modify_search_space(search_space, node)

        # Apply reparameterization if enabled
        reparam_space = None
        if self.tuning_config.use_reparameterization and search_space:
            reparams = []
            if self.tuning_config.custom_reparameterizations:
                reparams.extend(self.tuning_config.custom_reparameterizations)
            # Get prebaked reparameterizations for this model
            prebaked = get_prebaked_reparameterization(
                node.estimator_class,
                search_space.parameter_names
            )
            reparams.extend(prebaked)

            if reparams:
                reparam_space = ReparameterizedSpace(search_space, reparams)
                search_space = reparam_space.build_transformed_space()

        # Optimize if search space exists
        if search_space and len(search_space) > 0:
            best_params, opt_result = self._optimize_node(
                node, ctx, search_space, reparam_space
            )
        else:
            best_params = dict(node.fixed_params)
            opt_result = None

        # Apply plugin post-tune modifications
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                best_params = plugin.post_tune(best_params, node, ctx)

        # Apply feature selection if configured
        ctx, best_params, selected_features, fs_opt_result = (
            self._apply_feature_selection(
                node, ctx, best_params, search_space, reparam_space
            )
        )
        if fs_opt_result is not None:
            opt_result = fs_opt_result

        # Apply estimator scaling if configured
        scaling_config = EstimatorScalingConfig(
            tuning_n_estimators=self.tuning_config.tuning_n_estimators,
            final_n_estimators=self.tuning_config.final_n_estimators,
            scaling_search=self.tuning_config.estimator_scaling_search,
            scaling_factors=self.tuning_config.estimator_scaling_factors,
        )
        scaler = EstimatorScaler(scaling_config, self.tuning_config.greater_is_better)

        cv_result = None
        if scaling_config.scaling_search:
            best_params, cv_result = scaler.search_scaling(
                node, ctx, best_params,
                lambda params: self._cross_validate(node, ctx, params),
            )
        elif (scaling_config.tuning_n_estimators is not None
                and scaling_config.final_n_estimators is not None):
            best_params = scaler.apply_fixed_scaling(node, best_params)

        # Final CV with best params (skip if already done in scaling search)
        if cv_result is None:
            cv_result = self._cross_validate(node, ctx, best_params)

        return FittedNode(
            node=node,
            cv_result=cv_result,
            best_params=best_params,
            optimization_result=opt_result,
            selected_features=selected_features,
        )

    def _apply_feature_selection(
        self,
        node: ModelNode,
        ctx: DataContext,
        best_params: Dict[str, Any],
        search_space,
        reparam_space,
    ) -> Tuple[DataContext, Dict[str, Any], Optional[List[str]], Optional[OptimizationResult]]:
        """Apply feature selection and optional re-tuning.

        Returns:
            Tuple of (updated context, updated best_params,
            selected_features, updated opt_result or None).
        """
        selected_features = None
        if (self.tuning_config.feature_selection and
                self.tuning_config.feature_selection.enabled):
            selector = FeatureSelector(self.tuning_config.feature_selection)
            selection_result = selector.select_for_node(node, ctx, best_params)
            selected_features = selection_result.selected_features

            # Log feature selection results
            log_feature_selection(
                logger, node.name, ctx.feature_cols, selected_features
            )

            # Filter context to selected features
            if selected_features:
                ctx = ctx.with_feature_cols(selected_features)

            # Retune if configured
            if self.tuning_config.feature_selection.retune_after_pruning:
                # Use the original (non-reparameterized) space for narrowing,
                # since best_params contains original param names.
                retune_space = (
                    reparam_space.original_space if reparam_space else search_space
                )
                if retune_space and len(retune_space) > 0:
                    # Narrow search space around previous best, biased towards less regularization
                    # (since removing features is itself a form of regularization)
                    narrowed_space = retune_space.narrow_around(
                        center=best_params,
                        factor=0.5,
                        regularization_bias=0.25,
                    )
                    logger.info(
                        f"Re-tuning '{node.name}' with {len(selected_features)} features "
                        f"(narrowed search space around previous best)"
                    )
                    best_params, opt_result = self._optimize_node(
                        node, ctx, narrowed_space, None  # Don't use reparam on narrowed space
                    )
                    return ctx, best_params, selected_features, opt_result

        return ctx, best_params, selected_features, None

    def _optimize_node(
        self,
        node: ModelNode,
        ctx: DataContext,
        search_space,
        reparam_space: Optional[ReparameterizedSpace] = None,
    ) -> Tuple[Dict[str, Any], OptimizationResult]:
        """Run hyperparameter optimization for a node."""
        # Get the original search space for type conversion
        original_space = (
            reparam_space.original_space if reparam_space else search_space
        )

        def convert_param_types(params: Dict[str, Any]) -> Dict[str, Any]:
            """Convert parameters back to their original types."""
            from sklearn_meta.search.parameter import IntParameter

            converted = {}
            for name, value in params.items():
                param = original_space.get_parameter(name)
                if param is not None and isinstance(param, IntParameter):
                    converted[name] = int(round(value))
                else:
                    converted[name] = value
            return converted

        _use_tuning_n_estimators = (
            self.tuning_config.tuning_n_estimators is not None
            and supports_param(node.estimator_class, "n_estimators")
        )

        def objective(params: Dict[str, Any]) -> float:
            # Inverse transform if reparameterized
            if reparam_space:
                params = reparam_space.inverse_transform(params)
                params = convert_param_types(params)

            # Merge with fixed params
            all_params = dict(node.fixed_params)
            all_params.update(params)

            # Override n_estimators for faster tuning if configured
            if _use_tuning_n_estimators:
                all_params["n_estimators"] = self.tuning_config.tuning_n_estimators

            # Cross-validate
            cv_result = self._cross_validate(node, ctx, all_params)

            # Return appropriate value based on optimization direction
            score = cv_result.mean_score
            if self.tuning_config.greater_is_better:
                return -score  # Minimize negative score
            return score

        opt_result = self.search_backend.optimize(
            objective=objective,
            search_space=search_space,
            n_trials=self.tuning_config.n_trials,
            timeout=self.tuning_config.timeout,
            study_name=f"{node.name}_tuning",
            early_stopping_rounds=self.tuning_config.early_stopping_rounds,
        )

        # Inverse transform best params if reparameterized
        best_params_transformed = opt_result.best_params
        if reparam_space:
            best_params_transformed = reparam_space.inverse_transform(best_params_transformed)
            best_params_transformed = convert_param_types(best_params_transformed)

        # Merge best params with fixed params
        best_params = dict(node.fixed_params)
        best_params.update(best_params_transformed)

        return best_params, opt_result

    def _cross_validate(
        self, node: ModelNode, ctx: DataContext, params: Dict[str, Any]
    ) -> CVResult:
        """Run cross-validation for a node with given parameters."""
        cache_key = id(ctx)
        if cache_key not in self._folds_cache:
            self._folds_cache[cache_key] = self.data_manager.create_folds(ctx)
        folds = self._folds_cache[cache_key]

        # Parallel fold fitting if executor available with multiple workers
        if self.executor is not None and self.executor.n_workers > 1:
            # Create a function that captures the fixed arguments
            def fit_fold_task(fold: CVFold) -> FoldResult:
                return self._fit_fold(node, ctx, fold, params)

            fold_results = self.executor.map(fit_fold_task, folds)
        else:
            # Sequential fallback
            fold_results = []
            for fold in folds:
                result = self._fit_fold(node, ctx, fold, params)
                fold_results.append(result)

        # Audit logging (after parallel section)
        if self.audit_logger:
            for result in fold_results:
                self.audit_logger.log_fold(
                    node_name=node.name,
                    fold=result.fold,
                    score=result.val_score,
                    fit_time=result.fit_time,
                    params=params,
                )

        return self.data_manager.aggregate_cv_result(node.name, fold_results, ctx)

    def _fit_fold(
        self,
        node: ModelNode,
        ctx: DataContext,
        fold: CVFold,
        params: Dict[str, Any],
    ) -> FoldResult:
        """Fit a model on a single CV fold."""
        train_ctx, val_ctx = self.data_manager.align_to_fold(ctx, fold)

        # Call plugin on_fold_start hooks
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                plugin.on_fold_start(fold.fold_idx, node, train_ctx)

        # Check cache before fitting
        cache_key = None
        if self.fit_cache:
            cache_key = self.fit_cache.cache_key(node, params, train_ctx)
            cached_model = self.fit_cache.get(cache_key)
            if cached_model is not None:
                # Use cached model
                model = cached_model
                fit_time = 0.0

                # Get predictions and score
                pred_start = time.time()
                val_predictions = node.get_output(model, val_ctx.X)
                predict_time = time.time() - pred_start

                val_score = self._calculate_score(val_ctx.y, val_predictions, node)

                return FoldResult(
                    fold=fold,
                    model=model,
                    val_predictions=val_predictions,
                    val_score=val_score,
                    fit_time=fit_time,
                    predict_time=predict_time,
                    params=params,
                )

        # Create and fit model
        model = node.create_estimator(params)

        # Inject distillation objective if applicable
        if node.is_distilled and train_ctx.soft_targets is not None:
            from sklearn_meta.core.model.distillation import build_distillation_objective
            custom_obj = build_distillation_objective(
                train_ctx.soft_targets, node.distillation_config
            )
            model.set_params(objective=custom_obj)

        # Apply plugin fit param modifications
        fit_params = dict(node.fit_params)
        if self.plugin_registry:
            fit_params = self.plugin_registry.apply_modify_fit_params(
                node.estimator_class, fit_params, train_ctx
            )

        start_time = time.time()
        model.fit(train_ctx.X, train_ctx.y, **fit_params)
        fit_time = time.time() - start_time

        # Apply plugin post-fit modifications
        if self.plugin_registry:
            model = self.plugin_registry.apply_post_fit(
                node.estimator_class, model, node, train_ctx
            )

        # Store in cache after fitting
        if self.fit_cache and cache_key:
            self.fit_cache.put(cache_key, model)

        # Get predictions and score
        pred_start = time.time()
        val_predictions = node.get_output(model, val_ctx.X)
        predict_time = time.time() - pred_start

        # Calculate score
        val_score = self._calculate_score(val_ctx.y, val_predictions, node)

        # Call plugin on_fold_end hooks
        if self.plugin_registry:
            for plugin in self.plugin_registry.get_plugins_for(node.estimator_class):
                plugin.on_fold_end(fold.fold_idx, model, val_score, node)

        return FoldResult(
            fold=fold,
            model=model,
            val_predictions=val_predictions,
            val_score=val_score,
            fit_time=fit_time,
            predict_time=predict_time,
            params=params,
        )

    def _calculate_score(self, y_true, y_pred, node=None) -> float:
        """Calculate the evaluation score."""
        scorer = self._get_scorer()

        is_clf = False
        if node is not None:
            try:
                is_clf = issubclass(node.estimator_class, ClassifierMixin)
            except TypeError:
                is_clf = False

        if is_clf:
            wrapper = _ClassifierPredictionWrapper(y_pred, np.unique(y_true))
        else:
            wrapper = _PredictionWrapper(y_pred)

        # Use scorer properly - this respects _sign, _kwargs, and response_method
        # We pass y_true as X since the wrapper ignores it
        return scorer(wrapper, y_true, y_true)

    def _inject_soft_targets(
        self,
        ctx: DataContext,
        node: ModelNode,
        oof_cache: Dict[str, np.ndarray],
    ) -> DataContext:
        """Inject teacher soft targets into context for distillation."""
        # Find teacher from DISTILL edge
        teacher_name = None
        for edge in self.graph.get_upstream(node.name):
            if edge.dep_type == DependencyType.DISTILL:
                teacher_name = edge.source
                break

        if teacher_name is None:
            raise ValueError(
                f"Node '{node.name}' has distillation_config but no DISTILL "
                f"edge in the graph"
            )

        if teacher_name not in oof_cache:
            raise ValueError(
                f"Teacher '{teacher_name}' OOF predictions not found. "
                f"Ensure the teacher is fitted before the student."
            )

        teacher_oof = oof_cache[teacher_name]

        # Extract positive-class probability
        if teacher_oof.ndim == 2 and teacher_oof.shape[1] == 2:
            soft_targets = teacher_oof[:, 1]
        elif teacher_oof.ndim == 1:
            soft_targets = teacher_oof
        else:
            raise ValueError(
                f"Teacher '{teacher_name}' OOF has unexpected shape "
                f"{teacher_oof.shape}. Expected (n,) or (n, 2)."
            )

        return ctx.with_soft_targets(soft_targets)

    def _prepare_context_with_oof(
        self,
        ctx: DataContext,
        node_names: List[str],
        oof_cache: Dict[str, np.ndarray],
    ) -> DataContext:
        """Prepare context by adding OOF predictions from upstream nodes."""
        # Collect all upstream edges for the target nodes, preserving insertion
        # order so that feature columns match the order used at predict time
        # (which iterates graph.get_upstream in list order).
        seen = set()
        all_upstream = []
        for node_name in node_names:
            for edge in self.graph.get_upstream(node_name):
                if edge.dep_type == DependencyType.DISTILL:
                    continue  # handled per-node, not as features
                if edge not in seen:
                    seen.add(edge)
                    all_upstream.append(edge)

        if not all_upstream:
            return ctx

        # Add upstream predictions to context
        predictions = {}
        for edge in all_upstream:
            if edge.source in oof_cache:
                predictions[edge.feature_name] = oof_cache[edge.source]

        if predictions:
            return ctx.augment_with_predictions(predictions, prefix="")

        return ctx

    def __repr__(self) -> str:
        return (
            f"TuningOrchestrator(graph={self.graph}, "
            f"strategy={self.tuning_config.strategy.value})"
        )
