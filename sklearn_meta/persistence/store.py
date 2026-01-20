"""ArtifactStore: Storage for models, parameters, and CV ensembles."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn_meta.core.tuning.orchestrator import FittedGraph, FittedNode


@dataclass
class ArtifactMetadata:
    """Metadata for a stored artifact."""

    artifact_id: str
    artifact_type: str
    created_at: str
    node_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


class ArtifactStore:
    """
    Storage for models, parameters, and CV ensembles.

    This class provides a simple file-based storage system for ML artifacts.
    It supports:
    - Saving and loading fitted models
    - Storing hyperparameters and metrics
    - Versioning and metadata tracking
    """

    def __init__(self, base_path: str = ".sklearn_meta_artifacts") -> None:
        """
        Initialize the artifact store.

        Args:
            base_path: Base directory for storing artifacts.
        """
        self.base_path = Path(base_path)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        (self.base_path / "models").mkdir(parents=True, exist_ok=True)
        (self.base_path / "params").mkdir(parents=True, exist_ok=True)
        (self.base_path / "graphs").mkdir(parents=True, exist_ok=True)
        (self.base_path / "metadata").mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        node_name: str,
        fold_idx: int = 0,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save a fitted model.

        Args:
            model: The fitted model to save.
            node_name: Name of the model node.
            fold_idx: CV fold index.
            params: Hyperparameters used.
            metrics: Performance metrics.
            tags: Additional metadata tags.

        Returns:
            Artifact ID.
        """
        artifact_id = self._generate_id(node_name, fold_idx, params)

        # Save model
        model_path = self.base_path / "models" / f"{artifact_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type="model",
            created_at=datetime.now().isoformat(),
            node_name=node_name,
            params=params or {},
            metrics=metrics or {},
            tags=tags or {},
        )
        self._save_metadata(metadata)

        return artifact_id

    def load_model(self, artifact_id: str) -> Any:
        """
        Load a saved model.

        Args:
            artifact_id: The artifact ID.

        Returns:
            The loaded model.
        """
        model_path = self.base_path / "models" / f"{artifact_id}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {artifact_id}")

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def save_fitted_node(
        self,
        fitted_node: FittedNode,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Save all models from a fitted node.

        Args:
            fitted_node: The fitted node to save.
            tags: Additional metadata tags.

        Returns:
            List of artifact IDs for each fold model.
        """
        artifact_ids = []
        for fold_idx, (fold_result, model) in enumerate(
            zip(fitted_node.cv_result.fold_results, fitted_node.models)
        ):
            artifact_id = self.save_model(
                model=model,
                node_name=fitted_node.node.name,
                fold_idx=fold_idx,
                params=fitted_node.best_params,
                metrics={"val_score": fold_result.val_score},
                tags=tags,
            )
            artifact_ids.append(artifact_id)
        return artifact_ids

    def save_fitted_graph(
        self,
        fitted_graph: FittedGraph,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save an entire fitted graph.

        Args:
            fitted_graph: The fitted graph to save.
            name: Name for this graph version.
            tags: Additional metadata tags.

        Returns:
            Graph artifact ID.
        """
        graph_id = f"graph_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save all nodes
        node_artifacts = {}
        for node_name, fitted_node in fitted_graph.fitted_nodes.items():
            node_artifacts[node_name] = self.save_fitted_node(fitted_node, tags)

        # Save graph structure and references
        graph_data = {
            "graph_id": graph_id,
            "created_at": datetime.now().isoformat(),
            "node_artifacts": node_artifacts,
            "total_time": fitted_graph.total_time,
            "tuning_config": {
                "strategy": fitted_graph.tuning_config.strategy.value,
                "n_trials": fitted_graph.tuning_config.n_trials,
                "metric": fitted_graph.tuning_config.metric,
            },
            "tags": tags or {},
        }

        graph_path = self.base_path / "graphs" / f"{graph_id}.json"
        with open(graph_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        return graph_id

    def save_params(
        self,
        params: Dict[str, Any],
        node_name: str,
        description: str = "",
    ) -> str:
        """
        Save hyperparameters.

        Args:
            params: Parameters to save.
            node_name: Name of the model node.
            description: Optional description.

        Returns:
            Artifact ID.
        """
        artifact_id = self._generate_id(node_name, 0, params)

        params_data = {
            "artifact_id": artifact_id,
            "node_name": node_name,
            "params": params,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }

        params_path = self.base_path / "params" / f"{artifact_id}.json"
        with open(params_path, "w") as f:
            json.dump(params_data, f, indent=2)

        return artifact_id

    def load_params(self, artifact_id: str) -> Dict[str, Any]:
        """
        Load saved parameters.

        Args:
            artifact_id: The artifact ID.

        Returns:
            Dictionary of parameters.
        """
        params_path = self.base_path / "params" / f"{artifact_id}.json"
        if not params_path.exists():
            raise FileNotFoundError(f"Params not found: {artifact_id}")

        with open(params_path, "r") as f:
            data = json.load(f)
        return data["params"]

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> List[ArtifactMetadata]:
        """
        List stored artifacts.

        Args:
            artifact_type: Filter by type ("model", "params", "graph").
            node_name: Filter by node name.

        Returns:
            List of artifact metadata.
        """
        artifacts = []

        metadata_dir = self.base_path / "metadata"
        if metadata_dir.exists():
            for meta_file in metadata_dir.glob("*.json"):
                with open(meta_file, "r") as f:
                    data = json.load(f)
                    metadata = ArtifactMetadata(**data)

                    if artifact_type and metadata.artifact_type != artifact_type:
                        continue
                    if node_name and metadata.node_name != node_name:
                        continue

                    artifacts.append(metadata)

        return artifacts

    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact.

        Args:
            artifact_id: The artifact ID.

        Returns:
            True if deleted, False if not found.
        """
        deleted = False

        # Try to delete from each location
        for subdir in ["models", "params"]:
            for ext in [".pkl", ".json"]:
                path = self.base_path / subdir / f"{artifact_id}{ext}"
                if path.exists():
                    path.unlink()
                    deleted = True

        # Delete metadata
        meta_path = self.base_path / "metadata" / f"{artifact_id}.json"
        if meta_path.exists():
            meta_path.unlink()

        return deleted

    def _generate_id(
        self,
        node_name: str,
        fold_idx: int,
        params: Optional[Dict[str, Any]],
    ) -> str:
        """Generate a unique artifact ID."""
        content = f"{node_name}_{fold_idx}_{json.dumps(params or {}, sort_keys=True)}"
        hash_str = hashlib.md5(content.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{node_name}_{fold_idx}_{timestamp}_{hash_str}"

    def _save_metadata(self, metadata: ArtifactMetadata) -> None:
        """Save artifact metadata."""
        meta_path = self.base_path / "metadata" / f"{metadata.artifact_id}.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "artifact_id": metadata.artifact_id,
                    "artifact_type": metadata.artifact_type,
                    "created_at": metadata.created_at,
                    "node_name": metadata.node_name,
                    "params": metadata.params,
                    "metrics": metadata.metrics,
                    "tags": metadata.tags,
                },
                f,
                indent=2,
            )

    def __repr__(self) -> str:
        return f"ArtifactStore(base_path={self.base_path})"
