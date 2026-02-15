"""ArtifactStore: Abstract interface for artifact persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn_meta.core.tuning.orchestrator import FittedGraph, FittedNode


class ArtifactStore(ABC):
    """
    Abstract interface for storing models, parameters, and CV ensembles.

    Subclasses implement a concrete backend (filesystem, cloud, database, etc.).
    """

    @abstractmethod
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

    @abstractmethod
    def load_model(self, artifact_id: str) -> Any:
        """
        Load a saved model.

        Args:
            artifact_id: The artifact ID.

        Returns:
            The loaded model.
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List stored artifacts.

        Args:
            artifact_type: Filter by type ("model", "params", "graph").
            node_name: Filter by node name.

        Returns:
            List of artifact metadata dicts.
        """

    @abstractmethod
    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact.

        Args:
            artifact_id: The artifact ID.

        Returns:
            True if deleted, False if not found.
        """
