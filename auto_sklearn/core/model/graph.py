"""ModelGraph: DAG of model nodes with topological ordering."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set

from auto_sklearn.core.model.dependency import DependencyEdge, DependencyType
from auto_sklearn.core.model.node import ModelNode


class CycleError(Exception):
    """Raised when a cycle is detected in the graph."""

    pass


class ModelGraph:
    """
    Directed acyclic graph of model nodes.

    This class manages the structure of the ML pipeline, including:
    - Node storage and lookup
    - Dependency edges between nodes
    - Topological ordering for execution
    - Layer extraction for layer-wise optimization
    """

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self._nodes: Dict[str, ModelNode] = {}
        self._edges: List[DependencyEdge] = []
        self._adjacency: Dict[str, List[DependencyEdge]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[DependencyEdge]] = defaultdict(list)

    def add_node(self, node: ModelNode) -> None:
        """
        Add a model node to the graph.

        Args:
            node: The model node to add.

        Raises:
            ValueError: If a node with the same name already exists.
        """
        if node.name in self._nodes:
            raise ValueError(f"Node '{node.name}' already exists in graph")
        self._nodes[node.name] = node

    def add_edge(self, edge: DependencyEdge) -> None:
        """
        Add a dependency edge to the graph.

        Args:
            edge: The dependency edge to add.

        Raises:
            ValueError: If source or target node doesn't exist, or if edge creates a cycle.
        """
        if edge.source not in self._nodes:
            raise ValueError(f"Source node '{edge.source}' not found in graph")
        if edge.target not in self._nodes:
            raise ValueError(f"Target node '{edge.target}' not found in graph")

        # Add edge
        self._edges.append(edge)
        self._adjacency[edge.source].append(edge)
        self._reverse_adjacency[edge.target].append(edge)

        # Check for cycles
        try:
            self.topological_order()
        except CycleError:
            # Remove the edge that caused the cycle
            self._edges.pop()
            self._adjacency[edge.source].pop()
            self._reverse_adjacency[edge.target].pop()
            raise CycleError(
                f"Adding edge {edge.source} -> {edge.target} would create a cycle"
            )

    def get_node(self, name: str) -> ModelNode:
        """
        Get a node by name.

        Args:
            name: Node name.

        Returns:
            The model node.

        Raises:
            KeyError: If node doesn't exist.
        """
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not found in graph")
        return self._nodes[name]

    def get_upstream(self, node_name: str) -> List[DependencyEdge]:
        """
        Get all edges pointing to a node (its dependencies).

        Args:
            node_name: Name of the target node.

        Returns:
            List of edges where this node is the target.
        """
        return list(self._reverse_adjacency[node_name])

    def get_downstream(self, node_name: str) -> List[DependencyEdge]:
        """
        Get all edges originating from a node (nodes that depend on it).

        Args:
            node_name: Name of the source node.

        Returns:
            List of edges where this node is the source.
        """
        return list(self._adjacency[node_name])

    def topological_order(self) -> List[str]:
        """
        Get nodes in topological order (dependencies before dependents).

        Returns:
            List of node names in topological order.

        Raises:
            CycleError: If the graph contains a cycle.
        """
        # Kahn's algorithm
        in_degree = {name: 0 for name in self._nodes}
        for edge in self._edges:
            in_degree[edge.target] += 1

        # Start with nodes that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for edge in self._adjacency[node]:
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(result) != len(self._nodes):
            raise CycleError("Graph contains a cycle")

        return result

    def get_layers(self) -> List[List[str]]:
        """
        Get nodes organized into layers for layer-wise optimization.

        Layer 0 contains nodes with no dependencies.
        Layer N contains nodes whose dependencies are all in layers < N.

        Returns:
            List of layers, where each layer is a list of node names.
        """
        # Calculate the layer (longest path from any source) for each node
        node_layers: Dict[str, int] = {}

        for node_name in self.topological_order():
            upstream = self.get_upstream(node_name)
            if not upstream:
                node_layers[node_name] = 0
            else:
                max_upstream_layer = max(node_layers[e.source] for e in upstream)
                node_layers[node_name] = max_upstream_layer + 1

        # Group nodes by layer
        max_layer = max(node_layers.values()) if node_layers else -1
        layers: List[List[str]] = [[] for _ in range(max_layer + 1)]
        for node_name, layer in node_layers.items():
            layers[layer].append(node_name)

        return layers

    def get_root_nodes(self) -> List[str]:
        """Get nodes with no dependencies (layer 0)."""
        return [name for name in self._nodes if not self._reverse_adjacency[name]]

    def get_leaf_nodes(self) -> List[str]:
        """Get nodes with no dependents (final outputs)."""
        return [name for name in self._nodes if not self._adjacency[name]]

    def validate(self) -> List[str]:
        """
        Validate the graph structure.

        Returns:
            List of validation warnings (empty if valid).

        Raises:
            CycleError: If the graph contains a cycle.
            ValueError: If the graph has critical issues.
        """
        warnings = []

        # Check for cycles (will raise CycleError if found)
        self.topological_order()

        # Check for missing dependencies in search spaces
        for node_name, node in self._nodes.items():
            upstream = self.get_upstream(node_name)
            for edge in upstream:
                if edge.dep_type == DependencyType.PREDICTION:
                    # Check if the source can produce predictions
                    source_node = self._nodes[edge.source]
                    if not hasattr(source_node.estimator_class, "predict"):
                        warnings.append(
                            f"Node '{edge.source}' used as prediction dependency "
                            f"but estimator lacks 'predict' method"
                        )

        # Check for orphaned nodes (no inputs or outputs)
        for node_name in self._nodes:
            if (
                not self._adjacency[node_name]
                and not self._reverse_adjacency[node_name]
                and len(self._nodes) > 1
            ):
                warnings.append(f"Node '{node_name}' has no connections")

        return warnings

    def subgraph(self, node_names: Set[str]) -> ModelGraph:
        """
        Create a subgraph containing only the specified nodes.

        Args:
            node_names: Set of node names to include.

        Returns:
            New ModelGraph with only the specified nodes and relevant edges.
        """
        subgraph = ModelGraph()

        # Add nodes
        for name in node_names:
            if name in self._nodes:
                subgraph.add_node(self._nodes[name])

        # Add edges between included nodes
        for edge in self._edges:
            if edge.source in node_names and edge.target in node_names:
                subgraph.add_edge(edge)

        return subgraph

    def ancestors(self, node_name: str) -> Set[str]:
        """
        Get all ancestor nodes (transitive dependencies).

        Args:
            node_name: Name of the node.

        Returns:
            Set of all ancestor node names.
        """
        ancestors = set()
        to_visit = [e.source for e in self.get_upstream(node_name)]

        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(e.source for e in self.get_upstream(current))

        return ancestors

    def descendants(self, node_name: str) -> Set[str]:
        """
        Get all descendant nodes (transitive dependents).

        Args:
            node_name: Name of the node.

        Returns:
            Set of all descendant node names.
        """
        descendants = set()
        to_visit = [e.target for e in self.get_downstream(node_name)]

        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(e.target for e in self.get_downstream(current))

        return descendants

    @property
    def nodes(self) -> Dict[str, ModelNode]:
        """Dictionary of all nodes."""
        return dict(self._nodes)

    @property
    def edges(self) -> List[DependencyEdge]:
        """List of all edges."""
        return list(self._edges)

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    def __contains__(self, name: str) -> bool:
        """Check if a node exists."""
        return name in self._nodes

    def __iter__(self) -> Iterator[str]:
        """Iterate over node names in topological order."""
        return iter(self.topological_order())

    def __repr__(self) -> str:
        return f"ModelGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
