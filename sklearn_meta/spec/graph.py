"""GraphSpec: DAG of model nodes with topological ordering."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable, Dict, Iterator, List, Optional, Set

from sklearn_meta.spec.dependency import DependencyEdge, DependencyType
from sklearn_meta.spec.node import NodeSpec


class CycleError(Exception):
    """Raised when a cycle is detected in the graph."""

    pass


class GraphSpec:
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
        self._nodes: Dict[str, NodeSpec] = {}
        self._edges: List[DependencyEdge] = []
        self._adjacency: Dict[str, List[DependencyEdge]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[DependencyEdge]] = defaultdict(list)

    def add_node(self, node: NodeSpec) -> None:
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

    def _can_reach(self, source: str, target: str) -> bool:
        """Check if source can reach target via existing forward edges (DFS)."""
        visited: Set[str] = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            for edge in self._adjacency.get(node, []):
                stack.append(edge.target)
        return False

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

        # Check for cycles before adding: would target -> source path exist?
        if self._can_reach(edge.target, edge.source):
            raise CycleError(
                f"Adding edge {edge.source} -> {edge.target} would create a cycle"
            )

        # Safe to add edge
        self._edges.append(edge)
        self._adjacency[edge.source].append(edge)
        self._reverse_adjacency[edge.target].append(edge)

    def get_node(self, name: str) -> NodeSpec:
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
        queue = deque(name for name, degree in in_degree.items() if degree == 0)
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for edge in self._adjacency[node]:
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(result) != len(self._nodes):
            raise CycleError("Graph contains a cycle")

        return result

    def _compute_layers(
        self,
        edge_filter: Optional[Callable[[DependencyEdge], bool]] = None,
    ) -> List[List[str]]:
        """Compute layers with an optional edge filter.

        Args:
            edge_filter: When provided, only edges for which the callable
                returns ``True`` are considered when assigning layers.
        """
        node_layers: Dict[str, int] = {}

        for node_name in self.topological_order():
            upstream = self.get_upstream(node_name)
            if edge_filter is not None:
                upstream = [e for e in upstream if edge_filter(e)]
            if not upstream:
                node_layers[node_name] = 0
            else:
                max_upstream_layer = max(node_layers[e.source] for e in upstream)
                node_layers[node_name] = max_upstream_layer + 1

        max_layer = max(node_layers.values()) if node_layers else -1
        layers: List[List[str]] = [[] for _ in range(max_layer + 1)]
        for node_name, layer in node_layers.items():
            layers[layer].append(node_name)

        return layers

    def get_layers(self) -> List[List[str]]:
        """
        Get nodes organized into layers for layer-wise optimization.

        Layer 0 contains nodes with no dependencies.
        Layer N contains nodes whose dependencies are all in layers < N.

        Returns:
            List of layers, where each layer is a list of node names.
        """
        return self._compute_layers()

    def get_training_layers(self) -> List[List[str]]:
        """
        Get nodes organized into fit-time layers.

        Unlike :meth:`get_layers`, this ignores edges that do not block
        training (for example, conditional-sample edges that use observed
        values during training).
        """
        return self._compute_layers(edge_filter=lambda e: e.blocks_training())

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

    def subgraph(self, node_names: Set[str]) -> GraphSpec:
        """
        Create a subgraph containing only the specified nodes.

        Args:
            node_names: Set of node names to include.

        Returns:
            New GraphSpec with only the specified nodes and relevant edges.
        """
        sub = GraphSpec()

        # Add nodes
        for name in node_names:
            if name in self._nodes:
                sub.add_node(self._nodes[name])

        # Add edges between included nodes
        for edge in self._edges:
            if edge.source in node_names and edge.target in node_names:
                sub.add_edge(edge)

        return sub

    def ancestors(self, node_name: str) -> Set[str]:
        """
        Get all ancestor nodes (transitive dependencies).

        Args:
            node_name: Name of the node.

        Returns:
            Set of all ancestor node names.
        """
        result: Set[str] = set()
        to_visit = [e.source for e in self.get_upstream(node_name)]

        while to_visit:
            current = to_visit.pop()
            if current not in result:
                result.add(current)
                to_visit.extend(e.source for e in self.get_upstream(current))

        return result

    def descendants(self, node_name: str) -> Set[str]:
        """
        Get all descendant nodes (transitive dependents).

        Args:
            node_name: Name of the node.

        Returns:
            Set of all descendant node names.
        """
        result: Set[str] = set()
        to_visit = [e.target for e in self.get_downstream(node_name)]

        while to_visit:
            current = to_visit.pop()
            if current not in result:
                result.add(current)
                to_visit.extend(e.target for e in self.get_downstream(current))

        return result

    @property
    def nodes(self) -> Dict[str, NodeSpec]:
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
        return f"GraphSpec(nodes={len(self._nodes)}, edges={len(self._edges)})"
