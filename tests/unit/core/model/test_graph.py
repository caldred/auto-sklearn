"""Tests for ModelGraph."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn_meta.core.model.graph import CycleError, ModelGraph
from sklearn_meta.core.model.node import ModelNode
from sklearn_meta.core.model.dependency import DependencyEdge, DependencyType


class TestModelGraphBasics:
    """Tests for basic ModelGraph operations."""

    def test_empty_graph(self):
        """Verify empty graph has no nodes or edges."""
        graph = ModelGraph()

        assert len(graph) == 0
        assert len(graph.edges) == 0

    def test_add_node(self):
        """Verify node can be added to graph."""
        graph = ModelGraph()
        node = ModelNode(name="test", estimator_class=LogisticRegression)

        graph.add_node(node)

        assert len(graph) == 1
        assert "test" in graph
        assert graph.get_node("test") is node

    def test_add_duplicate_node_raises(self):
        """Verify adding duplicate node raises error."""
        graph = ModelGraph()
        node1 = ModelNode(name="test", estimator_class=LogisticRegression)
        node2 = ModelNode(name="test", estimator_class=RandomForestClassifier)

        graph.add_node(node1)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node2)

    def test_get_node_not_found_raises(self):
        """Verify getting non-existent node raises error."""
        graph = ModelGraph()

        with pytest.raises(KeyError, match="not found"):
            graph.get_node("nonexistent")

    def test_contains(self):
        """Verify __contains__ works correctly."""
        graph = ModelGraph()
        node = ModelNode(name="test", estimator_class=LogisticRegression)
        graph.add_node(node)

        assert "test" in graph
        assert "nonexistent" not in graph


class TestModelGraphEdges:
    """Tests for edge operations."""

    def test_add_edge(self):
        """Verify edge can be added between nodes."""
        graph = ModelGraph()
        node_a = ModelNode(name="A", estimator_class=LogisticRegression)
        node_b = ModelNode(name="B", estimator_class=LogisticRegression)

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_edge(DependencyEdge(source="A", target="B"))

        assert len(graph.edges) == 1

    def test_add_edge_missing_source_raises(self):
        """Verify adding edge with missing source raises error."""
        graph = ModelGraph()
        node_b = ModelNode(name="B", estimator_class=LogisticRegression)
        graph.add_node(node_b)

        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge(DependencyEdge(source="A", target="B"))

    def test_add_edge_missing_target_raises(self):
        """Verify adding edge with missing target raises error."""
        graph = ModelGraph()
        node_a = ModelNode(name="A", estimator_class=LogisticRegression)
        graph.add_node(node_a)

        with pytest.raises(ValueError, match="Target node"):
            graph.add_edge(DependencyEdge(source="A", target="B"))

    def test_get_upstream(self):
        """Verify get_upstream returns correct edges."""
        graph = ModelGraph()
        for name in ["A", "B", "C"]:
            graph.add_node(ModelNode(name=name, estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(source="A", target="C"))
        graph.add_edge(DependencyEdge(source="B", target="C"))

        upstream = graph.get_upstream("C")

        assert len(upstream) == 2
        sources = {e.source for e in upstream}
        assert sources == {"A", "B"}

    def test_get_downstream(self):
        """Verify get_downstream returns correct edges."""
        graph = ModelGraph()
        for name in ["A", "B", "C"]:
            graph.add_node(ModelNode(name=name, estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(source="A", target="B"))
        graph.add_edge(DependencyEdge(source="A", target="C"))

        downstream = graph.get_downstream("A")

        assert len(downstream) == 2
        targets = {e.target for e in downstream}
        assert targets == {"B", "C"}


class TestModelGraphTopologicalSort:
    """Tests for topological ordering."""

    def test_topological_sort_linear(self, linear_graph):
        """Verify linear A->B->C produces [A, B, C]."""
        order = linear_graph.topological_order()

        assert order == ["A", "B", "C"]

    def test_topological_sort_diamond(self, diamond_graph):
        """Verify diamond graph has dependencies before dependents."""
        order = diamond_graph.topological_order()

        # A must come before B and C
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")

        # B and C must come before D
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_topological_sort_independent_nodes(self, two_model_graph):
        """Verify independent nodes are in topological order."""
        order = two_model_graph.topological_order()

        # Both should be present, order doesn't matter for independent nodes
        assert len(order) == 2
        assert set(order) == {"rf", "lr"}

    def test_cycle_detection(self):
        """Verify cycle detection raises CycleError."""
        graph = ModelGraph()
        for name in ["A", "B", "C"]:
            graph.add_node(ModelNode(name=name, estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(source="A", target="B"))
        graph.add_edge(DependencyEdge(source="B", target="C"))

        # This should create a cycle
        with pytest.raises(CycleError, match="cycle"):
            graph.add_edge(DependencyEdge(source="C", target="A"))

    def test_cycle_detection_preserves_graph(self):
        """Verify failed cycle edge doesn't modify graph."""
        graph = ModelGraph()
        for name in ["A", "B"]:
            graph.add_node(ModelNode(name=name, estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(source="A", target="B"))

        try:
            graph.add_edge(DependencyEdge(source="B", target="A"))
        except CycleError:
            pass

        # Graph should still be valid
        assert len(graph.edges) == 1
        order = graph.topological_order()
        assert order == ["A", "B"]


class TestModelGraphLayers:
    """Tests for layer extraction."""

    def test_get_layers_single_node(self, simple_graph):
        """Verify single node is in layer 0."""
        layers = simple_graph.get_layers()

        assert len(layers) == 1
        assert layers[0] == ["rf"]

    def test_get_layers_independent_nodes(self, two_model_graph):
        """Verify independent nodes are in same layer."""
        layers = two_model_graph.get_layers()

        assert len(layers) == 1
        assert set(layers[0]) == {"rf", "lr"}

    def test_get_layers_stacking(self, stacking_graph):
        """Verify stacking graph has base and meta layers."""
        layers = stacking_graph.get_layers()

        assert len(layers) == 2

        # Layer 0: base models
        assert set(layers[0]) == {"rf_base", "lr_base"}

        # Layer 1: meta model
        assert layers[1] == ["meta"]

    def test_get_layers_linear(self, linear_graph):
        """Verify linear graph has 3 layers."""
        layers = linear_graph.get_layers()

        assert len(layers) == 3
        assert layers[0] == ["A"]
        assert layers[1] == ["B"]
        assert layers[2] == ["C"]

    def test_get_layers_diamond(self, diamond_graph):
        """Verify diamond graph layers."""
        layers = diamond_graph.get_layers()

        assert len(layers) == 3
        assert layers[0] == ["A"]
        assert set(layers[1]) == {"B", "C"}
        assert layers[2] == ["D"]


class TestModelGraphRootAndLeaf:
    """Tests for root and leaf node extraction."""

    def test_get_root_nodes(self, stacking_graph):
        """Verify root nodes have no dependencies."""
        roots = stacking_graph.get_root_nodes()

        assert set(roots) == {"rf_base", "lr_base"}

    def test_get_leaf_nodes(self, stacking_graph):
        """Verify leaf nodes have no dependents."""
        leaves = stacking_graph.get_leaf_nodes()

        assert leaves == ["meta"]

    def test_single_node_is_both_root_and_leaf(self, simple_graph):
        """Verify single node is both root and leaf."""
        roots = simple_graph.get_root_nodes()
        leaves = simple_graph.get_leaf_nodes()

        assert roots == ["rf"]
        assert leaves == ["rf"]


class TestModelGraphValidation:
    """Tests for graph validation."""

    def test_validate_valid_graph(self, stacking_graph):
        """Verify valid graph returns no errors."""
        warnings = stacking_graph.validate()

        # May have warnings but should not raise
        assert isinstance(warnings, list)

    def test_validate_detects_orphan(self):
        """Verify validation warns about orphaned nodes."""
        graph = ModelGraph()
        graph.add_node(ModelNode(name="connected", estimator_class=LogisticRegression))
        graph.add_node(ModelNode(name="orphan", estimator_class=LogisticRegression))

        warnings = graph.validate()

        orphan_warning = any("orphan" in w.lower() or "no connections" in w.lower() for w in warnings)
        assert orphan_warning


class TestModelGraphSubgraph:
    """Tests for subgraph extraction."""

    def test_subgraph_single_node(self, stacking_graph):
        """Verify subgraph with single node."""
        subgraph = stacking_graph.subgraph({"rf_base"})

        assert len(subgraph) == 1
        assert "rf_base" in subgraph
        assert len(subgraph.edges) == 0

    def test_subgraph_preserves_edges(self, stacking_graph):
        """Verify subgraph preserves edges between included nodes."""
        subgraph = stacking_graph.subgraph({"rf_base", "meta"})

        assert len(subgraph) == 2
        assert len(subgraph.edges) == 1
        assert subgraph.edges[0].source == "rf_base"
        assert subgraph.edges[0].target == "meta"

    def test_subgraph_excludes_external_edges(self, stacking_graph):
        """Verify subgraph excludes edges to nodes not in subgraph."""
        subgraph = stacking_graph.subgraph({"rf_base", "lr_base"})

        assert len(subgraph) == 2
        assert len(subgraph.edges) == 0  # No edges between rf_base and lr_base


class TestModelGraphAncestorsDescendants:
    """Tests for ancestor and descendant extraction."""

    def test_ancestors_linear(self, linear_graph):
        """Verify ancestors in linear graph."""
        ancestors_c = linear_graph.ancestors("C")

        assert ancestors_c == {"A", "B"}

    def test_ancestors_root_empty(self, linear_graph):
        """Verify root node has no ancestors."""
        ancestors_a = linear_graph.ancestors("A")

        assert ancestors_a == set()

    def test_descendants_linear(self, linear_graph):
        """Verify descendants in linear graph."""
        descendants_a = linear_graph.descendants("A")

        assert descendants_a == {"B", "C"}

    def test_descendants_leaf_empty(self, linear_graph):
        """Verify leaf node has no descendants."""
        descendants_c = linear_graph.descendants("C")

        assert descendants_c == set()

    def test_ancestors_diamond(self, diamond_graph):
        """Verify ancestors in diamond graph."""
        ancestors_d = diamond_graph.ancestors("D")

        assert ancestors_d == {"A", "B", "C"}

    def test_descendants_diamond(self, diamond_graph):
        """Verify descendants in diamond graph."""
        descendants_a = diamond_graph.descendants("A")

        assert descendants_a == {"B", "C", "D"}


class TestModelGraphProperties:
    """Tests for graph properties."""

    def test_nodes_property(self, stacking_graph):
        """Verify nodes property returns dict copy."""
        nodes = stacking_graph.nodes

        assert len(nodes) == 3
        assert "rf_base" in nodes
        assert "lr_base" in nodes
        assert "meta" in nodes

    def test_edges_property(self, stacking_graph):
        """Verify edges property returns list copy."""
        edges = stacking_graph.edges

        assert len(edges) == 2

    def test_len(self, stacking_graph):
        """Verify __len__ returns node count."""
        assert len(stacking_graph) == 3


class TestModelGraphIteration:
    """Tests for graph iteration."""

    def test_iter_topological_order(self, linear_graph):
        """Verify iteration is in topological order."""
        nodes = list(linear_graph)

        assert nodes == ["A", "B", "C"]


class TestModelGraphRepr:
    """Tests for graph representation."""

    def test_repr(self, stacking_graph):
        """Verify repr is informative."""
        repr_str = repr(stacking_graph)

        assert "ModelGraph" in repr_str
        assert "nodes=3" in repr_str
        assert "edges=2" in repr_str
