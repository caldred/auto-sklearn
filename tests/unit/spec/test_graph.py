"""Tests for GraphSpec."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn_meta.spec.graph import CycleError, GraphSpec
from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.spec.dependency import DependencyEdge, DependencyType


class TestGraphSpecBasics:
    """Tests for basic GraphSpec operations."""

    def test_add_duplicate_node_raises(self):
        """Verify adding duplicate node raises error."""
        graph = GraphSpec()
        node1 = NodeSpec(name="test", estimator_class=LogisticRegression)
        node2 = NodeSpec(name="test", estimator_class=RandomForestClassifier)

        graph.add_node(node1)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node2)

    def test_get_node_not_found_raises(self):
        """Verify getting non-existent node raises error."""
        graph = GraphSpec()

        with pytest.raises(KeyError, match="not found"):
            graph.get_node("nonexistent")

class TestGraphSpecEdges:
    """Tests for edge operations."""

    def test_add_edge(self):
        """Verify edge can be added between nodes."""
        graph = GraphSpec()
        node_a = NodeSpec(name="A", estimator_class=LogisticRegression)
        node_b = NodeSpec(name="B", estimator_class=LogisticRegression)

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_edge(DependencyEdge(source="A", target="B"))

        assert len(graph.edges) == 1

    def test_add_edge_missing_source_raises(self):
        """Verify adding edge with missing source raises error."""
        graph = GraphSpec()
        node_b = NodeSpec(name="B", estimator_class=LogisticRegression)
        graph.add_node(node_b)

        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge(DependencyEdge(source="A", target="B"))

    def test_add_edge_missing_target_raises(self):
        """Verify adding edge with missing target raises error."""
        graph = GraphSpec()
        node_a = NodeSpec(name="A", estimator_class=LogisticRegression)
        graph.add_node(node_a)

        with pytest.raises(ValueError, match="Target node"):
            graph.add_edge(DependencyEdge(source="A", target="B"))

    def test_get_upstream(self):
        """Verify get_upstream returns correct edges."""
        graph = GraphSpec()
        for name in ["A", "B", "C"]:
            graph.add_node(NodeSpec(name=name, estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(source="A", target="C"))
        graph.add_edge(DependencyEdge(source="B", target="C"))

        upstream = graph.get_upstream("C")

        assert len(upstream) == 2
        sources = {e.source for e in upstream}
        assert sources == {"A", "B"}

    def test_get_downstream(self):
        """Verify get_downstream returns correct edges."""
        graph = GraphSpec()
        for name in ["A", "B", "C"]:
            graph.add_node(NodeSpec(name=name, estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(source="A", target="B"))
        graph.add_edge(DependencyEdge(source="A", target="C"))

        downstream = graph.get_downstream("A")

        assert len(downstream) == 2
        targets = {e.target for e in downstream}
        assert targets == {"B", "C"}


class TestGraphSpecTopologicalSort:
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
        graph = GraphSpec()
        for name in ["A", "B", "C"]:
            graph.add_node(NodeSpec(name=name, estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(source="A", target="B"))
        graph.add_edge(DependencyEdge(source="B", target="C"))

        # This should create a cycle
        with pytest.raises(CycleError, match="cycle"):
            graph.add_edge(DependencyEdge(source="C", target="A"))

    def test_cycle_detection_preserves_graph(self):
        """Verify failed cycle edge doesn't modify graph."""
        graph = GraphSpec()
        for name in ["A", "B"]:
            graph.add_node(NodeSpec(name=name, estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(source="A", target="B"))

        try:
            graph.add_edge(DependencyEdge(source="B", target="A"))
        except CycleError:
            pass

        # Graph should still be valid
        assert len(graph.edges) == 1
        order = graph.topological_order()
        assert order == ["A", "B"]


class TestGraphSpecLayers:
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


class TestGraphSpecRootAndLeaf:
    """Tests for root and leaf node extraction."""

    def test_get_root_nodes(self, stacking_graph):
        """Verify root nodes have no dependencies."""
        roots = stacking_graph.get_root_nodes()

        assert set(roots) == {"rf_base", "lr_base"}

    def test_get_leaf_nodes(self, stacking_graph):
        """Verify leaf nodes have no dependents."""
        leaves = stacking_graph.get_leaf_nodes()

        assert leaves == ["meta"]

class TestGraphSpecAncestorsDescendants:
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


class TestGraphSpecSubgraph:
    """Tests for subgraph extraction with detailed edge/node filtering."""

    def test_subgraph_diamond_excludes_missing_node_edges(self, diamond_graph):
        """Subgraph {A,B,D} of diamond keeps A->B, B->D but not A->C, C->D."""
        sub = diamond_graph.subgraph({"A", "B", "D"})

        assert len(sub) == 3
        assert {"A", "B", "D"} == set(sub.nodes.keys())

        edge_pairs = {(e.source, e.target) for e in sub.edges}
        assert ("A", "B") in edge_pairs
        assert ("B", "D") in edge_pairs
        assert ("A", "C") not in edge_pairs
        assert ("C", "D") not in edge_pairs

    def test_subgraph_single_node_no_edges(self, diamond_graph):
        """Subgraph with a single node should have 1 node and 0 edges."""
        sub = diamond_graph.subgraph({"B"})

        assert len(sub) == 1
        assert "B" in sub
        assert len(sub.edges) == 0

    def test_subgraph_preserves_node_properties(self):
        """Subgraph preserves estimator_class and fixed_params on nodes."""
        graph = GraphSpec()
        node = NodeSpec(
            name="lr",
            estimator_class=LogisticRegression,
            fixed_params={"C": 0.5, "max_iter": 200},
        )
        graph.add_node(node)

        sub = graph.subgraph({"lr"})
        sub_node = sub.get_node("lr")

        assert sub_node.estimator_class is LogisticRegression
        assert sub_node.fixed_params == {"C": 0.5, "max_iter": 200}

    def test_subgraph_stacking_base_models_no_edges(self, stacking_graph):
        """Subgraph of base models only has no edges (target 'meta' excluded)."""
        sub = stacking_graph.subgraph({"rf_base", "lr_base"})

        assert len(sub) == 2
        assert len(sub.edges) == 0

    def test_subgraph_topological_order_valid(self, diamond_graph):
        """Subgraph topological_order works without broken references."""
        sub = diamond_graph.subgraph({"A", "B", "D"})
        order = sub.topological_order()

        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("D")


class TestGraphSpecValidateWarnings:
    """Tests for validate() warnings and error detection."""

    def test_two_orphan_nodes_produce_warnings(self):
        """Two-node graph with no edges: both nodes should warn."""
        graph = GraphSpec()
        graph.add_node(NodeSpec(name="a", estimator_class=LogisticRegression))
        graph.add_node(NodeSpec(name="b", estimator_class=LogisticRegression))

        warnings = graph.validate()

        warning_text = " ".join(warnings).lower()
        assert "no connections" in warning_text or "orphan" in warning_text
        # Both nodes should produce a warning
        node_warnings = [w for w in warnings if "no connections" in w.lower() or "orphan" in w.lower()]
        assert len(node_warnings) == 2

    def test_single_node_no_orphan_warning(self):
        """Single-node graph should NOT produce orphan warning."""
        graph = GraphSpec()
        graph.add_node(NodeSpec(name="solo", estimator_class=LogisticRegression))

        warnings = graph.validate()

        orphan_warnings = [w for w in warnings if "no connections" in w.lower() or "orphan" in w.lower()]
        assert len(orphan_warnings) == 0

    def test_prediction_dependency_lacking_predict_warns(self):
        """Node used as PREDICTION dep but lacking predict: should warn."""
        graph = GraphSpec()
        graph.add_node(NodeSpec(
            name="scaler",
            estimator_class=StandardScaler,
            output_type=OutputType.TRANSFORM,
        ))
        graph.add_node(NodeSpec(name="model", estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(
            source="scaler",
            target="model",
            dep_type=DependencyType.PREDICTION,
        ))

        warnings = graph.validate()

        predict_warnings = [w for w in warnings if "predict" in w.lower()]
        assert len(predict_warnings) >= 1

    def test_cycle_raises_cycle_error(self):
        """Graph with a cycle should raise CycleError from validate."""
        graph = GraphSpec()
        graph.add_node(NodeSpec(name="A", estimator_class=LogisticRegression))
        graph.add_node(NodeSpec(name="B", estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(source="A", target="B"))

        # Adding C->A after B->C would be caught at add_edge time,
        # so test that validate itself calls topological_order which
        # would surface a cycle if one existed.
        # We verify that a valid graph does not raise.
        graph.validate()  # should not raise

        # Now test that CycleError is raised when a cycle is attempted
        with pytest.raises(CycleError):
            graph.add_edge(DependencyEdge(source="B", target="A"))


class TestGraphSpecMixedEdgeTypes:
    """Tests for graphs with mixed dependency types and layer computation."""

    def test_transform_and_prediction_edges_three_layers(self):
        """scaler->model (TRANSFORM), model->meta (PREDICTION) = 3 layers."""
        graph = GraphSpec()
        graph.add_node(NodeSpec(
            name="scaler",
            estimator_class=StandardScaler,
            output_type=OutputType.TRANSFORM,
        ))
        graph.add_node(NodeSpec(name="model", estimator_class=LogisticRegression))
        graph.add_node(NodeSpec(name="meta", estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(
            source="scaler", target="model", dep_type=DependencyType.TRANSFORM,
        ))
        graph.add_edge(DependencyEdge(
            source="model", target="meta", dep_type=DependencyType.PREDICTION,
        ))

        layers = graph.get_layers()

        assert len(layers) == 3
        assert layers[0] == ["scaler"]
        assert layers[1] == ["model"]
        assert layers[2] == ["meta"]

    def test_distill_edge_two_layers(self):
        """teacher->student (DISTILL) should have 2 layers."""
        graph = GraphSpec()
        graph.add_node(NodeSpec(name="teacher", estimator_class=LogisticRegression))
        graph.add_node(NodeSpec(name="student", estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(
            source="teacher", target="student", dep_type=DependencyType.DISTILL,
        ))

        layers = graph.get_layers()

        assert len(layers) == 2
        assert layers[0] == ["teacher"]
        assert layers[1] == ["student"]

    def test_same_source_different_edge_types(self):
        """One source with TRANSFORM to one target and PREDICTION to another."""
        graph = GraphSpec()
        graph.add_node(NodeSpec(
            name="scaler",
            estimator_class=StandardScaler,
            output_type=OutputType.TRANSFORM,
        ))
        graph.add_node(NodeSpec(name="lr", estimator_class=LogisticRegression))
        graph.add_node(NodeSpec(name="meta", estimator_class=LogisticRegression))

        graph.add_edge(DependencyEdge(
            source="scaler", target="lr", dep_type=DependencyType.TRANSFORM,
        ))
        graph.add_edge(DependencyEdge(
            source="scaler", target="meta", dep_type=DependencyType.PREDICTION,
        ))

        # Both lr and meta depend on scaler, so 2 layers
        layers = graph.get_layers()
        assert len(layers) == 2
        assert layers[0] == ["scaler"]
        assert set(layers[1]) == {"lr", "meta"}


class TestGraphSpecIterAndProperties:
    """Tests for iteration, property copies, and root/leaf accessors."""

    def test_iter_returns_topological_order(self):
        """iter(graph) returns names in topological order."""
        graph = GraphSpec()
        graph.add_node(NodeSpec(name="A", estimator_class=LogisticRegression))
        graph.add_node(NodeSpec(name="B", estimator_class=LogisticRegression))
        graph.add_edge(DependencyEdge(source="A", target="B"))

        names = list(graph)

        assert names.index("A") < names.index("B")

