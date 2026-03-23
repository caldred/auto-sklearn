"""Tests for GraphRunner helper behavior."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from sklearn_meta.data.view import DataView
from sklearn_meta.engine.runner import GraphRunner
from sklearn_meta.spec.dependency import DependencyEdge, DependencyType
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec


def _base_view() -> DataView:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = np.array([0.0, 1.0, 0.0, 1.0])
    return DataView.from_Xy(X, y)


def test_add_oof_overlays_preserves_parallel_edges_with_distinct_feature_names():
    view = _base_view()

    graph = GraphSpec()
    graph.add_node(NodeSpec(name="base", estimator_class=LinearRegression))
    graph.add_node(NodeSpec(name="meta", estimator_class=LinearRegression))
    graph.add_edge(
        DependencyEdge(
            source="base",
            target="meta",
            dep_type=DependencyType.PREDICTION,
            column_name="p1",
        )
    )
    graph.add_edge(
        DependencyEdge(
            source="base",
            target="meta",
            dep_type=DependencyType.PREDICTION,
            column_name="p2",
        )
    )

    overlay_view = GraphRunner._add_oof_overlays(
        view,
        ["meta"],
        {"base": np.array([0.1, 0.2, 0.3, 0.4])},
        graph,
    )
    batch = overlay_view.materialize()

    assert "p1" in batch.feature_names
    assert "p2" in batch.feature_names


def test_add_oof_overlays_raises_on_conflicting_feature_names():
    view = _base_view()

    graph = GraphSpec()
    graph.add_node(NodeSpec(name="base_a", estimator_class=LinearRegression))
    graph.add_node(NodeSpec(name="base_b", estimator_class=LinearRegression))
    graph.add_node(NodeSpec(name="meta", estimator_class=LinearRegression))
    graph.add_edge(
        DependencyEdge(
            source="base_a",
            target="meta",
            dep_type=DependencyType.PREDICTION,
            column_name="shared_feature",
        )
    )
    graph.add_edge(
        DependencyEdge(
            source="base_b",
            target="meta",
            dep_type=DependencyType.PREDICTION,
            column_name="shared_feature",
        )
    )

    with pytest.raises(ValueError, match="Conflicting overlay feature name 'shared_feature'"):
        GraphRunner._add_oof_overlays(
            view,
            ["meta"],
            {
                "base_a": np.array([0.1, 0.2, 0.3, 0.4]),
                "base_b": np.array([0.4, 0.3, 0.2, 0.1]),
            },
            graph,
        )
