"""InferenceCompiler: compile a TrainingRun into lightweight inference graphs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from sklearn_meta.artifacts.inference import (
    InferenceGraph,
    JointQuantileInferenceGraph,
    QuantileFittedNode,
)
from sklearn_meta.artifacts.training import (
    QuantileNodeRunResult,
    TrainingRun,
)

if TYPE_CHECKING:
    pass


class InferenceCompiler:
    @staticmethod
    def compile(run: TrainingRun) -> InferenceGraph:
        from sklearn_meta.spec.quantile import JointQuantileGraphSpec

        if isinstance(run.graph, JointQuantileGraphSpec):
            return InferenceCompiler.compile_quantile(run)

        node_models: Dict[str, List[Any]] = {}
        selected_features: Dict[str, Optional[List[str]]] = {}
        node_params: Dict[str, Dict[str, Any]] = {}
        for name, result in run.node_results.items():
            node_models[name] = result.models
            selected_features[name] = result.selected_features
            node_params[name] = result.best_params
        return InferenceGraph(
            graph=run.graph,
            node_models=node_models,
            selected_features=selected_features,
            node_params=node_params,
        )

    @staticmethod
    def compile_quantile(run: TrainingRun) -> JointQuantileInferenceGraph:
        """Build a JointQuantileInferenceGraph from QuantileNodeRunResults in run.node_results."""
        from sklearn_meta.spec.quantile import JointQuantileGraphSpec

        graph = run.graph
        if not isinstance(graph, JointQuantileGraphSpec):
            raise TypeError(
                "compile_quantile requires a TrainingRun whose graph is a JointQuantileGraphSpec"
            )

        sampler = graph.create_quantile_sampler()

        fitted_nodes: Dict[str, QuantileFittedNode] = {}
        for prop_name in graph.property_order:
            node_name = f"quantile_{prop_name}"
            result = run.node_results.get(node_name)
            if result is None:
                raise ValueError(f"Missing result for quantile node '{node_name}'")
            if not isinstance(result, QuantileNodeRunResult):
                raise TypeError(
                    f"Expected QuantileNodeRunResult for '{node_name}', "
                    f"got {type(result).__name__}"
                )

            q_node = graph.get_quantile_node(prop_name)

            fitted_nodes[prop_name] = QuantileFittedNode(
                quantile_models=result.quantile_models,
                quantile_levels=q_node.quantile_levels,
                selected_features=result.selected_features,
            )

        return JointQuantileInferenceGraph(
            graph=graph,
            fitted_nodes=fitted_nodes,
            quantile_sampler=sampler,
        )
