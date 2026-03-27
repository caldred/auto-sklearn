"""GraphRunner: Orchestrates the entire training process over a GraphSpec."""

from __future__ import annotations

import datetime
import hashlib
import logging
import time
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from sklearn_meta.artifacts.training import (
    NodeRunResult,
    RunMetadata,
    TrainingRun,
)
from sklearn_meta.data.view import DataView
from sklearn_meta.execution.training import (
    NodeTrainingJobBuilder,
    NodeTrainingResultReconstructor,
    get_trainer,
)
from sklearn_meta.engine.cv import CVEngine
from sklearn_meta.engine.search import SearchService
from sklearn_meta.engine.selection import FeatureSelectionService
from sklearn_meta.engine.strategy import OptimizationStrategy
from sklearn_meta.runtime.config import RunConfig
from sklearn_meta.runtime.services import RuntimeServices
from sklearn_meta.spec.dependency import DependencyType
from sklearn_meta.spec.node import NodeSpec

if TYPE_CHECKING:
    from sklearn_meta.spec.graph import GraphSpec

logger = logging.getLogger(__name__)


class GraphRunner:
    """Executes a GraphSpec with a RunConfig and RuntimeServices.

    The runner orchestrates the full training process:

    1. Validate graph
    2. Create CVEngine, SearchService, FeatureSelectionService
    3. Get layers from graph based on config.tuning.strategy
    4. For each layer:
       a. Add OOF overlays from upstream nodes
       b. For each node:
          - Choose trainer (Standard or Quantile)
          - Check conditional
          - Inject distillation soft targets into aux
          - trainer.fit_node() -> NodeRunResult
       c. Cache OOF predictions
    5. Build RunMetadata
    6. Assemble TrainingRun
    """

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services

    def fit(
        self,
        graph: GraphSpec,
        data: DataView,
        config: RunConfig,
    ) -> TrainingRun:
        """Fit the entire model graph.

        Args:
            graph: The model graph to fit.
            data: DataView containing features, targets, and auxiliary data.
            config: Run configuration (CV, tuning, feature selection, etc.).

        Returns:
            A TrainingRun containing all node results and metadata.

        Raises:
            ValueError: If the graph is invalid or the strategy is unsupported.
        """
        start_time = time.time()
        graph.validate()

        # Create engine components
        cv_engine = CVEngine(config.cv)
        search_service = SearchService(self.services.search_backend)
        selection_service: Optional[FeatureSelectionService] = None
        if config.feature_selection is not None and config.feature_selection.enabled:
            selection_service = FeatureSelectionService(config.feature_selection)

        # Get ordered layers based on strategy
        ordered_layers = self._get_ordered_layers(graph, config)

        # Fit nodes layer by layer
        node_results: Dict[str, NodeRunResult] = {}
        oof_cache: Dict[str, np.ndarray] = {}

        for layer_idx, layer in enumerate(ordered_layers):
            if config.verbosity >= 1 and len(ordered_layers) > 1:
                if self.services.audit_logger is not None:
                    self.services.audit_logger.log_layer_start(layer_idx, layer)

            # Add OOF overlays from upstream nodes
            layer_data = self._add_oof_overlays(data, layer, oof_cache, graph)
            dispatchable_jobs = []
            local_nodes: List[str] = []

            for node_name in layer:
                node = graph.get_node(node_name)

                if node.is_conditional and not node.should_run(layer_data):
                    continue

                if self.services.training_dispatcher is not None and self._is_dispatchable(
                    node, graph,
                ):
                    node_data = self._prepare_node_data(
                        layer_data, node, oof_cache, graph,
                    )
                    dispatchable_jobs.append(
                        NodeTrainingJobBuilder.build_for_dispatch(
                            node=node,
                            node_data=node_data,
                            config=config,
                            services=self.services,
                        )
                    )
                    if self.services.audit_logger is not None:
                        self.services.audit_logger.log_node_start(node_name)
                else:
                    local_nodes.append(node_name)

            if dispatchable_jobs and self.services.training_dispatcher is not None:
                dispatched_results = self.services.training_dispatcher.dispatch(
                    dispatchable_jobs, self.services,
                )
                for job, result in zip(dispatchable_jobs, dispatched_results):
                    reconstructed = NodeTrainingResultReconstructor.reconstruct(
                        job, result,
                    )
                    node_results[result.node_name] = reconstructed
                    oof_cache[result.node_name] = reconstructed.oof_predictions
                    if self.services.audit_logger is not None:
                        self.services.audit_logger.log_node_complete(
                            result.node_name,
                            reconstructed.mean_score,
                            reconstructed.best_params,
                            reconstructed.cv_result.total_fit_time,
                        )

            for node_name in local_nodes:
                node = graph.get_node(node_name)
                if self.services.audit_logger is not None:
                    self.services.audit_logger.log_node_start(node_name)

                result = self._fit_node_inline(
                    node=node,
                    layer_data=layer_data,
                    oof_cache=oof_cache,
                    graph=graph,
                    config=config,
                    cv_engine=cv_engine,
                    search_service=search_service,
                    selection_service=selection_service,
                )

                node_results[node_name] = result
                oof_cache[node_name] = result.oof_predictions

                if self.services.audit_logger is not None:
                    self.services.audit_logger.log_node_complete(
                        node_name,
                        result.mean_score,
                        result.best_params,
                        result.cv_result.total_fit_time,
                    )

        total_time = time.time() - start_time

        # Build metadata
        metadata = self._build_metadata(graph, data, config, node_results)

        return TrainingRun(
            graph=graph,
            config=config,
            node_results=node_results,
            metadata=metadata,
            total_time=total_time,
        )

    # ------------------------------------------------------------------
    # Strategy routing
    # ------------------------------------------------------------------

    def _get_ordered_layers(
        self,
        graph: GraphSpec,
        config: RunConfig,
    ) -> List[List[str]]:
        """Determine execution layers based on the optimization strategy.

        Args:
            graph: The model graph.
            config: Run configuration.

        Returns:
            List of layers, each layer being a list of node names.
        """
        strategy = config.tuning.strategy

        if strategy == OptimizationStrategy.LAYER_BY_LAYER:
            if self.services.training_dispatcher is not None:
                return graph.get_training_layers()
            return graph.get_layers()
        elif strategy == OptimizationStrategy.GREEDY:
            return [[name] for name in graph.topological_order()]
        elif strategy == OptimizationStrategy.NONE:
            return [[name] for name in graph.topological_order()]
        else:
            raise ValueError(f"Unsupported optimization strategy: {strategy}")

    def _fit_node_inline(
        self,
        node: NodeSpec,
        layer_data: DataView,
        oof_cache: Dict[str, np.ndarray],
        graph: GraphSpec,
        config: RunConfig,
        cv_engine: CVEngine,
        search_service: SearchService,
        selection_service: Optional[FeatureSelectionService],
    ) -> NodeRunResult:
        node_data = self._prepare_node_data(layer_data, node, oof_cache, graph)
        trainer = get_trainer(node)
        return trainer.fit_node(
            node,
            node_data,
            config,
            self.services,
            cv_engine,
            search_service,
            selection_service,
        )

    @staticmethod
    def _prepare_node_data(
        layer_data: DataView,
        node: NodeSpec,
        oof_cache: Dict[str, np.ndarray],
        graph: GraphSpec,
    ) -> DataView:
        node_data = layer_data
        if node.feature_cols is not None:
            node_data = node_data.select_features(node.feature_cols)

        if node.is_distilled:
            node_data = GraphRunner._inject_soft_targets(
                node_data, node, oof_cache, graph,
            )

        return GraphRunner._inject_conditional_samples(node_data, node, graph)

    @staticmethod
    def _is_dispatchable(node: NodeSpec, graph: GraphSpec) -> bool:
        if node.is_conditional:
            return False
        if node.is_distilled:
            return False
        return NodeTrainingJobBuilder.is_dispatchable(node, graph)

    # ------------------------------------------------------------------
    # OOF overlay injection
    # ------------------------------------------------------------------

    @staticmethod
    def _add_oof_overlays(
        data: DataView,
        node_names: List[str],
        oof_cache: Dict[str, np.ndarray],
        graph: GraphSpec,
    ) -> DataView:
        """Add OOF predictions from upstream nodes as overlays.

        Skips DISTILL and CONDITIONAL_SAMPLE edges (handled per-node).

        Args:
            data: The base DataView.
            node_names: Names of nodes in the current layer.
            oof_cache: Cached OOF predictions keyed by node name.
            graph: The model graph.

        Returns:
            DataView with upstream OOF predictions added as overlay columns.
        """
        predictions: Dict[str, np.ndarray] = {}
        overlay_sources: Dict[str, str] = {}
        overlay_types: Dict[str, DependencyType] = {}
        overlay_edge_refs: Dict[str, str] = {}

        for node_name in node_names:
            for edge in graph.get_upstream(node_name):
                if edge.dep_type == DependencyType.DISTILL:
                    continue
                if edge.dep_type == DependencyType.CONDITIONAL_SAMPLE:
                    continue  # handled per-node
                if edge.source not in oof_cache:
                    continue

                feature_name = edge.feature_name
                edge_ref = (
                    f"{edge.source}->{edge.target}"
                    f" ({edge.dep_type.value}, feature='{feature_name}')"
                )

                if feature_name not in predictions:
                    predictions[feature_name] = oof_cache[edge.source]
                    overlay_sources[feature_name] = edge.source
                    overlay_types[feature_name] = edge.dep_type
                    overlay_edge_refs[feature_name] = edge_ref
                    continue

                same_source = overlay_sources[feature_name] == edge.source
                same_type = overlay_types[feature_name] == edge.dep_type

                if same_source and same_type:
                    continue

                raise ValueError(
                    "Conflicting overlay feature name "
                    f"'{feature_name}' from edges "
                    f"{overlay_edge_refs[feature_name]} and {edge_ref}"
                )

        if predictions:
            return data.with_overlays(predictions)
        return data

    # ------------------------------------------------------------------
    # Distillation soft-target injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_soft_targets(
        data: DataView,
        node: NodeSpec,
        oof_cache: Dict[str, np.ndarray],
        graph: GraphSpec,
    ) -> DataView:
        """Inject teacher soft targets into aux for distillation.

        Finds the DISTILL edge pointing to *node*, looks up the teacher's
        OOF predictions, and injects them as the ``soft_targets`` aux channel.

        Args:
            data: The current DataView for this node.
            node: The distilled student node.
            oof_cache: Cached OOF predictions keyed by node name.
            graph: The model graph.

        Returns:
            DataView with ``soft_targets`` aux channel added.

        Raises:
            ValueError: If no DISTILL edge or teacher OOF is missing.
        """
        teacher_name: Optional[str] = None
        for edge in graph.get_upstream(node.name):
            if edge.dep_type == DependencyType.DISTILL:
                teacher_name = edge.source
                break

        if teacher_name is None:
            raise ValueError(
                f"Node '{node.name}' has distillation_config but no DISTILL edge"
            )

        if teacher_name not in oof_cache:
            raise ValueError(
                f"Teacher '{teacher_name}' OOF predictions not found"
            )

        teacher_oof = oof_cache[teacher_name]

        # For binary classification, extract the positive-class column.
        # For multiclass, pass the full 2-D probability matrix through.
        if teacher_oof.ndim == 1:
            soft_targets = teacher_oof
        elif teacher_oof.ndim == 2 and teacher_oof.shape[1] == 2:
            soft_targets = teacher_oof[:, 1]
        else:
            # teacher_oof.ndim == 2, shape[1] > 2 → multiclass
            soft_targets = teacher_oof

        return data.with_aux("soft_targets", soft_targets)

    # ------------------------------------------------------------------
    # Conditional sample injection (joint quantile training)
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_conditional_samples(
        data: DataView,
        node: NodeSpec,
        graph: GraphSpec,
    ) -> DataView:
        """Inject actual target values as features for CONDITIONAL_SAMPLE edges.

        During training, the actual observed values of the upstream property
        are used as conditioning features rather than predictions. This
        supports the joint quantile regression pattern.

        Args:
            data: The current DataView for this node.
            node: The node to inject conditional samples for.
            graph: The model graph.

        Returns:
            DataView with conditional sample overlays added.
        """
        for edge in graph.get_upstream(node.name):
            if edge.dep_type == DependencyType.CONDITIONAL_SAMPLE:
                config = edge.conditional_config
                if config is not None and config.use_actual_during_training:
                    # The target for the upstream property should be bound
                    # on the data view
                    target_ref = data.targets.get(config.property_name)
                    if target_ref is not None:
                        resolved = data.resolve_channel(target_ref)
                        # Add as full-length overlay — need dataset-length array
                        if data.row_sel is None:
                            data = data.with_overlay(
                                edge.feature_name, resolved,
                            )
                        else:
                            # Build a full-length array then add as overlay
                            full = np.zeros(data.dataset.n_rows)
                            full[data.row_sel] = resolved
                            data = data.with_overlay(
                                edge.feature_name, full,
                            )
        return data

    # ------------------------------------------------------------------
    # Metadata assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(
        graph: GraphSpec,
        data: DataView,
        config: RunConfig,
        node_results: Dict[str, NodeRunResult],
    ) -> RunMetadata:
        """Build RunMetadata from the completed training run.

        Args:
            graph: The model graph.
            data: The original DataView.
            config: Run configuration.
            node_results: Results from each fitted node.

        Returns:
            RunMetadata summarising the training run.
        """
        import sklearn_meta

        batch = data.materialize()
        total_trials = sum(
            r.optimization_result.n_trials
            for r in node_results.values()
            if r.optimization_result is not None
        )
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(batch.X).values.tobytes()
        ).hexdigest()

        return RunMetadata(
            timestamp=datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat(),
            sklearn_meta_version=sklearn_meta.__version__,
            data_shape=(batch.n_samples, len(batch.feature_names)),
            feature_names=batch.feature_names,
            cv_config=config.cv.to_dict(),
            tuning_config_summary={
                "strategy": config.tuning.strategy.value,
                "n_trials": config.tuning.n_trials,
                "metric": config.tuning.metric,
                "greater_is_better": config.tuning.greater_is_better,
            },
            total_trials=total_trials,
            data_hash=data_hash,
            random_state=config.cv.random_state,
        )
