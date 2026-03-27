"""Inference artifacts: lightweight graphs for prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from sklearn_meta.engine.estimator_factory import get_output
from sklearn_meta.spec.dependency import DependencyType
from sklearn_meta.spec.graph import GraphSpec

if TYPE_CHECKING:
    from sklearn_meta.spec.quantile import JointQuantileGraphSpec
    from sklearn_meta.spec.quantile_sampler import QuantileSampler


@dataclass
class InferenceGraph:
    graph: GraphSpec
    node_models: Dict[str, List[Any]]  # node_name -> fold models
    selected_features: Dict[str, Optional[List[str]]]
    node_params: Dict[str, Dict[str, Any]]

    def predict(self, X: pd.DataFrame, node_name: Optional[str] = None) -> np.ndarray:
        target_node = self._resolve_target_node(node_name)
        return self._predict_node(X, target_node)

    def predict_proba(
        self,
        X: pd.DataFrame,
        node_name: Optional[str] = None,
    ) -> np.ndarray:
        target_node = self._resolve_target_node(node_name)
        return self._predict_node(X, target_node, final_output_mode="predict_proba")

    def _resolve_target_node(self, node_name: Optional[str]) -> str:
        if node_name is not None:
            return node_name

        leaves = self.graph.get_leaf_nodes()
        if not leaves:
            raise ValueError("Graph has no leaf nodes")
        return leaves[0]

    def _predict_node(
        self,
        X: pd.DataFrame,
        node_name: str,
        cache: Optional[Dict[object, np.ndarray]] = None,
        final_output_mode: str = "configured",
    ) -> np.ndarray:
        if cache is None:
            cache = {}
        cache_key = (node_name, final_output_mode)
        if cache_key in cache:
            return cache[cache_key]
        if node_name in cache and final_output_mode == "configured":
            return cache[node_name]

        node = self.graph.get_node(node_name)

        # Get upstream predictions and augment features
        upstream_edges = self.graph.get_upstream(node_name)
        X_augmented = X.copy()

        for edge in upstream_edges:
            if edge.dep_type in (DependencyType.PREDICTION, DependencyType.PROBA, DependencyType.FEATURE):
                upstream_preds = self._predict_node(X, edge.source, cache)
                col_name = edge.feature_name
                if upstream_preds.ndim == 1:
                    X_augmented[col_name] = upstream_preds
                else:
                    for i in range(upstream_preds.shape[1]):
                        X_augmented[f"{col_name}_{i}"] = upstream_preds[:, i]
            elif edge.dep_type == DependencyType.TRANSFORM:
                upstream_preds = self._predict_node(X, edge.source, cache)
                X_augmented = upstream_preds
            # BASE_MARGIN, DISTILL: training-only, skip at inference

        # Filter to selected features
        sel_feats = self.selected_features.get(node_name)
        if sel_feats is not None:
            X_augmented = X_augmented[sel_feats]

        # Ensemble predictions from fold models
        predictions = []
        for model in self.node_models[node_name]:
            predictions.append(
                self._get_model_output(
                    node,
                    model,
                    X_augmented,
                    final_output_mode=final_output_mode,
                )
            )

        result = np.mean(predictions, axis=0)
        if final_output_mode == "configured":
            cache[node_name] = result
        cache[cache_key] = result
        return result

    def _get_model_output(
        self,
        node,
        model: Any,
        X: pd.DataFrame,
        final_output_mode: str = "configured",
    ) -> np.ndarray:
        if final_output_mode == "configured":
            return get_output(node, model, X)

        if final_output_mode == "predict_proba":
            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    f"Node '{node.name}' does not support probability prediction: "
                    f"{type(model).__name__} has no 'predict_proba' method"
                )
            return model.predict_proba(X)

        raise ValueError(f"Unknown final output mode: {final_output_mode}")

    def save(self, path) -> None:
        import joblib
        from pathlib import Path
        from sklearn_meta.persistence.manifest import write_manifest, to_json_safe

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for node_name, models in self.node_models.items():
            node_dir = path / "nodes" / node_name
            node_dir.mkdir(parents=True, exist_ok=True)
            for i, model in enumerate(models):
                joblib.dump(model, node_dir / f"fold_{i}.joblib")

        manifest = {
            "version": 3,
            "type": "inference",
            "graph": {
                "nodes": [to_json_safe(n.to_dict(), path=f"graph.nodes.{n.name}")
                          for n in self.graph.nodes.values()
                          if n.name in self.node_models],
                "edges": [to_json_safe(e.to_dict(), path=f"graph.edges.{e.source}->{e.target}")
                          for e in self.graph.edges
                          if e.source in self.node_models and e.target in self.node_models],
            },
            "nodes": {
                name: {
                    "n_folds": len(models),
                    "selected_features": self.selected_features.get(name),
                    "params": to_json_safe(self.node_params.get(name, {}), path=f"nodes.{name}.params"),
                }
                for name, models in self.node_models.items()
            },
        }
        write_manifest(path, manifest)

    @classmethod
    def load(cls, path) -> InferenceGraph:
        import joblib
        from pathlib import Path
        from sklearn_meta.persistence.manifest import read_manifest
        from sklearn_meta.spec.node import NodeSpec
        from sklearn_meta.spec.graph import GraphSpec as _GraphSpec
        from sklearn_meta.spec.dependency import DependencyEdge

        path = Path(path)
        manifest = read_manifest(path)

        graph = _GraphSpec()
        for node_data in manifest["graph"]["nodes"]:
            graph.add_node(NodeSpec.from_dict(node_data))
        for edge_data in manifest["graph"]["edges"]:
            graph.add_edge(DependencyEdge.from_dict(edge_data))

        node_models: Dict[str, List[Any]] = {}
        selected_features: Dict[str, Optional[List[str]]] = {}
        node_params: Dict[str, Dict[str, Any]] = {}

        for name, meta in manifest["nodes"].items():
            n_folds = meta["n_folds"]
            node_dir = path / "nodes" / name
            models = [joblib.load(node_dir / f"fold_{i}.joblib") for i in range(n_folds)]
            node_models[name] = models
            selected_features[name] = meta.get("selected_features")
            node_params[name] = meta.get("params", {})

        return cls(graph=graph, node_models=node_models,
                   selected_features=selected_features, node_params=node_params)


@dataclass
class QuantileFittedNode:
    """Inference-only fitted quantile node."""
    quantile_models: Dict[float, List[Any]]
    quantile_levels: List[float]
    selected_features: Optional[List[str]] = None

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        if self.selected_features is not None:
            # Only filter to features that exist in X — conditioning
            # columns may be added by the joint inference loop.
            available = [f for f in self.selected_features if f in X.columns]
            extra = [c for c in X.columns if c not in self.selected_features]
            X = X[available + extra]
        predictions = np.zeros((len(X), len(self.quantile_levels)))
        for q_idx, tau in enumerate(self.quantile_levels):
            fold_preds = [model.predict(X) for model in self.quantile_models[tau]]
            predictions[:, q_idx] = np.mean(fold_preds, axis=0)
        return predictions


@dataclass
class JointQuantileInferenceGraph:
    graph: JointQuantileGraphSpec  # from sklearn_meta.spec.quantile
    fitted_nodes: Dict[str, QuantileFittedNode]  # property -> fitted node
    quantile_sampler: QuantileSampler  # from sklearn_meta.spec.quantile_sampler

    @property
    def property_order(self) -> List[str]:
        """Convenience access to the graph property order."""
        return self.graph.property_order

    def get_property_quantiles(
        self,
        X: pd.DataFrame,
        property_name: str,
    ) -> np.ndarray:
        """Return all quantile predictions for a single property.

        Downstream properties are conditioned on the median predictions of all
        upstream properties, matching the sequential joint-inference flow.
        """
        X_augmented = X.copy()

        for current_property in self.graph.property_order:
            fitted_node = self.fitted_nodes[current_property]
            quantile_preds = fitted_node.predict_quantiles(X_augmented)

            if current_property == property_name:
                return quantile_preds

            medians = self.quantile_sampler.get_median(
                fitted_node.quantile_levels,
                quantile_preds,
            )
            X_augmented[f"cond_{current_property}"] = medians

        raise KeyError(f"Property '{property_name}' not found in joint quantile graph")

    def sample_joint(self, X: pd.DataFrame, n_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Sample from the joint distribution of all properties.

        Args:
            X: Input features.
            n_samples: Number of samples (uses sampler default if None).

        Returns:
            Dict mapping property name to sampled values of shape (n_data_points, n_samples).
        """
        if n_samples is not None:
            self.quantile_sampler.n_samples = n_samples
        self.quantile_sampler.reset_samples()

        samples: Dict[str, np.ndarray] = {}
        X_augmented = X.copy()

        for prop_name in self.graph.property_order:
            fitted_node = self.fitted_nodes[prop_name]
            quantile_preds = fitted_node.predict_quantiles(X_augmented)

            prop_samples = self.quantile_sampler.sample_property_batched(
                prop_name,
                quantile_levels=fitted_node.quantile_levels,
                quantile_predictions=quantile_preds,
            )
            samples[prop_name] = prop_samples

            # Condition downstream properties on sampled medians
            conditioning_values = self.quantile_sampler.get_median(
                fitted_node.quantile_levels, quantile_preds,
            )
            X_augmented[f"cond_{prop_name}"] = conditioning_values

        return samples

    def predict_median(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict median for all properties."""
        results: Dict[str, np.ndarray] = {}
        X_augmented = X.copy()

        for prop_name in self.graph.property_order:
            fitted_node = self.fitted_nodes[prop_name]
            quantile_preds = fitted_node.predict_quantiles(X_augmented)
            medians = self.quantile_sampler.get_median(
                fitted_node.quantile_levels, quantile_preds,
            )
            results[prop_name] = medians
            X_augmented[f"cond_{prop_name}"] = medians

        return results

    def predict_quantile(self, X: pd.DataFrame, q: float) -> Dict[str, np.ndarray]:
        """Predict a specific quantile for all properties."""
        results: Dict[str, np.ndarray] = {}
        X_augmented = X.copy()

        for prop_name in self.graph.property_order:
            fitted_node = self.fitted_nodes[prop_name]
            quantile_preds = fitted_node.predict_quantiles(X_augmented)
            q_values = self.quantile_sampler.get_quantile(
                q, fitted_node.quantile_levels, quantile_preds,
            )
            results[prop_name] = q_values
            # Use median for conditioning regardless of requested quantile
            medians = self.quantile_sampler.get_median(
                fitted_node.quantile_levels, quantile_preds,
            )
            X_augmented[f"cond_{prop_name}"] = medians

        return results

    def save(self, path) -> None:
        import joblib
        from pathlib import Path
        from sklearn_meta.persistence.manifest import write_manifest

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for prop_name, fitted_node in self.fitted_nodes.items():
            node_dir = path / "nodes" / prop_name
            node_dir.mkdir(parents=True, exist_ok=True)
            for tau, models in fitted_node.quantile_models.items():
                for i, model in enumerate(models):
                    joblib.dump(model, node_dir / f"q{tau:.4f}_fold_{i}.joblib")

        manifest = {
            "version": 3,
            "type": "joint_quantile_inference",
            "property_order": self.graph.property_order,
            "quantile_levels": self.graph.quantile_levels,
            "sampler": {
                "strategy": self.quantile_sampler.strategy.value,
                "n_samples": self.quantile_sampler.n_samples,
                "random_state": self.quantile_sampler.random_state,
            },
            "nodes": {
                prop_name: {
                    "quantile_levels": fitted_node.quantile_levels,
                    "n_folds": len(next(iter(fitted_node.quantile_models.values()))) if fitted_node.quantile_models else 0,
                    "selected_features": fitted_node.selected_features,
                }
                for prop_name, fitted_node in self.fitted_nodes.items()
            },
        }
        write_manifest(path, manifest)

    @classmethod
    def load(cls, path) -> JointQuantileInferenceGraph:
        import joblib
        from pathlib import Path
        from sklearn_meta.persistence.manifest import read_manifest
        from sklearn_meta.spec.quantile import JointQuantileGraphSpec as _JQGraphSpec, JointQuantileConfig
        from sklearn_meta.spec.quantile_sampler import QuantileSampler as _QuantileSampler, SamplingStrategy

        path = Path(path)
        manifest = read_manifest(path)

        # Reconstruct sampler
        sampler_data = manifest["sampler"]
        sampler = _QuantileSampler(
            strategy=SamplingStrategy(sampler_data["strategy"]),
            n_samples=sampler_data["n_samples"],
            random_state=sampler_data.get("random_state"),
        )

        # Load fitted nodes
        fitted_nodes: Dict[str, QuantileFittedNode] = {}
        for prop_name, node_meta in manifest["nodes"].items():
            quantile_levels = node_meta["quantile_levels"]
            n_folds = node_meta["n_folds"]
            node_dir = path / "nodes" / prop_name

            quantile_models: Dict[float, List[Any]] = {}
            for tau in quantile_levels:
                models = [joblib.load(node_dir / f"q{tau:.4f}_fold_{i}.joblib") for i in range(n_folds)]
                quantile_models[tau] = models

            fitted_nodes[prop_name] = QuantileFittedNode(
                quantile_models=quantile_models,
                quantile_levels=quantile_levels,
                selected_features=node_meta.get("selected_features"),
            )

        # Reconstruct graph config (minimal -- estimator_class not available)
        config = JointQuantileConfig(
            property_names=manifest["property_order"],
            quantile_levels=manifest["quantile_levels"],
        )
        graph = _JQGraphSpec(config)

        return cls(graph=graph, fitted_nodes=fitted_nodes, quantile_sampler=sampler)
