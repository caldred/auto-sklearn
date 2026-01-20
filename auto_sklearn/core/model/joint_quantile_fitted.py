"""JointQuantileFittedGraph: Inference interface for joint quantile regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from auto_sklearn.core.model.joint_quantile_graph import JointQuantileGraph
from auto_sklearn.core.model.quantile_sampler import QuantileSampler

if TYPE_CHECKING:
    from auto_sklearn.core.tuning.joint_quantile_orchestrator import (
        FittedQuantileNode,
        JointQuantileFitResult,
    )


@dataclass
class JointQuantileFittedGraph:
    """
    Inference interface for fitted joint quantile models.

    Provides methods for:
    - Joint sampling from the multivariate distribution
    - Point predictions (median, specific quantiles)
    - Quantile predictions for each property

    The joint sampling uses the chain rule decomposition:
    P(Y₁, Y₂, ..., Yₙ | X) = P(Y₁|X) × P(Y₂|X,Y₁) × P(Y₃|X,Y₁,Y₂) × ...

    During inference, we sample from each conditional and feed the
    sampled values forward as conditioning features.

    Attributes:
        graph: The JointQuantileGraph defining the structure.
        fitted_nodes: Dict mapping property names to FittedQuantileNode.
        quantile_sampler: Sampler for converting quantile predictions to samples.

    Example:
        # Create from fit result
        fitted = JointQuantileFittedGraph.from_fit_result(fit_result)

        # Sample from joint distribution
        samples = fitted.sample_joint(X_test, n_samples=1000)
        # Returns: (n_test, 1000, n_properties)

        # Get median predictions
        medians = fitted.predict_median(X_test)
        # Returns: (n_test, n_properties)

        # Get specific quantile
        q90 = fitted.predict_quantile(X_test, q=0.9)
        # Returns: (n_test, n_properties)
    """

    graph: JointQuantileGraph
    fitted_nodes: Dict[str, "FittedQuantileNode"]
    quantile_sampler: QuantileSampler = field(default_factory=QuantileSampler)

    @classmethod
    def from_fit_result(cls, fit_result: "JointQuantileFitResult") -> "JointQuantileFittedGraph":
        """
        Create a fitted graph from a fit result.

        Args:
            fit_result: Result from JointQuantileOrchestrator.fit().

        Returns:
            JointQuantileFittedGraph ready for inference.
        """
        sampler = fit_result.graph.create_quantile_sampler()
        return cls(
            graph=fit_result.graph,
            fitted_nodes=fit_result.fitted_nodes,
            quantile_sampler=sampler,
        )

    def sample_joint(
        self,
        X: pd.DataFrame,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample from the joint distribution P(Y₁, Y₂, ..., Yₙ | X).

        Uses sequential sampling: for each sample path, we:
        1. Predict quantiles for Y₁ given X
        2. Sample Ŷ₁ from the quantile distribution
        3. Predict quantiles for Y₂ given X and sampled Ŷ₁
        4. Sample Ŷ₂ using the SAME uniform random value
        5. Continue for all properties...

        Using the same uniform samples across all properties ensures
        consistent sampling paths that capture the correlation structure.

        Args:
            X: Input features of shape (n_data_points, n_features).
            n_samples: Number of samples per data point. If None, uses
                      the sampler's configured n_samples.

        Returns:
            Samples of shape (n_data_points, n_samples, n_properties).
        """
        n_data = len(X)
        n_props = self.graph.n_properties
        n_samples = n_samples or self.quantile_sampler.n_samples

        # Reset sampler for fresh uniform samples
        if n_samples != self.quantile_sampler.n_samples:
            sampler = QuantileSampler(
                strategy=self.quantile_sampler.strategy,
                n_samples=n_samples,
                random_state=None,  # New random samples
            )
        else:
            sampler = self.quantile_sampler
            sampler.reset_samples()

        # Pre-generate uniform samples for consistent paths
        uniform_samples = sampler.uniform_samples

        # Result array: (n_data, n_samples, n_properties)
        samples = np.zeros((n_data, n_samples, n_props))

        # Sample sequentially through the chain
        for prop_idx, prop_name in enumerate(self.graph.property_order):
            fitted_node = self.fitted_nodes[prop_name]
            quantile_levels = np.array(fitted_node.quantile_levels)

            # Prepare features with conditioning on previous samples
            X_cond = self._prepare_conditional_features(
                X, prop_name, samples, prop_idx
            )

            # Predict quantiles for each sample path
            # For efficiency, we batch predict for all sample paths together
            # X_cond has shape (n_data * n_samples, n_features)
            quantile_preds = fitted_node.predict_quantiles(X_cond)
            # Shape: (n_data * n_samples, n_quantiles)

            # Reshape to (n_data, n_samples, n_quantiles)
            quantile_preds = quantile_preds.reshape(n_data, n_samples, -1)

            # Sample from each distribution using the same uniform values
            for i in range(n_data):
                for j in range(n_samples):
                    samples[i, j, prop_idx] = np.interp(
                        uniform_samples[j],
                        quantile_levels,
                        quantile_preds[i, j],
                    )

        return samples

    def _prepare_conditional_features(
        self,
        X: pd.DataFrame,
        current_prop: str,
        samples: np.ndarray,
        current_prop_idx: int,
    ) -> pd.DataFrame:
        """
        Prepare features with conditioning on previous samples.

        Args:
            X: Original input features.
            current_prop: Current property being sampled.
            samples: Samples so far, shape (n_data, n_samples, n_props).
            current_prop_idx: Index of current property.

        Returns:
            Expanded features with conditioning, shape (n_data * n_samples, n_features).
        """
        upstream_props = self.graph.get_conditioning_properties(current_prop)
        n_data = len(X)
        n_samples = samples.shape[1]

        if not upstream_props:
            # No conditioning - just repeat X for each sample path
            X_repeated = pd.concat([X] * n_samples, ignore_index=True)
            return X_repeated

        # Create expanded features for all sample paths
        X_list = []
        for j in range(n_samples):
            X_sample = X.copy()
            for up_idx, up_prop in enumerate(upstream_props):
                prop_idx = self.graph.property_order.index(up_prop)
                X_sample[f"cond_{up_prop}"] = samples[:, j, prop_idx]
            X_list.append(X_sample)

        return pd.concat(X_list, ignore_index=True)

    def sample_joint_efficient(
        self,
        X: pd.DataFrame,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Efficiently sample from joint distribution using batched operations.

        This is a more memory-efficient version that processes samples
        in batches to avoid creating very large intermediate arrays.

        Args:
            X: Input features.
            n_samples: Number of samples per data point.

        Returns:
            Samples of shape (n_data_points, n_samples, n_properties).
        """
        n_data = len(X)
        n_props = self.graph.n_properties
        n_samples = n_samples or self.quantile_sampler.n_samples

        # Reset sampler
        self.quantile_sampler.reset_samples()
        uniform_samples = self.quantile_sampler.uniform_samples[:n_samples]

        # Result array
        samples = np.zeros((n_data, n_samples, n_props))

        # Process each data point
        for i in range(n_data):
            X_point = X.iloc[[i]]

            for prop_idx, prop_name in enumerate(self.graph.property_order):
                fitted_node = self.fitted_nodes[prop_name]
                quantile_levels = np.array(fitted_node.quantile_levels)

                # Prepare features for this point with all sample paths
                upstream_props = self.graph.get_conditioning_properties(prop_name)

                if not upstream_props:
                    # No conditioning - predict quantiles once
                    quantile_preds = fitted_node.predict_quantiles(X_point)[0]
                    # Sample all paths from same distribution
                    for j in range(n_samples):
                        samples[i, j, prop_idx] = np.interp(
                            uniform_samples[j],
                            quantile_levels,
                            quantile_preds,
                        )
                else:
                    # Need to predict for each sample path separately
                    for j in range(n_samples):
                        X_cond = X_point.copy()
                        for up_prop in upstream_props:
                            up_idx = self.graph.property_order.index(up_prop)
                            X_cond[f"cond_{up_prop}"] = samples[i, j, up_idx]

                        quantile_preds = fitted_node.predict_quantiles(X_cond)[0]
                        samples[i, j, prop_idx] = np.interp(
                            uniform_samples[j],
                            quantile_levels,
                            quantile_preds,
                        )

        return samples

    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict median values for all properties.

        Uses the median quantile predictions, conditioning on
        median values from upstream properties.

        Args:
            X: Input features.

        Returns:
            Median predictions of shape (n_data_points, n_properties).
        """
        return self.predict_quantile(X, q=0.5)

    def predict_quantile(self, X: pd.DataFrame, q: float) -> np.ndarray:
        """
        Predict values at a specific quantile for all properties.

        Args:
            X: Input features.
            q: Quantile level (0 < q < 1).

        Returns:
            Quantile predictions of shape (n_data_points, n_properties).
        """
        n_data = len(X)
        n_props = self.graph.n_properties

        predictions = np.zeros((n_data, n_props))

        # Track predictions for conditioning
        prop_preds: Dict[str, np.ndarray] = {}

        for prop_idx, prop_name in enumerate(self.graph.property_order):
            fitted_node = self.fitted_nodes[prop_name]
            quantile_levels = np.array(fitted_node.quantile_levels)

            # Prepare features with conditioning
            upstream_props = self.graph.get_conditioning_properties(prop_name)
            X_cond = X.copy()
            for up_prop in upstream_props:
                X_cond[f"cond_{up_prop}"] = prop_preds[up_prop]

            # Predict quantiles and interpolate to desired level
            quantile_preds = fitted_node.predict_quantiles(X_cond)
            # Shape: (n_data, n_quantiles)

            # Interpolate to get value at quantile q
            prop_pred = np.zeros(n_data)
            for i in range(n_data):
                prop_pred[i] = np.interp(q, quantile_levels, quantile_preds[i])

            predictions[:, prop_idx] = prop_pred
            prop_preds[prop_name] = prop_pred

        return predictions

    def predict_quantiles_all(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Predict multiple quantile levels for all properties.

        Args:
            X: Input features.
            quantiles: List of quantile levels. If None, uses the
                      graph's configured quantile levels.

        Returns:
            Predictions of shape (n_data_points, n_quantiles, n_properties).
        """
        if quantiles is None:
            quantiles = self.graph.quantile_levels

        n_data = len(X)
        n_quantiles = len(quantiles)
        n_props = self.graph.n_properties

        predictions = np.zeros((n_data, n_quantiles, n_props))

        for q_idx, q in enumerate(quantiles):
            predictions[:, q_idx, :] = self.predict_quantile(X, q)

        return predictions

    def get_property_quantiles(
        self,
        X: pd.DataFrame,
        property_name: str,
    ) -> np.ndarray:
        """
        Get all quantile predictions for a specific property.

        Note: This conditions on median values of upstream properties.

        Args:
            X: Input features.
            property_name: Name of the property.

        Returns:
            Quantile predictions of shape (n_data_points, n_quantiles).
        """
        fitted_node = self.fitted_nodes[property_name]
        upstream_props = self.graph.get_conditioning_properties(property_name)

        # Get conditioning values (medians of upstream properties)
        X_cond = X.copy()
        for up_prop in upstream_props:
            up_fitted = self.fitted_nodes[up_prop]
            # Recursively get median predictions
            up_medians = self.get_property_quantiles(X, up_prop)
            median_idx = len(up_fitted.quantile_levels) // 2
            X_cond[f"cond_{up_prop}"] = up_medians[:, median_idx]

        return fitted_node.predict_quantiles(X_cond)

    @property
    def property_order(self) -> List[str]:
        """Property ordering used in the model."""
        return self.graph.property_order

    @property
    def n_properties(self) -> int:
        """Number of properties."""
        return self.graph.n_properties

    @property
    def quantile_levels(self) -> List[float]:
        """Quantile levels used in the model."""
        return self.graph.quantile_levels

    def __repr__(self) -> str:
        return (
            f"JointQuantileFittedGraph(properties={self.property_order}, "
            f"n_quantiles={len(self.quantile_levels)})"
        )
