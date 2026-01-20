"""ShadowFeatureSelector: Entropy-matched shadow features for feature pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from sklearn_meta.selection.importance import ImportanceExtractor, ImportanceRegistry

if TYPE_CHECKING:
    pass


@dataclass
class ShadowResult:
    """Result from shadow feature selection."""

    features_to_keep: List[str]
    features_to_drop: List[str]
    feature_importances: Dict[str, float]
    shadow_importances: Dict[str, float]
    feature_to_shadow: Dict[str, str]
    threshold_used: float


class ShadowFeatureSelector:
    """
    Creates noise features matched to real feature entropy distributions,
    then prunes features with importance below their shadow counterpart.

    Key insight: Random noise with similar "information structure" provides
    a calibrated baseline. Features that can't beat calibrated noise are
    likely not generalizing.

    Algorithm:
    1. Compute entropy for each feature using quantile-based estimation
    2. Cluster features by entropy into k groups
    3. Generate shadow (noise) features matching each cluster's entropy
    4. Fit model on augmented data (real + shadow features)
    5. Compare real feature importance to its matched shadow
    6. Drop features where: importance < threshold_mult * shadow_importance
    """

    def __init__(
        self,
        importance_extractor: Optional[ImportanceExtractor] = None,
        n_shadows: int = 5,
        n_clusters: int = 5,
        threshold_mult: float = 1.414,  # sqrt(2)
        random_state: int = 42,
    ) -> None:
        """
        Initialize the shadow feature selector.

        Args:
            importance_extractor: Extractor for feature importance.
                                  Uses TreeImportanceExtractor by default.
            n_shadows: Number of shadow features per entropy cluster.
            n_clusters: Number of entropy clusters.
            threshold_mult: Multiplier for shadow importance threshold.
            random_state: Random seed for reproducibility.
        """
        self.importance_extractor = importance_extractor
        self.n_shadows = n_shadows
        self.n_clusters = n_clusters
        self.threshold_mult = threshold_mult
        self.random_state = random_state

        self._importance_registry = ImportanceRegistry()

    def _compute_entropy(self, col: pd.Series) -> float:
        """
        Compute entropy using quantile-based estimation.

        Uses 256 quantiles to estimate the distribution and compute
        Shannon entropy.

        Args:
            col: Feature column.

        Returns:
            Estimated entropy value.
        """
        # Handle NaN values
        col_clean = col.fillna(0)

        # Use quantiles to estimate distribution
        qs = np.linspace(0, 1, 256)
        try:
            q_values = col_clean.quantile(q=qs)
            _, counts = np.unique(q_values, return_counts=True)
        except Exception:
            return 0.0

        # Compute probabilities
        p = counts / counts.sum()

        # Shannon entropy
        entropy = -np.sum(p * np.log2(p + 1e-12))
        return float(entropy)

    def _cluster_features_by_entropy(
        self,
        X: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[int, List[str]]:
        """
        Cluster features by their entropy values.

        Args:
            X: Feature DataFrame.
            feature_cols: List of feature columns to cluster.

        Returns:
            Dictionary mapping cluster index to list of feature names.
        """
        # Compute entropy for each feature
        entropies = {}
        for col in feature_cols:
            entropies[col] = self._compute_entropy(X[col])

        # Convert to array for clustering
        entropy_values = np.array(list(entropies.values())).reshape(-1, 1)

        # Use KMeans for clustering
        from sklearn.cluster import KMeans

        n_clusters = min(self.n_clusters, len(feature_cols))
        if n_clusters < 1:
            n_clusters = 1

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(entropy_values)

        # Group features by cluster
        clusters: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        for col, label in zip(feature_cols, labels):
            clusters[label].append(col)

        return clusters

    def _create_shadow_features(
        self,
        X: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Create shadow features matched to real feature entropy.

        Args:
            X: Feature DataFrame.
            feature_cols: List of feature columns.

        Returns:
            Tuple of (augmented DataFrame, mapping from feature to shadow name).
        """
        np.random.seed(self.random_state)

        # Cluster features by entropy
        clusters = self._cluster_features_by_entropy(X, feature_cols)

        # Create shadow features for each cluster
        shadow_cols = {}
        feature_to_shadow = {}

        for cluster_idx, cluster_features in clusters.items():
            if not cluster_features:
                continue

            # Compute mean entropy for this cluster
            mean_entropy = np.mean(
                [self._compute_entropy(X[col]) for col in cluster_features]
            )

            # Create shadow features with similar entropy
            for shadow_idx in range(self.n_shadows):
                shadow_name = f"__shadow_{cluster_idx}_{shadow_idx}__"

                # Generate noise that approximately matches the entropy
                # Higher entropy = more uniform distribution
                # Lower entropy = more concentrated distribution
                if mean_entropy > 4:  # High entropy
                    shadow_data = np.random.randn(len(X))
                elif mean_entropy > 2:  # Medium entropy
                    shadow_data = np.random.exponential(1.0, len(X))
                else:  # Low entropy
                    shadow_data = np.random.choice(
                        np.arange(5), size=len(X), p=[0.5, 0.3, 0.1, 0.05, 0.05]
                    ).astype(float)

                shadow_cols[shadow_name] = shadow_data

            # Map features to shadow (use first shadow in cluster)
            shadow_base = f"__shadow_{cluster_idx}_0__"
            for feat in cluster_features:
                feature_to_shadow[feat] = shadow_base

        # Create augmented DataFrame
        X_augmented = X.copy()
        for col_name, col_data in shadow_cols.items():
            X_augmented[col_name] = col_data

        return X_augmented, feature_to_shadow

    def fit_select(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None,
        importance_type: str = "gain",
    ) -> ShadowResult:
        """
        Fit model with shadow features and select important features.

        Args:
            model: Unfitted sklearn-compatible estimator.
            X: Feature DataFrame.
            y: Target Series.
            feature_cols: List of feature columns to evaluate.
                         If None, uses all columns.
            importance_type: Type of importance to use.

        Returns:
            ShadowResult with features to keep/drop and importances.
        """
        if feature_cols is None:
            feature_cols = list(X.columns)

        # Create shadow features
        X_augmented, feature_to_shadow = self._create_shadow_features(X, feature_cols)

        # Fit model on augmented data
        model.fit(X_augmented, y)

        # Get extractor
        if self.importance_extractor is not None:
            extractor = self.importance_extractor
        else:
            extractor = self._importance_registry.get_extractor(model)

        # Extract importance for all features
        all_features = list(X_augmented.columns)
        all_importance = extractor.extract(model, all_features, importance_type=importance_type)

        # Separate real and shadow importances
        feature_importances = {f: all_importance.get(f, 0.0) for f in feature_cols}
        shadow_importances = {
            f: all_importance.get(f, 0.0)
            for f in all_importance
            if f.startswith("__shadow_")
        }

        # Determine threshold and select features
        features_to_keep = []
        features_to_drop = []

        for feat in feature_cols:
            feat_imp = feature_importances[feat]
            shadow_name = feature_to_shadow[feat]
            shadow_imp = shadow_importances.get(shadow_name, 0.0)

            threshold = self.threshold_mult * shadow_imp

            if feat_imp >= threshold:
                features_to_keep.append(feat)
            else:
                features_to_drop.append(feat)

        # Get average shadow importance as threshold reference
        avg_shadow_imp = np.mean(list(shadow_importances.values())) if shadow_importances else 0.0
        threshold_used = self.threshold_mult * avg_shadow_imp

        return ShadowResult(
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            feature_importances=feature_importances,
            shadow_importances=shadow_importances,
            feature_to_shadow=feature_to_shadow,
            threshold_used=threshold_used,
        )

    def select_features(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Convenience method to get just the features to keep.

        Args:
            model: Unfitted sklearn-compatible estimator.
            X: Feature DataFrame.
            y: Target Series.
            feature_cols: List of feature columns to evaluate.

        Returns:
            List of feature names to keep.
        """
        result = self.fit_select(model, X, y, feature_cols)
        return result.features_to_keep

    def __repr__(self) -> str:
        return (
            f"ShadowFeatureSelector(n_shadows={self.n_shadows}, "
            f"threshold_mult={self.threshold_mult})"
        )
