"""ShadowFeatureSelector: round-based paired-shadow pruning."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn_meta.selection.importance import (
    ImportanceExtractor,
    ImportanceRegistry,
    PermutationImportanceExtractor,
)

logger = logging.getLogger(__name__)


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
    Round-based shadow feature selection.

    Uses `n_shadows` rounds. In each round, shadows are added for roughly
    `1 / n_shadows` of real features (random partition). Each selected feature
    gets a paired random shadow matched to its empirical distribution and sampled
    from a global covariance model.
    """

    def __init__(
        self,
        importance_extractor: Optional[ImportanceExtractor] = None,
        n_shadows: int = 5,
        n_clusters: Optional[int] = None,
        threshold_mult: float = 1.414,  # sqrt(2)
        random_state: int = 42,
        covariance_sample_size: int = 50000,
    ) -> None:
        self.importance_extractor = importance_extractor
        self.n_shadows = max(1, int(n_shadows))
        # Compatibility attribute retained for callers/tests from the previous
        # entropy-clustering implementation.
        self.n_clusters = max(
            1, int(n_clusters if n_clusters is not None else self.n_shadows)
        )
        self.threshold_mult = threshold_mult
        self.random_state = random_state
        self.covariance_sample_size = covariance_sample_size

        self._importance_registry = ImportanceRegistry()

    def _compute_entropy(self, values: pd.Series) -> float:
        """Estimate Shannon entropy in bits for a feature column.

        Uses exact counts for low-cardinality columns and histogram binning for
        higher-cardinality numeric data. This preserves the older helper API
        used by tests and ad hoc inspection code.
        """
        series = pd.Series(values).dropna()
        if series.empty:
            return 0.0

        n_unique = int(series.nunique(dropna=True))
        if n_unique <= 64:
            probs = (
                series.value_counts(normalize=True, dropna=True)
                .to_numpy(dtype=float)
            )
        else:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                probs = (
                    series.astype(str)
                    .value_counts(normalize=True, dropna=True)
                    .to_numpy(dtype=float)
                )
            else:
                counts, _ = np.histogram(numeric.to_numpy(dtype=float), bins=64)
                probs = counts[counts > 0].astype(float)
                probs /= probs.sum()

        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def _cluster_features_by_entropy(
        self,
        X: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[int, List[str]]:
        """Group features into entropy bands.

        This is a compatibility helper for the previous API. The current
        pruning algorithm does not depend on these clusters.
        """
        if not feature_cols:
            return {}

        n_clusters = min(self.n_clusters, len(feature_cols))
        ranked = sorted(
            feature_cols,
            key=lambda col: (self._compute_entropy(X[col]), col),
        )

        clusters: Dict[int, List[str]] = {idx: [] for idx in range(n_clusters)}
        for idx, feature in enumerate(ranked):
            clusters[idx % n_clusters].append(feature)
        return clusters

    def _create_shadow_features(
        self,
        X: pd.DataFrame,
        feature_cols: List[str],
    ) -> tuple[pd.DataFrame, Dict[str, str]]:
        """Build an augmented DataFrame with one compatibility shadow per feature."""
        if not feature_cols:
            return X.copy(), {}

        X_base = X[feature_cols].copy()
        rng = np.random.default_rng(self.random_state)
        z_shadow = self._sample_shadow_latents(X_base, feature_cols, rng)

        shadow_cols: Dict[str, np.ndarray] = {}
        feature_to_shadow: Dict[str, str] = {}
        for idx, feat in enumerate(feature_cols):
            shadow_name = f"__shadow_{idx}_{feat}"
            ref = (
                pd.to_numeric(X_base[feat], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            shadow_cols[shadow_name] = self._map_to_empirical_distribution(
                z_shadow[:, idx],
                ref,
            )
            feature_to_shadow[feat] = shadow_name

        shadow_df = pd.DataFrame(shadow_cols, index=X_base.index)
        return pd.concat([X_base, shadow_df], axis=1), feature_to_shadow

    @staticmethod
    def _standardize(values: np.ndarray) -> np.ndarray:
        """Return z-scored values with safe handling of near-zero variance."""
        arr = np.asarray(values, dtype=float)
        std = float(np.std(arr))
        if std < 1e-12:
            return np.zeros_like(arr, dtype=float)
        return (arr - float(np.mean(arr))) / std

    @staticmethod
    def _map_to_empirical_distribution(z: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Monotonic rank-map from latent Gaussian values to empirical feature values."""
        z = np.asarray(z, dtype=float)
        ref = np.asarray(ref, dtype=float)

        if len(z) == 0:
            return z
        if len(ref) == 0:
            return np.zeros_like(z)

        order = np.argsort(z, kind="mergesort")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(z))

        ref_sorted = np.sort(ref)
        if len(z) == 1:
            return np.array([ref_sorted[len(ref_sorted) // 2]], dtype=float)

        q = ranks / max(len(z) - 1, 1)
        idx = np.clip((q * (len(ref_sorted) - 1)).astype(int), 0, len(ref_sorted) - 1)
        return ref_sorted[idx]

    def _build_round_plan(
        self,
        feature_cols: List[str],
        rng: np.random.Generator,
    ) -> List[List[str]]:
        """
        Build `n_shadows` rounds where each round shadows about `1/n_shadows`
        of the feature set.
        """
        n_features = len(feature_cols)
        if n_features == 0:
            return []

        n_rounds = self.n_shadows
        subset_size = max(1, int(math.ceil(n_features / n_rounds)))

        perm = list(rng.permutation(feature_cols))
        rounds: List[List[str]] = []
        for r in range(n_rounds):
            start = r * subset_size
            end = min((r + 1) * subset_size, n_features)
            chunk = perm[start:end]
            if len(chunk) < subset_size:
                needed = subset_size - len(chunk)
                chunk = chunk + perm[:needed]
            rounds.append(chunk)
        return rounds

    def _sample_shadow_latents(
        self,
        X: pd.DataFrame,
        feature_cols: List[str],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample latent Gaussian shadows using global feature covariance."""
        n_rows = len(X)
        n_features = len(feature_cols)
        if n_features == 0:
            return np.zeros((n_rows, 0), dtype=float)
        if n_features == 1:
            return rng.standard_normal((n_rows, 1))

        X_num_full = pd.DataFrame(
            {
                col: pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)
                for col in feature_cols
            },
            index=X.index,
        )

        if len(X_num_full) > self.covariance_sample_size:
            sample_rng = np.random.default_rng(self.random_state)
            sample_idx = sample_rng.choice(
                len(X_num_full),
                size=self.covariance_sample_size,
                replace=False,
            )
            X_num_cov = X_num_full.iloc[sample_idx]
        else:
            X_num_cov = X_num_full

        Z_cov = np.column_stack(
            [self._standardize(X_num_cov[col].to_numpy(dtype=float)) for col in feature_cols]
        )
        cov = np.cov(Z_cov, rowvar=False)
        cov = np.atleast_2d(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0))
        cov = cov + np.eye(cov.shape[0]) * 1e-6

        try:
            return rng.multivariate_normal(
                mean=np.zeros(n_features, dtype=float),
                cov=cov,
                size=n_rows,
            )
        except np.linalg.LinAlgError:
            return rng.standard_normal((n_rows, n_features))

    def fit_select(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None,
        importance_type: str = "gain",
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> ShadowResult:
        """Fit model with shadow rounds and select important features."""
        if feature_cols is None:
            feature_cols = list(X.columns)
        if not feature_cols:
            return ShadowResult(
                features_to_keep=[],
                features_to_drop=[],
                feature_importances={},
                shadow_importances={},
                feature_to_shadow={},
                threshold_used=0.0,
            )

        X_base = X[feature_cols]
        feature_idx = {feat: idx for idx, feat in enumerate(feature_cols)}
        rng = np.random.default_rng(self.random_state)
        round_plan = self._build_round_plan(feature_cols, rng)

        if self.importance_extractor is not None:
            extractor = self.importance_extractor
        else:
            model.fit(X_base, y)
            extractor = self._importance_registry.get_extractor(model)

        real_imp_sum: Dict[str, float] = {f: 0.0 for f in feature_cols}
        shadow_imp_sum: Dict[str, float] = {f: 0.0 for f in feature_cols}
        counts: Dict[str, int] = {f: 0 for f in feature_cols}
        feature_to_shadow = {f: f"__shadow_avg__{f}" for f in feature_cols}

        needs_val = isinstance(extractor, PermutationImportanceExtractor) and X_val is not None and y_val is not None
        X_val_base = X_val[feature_cols] if needs_val else None

        for round_idx, selected in enumerate(round_plan):
            z_shadow = self._sample_shadow_latents(
                X=X_base,
                feature_cols=feature_cols,
                rng=np.random.default_rng(self.random_state + round_idx + 1),
            )
            round_shadow_map: Dict[str, str] = {}
            shadow_cols: Dict[str, np.ndarray] = {}

            for feat in selected:
                idx = feature_idx[feat]
                shadow_name = f"__shadow_r{round_idx}_{idx}__"
                ref = pd.to_numeric(X_base[feat], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                shadow_cols[shadow_name] = self._map_to_empirical_distribution(z_shadow[:, idx], ref)
                round_shadow_map[feat] = shadow_name

            shadow_df = pd.DataFrame(shadow_cols, index=X_base.index)
            X_augmented = pd.concat([X_base, shadow_df], axis=1)

            model.fit(X_augmented, y)

            # Build extract kwargs; for PermutationImportanceExtractor pass
            # validation data with matching shadow columns.
            extract_kwargs: Dict[str, Any] = {"importance_type": importance_type}
            if needs_val:
                z_shadow_val = self._sample_shadow_latents(
                    X=X_val_base,
                    feature_cols=feature_cols,
                    rng=np.random.default_rng(self.random_state + round_idx + 1),
                )
                val_shadow_cols: Dict[str, np.ndarray] = {}
                for feat in selected:
                    idx = feature_idx[feat]
                    shadow_name = f"__shadow_r{round_idx}_{idx}__"
                    ref = pd.to_numeric(X_base[feat], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    val_shadow_cols[shadow_name] = self._map_to_empirical_distribution(
                        z_shadow_val[:, idx], ref
                    )
                val_shadow_df = pd.DataFrame(val_shadow_cols, index=X_val_base.index)
                X_val_augmented = pd.concat([X_val_base, val_shadow_df], axis=1)
                extract_kwargs["X_val"] = X_val_augmented
                extract_kwargs["y_val"] = y_val

            all_importance = extractor.extract(
                model,
                list(X_augmented.columns),
                **extract_kwargs,
            )

            for feat in selected:
                sh_name = round_shadow_map[feat]
                real_imp_sum[feat] += float(all_importance.get(feat, 0.0))
                shadow_imp_sum[feat] += float(all_importance.get(sh_name, 0.0))
                counts[feat] += 1

        missing = [f for f in feature_cols if counts[f] == 0]
        if missing:
            logger.warning("Shadow rounds missed %d features", len(missing))

        feature_importances = {
            feat: (real_imp_sum[feat] / counts[feat] if counts[feat] > 0 else 0.0)
            for feat in feature_cols
        }
        shadow_importances = {
            feature_to_shadow[feat]: (
                shadow_imp_sum[feat] / counts[feat] if counts[feat] > 0 else 0.0
            )
            for feat in feature_cols
        }

        features_to_keep: List[str] = []
        features_to_drop: List[str] = []
        thresholds: List[float] = []
        comparisons = []

        for feat in feature_cols:
            feat_imp = feature_importances[feat]
            paired_shadow = shadow_importances.get(feature_to_shadow[feat], 0.0)
            threshold = self.threshold_mult * paired_shadow
            thresholds.append(threshold)

            if feat_imp >= threshold:
                features_to_keep.append(feat)
                decision = "KEEP"
            else:
                features_to_drop.append(feat)
                decision = "DROP"

            ratio_to_threshold = (
                float("inf")
                if threshold == 0.0 and feat_imp > 0.0
                else (feat_imp / threshold if threshold > 0.0 else 0.0)
            )
            comparisons.append(
                {
                    "feature": feat,
                    "feature_importance": feat_imp,
                    "shadow_importance": paired_shadow,
                    "threshold": threshold,
                    "ratio_to_threshold": ratio_to_threshold,
                    "decision": decision,
                }
            )

        threshold_used = float(np.mean(thresholds)) if thresholds else 0.0
        avg_shadow_imp = float(np.mean(list(shadow_importances.values()))) if shadow_importances else 0.0
        per_round_fraction = 1.0 / float(self.n_shadows)

        logger.info(
            "Shadow pruning: %d/%d kept (base_mult=%.3f, rounds=%d, per_round_fraction=%.3f)",
            len(features_to_keep),
            len(feature_cols),
            self.threshold_mult,
            self.n_shadows,
            per_round_fraction,
        )
        logger.info(
            "Shadow baseline: avg_shadow_imp=%.6f, paired_shadow_thresholding, avg_threshold=%.6f",
            avg_shadow_imp,
            threshold_used,
        )

        top_n = 10
        kept_rows = sorted(
            [row for row in comparisons if row["decision"] == "KEEP"],
            key=lambda row: row["ratio_to_threshold"],
            reverse=True,
        )[:top_n]
        dropped_rows = sorted(
            [row for row in comparisons if row["decision"] == "DROP"],
            key=lambda row: row["ratio_to_threshold"],
        )[:top_n]

        if kept_rows:
            logger.info("Top kept (imp | shadow | threshold | ratio_to_threshold):")
            for row in kept_rows:
                logger.info(
                    "  KEEP %-34s %.6f | %.6f | %.6f | %.3f",
                    row["feature"][:34],
                    row["feature_importance"],
                    row["shadow_importance"],
                    row["threshold"],
                    row["ratio_to_threshold"],
                )

        if dropped_rows:
            logger.info("Top dropped (imp | shadow | threshold | ratio_to_threshold):")
            for row in dropped_rows:
                logger.info(
                    "  DROP %-34s %.6f | %.6f | %.6f | %.3f",
                    row["feature"][:34],
                    row["feature_importance"],
                    row["shadow_importance"],
                    row["threshold"],
                    row["ratio_to_threshold"],
                )

        return ShadowResult(
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            feature_importances=feature_importances,
            shadow_importances=shadow_importances,
            feature_to_shadow=feature_to_shadow,
            threshold_used=threshold_used,
        )

    def _build_group_representatives(
        self,
        X: pd.DataFrame,
        group_to_features: Dict[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        """Build one numeric representative array per group via standardize+mean."""
        group_reps: Dict[str, np.ndarray] = {}
        for group_name, members in group_to_features.items():
            group_matrix = np.column_stack([
                pd.to_numeric(X[feat], errors="coerce")
                .fillna(0.0).to_numpy(dtype=float)
                for feat in members
            ])
            if group_matrix.shape[1] == 1:
                group_reps[group_name] = group_matrix[:, 0]
            else:
                standardized = np.column_stack([
                    self._standardize(group_matrix[:, idx])
                    for idx in range(group_matrix.shape[1])
                ])
                group_reps[group_name] = standardized.mean(axis=1)
        return group_reps

    def fit_select_grouped(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        group_to_features: Dict[str, List[str]],
        feature_cols: Optional[List[str]] = None,
        importance_type: str = "gain",
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> ShadowResult:
        """Fit model with one shadow representative per group.

        This is the grouped cutover mode: once explicit grouping is present,
        each group is compared against a single shadow representative rather
        than one shadow per member feature.
        """
        if feature_cols is None:
            feature_cols = list(X.columns)
        if not feature_cols:
            return ShadowResult(
                features_to_keep=[],
                features_to_drop=[],
                feature_importances={},
                shadow_importances={},
                feature_to_shadow={},
                threshold_used=0.0,
            )

        feature_set = set(feature_cols)
        active_group_to_features: Dict[str, List[str]] = {}
        for group_name, members in group_to_features.items():
            active_members = [f for f in members if f in feature_set]
            if active_members:
                active_group_to_features[group_name] = active_members

        if not active_group_to_features:
            return self.fit_select(
                model=model,
                X=X,
                y=y,
                feature_cols=feature_cols,
                importance_type=importance_type,
                X_val=X_val,
                y_val=y_val,
            )

        X_base = X[feature_cols]
        rng = np.random.default_rng(self.random_state)
        group_names = list(active_group_to_features.keys())
        round_plan = self._build_round_plan(group_names, rng)

        if self.importance_extractor is not None:
            extractor = self.importance_extractor
        else:
            model.fit(X_base, y)
            extractor = self._importance_registry.get_extractor(model)

        # Build one numeric representative series per group.
        group_reps = self._build_group_representatives(X_base, active_group_to_features)

        X_group = pd.DataFrame(group_reps, index=X_base.index)
        group_idx = {group: idx for idx, group in enumerate(group_names)}

        real_group_imp_sum: Dict[str, float] = {group: 0.0 for group in group_names}
        shadow_group_imp_sum: Dict[str, float] = {group: 0.0 for group in group_names}
        counts: Dict[str, int] = {group: 0 for group in group_names}
        group_to_shadow_name: Dict[str, str] = {
            group: f"__shadow_group_avg__{group}" for group in group_names
        }

        needs_val = isinstance(extractor, PermutationImportanceExtractor) and X_val is not None and y_val is not None
        if needs_val:
            X_val_base = X_val[feature_cols]
            val_group_reps = self._build_group_representatives(X_val_base, active_group_to_features)
            X_val_group = pd.DataFrame(val_group_reps, index=X_val_base.index)
        else:
            X_val_base = None
            X_val_group = None

        for round_idx, selected_groups in enumerate(round_plan):
            z_shadow = self._sample_shadow_latents(
                X=X_group,
                feature_cols=group_names,
                rng=np.random.default_rng(self.random_state + round_idx + 1),
            )
            round_shadow_map: Dict[str, str] = {}
            shadow_cols: Dict[str, np.ndarray] = {}

            for group_name in selected_groups:
                idx = group_idx[group_name]
                shadow_name = f"__shadow_group_r{round_idx}_{idx}__"
                ref = X_group[group_name].to_numpy(dtype=float)
                shadow_cols[shadow_name] = self._map_to_empirical_distribution(
                    z_shadow[:, idx],
                    ref,
                )
                round_shadow_map[group_name] = shadow_name

            shadow_df = pd.DataFrame(shadow_cols, index=X_base.index)
            X_augmented = pd.concat([X_base, shadow_df], axis=1)

            model.fit(X_augmented, y)

            extract_kwargs: Dict[str, Any] = {"importance_type": importance_type}
            if needs_val:
                z_shadow_val = self._sample_shadow_latents(
                    X=X_val_group,
                    feature_cols=group_names,
                    rng=np.random.default_rng(self.random_state + round_idx + 1),
                )
                val_shadow_cols: Dict[str, np.ndarray] = {}
                for group_name_v in selected_groups:
                    idx_v = group_idx[group_name_v]
                    shadow_name_v = f"__shadow_group_r{round_idx}_{idx_v}__"
                    ref_v = X_group[group_name_v].to_numpy(dtype=float)
                    val_shadow_cols[shadow_name_v] = self._map_to_empirical_distribution(
                        z_shadow_val[:, idx_v], ref_v
                    )
                val_shadow_df = pd.DataFrame(val_shadow_cols, index=X_val_base.index)
                X_val_augmented = pd.concat([X_val_base, val_shadow_df], axis=1)
                extract_kwargs["X_val"] = X_val_augmented
                extract_kwargs["y_val"] = y_val

            all_importance = extractor.extract(
                model,
                list(X_augmented.columns),
                **extract_kwargs,
            )

            for group_name in selected_groups:
                members = active_group_to_features[group_name]
                group_real_imp = float(
                    np.mean([all_importance.get(member, 0.0) for member in members])
                )
                shadow_name = round_shadow_map[group_name]
                shadow_imp = float(all_importance.get(shadow_name, 0.0))

                real_group_imp_sum[group_name] += group_real_imp
                shadow_group_imp_sum[group_name] += shadow_imp
                counts[group_name] += 1

        group_importances: Dict[str, float] = {}
        shadow_importances: Dict[str, float] = {}
        for group_name in group_names:
            count = counts[group_name]
            group_importances[group_name] = (
                real_group_imp_sum[group_name] / count if count > 0 else 0.0
            )
            shadow_importances[group_to_shadow_name[group_name]] = (
                shadow_group_imp_sum[group_name] / count if count > 0 else 0.0
            )

        features_to_keep: List[str] = []
        features_to_drop: List[str] = []
        feature_importances: Dict[str, float] = {}
        feature_to_shadow: Dict[str, str] = {}
        thresholds: List[float] = []

        for group_name in group_names:
            group_imp = group_importances[group_name]
            shadow_name = group_to_shadow_name[group_name]
            group_shadow_imp = shadow_importances.get(shadow_name, 0.0)
            threshold = self.threshold_mult * group_shadow_imp
            thresholds.append(threshold)

            members = active_group_to_features[group_name]
            keep_group = group_imp >= threshold

            for feature in members:
                feature_importances[feature] = group_imp
                feature_to_shadow[feature] = shadow_name
                if keep_group:
                    features_to_keep.append(feature)
                else:
                    features_to_drop.append(feature)

        # Preserve deterministic order according to feature_cols.
        kept_set = set(features_to_keep)
        features_to_keep = [f for f in feature_cols if f in kept_set]
        features_to_drop = [f for f in feature_cols if f not in kept_set]

        threshold_used = float(np.mean(thresholds)) if thresholds else 0.0
        logger.info(
            "Grouped shadow pruning: %d/%d kept across %d groups (base_mult=%.3f)",
            len(features_to_keep),
            len(feature_cols),
            len(group_names),
            self.threshold_mult,
        )

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
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> List[str]:
        """Convenience method to get just the features to keep."""
        result = self.fit_select(model, X, y, feature_cols, X_val=X_val, y_val=y_val)
        return result.features_to_keep

    def __repr__(self) -> str:
        return (
            f"ShadowFeatureSelector(n_shadows={self.n_shadows}, "
            f"threshold_mult={self.threshold_mult})"
        )
