# Code Review Backlog

Items identified during code review that were considered but deferred. Organized by category.

## Redundant State

### RunMetadata.cv_config duplicates TuningConfig.cv_config
`RunMetadata` serializes `TuningConfig.cv_config` into its own `cv_config` field, but `TuningConfig` itself is already fully persisted in the manifest. After load, `fitted_graph.tuning_config.cv_config` and `fitted_graph.metadata.cv_config` carry the same data. The metadata field is derivable from the tuning config.

**Why deferred:** Intentional design — `RunMetadata` is a self-contained snapshot record. Removing it would couple metadata consumers to `TuningConfig` internals.

### RunMetadata.tuning_config_summary partially duplicates TuningConfig
The summary stores `strategy`, `n_trials`, `metric`, and `greater_is_better` — all also preserved in the separately persisted `tuning_config`.

**Why deferred:** Same reasoning as above. The summary is a convenience for quick inspection without deserializing the full config.

## Parameter Sprawl

### TuningConfig has four flat estimator-scaling fields instead of EstimatorScalingConfig
`TuningConfig` has `tuning_n_estimators`, `final_n_estimators`, `estimator_scaling_search`, and `estimator_scaling_factors` as flat fields, mirroring the `EstimatorScalingConfig` dataclass. The orchestrator manually assembles an `EstimatorScalingConfig` from them in `_fit_node`.

**Why deferred:** Changing `TuningConfig`'s public API is a breaking change for existing users. Would need a deprecation path.

## Code Reuse

### Test fixture duplication across test files
`test_predict_contract.py`, `test_run_metadata.py`, and `test_fitted_graph_serialization.py` each define their own helpers for creating `DataContext`, `TuningConfig`, and `FittedGraph` instances. These could converge on shared conftest fixtures.

**Why deferred:** Normal test isolation pattern. Shared fixtures add coupling between test files and make individual tests harder to understand in isolation.

### distillation.py reimplements supports_param pattern
`validate_distillation_estimator` in `core/model/distillation.py` does its own `inspect.signature(estimator_class.__init__)` check, duplicating the logic now centralized in `supports_param()` from `core/tuning/estimator_scaling.py`.

**Why deferred:** Would require a cross-layer import from `core.model` into `core.tuning`, or moving `supports_param` to a shared utility module.

### Fold-building loop duplicated in test_fitted_graph_serialization.py
The fold-construction loop (creating `CVFold`, `FoldResult`, `CVResult`) appears three times within `test_fitted_graph_serialization.py` with minor variations.

**Why deferred:** Test-only duplication. Could extract a local `_make_fold_results` helper within that file.

## Leaky Abstractions

### FittedGraph exposes _training_artifacts_available as constructor argument
The private field `_training_artifacts_available` is passed as an explicit keyword argument in the `load()` classmethod's `cls(...)` call. This leaks internal bookkeeping through the dataclass interface.

**Why deferred:** The `load()` classmethod is the only caller that sets this. Making it truly private would require post-init mutation or a factory pattern, adding complexity for minimal gain.

### _build_inference_only_cv_result creates structurally hollow CVResult objects
Loaded (inference-only) graphs get `CVResult` objects with empty `train_indices`, `val_indices`, and `val_predictions`. Callers accessing these fields get empty arrays rather than a clear error or sentinel type.

**Why deferred:** Introducing a distinct `InferenceCVResult` type or making `cv_result` optional would be a larger refactor touching many callsites. The current approach works — the empty arrays are harmless for inference use cases.

## Efficiency

### data_hash computation is O(n_samples x n_features) on every fit()
`pd.util.hash_pandas_object(ctx.X).values.tobytes()` computes a per-row hash across all columns, added unconditionally to every training run.

**Why deferred:** Intentional feature for reproducibility tracking. Could be made lazy/optional in the future if profiling shows it's a bottleneck on very large datasets.

### supports_param calls inspect.signature on every invocation
`inspect.signature(estimator_class.__init__)` does introspection each time. Could be cached with `functools.lru_cache`.

**Why deferred:** Called only 1-2 times per node per tuning run. Not a hot path.

### _deserialize_tuning_config instantiates a throwaway TuningConfig for defaults
Constructs `TuningConfig()` solely to read its field defaults, then discards it.

**Why deferred:** `TuningConfig` is a lightweight dataclass with no side effects. The cost is negligible.

### EstimatorScaler constructed unconditionally for every node
`EstimatorScalingConfig` and `EstimatorScaler` are created in `_fit_node` even when scaling is disabled.

**Why deferred:** Two lightweight dataclass allocations per node. Guarding construction behind a condition would add branching complexity for negligible savings.
