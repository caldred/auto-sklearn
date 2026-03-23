# Code Review Backlog

Items identified during code review that were considered but deferred. Organized by category.

## Redundant State

### RunMetadata.cv_config duplicates RunConfig.cv
`RunMetadata` serializes `RunConfig.cv` into its own `cv_config` field, but `RunConfig` is already fully persisted in the manifest. After load, `training_run.config.cv` and `training_run.metadata.cv_config` carry the same data. The metadata field is derivable from the run config.

**Why deferred:** Intentional design -- `RunMetadata` is a self-contained snapshot record. Removing it would couple metadata consumers to `RunConfig` internals.

### RunMetadata.tuning_config_summary partially duplicates TuningConfig
The summary stores `strategy`, `n_trials`, `metric`, and `greater_is_better` -- all also preserved in the separately persisted `RunConfig.tuning`.

**Why deferred:** Same reasoning as above. The summary is a convenience for quick inspection without deserializing the full config.

## Code Reuse

### Test fixture duplication across test files
`test_predict_contract.py`, `test_run_metadata.py`, and `test_fitted_graph_serialization.py` each define their own helpers for creating `DataView`, `RunConfig`, and `TrainingRun` instances. These could converge on shared conftest fixtures.

**Why deferred:** Normal test isolation pattern. Shared fixtures add coupling between test files and make individual tests harder to understand in isolation.

### Fold-building loop duplicated in test_fitted_graph_serialization.py
The fold-construction loop (creating `CVFold`, `FoldResult`, `CVResult`) appears three times within `test_fitted_graph_serialization.py` with minor variations.

**Why deferred:** Test-only duplication. Could extract a local `_make_fold_results` helper within that file.

## Leaky Abstractions

### _build_inference_only_cv_result creates structurally hollow CVResult objects
Loaded (inference-only) `TrainingRun` instances get `CVResult` objects with empty `train_indices`, `val_indices`, and `val_predictions`. Callers accessing these fields get empty arrays rather than a clear error or sentinel type.

**Why deferred:** Introducing a distinct `InferenceCVResult` type or making `cv_result` optional would be a larger refactor touching many callsites. The current approach works -- the empty arrays are harmless for inference use cases. The separate `InferenceGraph` type avoids this issue for users who only need prediction.

## Efficiency

### data_hash computation is O(n_samples x n_features) on every fit()
`pd.util.hash_pandas_object(batch.X).values.tobytes()` computes a per-row hash across all columns, added unconditionally to every training run in `GraphRunner._build_metadata`.

**Why deferred:** Intentional feature for reproducibility tracking. Could be made lazy/optional in the future if profiling shows it's a bottleneck on very large datasets.

### supports_param calls inspect.signature on every invocation
`inspect.signature(estimator_class.__init__)` does introspection each time. Could be cached with `functools.lru_cache`.

**Why deferred:** Called only 1-2 times per node per tuning run. Not a hot path.

### EstimatorScaler constructed unconditionally for every node
`EstimatorScalingConfig` and `EstimatorScaler` are created in the standard trainer even when scaling is disabled.

**Why deferred:** Two lightweight dataclass allocations per node. Guarding construction behind a condition would add branching complexity for negligible savings.

## Resolved in v2

The following items from the original backlog have been resolved by the v2 architecture:

### ~~TuningConfig has four flat estimator-scaling fields instead of EstimatorScalingConfig~~
**Resolved.** `RunConfig` now composes `EstimatorScalingConfig` as a separate sub-config (`RunConfig.estimator_scaling`). No flat fields on `TuningConfig`.

### ~~distillation.py reimplements supports_param pattern~~
**Resolved.** The `core/` module tree was deleted. Distillation validation now uses the centralized utility from `engine/`.

### ~~FittedGraph exposes _training_artifacts_available as constructor argument~~
**Resolved.** `FittedGraph` no longer exists. `TrainingRun` handles full training artifacts, while `InferenceGraph` and `JointQuantileInferenceGraph` handle inference-only use cases. The artifact availability is determined by which type you load.

### ~~_deserialize_tuning_config instantiates a throwaway TuningConfig for defaults~~
**Resolved.** `TuningConfig` is now a frozen dataclass under `RunConfig` and is deserialized directly from manifest data in `TrainingRun.load()` without throwaway instances.
