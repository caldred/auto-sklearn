# Changelog

## 0.2.0

Breaking release that reorganizes the package around the new public surface:

- Public API is centered on top-level imports from `sklearn_meta`.
- Internal modules under `sklearn_meta.api` and `sklearn_meta.core.*` were removed.
- Core concepts were split into `spec`, `data`, `runtime`, `engine`, and `artifacts`.
- Training now flows through explicit graph, data, config, and runner objects.
- Inference artifacts are compiled from training runs instead of relying on the old fitted graph objects.

Upgrade notes:

- Migrate imports to top-level `sklearn_meta` exports where possible.
- Replace legacy `GraphBuilder(...).fit(...)` style usage with:
  - `GraphBuilder(...).compile()`
  - `DataView.from_Xy(...)`
  - `RunConfigBuilder(...).build()` or `RunConfig(...)`
  - `fit(graph, data, config)` or `GraphRunner(...).fit(...)`
- Downstream projects that still depend on removed internal paths should pin to `v0.1.0` or `release/0.1.x` until migrated.
