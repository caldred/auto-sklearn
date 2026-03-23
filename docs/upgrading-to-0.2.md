# Upgrading To 0.2

`0.2.0` is intentionally breaking.

## What changed

- `sklearn_meta.api` was removed.
- `sklearn_meta.core.*` modules were removed.
- Public concepts were reorganized into:
  - `sklearn_meta.spec`
  - `sklearn_meta.data`
  - `sklearn_meta.runtime`
  - `sklearn_meta.engine`
  - `sklearn_meta.artifacts`

## New training flow

Use the new explicit flow:

```python
from sklearn_meta import DataView, GraphBuilder, RunConfigBuilder, fit

graph = (
    GraphBuilder("example")
    .add_model("model", SomeEstimator)
    .fixed_params(random_state=42)
    .compile()
)

data = DataView.from_Xy(X_train, y_train, groups=groups)

config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="group")
    .tuning(n_trials=50, metric="neg_log_loss", greater_is_better=True)
    .build()
)

training_run = fit(graph, data, config)
inference = training_run.compile_inference()
predictions = inference.predict(X_test)
```

## What to do if you still depend on old imports

If your project imports from removed paths such as `sklearn_meta.api` or
`sklearn_meta.core.*`, stay on `v0.1.0` or the `release/0.1.x` branch until you
finish migrating to the new public API.
