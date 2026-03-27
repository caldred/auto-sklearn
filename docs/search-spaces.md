# Search Spaces

Search spaces define the hyperparameters to optimize during tuning. sklearn-meta provides two ways to define them: inline via `GraphBuilder`, or standalone via the `SearchSpace` class.

---

## Defining Search Spaces with GraphBuilder

The recommended approach. Define search spaces inline using `.param()`, `.int_param()`, and `.cat_param()`:

```python
from sklearn_meta.spec.builder import GraphBuilder
from sklearn.ensemble import RandomForestClassifier

graph = (
    GraphBuilder()
    .add_model("rf", RandomForestClassifier)
        .int_param("n_estimators", 50, 200)
        .int_param("max_depth", 3, 15)
        .param("min_samples_split", 0.01, 0.2)
        .cat_param("max_features", ["sqrt", "log2", None])
    .build()
)
```

### Builder Methods

| Method | Description | Example |
|--------|-------------|---------|
| `.param(name, low, high)` | Float hyperparameter | `.param("lr", 0.01, 0.3)` |
| `.param(name, low, high, log=True)` | Float with log scale | `.param("lr", 0.01, 0.3, log=True)` |
| `.int_param(name, low, high)` | Integer hyperparameter | `.int_param("depth", 3, 10)` |
| `.int_param(name, low, high, step=5)` | Integer with step size | `.int_param("units", 32, 512, step=32)` |
| `.cat_param(name, choices)` | Categorical hyperparameter | `.cat_param("solver", ["adam", "sgd"])` |
| `.search_space(space)` | Attach a pre-built SearchSpace | `.search_space(my_space)` |

---

## Defining Search Spaces with SearchSpace

For reusable or programmatically built spaces, use the `SearchSpace` class directly:

```python
from sklearn_meta.search.space import SearchSpace

space = (
    SearchSpace()
    .add_int("n_estimators", 50, 200)
    .add_float("learning_rate", 0.001, 0.1, log=True)
    .add_categorical("solver", ["adam", "sgd"])
)
```

Attach to a graph node with `.search_space()`:

```python
graph = (
    GraphBuilder()
    .add_model("rf", RandomForestClassifier)
        .search_space(rf_space)
    .add_model("xgb", XGBClassifier)
        .search_space(xgb_space)
    .build()
)
```

---

## Parameter Types

### Integer Parameters

```python
space.add_int("n_estimators", low=50, high=200)
space.add_int("max_depth", low=1, high=20)
```

**Options:**
- `log=True` -- Sample on log scale (useful for wide ranges)
- `step=5` -- Only sample multiples of 5

```python
space.add_int("n_estimators", 10, 1000, log=True)
space.add_int("hidden_units", 32, 512, step=32)  # 32, 64, 96, ...
```

### Float Parameters

```python
space.add_float("learning_rate", low=0.001, high=0.1)
space.add_float("l2_reg", low=1e-6, high=1e-2, log=True)
```

**Options:**
- `log=True` -- Sample on log scale
- `step=0.01` -- Discrete steps

```python
space.add_float("lr", 1e-5, 1e-1, log=True)
space.add_float("momentum", 0.8, 0.99, step=0.01)
```

### Categorical Parameters

```python
space.add_categorical("activation", ["relu", "tanh", "sigmoid"])
space.add_categorical("use_batch_norm", [True, False])
```

---

## Conditional Parameters

Parameters that only apply when another parameter has a specific value:

```python
from sklearn_meta.search.parameter import FloatParameter

space = SearchSpace()
space.add_categorical("optimizer", ["adam", "sgd", "rmsprop"])

# momentum only applies when optimizer="sgd"
space.add_conditional(
    name="momentum",
    parent_name="optimizer",
    parent_value="sgd",
    parameter=FloatParameter(name="momentum", low=0.0, high=0.99),
)

# beta1 only applies when optimizer="adam"
space.add_conditional(
    name="beta1",
    parent_name="optimizer",
    parent_value="adam",
    parameter=FloatParameter(name="beta1", low=0.8, high=0.99),
)
```

---

## Shorthand Notation

For quick space definitions:

```python
space = SearchSpace()
space.add_from_shorthand(
    n_estimators=(50, 200),           # Int (both bounds are integers)
    learning_rate=(0.001, 0.1),       # Float (at least one bound is float)
    lr_log=(0.001, 0.1, "log"),       # Float with log scale
    solver=["adam", "sgd", "lbfgs"],  # Categorical (list)
)
```

| Shorthand | Resulting Type |
|-----------|----------------|
| `(10, 100)` | IntParameter |
| `(0.1, 1.0)` | FloatParameter |
| `(0.001, 0.1, "log")` | FloatParameter with log=True |
| `["a", "b", "c"]` | CategoricalParameter |

---

## Creating from Dictionary

For configuration files or dynamic space creation:

```python
config = {
    "n_estimators": (50, 200),
    "learning_rate": (0.001, 0.1, "log"),
    "solver": ["adam", "sgd"],
}

space = SearchSpace.from_dict(config)
```

### Explicit Dictionary Format

For full control:

```python
config = {
    "learning_rate": {
        "type": "float",
        "low": 0.001,
        "high": 0.1,
        "log": True,
    },
    "n_estimators": {
        "type": "int",
        "low": 10,
        "high": 200,
        "step": 10,
    },
    "kernel": {
        "type": "categorical",
        "choices": ["rbf", "poly", "linear"],
    },
}

space = SearchSpace.from_dict(config)
```

---

## Space Operations

### Copy

```python
original = SearchSpace().add_int("n", 1, 10)
copy = original.copy()
copy.add_float("x", 0.0, 1.0)
assert len(original) == 1  # unaffected
```

### Merge

```python
space1 = SearchSpace().add_int("a", 1, 10)
space2 = SearchSpace().add_float("b", 0.0, 1.0)
space1.merge(space2)  # space1 now contains both 'a' and 'b'
```

On name conflicts, the merged space's parameters overwrite the original's.

### Narrow Around a Point

Create a focused search space around known-good parameters (useful for retuning after feature selection):

```python
best_params = {"learning_rate": 0.05, "max_depth": 7, "reg_lambda": 0.1}
narrowed = space.narrow_around(best_params, factor=0.5)
```

### Remove Parameters

```python
space.remove_parameter("b")
```

---

## Inspection

```python
space = (
    SearchSpace()
    .add_int("n", 1, 10)
    .add_float("lr", 0.001, 0.1)
)

space.parameter_names  # ['n', 'lr']
len(space)             # 2
"n" in space           # True

param = space.get_parameter("lr")
param.low   # 0.001
param.high  # 0.1
param.log   # False

for param in space:
    print(f"{param.name}: {param.low} to {param.high}")
```

---

## Best Practices

**Use log scale for learning rates and regularization.** Parameters that span orders of magnitude should always use `log=True`, otherwise the sampler wastes most trials on the upper end of the range:

```python
# Good: explores 0.001, 0.01, 0.1 evenly
space.add_float("lr", 0.001, 0.1, log=True)
space.add_float("alpha", 1e-6, 1.0, log=True)

# Bad: linear scale wastes samples on high values
space.add_float("lr", 0.001, 0.1)
```

For correlated parameters (e.g., learning_rate vs n_estimators), see [reparameterization](reparameterization.md).

---

## Next Steps

- [Tuning](tuning.md) -- How search spaces are optimized
- [Reparameterization](reparameterization.md) -- Transform correlated parameters
- [Model Graphs](model-graphs.md) -- Attach spaces to model nodes
- [Cross-Validation](cross-validation.md) -- Configure CV strategies used during tuning
