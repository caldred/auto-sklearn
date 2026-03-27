# Stacking

Stacking trains a "meta-learner" on the predictions of multiple "base models." sklearn-meta handles the tricky part -- generating out-of-fold predictions so the meta-learner never sees leaked training data.

---

## Quick Start

`stack()` builds and fits a stacking ensemble in one call:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_meta import stack

result = stack(
    base_models={
        "rf": (RandomForestClassifier, {"n_estimators": (50, 500), "max_depth": (3, 20)}),
        "gbm": (GradientBoostingClassifier, {"learning_rate": (0.01, 0.3, {"log": True})}),
    },
    meta_model=LogisticRegression,
    X=X_train, y=y_train,
    metric="roc_auc",
    n_trials=50,
)

result.predict(X_test)      # ensemble predictions
result.best_score_           # meta-learner's CV score
```

### stack() Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_models` | Dict of `{name: model_spec}` (see format below) | required |
| `meta_model` | Meta-learner model spec | required |
| `metric` | Scoring metric | `"neg_mean_squared_error"` |
| `n_trials` | Optuna trials per model | `50` |
| `cv` | Number of folds or a `CVConfig` | `5` |
| `stack_output` | Base-model output mode: `"auto"`, `"prediction"`, or `"proba"` | `"auto"` |
| `strategy` | CV strategy (auto-detected if omitted) | `None` |
| `groups` | Group labels for group CV | `None` |

Model specs can be:
- An estimator class: `RandomForestClassifier` (uses defaults, no tuning)
- An estimator instance: `RandomForestClassifier(n_estimators=200)` (uses those params, no tuning)
- A tuple: `(EstimatorClass, params_dict)` for tuning
- A tuple: `(EstimatorClass, params_dict, fixed_params_dict)` for tuning with fixed params

In `stack_output="auto"` mode, classifier base models use probabilities when they expose `predict_proba()`, while regressors and non-probabilistic classifiers use direct predictions. Use `stack_output="prediction"` to force label stacking, or `stack_output="proba"` to require probabilities from every base model.

---

## How Stacking Works

The key challenge with stacking is **data leakage**: if base models predict on data they trained on, the meta-learner sees overfit predictions and learns overfit patterns.

sklearn-meta solves this with **out-of-fold (OOF) predictions**:

1. Split data into K folds
2. For each fold, train base models on the other K-1 folds
3. Predict on the held-out fold
4. Combine predictions -- every training sample has a prediction from a model that never saw it
5. Train the meta-learner on these OOF predictions

At prediction time, all fold models predict and their outputs are aggregated before being passed to the meta-learner. Regressors are averaged, while classifier leaves return valid class labels rather than averaged numeric label values.

### Layer-by-Layer Optimization

For stacking graphs, sklearn-meta tunes models **layer by layer**:

1. **Layer 0**: Tune all base models (independently, can be parallelized)
2. **Generate OOF predictions** from the tuned base models
3. **Layer 1**: Tune the meta-learner on OOF predictions

This ensures the meta-learner is tuned on realistic inputs.

---

## Using the Explicit API

For multi-level stacking, mixed dependency types, or fine-grained control, use `GraphBuilder`:

```python
from sklearn.svm import SVC
from sklearn_meta import GraphBuilder, RunConfigBuilder, fit

graph = (
    GraphBuilder("stacking_pipeline")
    # Base models
    .add_model("rf", RandomForestClassifier)
        .int_param("n_estimators", 50, 500)
        .int_param("max_depth", 3, 20)
        .fixed_params(random_state=42)
    .add_model("gbm", GradientBoostingClassifier)
        .param("learning_rate", 0.01, 0.3, log=True)
        .int_param("max_depth", 3, 10)
    .add_model("svm", SVC)
        .param("C", 0.1, 100.0, log=True)
        .fixed_params(probability=True)
    # Meta-learner stacking on base model probabilities
    .add_model("meta", LogisticRegression)
        .param("C", 0.01, 100.0, log=True)
        .stacks_proba("rf", "gbm", "svm")
    .build()
)

config = (
    RunConfigBuilder()
    .cv(n_splits=5, strategy="stratified")
    .tuning(n_trials=50, metric="roc_auc")
    .build()
)

result = fit(graph, X_train, y_train, config)
```

### Dependency Methods

| Method | What it passes to the meta-learner |
|--------|------------------------------------|
| `.stacks("base")` | Class predictions (labels or regression values) |
| `.stacks_proba("base")` | Class probabilities -- usually better for classification |

Both accept multiple sources: `.stacks_proba("rf", "gbm", "svm")`.

For classification, `.stacks_proba()` is generally preferred because it preserves confidence information.

---

## Multi-Level Stacking

For deeper architectures, chain multiple meta-learner layers:

```python
from sklearn.linear_model import Ridge

graph = (
    GraphBuilder("deep_stack")
    # Layer 0: Base models
    .add_model("rf", RandomForestClassifier)
        .int_param("n_estimators", 50, 200)
    .add_model("xgb", XGBClassifier)
        .int_param("n_estimators", 50, 200)
    .add_model("lgbm", LGBMClassifier)
        .int_param("n_estimators", 50, 200)
    # Layer 1: Intermediate meta-learners
    .add_model("meta1", LogisticRegression)
        .param("C", 0.01, 100.0, log=True)
        .stacks_proba("rf", "xgb")
    .add_model("meta2", LogisticRegression)
        .param("C", 0.01, 100.0, log=True)
        .stacks_proba("lgbm")
    # Layer 2: Final meta-learner
    .add_model("final", Ridge)
        .param("alpha", 0.01, 100.0, log=True)
        .stacks_proba("meta1", "meta2")
    .build()
)
```

Each layer is tuned in order: base models first, then layer 1 meta-learners, then the final meta-learner.

---

## Working with Results

```python
# Predict through the full stack
predictions = result.predict(X_test)

# Shortcuts resolve to the leaf (meta-learner) node
result.best_score_       # meta-learner's CV score
result.best_params_      # meta-learner's best params
print(result.summary())  # summary of all nodes

# Access individual models
rf_result = result.node_results["rf"]
rf_result.best_params
rf_result.mean_score
rf_result.oof_predictions

# Predict from a specific node
rf_preds = result.predict(X_test, node_name="rf")

# Get probability predictions by averaging fold models
import numpy as np
models = result.node_results["rf"].models
probas = np.mean([m.predict_proba(X_test) for m in models], axis=0)
```

---

## Best Practices

1. **Use diverse base models** -- Mix model families (trees, linear, kernel-based). Three variations of random forest won't help much.
2. **Keep the meta-learner simple** -- `LogisticRegression` or `Ridge` work well. Complex meta-learners tend to overfit.
3. **Use `.stacks_proba()` for classification** -- Probabilities carry more information than class labels.
4. **Don't stack too deep** -- 2 layers (base + meta) is the sweet spot. 3 layers occasionally helps. 4+ rarely does.
5. **Ensure sufficient data** -- Stacking needs enough samples for reliable OOF predictions. Aim for 1000+ samples minimum.

---

## Next Steps

- [Model Graphs](model-graphs.md) -- Custom DAG architectures, dependency types, graph operations
- [Cross-Validation](cross-validation.md) -- CV strategies for stacking
- [Tuning](tuning.md) -- Optimization settings
