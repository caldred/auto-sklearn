# sklearn-meta

A Python library for automated machine learning with meta-learning capabilities, hyperparameter optimization, and model stacking.

## Features

- **Model Graph**: Define complex model pipelines as directed acyclic graphs (DAGs)
- **Hyperparameter Optimization**: Backend-agnostic search with Optuna integration
- **Reparameterization**: Orthogonal parameter transformations for better optimization
- **Cross-Validation**: Stratified, grouped, nested, and time-series CV strategies
- **Feature Selection**: Shadow feature-based selection with entropy matching
- **Stacking**: Multi-layer model stacking with out-of-fold predictions
- **Plugin System**: Extensible plugins for model-specific behavior (XGBoost, etc.)
- **Caching**: Hash-based caching for expensive model fitting operations
- **Audit Logging**: Comprehensive logging for tuning and training progress

## Installation

```bash
pip install -e .
```

### Optional Dependencies

For XGBoost support:
```bash
pip install xgboost
# On macOS, also install OpenMP:
brew install libomp
```

For LightGBM support:
```bash
pip install lightgbm
```

## Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.node import ModelNode
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.tuning.orchestrator import TuningConfig, TuningOrchestrator
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.search.backends.optuna import OptunaBackend

# Create search space
rf_space = SearchSpace()
rf_space.add_int("n_estimators", 50, 200)
rf_space.add_int("max_depth", 3, 10)

# Create model node
rf_node = ModelNode(
    name="rf",
    estimator_class=RandomForestClassifier,
    search_space=rf_space,
    fixed_params={"random_state": 42},
)

# Build model graph
graph = ModelGraph()
graph.add_node(rf_node)

# Configure tuning
cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED, random_state=42)
tuning_config = TuningConfig(
    strategy=OptimizationStrategy.LAYER_BY_LAYER,
    n_trials=20,
    cv_config=cv_config,
    metric="accuracy",
    greater_is_better=True,
)

# Create data context
ctx = DataContext(X=X_train, y=y_train)

# Run tuning
data_manager = DataManager(cv_config)
orchestrator = TuningOrchestrator(
    graph=graph,
    data_manager=data_manager,
    search_backend=OptunaBackend(),
    tuning_config=tuning_config,
)
fitted_graph = orchestrator.fit(ctx)

# Make predictions
predictions = fitted_graph.predict(X_test)
```

## Project Structure

```
sklearn_meta/
├── core/
│   ├── data/          # DataContext, CV, DataManager
│   ├── model/         # ModelNode, ModelGraph, Dependencies
│   └── tuning/        # TuningOrchestrator, Strategies
├── search/            # SearchSpace, Parameters, Backends
├── meta/              # Reparameterization, Prebaked configs
├── selection/         # Feature selection, Importance
├── plugins/           # Model-specific plugins
├── execution/         # Execution backends
├── persistence/       # Artifact storage, Caching
└── audit/             # Logging
```

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=sklearn_meta --cov-report=html
```

## License

MIT License
