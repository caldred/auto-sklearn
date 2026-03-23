"""ModelPlugin: Abstract base class for model-specific plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn_meta.data.view import DataView
    from sklearn_meta.data.batch import MaterializedBatch
    from sklearn_meta.spec.node import NodeSpec
    from sklearn_meta.search.space import SearchSpace


class ModelPlugin(ABC):
    """
    Abstract base class for model-specific plugins.

    Plugins provide lifecycle hooks for customizing model behavior:
    - Modify search space before optimization
    - Modify fit parameters before training
    - Post-process fitted models
    - Post-process tuning results

    Example usage:
        class XGBPlugin(ModelPlugin):
            def applies_to(self, cls):
                return hasattr(cls, 'get_booster')

            def modify_fit_params(self, params, batch):
                params['verbose'] = False
                return params
    """

    @property
    def name(self) -> str:
        """Plugin name for identification."""
        return self.__class__.__name__

    @abstractmethod
    def applies_to(self, estimator_class: Type) -> bool:
        """
        Check if this plugin applies to a given estimator class.

        Args:
            estimator_class: The estimator class to check.

        Returns:
            True if this plugin should be applied.
        """
        pass

    def modify_search_space(
        self,
        space: SearchSpace,
        node: NodeSpec,
    ) -> SearchSpace:
        """
        Modify the search space before optimization.

        This hook allows adding, removing, or modifying parameters
        in the search space.

        Args:
            space: The original search space.
            node: The node spec being tuned.

        Returns:
            Modified search space.
        """
        return space

    def modify_params(
        self,
        params: Dict[str, Any],
        node: NodeSpec,
    ) -> Dict[str, Any]:
        """
        Modify hyperparameters before model creation.

        Called after sampling from the search space, before creating
        the estimator instance.

        Args:
            params: The sampled/fixed parameters.
            node: The node spec.

        Returns:
            Modified parameters.
        """
        return params

    def modify_fit_params(
        self,
        params: Dict[str, Any],
        batch: MaterializedBatch,
    ) -> Dict[str, Any]:
        """
        Modify parameters passed to fit().

        This hook allows customizing fit behavior based on
        the materialized batch (e.g., adding eval_set, sample_weight).

        Args:
            params: The current fit parameters.
            batch: The materialized batch for this fit.

        Returns:
            Modified fit parameters.
        """
        return params

    def pre_fit(
        self,
        model: Any,
        node: NodeSpec,
        batch: MaterializedBatch,
    ) -> Any:
        """
        Pre-process model before fitting.

        Called after model creation but before fit().

        Args:
            model: The model instance.
            node: The node spec.
            batch: The materialized batch.

        Returns:
            The (possibly modified) model.
        """
        return model

    def post_fit(
        self,
        model: Any,
        node: NodeSpec,
        batch: MaterializedBatch,
    ) -> Any:
        """
        Post-process model after fitting.

        This hook allows model modifications after training
        (e.g., pruning, quantization, caching).

        Args:
            model: The fitted model.
            node: The node spec.
            batch: The materialized batch.

        Returns:
            The (possibly modified) model.
        """
        return model

    def post_tune(
        self,
        best_params: Dict[str, Any],
        node: NodeSpec,
        data: DataView,
    ) -> Dict[str, Any]:
        """
        Post-process tuning results.

        This hook allows additional optimization after the main
        hyperparameter search (e.g., multiplier tuning for XGBoost).

        Args:
            best_params: The best parameters found.
            node: The node spec.
            data: The data view.

        Returns:
            Potentially refined parameters.
        """
        return best_params

    def on_fold_start(
        self,
        fold_idx: int,
        node: NodeSpec,
        data: DataView,
    ) -> None:
        """
        Called at the start of each CV fold.

        Args:
            fold_idx: The fold index.
            node: The node spec.
            data: The data view for this fold.
        """
        pass

    def on_fold_end(
        self,
        fold_idx: int,
        model: Any,
        score: float,
        node: NodeSpec,
    ) -> None:
        """
        Called at the end of each CV fold.

        Args:
            fold_idx: The fold index.
            model: The fitted model.
            score: The validation score.
            node: The node spec.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CompositePlugin(ModelPlugin):
    """
    Plugin that combines multiple plugins.

    Applies all plugins in order, chaining their outputs.
    """

    def __init__(self, plugins: List[ModelPlugin]) -> None:
        """
        Initialize with a list of plugins.

        Args:
            plugins: List of plugins to combine.
        """
        self._plugins = plugins

    @property
    def name(self) -> str:
        names = [p.name for p in self._plugins]
        return f"Composite({', '.join(names)})"

    def applies_to(self, estimator_class: Type) -> bool:
        """Check if any plugin applies."""
        return any(p.applies_to(estimator_class) for p in self._plugins)

    def _get_applicable(self, estimator_class: Type) -> List[ModelPlugin]:
        """Get plugins that apply to the estimator."""
        return [p for p in self._plugins if p.applies_to(estimator_class)]

    def modify_search_space(self, space: SearchSpace, node: NodeSpec) -> SearchSpace:
        for plugin in self._get_applicable(node.estimator_class):
            space = plugin.modify_search_space(space, node)
        return space

    def modify_params(self, params: Dict[str, Any], node: NodeSpec) -> Dict[str, Any]:
        for plugin in self._get_applicable(node.estimator_class):
            params = plugin.modify_params(params, node)
        return params

    def modify_fit_params(self, params: Dict[str, Any], batch: MaterializedBatch) -> Dict[str, Any]:
        for plugin in self._plugins:
            params = plugin.modify_fit_params(params, batch)
        return params

    def pre_fit(self, model: Any, node: NodeSpec, batch: MaterializedBatch) -> Any:
        for plugin in self._get_applicable(node.estimator_class):
            model = plugin.pre_fit(model, node, batch)
        return model

    def post_fit(self, model: Any, node: NodeSpec, batch: MaterializedBatch) -> Any:
        for plugin in self._get_applicable(node.estimator_class):
            model = plugin.post_fit(model, node, batch)
        return model

    def post_tune(
        self, best_params: Dict[str, Any], node: NodeSpec, data: DataView
    ) -> Dict[str, Any]:
        for plugin in self._get_applicable(node.estimator_class):
            best_params = plugin.post_tune(best_params, node, data)
        return best_params

    def __repr__(self) -> str:
        return f"CompositePlugin(n_plugins={len(self._plugins)})"
