"""Tests for ModelPlugin and CompositePlugin base classes."""

import pytest
from typing import Any, Dict, Type

from sklearn_meta.plugins.base import ModelPlugin, CompositePlugin


class DummyPlugin(ModelPlugin):
    """Test plugin that applies to classes with 'target' in the name."""

    def __init__(self, suffix: str = ""):
        self.suffix = suffix
        self.calls = []

    def applies_to(self, estimator_class: Type) -> bool:
        return "Target" in estimator_class.__name__

    def modify_search_space(self, space, node):
        self.calls.append(("modify_search_space", node.name))
        return space

    def modify_params(self, params, node):
        self.calls.append(("modify_params", node.name))
        params = dict(params)
        params["modified_by"] = self.name + self.suffix
        return params

    def modify_fit_params(self, params, ctx):
        self.calls.append(("modify_fit_params", None))
        params = dict(params)
        params["fit_modified"] = True
        return params


class AlwaysPlugin(ModelPlugin):
    """Plugin that always applies."""

    def applies_to(self, estimator_class: Type) -> bool:
        return True

    def post_fit(self, model, node, ctx):
        if not hasattr(model, "_plugin_post_fit"):
            model._plugin_post_fit = []
        model._plugin_post_fit.append(self.name)
        return model


class NeverPlugin(ModelPlugin):
    """Plugin that never applies."""

    def applies_to(self, estimator_class: Type) -> bool:
        return False


class TargetEstimator:
    """Mock estimator class for testing."""
    pass


class OtherEstimator:
    """Mock estimator class for testing."""
    pass


class TestModelPluginBase:
    """Tests for ModelPlugin abstract base class."""

    def test_plugin_name_default(self):
        """Verify default name is class name."""
        plugin = AlwaysPlugin()
        assert plugin.name == "AlwaysPlugin"

    def test_plugin_repr(self):
        """Verify repr includes class name."""
        plugin = AlwaysPlugin()
        assert "AlwaysPlugin" in repr(plugin)

    def test_applies_to_abstract(self):
        """Verify applies_to must be implemented."""
        with pytest.raises(TypeError):
            ModelPlugin()

    def test_default_modify_search_space_passthrough(self, simple_search_space, rf_classifier_node):
        """Verify default modify_search_space returns space unchanged."""
        plugin = AlwaysPlugin()

        result = plugin.modify_search_space(simple_search_space, rf_classifier_node)

        assert result is simple_search_space

    def test_default_modify_params_passthrough(self, rf_classifier_node):
        """Verify default modify_params returns params unchanged."""
        plugin = AlwaysPlugin()
        params = {"a": 1, "b": 2}

        result = plugin.modify_params(params, rf_classifier_node)

        assert result == params

    def test_default_modify_fit_params_passthrough(self, data_context):
        """Verify default modify_fit_params returns params unchanged."""
        plugin = AlwaysPlugin()
        params = {"verbose": True}

        result = plugin.modify_fit_params(params, data_context)

        assert result == params

    def test_default_pre_fit_passthrough(self, rf_classifier_node, data_context):
        """Verify default pre_fit returns model unchanged."""
        plugin = AlwaysPlugin()
        model = object()

        result = plugin.pre_fit(model, rf_classifier_node, data_context)

        assert result is model

    def test_default_post_fit_passthrough(self, rf_classifier_node, data_context):
        """Verify default post_fit returns model unchanged."""
        plugin = NeverPlugin()  # Use NeverPlugin to avoid AlwaysPlugin's modification
        model = object()

        result = plugin.post_fit(model, rf_classifier_node, data_context)

        assert result is model

    def test_default_post_tune_passthrough(self, rf_classifier_node, data_context):
        """Verify default post_tune returns params unchanged."""
        plugin = AlwaysPlugin()
        params = {"n_estimators": 100}

        result = plugin.post_tune(params, rf_classifier_node, data_context)

        assert result == params

    def test_on_fold_start_noop(self, rf_classifier_node, data_context):
        """Verify on_fold_start does nothing by default."""
        plugin = AlwaysPlugin()

        # Should not raise
        plugin.on_fold_start(0, rf_classifier_node, data_context)

    def test_on_fold_end_noop(self, rf_classifier_node):
        """Verify on_fold_end does nothing by default."""
        plugin = AlwaysPlugin()

        # Should not raise
        plugin.on_fold_end(0, None, 0.85, rf_classifier_node)


class TestDummyPlugin:
    """Tests for custom plugin implementation."""

    def test_applies_to_target_class(self):
        """Verify applies_to returns True for matching class."""
        plugin = DummyPlugin()

        assert plugin.applies_to(TargetEstimator) is True

    def test_not_applies_to_other_class(self):
        """Verify applies_to returns False for non-matching class."""
        plugin = DummyPlugin()

        assert plugin.applies_to(OtherEstimator) is False

    def test_modify_params_adds_key(self, rf_classifier_node):
        """Verify modify_params adds the expected key."""
        plugin = DummyPlugin(suffix="_test")
        params = {"a": 1}

        result = plugin.modify_params(params, rf_classifier_node)

        assert result["modified_by"] == "DummyPlugin_test"
        assert result["a"] == 1

    def test_modify_params_does_not_mutate_original(self, rf_classifier_node):
        """Verify modify_params does not mutate the original dict."""
        plugin = DummyPlugin()
        original = {"a": 1}

        result = plugin.modify_params(original, rf_classifier_node)

        assert "modified_by" not in original
        assert "modified_by" in result

    def test_modify_fit_params_adds_key(self, data_context):
        """Verify modify_fit_params adds the expected key."""
        plugin = DummyPlugin()
        params = {}

        result = plugin.modify_fit_params(params, data_context)

        assert result["fit_modified"] is True

    def test_tracks_calls(self, rf_classifier_node, data_context):
        """Verify plugin tracks method calls."""
        plugin = DummyPlugin()

        plugin.modify_search_space(None, rf_classifier_node)
        plugin.modify_params({}, rf_classifier_node)
        plugin.modify_fit_params({}, data_context)

        assert len(plugin.calls) == 3
        assert plugin.calls[0][0] == "modify_search_space"
        assert plugin.calls[1][0] == "modify_params"
        assert plugin.calls[2][0] == "modify_fit_params"


class TestCompositePlugin:
    """Tests for CompositePlugin."""

    def test_composite_name_includes_children(self):
        """Verify composite name includes child plugin names."""
        plugins = [DummyPlugin(), AlwaysPlugin()]
        composite = CompositePlugin(plugins)

        assert "DummyPlugin" in composite.name
        assert "AlwaysPlugin" in composite.name
        assert "Composite" in composite.name

    def test_composite_repr(self):
        """Verify composite repr shows plugin count."""
        plugins = [DummyPlugin(), AlwaysPlugin()]
        composite = CompositePlugin(plugins)

        assert "n_plugins=2" in repr(composite)

    def test_applies_to_any_match(self):
        """Verify applies_to returns True if any child matches."""
        plugins = [DummyPlugin(), NeverPlugin()]
        composite = CompositePlugin(plugins)

        assert composite.applies_to(TargetEstimator) is True

    def test_applies_to_none_match(self):
        """Verify applies_to returns False if no child matches."""
        plugins = [NeverPlugin(), NeverPlugin()]
        composite = CompositePlugin(plugins)

        assert composite.applies_to(TargetEstimator) is False

    def test_modify_params_chains_applicable(self, rf_classifier_node):
        """Verify modify_params chains applicable plugins."""
        # Change rf_classifier_node to use TargetEstimator
        from unittest.mock import MagicMock
        node = MagicMock()
        node.name = "test"
        node.estimator_class = TargetEstimator

        plugin1 = DummyPlugin(suffix="_1")
        plugin2 = DummyPlugin(suffix="_2")
        composite = CompositePlugin([plugin1, plugin2])

        result = composite.modify_params({"a": 1}, node)

        # Last plugin's modification should be present
        assert result["modified_by"] == "DummyPlugin_2"

    def test_modify_fit_params_applies_all(self, data_context):
        """Verify modify_fit_params applies all plugins."""
        plugins = [DummyPlugin(), DummyPlugin()]
        composite = CompositePlugin(plugins)

        result = composite.modify_fit_params({}, data_context)

        assert result["fit_modified"] is True

    def test_post_fit_chains_applicable(self, data_context):
        """Verify post_fit chains applicable plugins."""
        from unittest.mock import MagicMock
        node = MagicMock()
        node.estimator_class = object

        plugins = [AlwaysPlugin(), AlwaysPlugin()]
        composite = CompositePlugin(plugins)

        class Model:
            pass

        model = Model()
        result = composite.post_fit(model, node, data_context)

        assert len(result._plugin_post_fit) == 2

    def test_empty_composite_passthrough(self, rf_classifier_node, data_context):
        """Verify empty composite acts as passthrough."""
        composite = CompositePlugin([])

        assert composite.applies_to(TargetEstimator) is False

        params = {"a": 1}
        result = composite.modify_params(params, rf_classifier_node)
        assert result == params

    def test_get_applicable_filters(self):
        """Verify _get_applicable filters correctly."""
        plugins = [DummyPlugin(), NeverPlugin(), AlwaysPlugin()]
        composite = CompositePlugin(plugins)

        applicable = composite._get_applicable(TargetEstimator)

        assert len(applicable) == 2
        assert any(isinstance(p, DummyPlugin) for p in applicable)
        assert any(isinstance(p, AlwaysPlugin) for p in applicable)


@pytest.fixture
def simple_search_space():
    """Simple search space for testing."""
    from sklearn_meta.search.space import SearchSpace
    space = SearchSpace()
    space.add_int("n_estimators", 10, 100)
    return space
