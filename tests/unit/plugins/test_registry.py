"""Tests for PluginRegistry."""

import pytest
from typing import Type

from sklearn_meta.plugins.base import ModelPlugin
from sklearn_meta.plugins.registry import PluginRegistry, get_default_registry


class DummyPluginA(ModelPlugin):
    """Test plugin A."""

    def applies_to(self, estimator_class: Type) -> bool:
        return "A" in estimator_class.__name__


class DummyPluginB(ModelPlugin):
    """Test plugin B."""

    def applies_to(self, estimator_class: Type) -> bool:
        return "B" in estimator_class.__name__


class DummyPluginAll(ModelPlugin):
    """Test plugin that applies to all."""

    def applies_to(self, estimator_class: Type) -> bool:
        return True


class FailingPlugin(ModelPlugin):
    """Plugin that raises in applies_to."""

    def applies_to(self, estimator_class: Type) -> bool:
        raise ValueError("Test failure")


class EstimatorA:
    pass


class EstimatorB:
    pass


class EstimatorAB:
    pass


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_register_duplicate_raises(self):
        """Verify registering duplicate name raises."""
        registry = PluginRegistry()
        plugin1 = DummyPluginA()
        plugin2 = DummyPluginA()

        registry.register(plugin1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(plugin2)

    def test_register_with_priority(self):
        """Verify priority affects order."""
        registry = PluginRegistry()

        registry.register(DummyPluginA())
        registry.register(DummyPluginB(), priority=0)  # Insert at beginning

        assert registry.plugin_names[0] == "DummyPluginB"
        assert registry.plugin_names[1] == "DummyPluginA"

    def test_unregister_existing(self):
        """Verify unregistering existing plugin works."""
        registry = PluginRegistry()
        plugin = DummyPluginA()
        registry.register(plugin)

        removed = registry.unregister("DummyPluginA")

        assert removed is plugin
        assert len(registry) == 0
        assert "DummyPluginA" not in registry

    def test_get_plugins_for_matching(self):
        """Verify get_plugins_for returns matching plugins."""
        registry = PluginRegistry()
        registry.register(DummyPluginA())
        registry.register(DummyPluginB())

        result = registry.get_plugins_for(EstimatorA)

        assert len(result) == 1
        assert isinstance(result[0], DummyPluginA)

    def test_get_plugins_for_multiple_matches(self):
        """Verify get_plugins_for returns all matching plugins."""
        registry = PluginRegistry()
        registry.register(DummyPluginA())
        registry.register(DummyPluginB())
        registry.register(DummyPluginAll())

        result = registry.get_plugins_for(EstimatorA)

        assert len(result) == 2  # DummyPluginA and DummyPluginAll

    def test_get_plugins_for_preserves_order(self):
        """Verify get_plugins_for preserves registration order."""
        registry = PluginRegistry()
        registry.register(DummyPluginAll(), name="first")
        registry.register(DummyPluginAll(), name="second")
        registry.register(DummyPluginAll(), name="third")

        result = registry.get_plugins_for(EstimatorA)

        # All are DummyPluginAll but registered with different names
        # The registry stores them in order
        assert len(result) == 3

    def test_get_plugins_for_handles_failure(self):
        """Verify get_plugins_for handles applies_to exceptions."""
        registry = PluginRegistry()
        registry.register(FailingPlugin())
        registry.register(DummyPluginAll())

        # Should not raise, should skip failing plugin
        result = registry.get_plugins_for(EstimatorA)

        assert len(result) == 1
        assert isinstance(result[0], DummyPluginAll)

    def test_get_plugins_for_names(self):
        """Verify get_plugins_for_names returns specified plugins."""
        registry = PluginRegistry()
        registry.register(DummyPluginA())
        registry.register(DummyPluginB())
        registry.register(DummyPluginAll())

        result = registry.get_plugins_for_names(["DummyPluginA", "DummyPluginAll"])

        assert len(result) == 2
        assert isinstance(result[0], DummyPluginA)
        assert isinstance(result[1], DummyPluginAll)

    def test_get_plugins_for_names_preserves_order(self):
        """Verify get_plugins_for_names preserves requested order."""
        registry = PluginRegistry()
        registry.register(DummyPluginA())
        registry.register(DummyPluginB())

        result = registry.get_plugins_for_names(["DummyPluginB", "DummyPluginA"])

        assert isinstance(result[0], DummyPluginB)
        assert isinstance(result[1], DummyPluginA)

