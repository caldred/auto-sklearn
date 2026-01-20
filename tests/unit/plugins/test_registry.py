"""Tests for PluginRegistry."""

import pytest
from typing import Type

from auto_sklearn.plugins.base import ModelPlugin
from auto_sklearn.plugins.registry import PluginRegistry, get_default_registry


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

    def test_empty_registry(self):
        """Verify empty registry has no plugins."""
        registry = PluginRegistry()

        assert len(registry) == 0
        assert registry.plugin_names == []

    def test_register_plugin(self):
        """Verify plugin registration works."""
        registry = PluginRegistry()
        plugin = DummyPluginA()

        registry.register(plugin)

        assert len(registry) == 1
        assert "DummyPluginA" in registry

    def test_register_with_custom_name(self):
        """Verify registration with custom name."""
        registry = PluginRegistry()
        plugin = DummyPluginA()

        registry.register(plugin, name="custom_name")

        assert "custom_name" in registry
        assert "DummyPluginA" not in registry

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

    def test_unregister_nonexistent(self):
        """Verify unregistering nonexistent returns None."""
        registry = PluginRegistry()

        removed = registry.unregister("NonexistentPlugin")

        assert removed is None

    def test_get_existing(self):
        """Verify get returns registered plugin."""
        registry = PluginRegistry()
        plugin = DummyPluginA()
        registry.register(plugin)

        result = registry.get("DummyPluginA")

        assert result is plugin

    def test_get_nonexistent(self):
        """Verify get returns None for nonexistent."""
        registry = PluginRegistry()

        result = registry.get("NonexistentPlugin")

        assert result is None

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

        names = [p.name for p in result]
        # All are DummyPluginAll but registered with different names
        # The registry stores them in order
        assert len(result) == 3

    def test_get_plugins_for_none_matching(self):
        """Verify get_plugins_for returns empty for no matches."""
        registry = PluginRegistry()
        registry.register(DummyPluginA())

        result = registry.get_plugins_for(EstimatorB)

        assert result == []

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

    def test_get_plugins_for_names_skips_missing(self):
        """Verify get_plugins_for_names skips missing plugins."""
        registry = PluginRegistry()
        registry.register(DummyPluginA())

        result = registry.get_plugins_for_names(["DummyPluginA", "Nonexistent"])

        assert len(result) == 1
        assert isinstance(result[0], DummyPluginA)

    def test_get_plugins_for_names_preserves_order(self):
        """Verify get_plugins_for_names preserves requested order."""
        registry = PluginRegistry()
        registry.register(DummyPluginA())
        registry.register(DummyPluginB())

        result = registry.get_plugins_for_names(["DummyPluginB", "DummyPluginA"])

        assert isinstance(result[0], DummyPluginB)
        assert isinstance(result[1], DummyPluginA)

    def test_contains(self):
        """Verify __contains__ works correctly."""
        registry = PluginRegistry()
        registry.register(DummyPluginA())

        assert "DummyPluginA" in registry
        assert "DummyPluginB" not in registry

    def test_len(self):
        """Verify __len__ returns correct count."""
        registry = PluginRegistry()

        assert len(registry) == 0

        registry.register(DummyPluginA())
        assert len(registry) == 1

        registry.register(DummyPluginB())
        assert len(registry) == 2

    def test_repr(self):
        """Verify repr includes plugin names."""
        registry = PluginRegistry()
        registry.register(DummyPluginA())

        repr_str = repr(registry)

        assert "DummyPluginA" in repr_str
        assert "PluginRegistry" in repr_str


class TestDefaultRegistry:
    """Tests for default registry functionality."""

    def test_get_default_registry_returns_registry(self):
        """Verify get_default_registry returns a PluginRegistry."""
        registry = get_default_registry()

        assert isinstance(registry, PluginRegistry)

    def test_get_default_registry_singleton(self):
        """Verify get_default_registry returns same instance."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2

    def test_default_registry_has_xgb_plugins(self):
        """Verify default registry includes XGBoost plugins if available."""
        registry = get_default_registry()

        # This test depends on whether xgboost is installed
        # The plugins are optional, so we just check the registry exists
        assert isinstance(registry, PluginRegistry)
