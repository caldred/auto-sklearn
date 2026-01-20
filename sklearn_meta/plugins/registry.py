"""PluginRegistry: Registry for model plugins."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from sklearn_meta.plugins.base import ModelPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Registry for model plugins.

    Manages plugin registration, lookup, and lifecycle.
    Plugins are matched to estimators using their `applies_to` method.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._plugins: Dict[str, ModelPlugin] = {}
        self._plugin_order: List[str] = []

    def register(
        self,
        plugin: ModelPlugin,
        name: Optional[str] = None,
        priority: int = -1,
    ) -> None:
        """
        Register a plugin.

        Args:
            plugin: The plugin to register.
            name: Optional name override (defaults to plugin.name).
            priority: Position in priority list (-1 for end).
        """
        plugin_name = name or plugin.name

        if plugin_name in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' is already registered")

        self._plugins[plugin_name] = plugin

        if priority == -1:
            self._plugin_order.append(plugin_name)
        else:
            self._plugin_order.insert(priority, plugin_name)

    def unregister(self, name: str) -> Optional[ModelPlugin]:
        """
        Unregister a plugin by name.

        Args:
            name: Plugin name.

        Returns:
            The removed plugin, or None if not found.
        """
        if name not in self._plugins:
            return None

        plugin = self._plugins.pop(name)
        self._plugin_order.remove(name)
        return plugin

    def get(self, name: str) -> Optional[ModelPlugin]:
        """
        Get a plugin by name.

        Args:
            name: Plugin name.

        Returns:
            The plugin, or None if not found.
        """
        return self._plugins.get(name)

    def get_plugins_for(self, estimator_class: Type) -> List[ModelPlugin]:
        """
        Get all plugins that apply to an estimator class.

        Args:
            estimator_class: The estimator class to check.

        Returns:
            List of applicable plugins in priority order.
        """
        applicable = []
        for name in self._plugin_order:
            plugin = self._plugins[name]
            try:
                if plugin.applies_to(estimator_class):
                    applicable.append(plugin)
            except Exception as e:
                # Skip plugins that fail the check
                logger.debug(f"Plugin '{name}' applicability check failed: {e}")
        return applicable

    def get_plugins_for_names(self, plugin_names: List[str]) -> List[ModelPlugin]:
        """
        Get plugins by name.

        Args:
            plugin_names: List of plugin names.

        Returns:
            List of plugins in the specified order.
        """
        plugins = []
        for name in plugin_names:
            if name in self._plugins:
                plugins.append(self._plugins[name])
        return plugins

    @property
    def plugin_names(self) -> List[str]:
        """List of registered plugin names in priority order."""
        return list(self._plugin_order)

    def __len__(self) -> int:
        """Number of registered plugins."""
        return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins

    def __repr__(self) -> str:
        return f"PluginRegistry(plugins={self._plugin_order})"


# Global default registry
_default_registry: Optional[PluginRegistry] = None


def get_default_registry() -> PluginRegistry:
    """Get the default global plugin registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PluginRegistry()
        _register_default_plugins(_default_registry)
    return _default_registry


def _register_default_plugins(registry: PluginRegistry) -> None:
    """Register default plugins."""
    try:
        from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin
        from sklearn_meta.plugins.xgboost.importance import XGBImportancePlugin

        registry.register(XGBMultiplierPlugin())
        registry.register(XGBImportancePlugin())
    except ImportError:
        pass  # XGBoost plugins require xgboost package
