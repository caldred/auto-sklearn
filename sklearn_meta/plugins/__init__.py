"""Plugin system for model-specific operations."""

from sklearn_meta.plugins.base import ModelPlugin
from sklearn_meta.plugins.registry import PluginRegistry

__all__ = ["ModelPlugin", "PluginRegistry"]
