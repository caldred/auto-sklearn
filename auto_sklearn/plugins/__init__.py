"""Plugin system for model-specific operations."""

from auto_sklearn.plugins.base import ModelPlugin
from auto_sklearn.plugins.registry import PluginRegistry

__all__ = ["ModelPlugin", "PluginRegistry"]
