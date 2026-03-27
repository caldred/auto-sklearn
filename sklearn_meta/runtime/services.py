"""RuntimeServices: Service wiring for training runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from sklearn_meta.search.backends.base import SearchBackend
from sklearn_meta.search.backends.optuna import OptunaBackend

if TYPE_CHECKING:
    from sklearn_meta.audit.logger import AuditLogger
    from sklearn_meta.execution.training import DispatchListener, TrainingDispatcher
    from sklearn_meta.persistence.cache import FitCache
    from sklearn_meta.plugins.registry import PluginRegistry


@dataclass
class RuntimeServices:
    """Services needed during a training run.

    The ``training_dispatcher`` is the primary extension point for cloud
    and distributed execution.  Third-party packages should implement the
    :class:`~sklearn_meta.execution.training.TrainingDispatcher` protocol
    and pass an instance here.  All other fields are optional supporting
    services.
    """

    search_backend: SearchBackend
    training_dispatcher: Optional["TrainingDispatcher"] = None
    plugin_registry: Optional["PluginRegistry"] = None
    audit_logger: Optional["AuditLogger"] = None
    fit_cache: Optional["FitCache"] = None
    dispatch_listener: Optional["DispatchListener"] = None

    @classmethod
    def default(cls) -> RuntimeServices:
        """OptunaBackend only. No plugins, no cache, no logger."""
        return cls(search_backend=OptunaBackend())
