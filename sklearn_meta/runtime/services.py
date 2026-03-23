"""RuntimeServices: Service wiring for training runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from sklearn_meta.search.backends.base import SearchBackend
from sklearn_meta.search.backends.optuna import OptunaBackend

if TYPE_CHECKING:
    from sklearn_meta.audit.logger import AuditLogger
    from sklearn_meta.execution.base import Executor
    from sklearn_meta.persistence.cache import FitCache
    from sklearn_meta.plugins.registry import PluginRegistry


@dataclass
class RuntimeServices:
    """Services needed during a training run."""

    search_backend: SearchBackend
    executor: Optional[Executor] = None
    plugin_registry: Optional[PluginRegistry] = None
    audit_logger: Optional[AuditLogger] = None
    fit_cache: Optional[FitCache] = None

    @classmethod
    def default(cls) -> RuntimeServices:
        """OptunaBackend only. No plugins, no executor, no cache, no logger."""
        return cls(search_backend=OptunaBackend())
