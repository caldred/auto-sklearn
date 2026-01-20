"""Execution backends for parallel and distributed computing."""

from sklearn_meta.execution.base import Executor
from sklearn_meta.execution.local import LocalExecutor

__all__ = ["Executor", "LocalExecutor"]
