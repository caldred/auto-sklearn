"""Execution backends for parallel and distributed computing."""

from auto_sklearn.execution.base import Executor
from auto_sklearn.execution.local import LocalExecutor

__all__ = ["Executor", "LocalExecutor"]
