"""LocalExecutor: Local execution backend using joblib or concurrent.futures."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, TypeVar

from auto_sklearn.execution.base import Executor

T = TypeVar("T")
R = TypeVar("R")


class LocalExecutor(Executor):
    """
    Local execution backend.

    Uses joblib for parallel execution when available, falls back to
    ThreadPoolExecutor for simple parallelism.
    """

    def __init__(
        self,
        n_workers: int = 1,
        backend: str = "threading",
        prefer: str = "processes",
    ) -> None:
        """
        Initialize the local executor.

        Args:
            n_workers: Number of parallel workers.
                      Use -1 for all CPUs, 0 for sequential execution.
            backend: Execution backend ("threading", "loky", "multiprocessing").
            prefer: Preferred backend for joblib ("threads" or "processes").
        """
        if n_workers == -1:
            n_workers = os.cpu_count() or 1
        elif n_workers == 0:
            n_workers = 1

        self._n_workers = n_workers
        self._backend = backend
        self._prefer = prefer
        self._thread_pool: Optional[ThreadPoolExecutor] = None

    @property
    def n_workers(self) -> int:
        """Number of workers."""
        return self._n_workers

    def map(self, fn: Callable[[T], R], items: List[T]) -> List[R]:
        """
        Apply a function to a list of items in parallel.

        Args:
            fn: Function to apply.
            items: Items to process.

        Returns:
            List of results.
        """
        if not items:
            return []

        if self._n_workers == 1:
            return [fn(item) for item in items]

        try:
            from joblib import Parallel, delayed

            results = Parallel(
                n_jobs=self._n_workers,
                backend=self._backend,
                prefer=self._prefer,
            )(delayed(fn)(item) for item in items)
            return list(results)

        except ImportError:
            # Fall back to ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self._n_workers) as pool:
                return list(pool.map(fn, items))

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future[R]:
        """
        Submit a function for asynchronous execution.

        Args:
            fn: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Future representing the pending result.
        """
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self._n_workers)

        return self._thread_pool.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks.
        """
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=wait)
            self._thread_pool = None

    def __repr__(self) -> str:
        return f"LocalExecutor(n_workers={self._n_workers}, backend={self._backend})"


class SequentialExecutor(Executor):
    """
    Sequential (non-parallel) executor.

    Useful for debugging or when parallelism is not needed.
    """

    def map(self, fn: Callable[[T], R], items: List[T]) -> List[R]:
        """Apply function sequentially."""
        return [fn(item) for item in items]

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future[R]:
        """Execute function immediately and return completed future."""
        future: Future[R] = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait: bool = True) -> None:
        """No-op for sequential executor."""
        pass

    @property
    def n_workers(self) -> int:
        """Always 1 worker."""
        return 1

    def __repr__(self) -> str:
        return "SequentialExecutor()"
