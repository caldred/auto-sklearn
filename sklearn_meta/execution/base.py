"""Executor: Abstract base class for execution backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Callable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class Executor(ABC):
    """
    Abstract base class for execution backends.

    Executors handle the parallel or distributed execution of tasks.
    This abstraction allows for easy swapping between local execution,
    multiprocessing, or distributed backends (e.g., Ray, Dask).
    """

    @abstractmethod
    def map(self, fn: Callable[[T], R], items: List[T]) -> List[R]:
        """
        Apply a function to a list of items.

        Args:
            fn: Function to apply to each item.
            items: List of items to process.

        Returns:
            List of results.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete.
        """
        pass

    def __enter__(self) -> Executor:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown(wait=True)

    @property
    def n_workers(self) -> int:
        """Number of workers available."""
        return 1
