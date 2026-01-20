"""Tests for Executor abstract base class."""

import pytest
from abc import ABC
from concurrent.futures import Future

from auto_sklearn.execution.base import Executor


class ConcreteExecutor(Executor):
    """Minimal concrete implementation for testing."""

    def __init__(self, n_workers: int = 2):
        self._workers = n_workers
        self._shutdown_called = False

    def map(self, fn, items):
        return [fn(item) for item in items]

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait=True):
        self._shutdown_called = True

    @property
    def n_workers(self):
        return self._workers


class TestExecutorAbstract:
    """Tests for Executor abstract base class."""

    def test_executor_is_abstract(self):
        """Verify Executor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Executor()

    def test_map_must_be_implemented(self):
        """Verify map is abstract."""

        class PartialExecutor(Executor):
            def submit(self, fn, *args, **kwargs):
                pass

            def shutdown(self, wait=True):
                pass

        with pytest.raises(TypeError):
            PartialExecutor()

    def test_submit_must_be_implemented(self):
        """Verify submit is abstract."""

        class PartialExecutor(Executor):
            def map(self, fn, items):
                pass

            def shutdown(self, wait=True):
                pass

        with pytest.raises(TypeError):
            PartialExecutor()

    def test_shutdown_must_be_implemented(self):
        """Verify shutdown is abstract."""

        class PartialExecutor(Executor):
            def map(self, fn, items):
                pass

            def submit(self, fn, *args, **kwargs):
                pass

        with pytest.raises(TypeError):
            PartialExecutor()


class TestConcreteExecutor:
    """Tests for concrete Executor implementation."""

    def test_map_applies_function(self):
        """Verify map applies function to all items."""
        executor = ConcreteExecutor()

        result = executor.map(lambda x: x * 2, [1, 2, 3, 4])

        assert result == [2, 4, 6, 8]

    def test_map_empty_list(self):
        """Verify map handles empty list."""
        executor = ConcreteExecutor()

        result = executor.map(lambda x: x, [])

        assert result == []

    def test_map_preserves_order(self):
        """Verify map preserves item order."""
        executor = ConcreteExecutor()

        result = executor.map(str, [5, 4, 3, 2, 1])

        assert result == ["5", "4", "3", "2", "1"]

    def test_submit_returns_future(self):
        """Verify submit returns a Future."""
        executor = ConcreteExecutor()

        future = executor.submit(lambda: 42)

        assert isinstance(future, Future)

    def test_submit_future_result(self):
        """Verify future contains correct result."""
        executor = ConcreteExecutor()

        future = executor.submit(lambda x, y: x + y, 3, 4)

        assert future.result() == 7

    def test_submit_with_kwargs(self):
        """Verify submit passes kwargs correctly."""
        executor = ConcreteExecutor()

        def fn(a, b=10):
            return a + b

        future = executor.submit(fn, 5, b=20)

        assert future.result() == 25

    def test_submit_exception_captured(self):
        """Verify submit captures exceptions in future."""
        executor = ConcreteExecutor()

        def failing_fn():
            raise ValueError("Test error")

        future = executor.submit(failing_fn)

        with pytest.raises(ValueError, match="Test error"):
            future.result()

    def test_shutdown_callable(self):
        """Verify shutdown can be called."""
        executor = ConcreteExecutor()

        executor.shutdown(wait=True)

        assert executor._shutdown_called

    def test_n_workers_property(self):
        """Verify n_workers property returns correct value."""
        executor = ConcreteExecutor(n_workers=4)

        assert executor.n_workers == 4


class TestExecutorDefaultMethods:
    """Tests for Executor default method implementations."""

    def test_default_n_workers(self):
        """Verify default n_workers is 1."""
        # Access via class attribute
        assert Executor.n_workers.fget(ConcreteExecutor(n_workers=1)) == 1

    def test_is_distributed_default_false(self):
        """Verify is_distributed returns False by default."""
        executor = ConcreteExecutor()

        assert executor.is_distributed() is False


class TestExecutorContextManager:
    """Tests for Executor context manager protocol."""

    def test_context_manager_enter(self):
        """Verify __enter__ returns self."""
        executor = ConcreteExecutor()

        with executor as e:
            assert e is executor

    def test_context_manager_exit_calls_shutdown(self):
        """Verify __exit__ calls shutdown."""
        executor = ConcreteExecutor()

        with executor:
            pass

        assert executor._shutdown_called

    def test_context_manager_exit_on_exception(self):
        """Verify __exit__ calls shutdown even on exception."""
        executor = ConcreteExecutor()

        try:
            with executor:
                raise ValueError("Test")
        except ValueError:
            pass

        assert executor._shutdown_called

    def test_context_manager_with_work(self):
        """Verify context manager works with actual work."""
        executor = ConcreteExecutor()

        with executor as e:
            result = e.map(lambda x: x ** 2, [1, 2, 3])

        assert result == [1, 4, 9]
        assert executor._shutdown_called
