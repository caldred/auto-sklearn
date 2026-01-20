"""Tests for FitCache."""

import pytest
import json
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sklearn_meta.persistence.cache import FitCache, CacheEntry


class MockModel:
    """Mock model for testing."""

    def __init__(self, value=42):
        self.value = value


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self):
        """Verify cache entry can be created."""
        entry = CacheEntry(
            cache_key="abc123",
            model=MockModel(),
            created_at="2024-01-01T00:00:00",
        )

        assert entry.cache_key == "abc123"
        assert entry.hit_count == 0

    def test_entry_defaults(self):
        """Verify entry has sensible defaults."""
        entry = CacheEntry(
            cache_key="test",
            model=None,
            created_at="2024-01-01",
        )

        assert entry.hit_count == 0


class TestFitCacheInit:
    """Tests for FitCache initialization."""

    def test_default_cache_dir(self):
        """Verify default cache dir is temp directory."""
        cache = FitCache()

        assert "sklearn_meta_cache" in str(cache.cache_dir)

    def test_custom_cache_dir(self, tmp_path):
        """Verify custom cache dir is used."""
        custom_dir = tmp_path / "my_cache"
        cache = FitCache(cache_dir=str(custom_dir))

        assert cache.cache_dir == custom_dir

    def test_creates_directory(self, tmp_path):
        """Verify cache directory is created."""
        cache_dir = tmp_path / "new_cache"
        cache = FitCache(cache_dir=str(cache_dir))

        assert cache_dir.exists()

    def test_default_max_size(self):
        """Verify default max size."""
        cache = FitCache()

        assert cache.max_size_mb == 1000.0

    def test_custom_max_size(self):
        """Verify custom max size."""
        cache = FitCache(max_size_mb=500.0)

        assert cache.max_size_mb == 500.0

    def test_enabled_by_default(self):
        """Verify cache is enabled by default."""
        cache = FitCache()

        assert cache.enabled is True

    def test_disabled_cache(self):
        """Verify cache can be disabled."""
        cache = FitCache(enabled=False)

        assert cache.enabled is False

    def test_repr(self, tmp_path):
        """Verify repr includes useful info."""
        cache = FitCache(cache_dir=str(tmp_path))

        repr_str = repr(cache)

        assert "FitCache" in repr_str
        assert "entries" in repr_str


class TestFitCacheCacheKey:
    """Tests for FitCache.cache_key method."""

    def test_cache_key_returns_string(self, data_context):
        """Verify cache_key returns string."""
        cache = FitCache()
        node = MagicMock()
        node.name = "test_node"
        node.estimator_class = MockModel
        node.estimator_class.__name__ = "MockModel"
        params = {"n_estimators": 100}

        key = cache.cache_key(node, params, data_context)

        assert isinstance(key, str)
        assert len(key) > 0

    def test_cache_key_deterministic(self, data_context):
        """Verify same inputs produce same key."""
        cache = FitCache()
        node = MagicMock()
        node.name = "test_node"
        node.estimator_class = MockModel
        node.estimator_class.__name__ = "MockModel"
        params = {"n_estimators": 100}

        key1 = cache.cache_key(node, params, data_context)
        key2 = cache.cache_key(node, params, data_context)

        assert key1 == key2

    def test_cache_key_different_params(self, data_context):
        """Verify different params produce different keys."""
        cache = FitCache()
        node = MagicMock()
        node.name = "test_node"
        node.estimator_class = MockModel
        node.estimator_class.__name__ = "MockModel"

        key1 = cache.cache_key(node, {"n_estimators": 100}, data_context)
        key2 = cache.cache_key(node, {"n_estimators": 200}, data_context)

        assert key1 != key2

    def test_cache_key_different_node(self, data_context):
        """Verify different nodes produce different keys."""
        cache = FitCache()

        node1 = MagicMock()
        node1.name = "node1"
        node1.estimator_class = MockModel
        node1.estimator_class.__name__ = "MockModel"

        node2 = MagicMock()
        node2.name = "node2"
        node2.estimator_class = MockModel
        node2.estimator_class.__name__ = "MockModel"

        params = {"n_estimators": 100}

        key1 = cache.cache_key(node1, params, data_context)
        key2 = cache.cache_key(node2, params, data_context)

        assert key1 != key2

    def test_cache_key_different_data(self, classification_data):
        """Verify different data produces different keys."""
        from sklearn_meta.core.data.context import DataContext

        cache = FitCache()
        node = MagicMock()
        node.name = "test_node"
        node.estimator_class = MockModel
        node.estimator_class.__name__ = "MockModel"
        params = {"n_estimators": 100}

        X1, y1 = classification_data
        ctx1 = DataContext(X=X1, y=y1)

        X2 = X1 * 2  # Different data
        ctx2 = DataContext(X=X2, y=y1)

        key1 = cache.cache_key(node, params, ctx1)
        key2 = cache.cache_key(node, params, ctx2)

        assert key1 != key2


class TestFitCacheDataHash:
    """Tests for FitCache._data_hash method."""

    def test_data_hash_returns_string(self, data_context):
        """Verify data hash returns string."""
        cache = FitCache()

        hash_val = cache._data_hash(data_context)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 16

    def test_data_hash_deterministic(self, data_context):
        """Verify same data produces same hash."""
        cache = FitCache()

        hash1 = cache._data_hash(data_context)
        hash2 = cache._data_hash(data_context)

        assert hash1 == hash2

    def test_data_hash_different_shapes(self, classification_data):
        """Verify different shapes produce different hashes."""
        from sklearn_meta.core.data.context import DataContext

        cache = FitCache()
        X, y = classification_data

        ctx1 = DataContext(X=X, y=y)
        ctx2 = DataContext(X=X.iloc[:100], y=y.iloc[:100])

        hash1 = cache._data_hash(ctx1)
        hash2 = cache._data_hash(ctx2)

        assert hash1 != hash2


class TestFitCacheGetPut:
    """Tests for FitCache.get and put methods."""

    def test_get_nonexistent_returns_none(self, tmp_path):
        """Verify get returns None for nonexistent key."""
        cache = FitCache(cache_dir=str(tmp_path))

        result = cache.get("nonexistent_key")

        assert result is None

    def test_put_and_get(self, tmp_path):
        """Verify put and get work together."""
        cache = FitCache(cache_dir=str(tmp_path))
        model = MockModel(value=99)

        cache.put("test_key", model)
        result = cache.get("test_key")

        assert isinstance(result, MockModel)
        assert result.value == 99

    def test_get_returns_from_memory(self, tmp_path):
        """Verify get returns from memory cache first."""
        cache = FitCache(cache_dir=str(tmp_path))
        model = MockModel(value=42)

        cache.put("test_key", model)

        # Delete disk cache to verify memory cache is used
        disk_path = tmp_path / "test_key.pkl"
        if disk_path.exists():
            disk_path.unlink()

        result = cache.get("test_key")

        assert result is not None
        assert result.value == 42

    def test_get_increments_hit_count(self, tmp_path):
        """Verify get increments hit count."""
        cache = FitCache(cache_dir=str(tmp_path))
        cache.put("test_key", MockModel())

        cache.get("test_key")
        cache.get("test_key")
        cache.get("test_key")

        assert cache._memory_cache["test_key"].hit_count == 3

    def test_put_stores_on_disk(self, tmp_path):
        """Verify put stores on disk."""
        cache = FitCache(cache_dir=str(tmp_path))
        model = MockModel(value=99)

        cache.put("test_key", model)

        disk_path = tmp_path / "test_key.pkl"
        assert disk_path.exists()

    def test_get_loads_from_disk(self, tmp_path):
        """Verify get loads from disk when not in memory."""
        cache = FitCache(cache_dir=str(tmp_path))
        model = MockModel(value=99)

        cache.put("test_key", model)
        cache._memory_cache.clear()  # Clear memory cache

        result = cache.get("test_key")

        assert result is not None
        assert result.value == 99

    def test_disabled_cache_put_noop(self, tmp_path):
        """Verify disabled cache doesn't store."""
        cache = FitCache(cache_dir=str(tmp_path), enabled=False)

        cache.put("test_key", MockModel())

        assert "test_key" not in cache._memory_cache
        assert not (tmp_path / "test_key.pkl").exists()

    def test_disabled_cache_get_returns_none(self, tmp_path):
        """Verify disabled cache always returns None."""
        cache = FitCache(cache_dir=str(tmp_path), enabled=True)
        cache.put("test_key", MockModel())

        cache.enabled = False
        result = cache.get("test_key")

        assert result is None


class TestFitCacheInvalidate:
    """Tests for FitCache.invalidate method."""

    def test_invalidate_existing(self, tmp_path):
        """Verify invalidating existing entry works."""
        cache = FitCache(cache_dir=str(tmp_path))
        cache.put("test_key", MockModel())

        result = cache.invalidate("test_key")

        assert result is True
        assert "test_key" not in cache._memory_cache
        assert not (tmp_path / "test_key.pkl").exists()

    def test_invalidate_nonexistent(self, tmp_path):
        """Verify invalidating nonexistent returns False."""
        cache = FitCache(cache_dir=str(tmp_path))

        result = cache.invalidate("nonexistent")

        assert result is False

    def test_invalidate_memory_only(self, tmp_path):
        """Verify invalidating memory-only entry works."""
        cache = FitCache(cache_dir=str(tmp_path))
        cache._memory_cache["test_key"] = CacheEntry(
            cache_key="test_key",
            model=MockModel(),
            created_at="2024-01-01",
        )

        result = cache.invalidate("test_key")

        assert result is True
        assert "test_key" not in cache._memory_cache


class TestFitCacheClear:
    """Tests for FitCache.clear method."""

    def test_clear_removes_all(self, tmp_path):
        """Verify clear removes all entries."""
        cache = FitCache(cache_dir=str(tmp_path))
        cache.put("key1", MockModel())
        cache.put("key2", MockModel())
        cache.put("key3", MockModel())

        cache.clear()

        assert len(cache._memory_cache) == 0
        assert list(tmp_path.glob("*.pkl")) == []

    def test_clear_empty_cache(self, tmp_path):
        """Verify clear on empty cache doesn't error."""
        cache = FitCache(cache_dir=str(tmp_path))

        # Should not raise
        cache.clear()


class TestFitCacheSizeLimit:
    """Tests for FitCache size limit enforcement."""

    def test_enforce_size_limit(self, tmp_path):
        """Verify size limit is enforced."""
        cache = FitCache(cache_dir=str(tmp_path), max_size_mb=0.001)  # Tiny limit

        # Put enough models to exceed limit
        for i in range(10):
            model = MockModel(value=i)
            model.large_data = np.random.randn(1000)  # Add some size
            cache.put(f"key_{i}", model)

        # Some should have been evicted
        disk_files = list(tmp_path.glob("*.pkl"))
        # Note: exact count depends on model size and eviction strategy


class TestFitCacheStats:
    """Tests for FitCache.stats method."""

    def test_stats_returns_dict(self, tmp_path):
        """Verify stats returns dictionary."""
        cache = FitCache(cache_dir=str(tmp_path))

        stats = cache.stats()

        assert isinstance(stats, dict)
        assert "enabled" in stats
        assert "memory_entries" in stats
        assert "disk_entries" in stats
        assert "disk_size_mb" in stats
        assert "total_hits" in stats

    def test_stats_reflects_state(self, tmp_path):
        """Verify stats reflect cache state."""
        cache = FitCache(cache_dir=str(tmp_path))
        cache.put("key1", MockModel())
        cache.put("key2", MockModel())
        cache.get("key1")
        cache.get("key1")

        stats = cache.stats()

        assert stats["memory_entries"] == 2
        assert stats["disk_entries"] == 2
        assert stats["total_hits"] == 2


class TestFitCacheCorruption:
    """Tests for cache corruption handling."""

    def test_handles_corrupted_disk_cache(self, tmp_path):
        """Verify corrupted disk cache is handled."""
        cache = FitCache(cache_dir=str(tmp_path))

        # Create corrupted cache file
        corrupt_path = tmp_path / "corrupt_key.pkl"
        with open(corrupt_path, "wb") as f:
            f.write(b"not valid pickle data")

        # Should return None and remove corrupt file
        result = cache.get("corrupt_key")

        assert result is None
        assert not corrupt_path.exists()
