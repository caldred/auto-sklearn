"""Tests for FitCache."""

from unittest.mock import MagicMock

import numpy as np

from sklearn_meta.persistence.cache import FitCache, CacheEntry


class MockModel:
    """Mock model for testing."""

    def __init__(self, value=42):
        self.value = value


class TestFitCacheInit:
    """Tests for FitCache initialization."""

    def test_creates_directory(self, tmp_path):
        """Verify cache directory is created on init."""
        cache_dir = tmp_path / "new_cache"
        FitCache(cache_dir=str(cache_dir))

        assert cache_dir.exists()


class TestFitCacheCacheKey:
    """Tests for FitCache.cache_key method."""

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
        from sklearn_meta.data.view import DataView

        cache = FitCache()
        node = MagicMock()
        node.name = "test_node"
        node.estimator_class = MockModel
        node.estimator_class.__name__ = "MockModel"
        params = {"n_estimators": 100}

        X1, y1 = classification_data
        ctx1 = DataView.from_Xy(X1, y1)

        X2 = X1 * 2  # Different data
        ctx2 = DataView.from_Xy(X2, y1)

        key1 = cache.cache_key(node, params, ctx1)
        key2 = cache.cache_key(node, params, ctx2)

        assert key1 != key2


class TestFitCacheDataHash:
    """Tests for FitCache._data_hash method."""

    def test_data_hash_deterministic(self, data_context):
        """Verify same data produces same hash."""
        cache = FitCache()

        hash1 = cache._data_hash(data_context)
        hash2 = cache._data_hash(data_context)

        assert hash1 == hash2

    def test_data_hash_different_shapes(self, classification_data):
        """Verify different shapes produce different hashes."""
        from sklearn_meta.data.view import DataView

        cache = FitCache()
        X, y = classification_data

        ctx1 = DataView.from_Xy(X, y)
        ctx2 = DataView.from_Xy(X.iloc[:100], y.iloc[:100])

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


class TestFitCacheStats:
    """Tests for FitCache.stats method."""

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


class TestFitCacheSizeLimitEviction:
    """Verify that size-limit enforcement actually removes entries."""

    def test_evicts_oldest_entries_when_over_limit(self, tmp_path):
        """After exceeding max_size_mb, the oldest disk entries are removed."""
        # Each entry with 1000 doubles ≈ 0.01 MB; set limit so ~3-4 fit
        cache = FitCache(cache_dir=str(tmp_path), max_size_mb=0.03)

        for i in range(10):
            model = MockModel(value=i)
            model.large_data = np.random.randn(1000)
            cache.put(f"key_{i}", model)

        disk_files = list(tmp_path.glob("*.pkl"))
        assert len(disk_files) < 10, "Expected some entries to be evicted"
        assert len(disk_files) > 0, "Expected at least some entries to survive"

    def test_eviction_keeps_most_recent(self, tmp_path):
        """Most-recently-added entries survive eviction (LRU by mtime)."""
        cache = FitCache(cache_dir=str(tmp_path), max_size_mb=0.03)

        for i in range(10):
            model = MockModel(value=i)
            model.large_data = np.random.randn(1000)
            cache.put(f"key_{i}", model)

        # The last entry should still be accessible
        last = cache.get("key_9")
        assert last is not None
        assert last.value == 9


class TestFitCacheCorruptionRecovery:
    """Verify the cache remains functional after encountering corruption."""

    def test_put_works_after_corruption(self, tmp_path):
        """Cache can store and retrieve new entries after hitting corruption."""
        cache = FitCache(cache_dir=str(tmp_path))

        corrupt_path = tmp_path / "bad_key.pkl"
        corrupt_path.write_bytes(b"not valid pickle")

        # Hit the corruption
        assert cache.get("bad_key") is None
        assert not corrupt_path.exists()

        # Cache should still work for new entries
        cache.put("good_key", MockModel(value=77))
        result = cache.get("good_key")
        assert result is not None
        assert result.value == 77

    def test_get_from_disk_after_memory_eviction_survives_corruption(self, tmp_path):
        """Corrupted file is removed; next put/get works normally."""
        cache = FitCache(cache_dir=str(tmp_path))

        cache.put("k1", MockModel(value=1))

        # Corrupt the disk file while memory cache still has the entry
        disk_path = tmp_path / "k1.pkl"
        disk_path.write_bytes(b"garbage")

        # Memory cache still works
        assert cache.get("k1").value == 1

        # Clear memory cache to force disk read → triggers corruption handler
        cache._memory_cache.clear()
        assert cache.get("k1") is None
        assert not disk_path.exists()


class TestFitCacheDataHashCollisions:
    """Verify that data hashes distinguish data that differs only in values."""

    def test_same_shape_different_values_produce_different_hashes(self, classification_data):
        """Two DataViews with same shape but different X values hash differently."""
        from sklearn_meta.data.view import DataView

        cache = FitCache()
        X, y = classification_data

        ctx1 = DataView.from_Xy(X, y)
        ctx2 = DataView.from_Xy(X * 2.0, y)

        h1 = cache._data_hash(ctx1)
        h2 = cache._data_hash(ctx2)
        assert h1 != h2

    def test_same_X_different_y_produce_different_hashes(self, classification_data):
        """DataViews with same X but different y hash differently."""
        from sklearn_meta.data.view import DataView

        cache = FitCache()
        X, y = classification_data

        ctx1 = DataView.from_Xy(X, y)
        ctx2 = DataView.from_Xy(X, 1 - y)  # Flip labels

        h1 = cache._data_hash(ctx1)
        h2 = cache._data_hash(ctx2)
        assert h1 != h2

    def test_same_values_different_columns_produce_different_hashes(self, classification_data):
        """DataViews with same values but different column names hash differently."""
        import pandas as pd
        from sklearn_meta.data.view import DataView

        cache = FitCache()
        X, y = classification_data

        X_renamed = X.copy()
        X_renamed.columns = [f"renamed_{i}" for i in range(X.shape[1])]

        ctx1 = DataView.from_Xy(X, y)
        ctx2 = DataView.from_Xy(X_renamed, y)

        h1 = cache._data_hash(ctx1)
        h2 = cache._data_hash(ctx2)
        assert h1 != h2


