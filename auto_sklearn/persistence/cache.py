"""FitCache: Hash-based caching for model fitting."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from auto_sklearn.core.data.context import DataContext
    from auto_sklearn.core.model.node import ModelNode


@dataclass
class CacheEntry:
    """An entry in the fit cache."""

    cache_key: str
    model: Any
    created_at: str
    hit_count: int = 0


class FitCache:
    """
    Hash-based caching for expensive model fitting operations.

    The cache key is computed from:
    - Node name
    - Hyperparameters
    - Data hash (shape + sample of values for efficiency)

    This allows skipping redundant fits when:
    - Rerunning with the same data and params
    - Evaluating the same hyperparameters during optimization
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size_mb: float = 1000.0,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the fit cache.

        Args:
            cache_dir: Directory for cache storage. Uses temp dir if None.
            max_size_mb: Maximum cache size in megabytes.
            enabled: Whether caching is enabled.
        """
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "auto_sklearn_cache")

        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.enabled = enabled

        self._memory_cache: Dict[str, CacheEntry] = {}
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(
        self,
        node: ModelNode,
        params: Dict[str, Any],
        ctx: DataContext,
    ) -> str:
        """
        Compute a cache key for a model fit.

        Args:
            node: The model node.
            params: Hyperparameters.
            ctx: Data context.

        Returns:
            Cache key string.
        """
        # Components of the key
        components = {
            "node_name": node.name,
            "estimator_class": node.estimator_class.__name__,
            "params": json.dumps(params, sort_keys=True, default=str),
            "data_hash": self._data_hash(ctx),
        }

        key_str = json.dumps(components, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _data_hash(self, ctx: DataContext) -> str:
        """
        Compute a hash of the data context.

        Uses shape and a sample of values for efficiency.
        """
        hasher = hashlib.sha256()

        # Hash shape
        hasher.update(str(ctx.X.shape).encode())
        if ctx.y is not None:
            hasher.update(str(len(ctx.y)).encode())

        # Hash sample of X values (first and last 100 rows)
        if len(ctx.X) > 0:
            n_sample = min(100, len(ctx.X))
            sample_idx = list(range(n_sample)) + list(range(-n_sample, 0))
            sample_idx = [i for i in sample_idx if i < len(ctx.X)]

            X_sample = ctx.X.iloc[sample_idx].values
            hasher.update(X_sample.tobytes())

        # Hash sample of y values
        if ctx.y is not None and len(ctx.y) > 0:
            y_sample = ctx.y.iloc[:100].values
            hasher.update(y_sample.tobytes())

        # Hash column names
        hasher.update(",".join(ctx.X.columns).encode())

        return hasher.hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        """
        Get a cached model.

        Args:
            key: Cache key.

        Returns:
            Cached model if found, None otherwise.
        """
        if not self.enabled:
            return None

        # Check memory cache first
        if key in self._memory_cache:
            self._memory_cache[key].hit_count += 1
            return self._memory_cache[key].model

        # Check disk cache
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    model = pickle.load(f)
                # Add to memory cache
                self._memory_cache[key] = CacheEntry(
                    cache_key=key,
                    model=model,
                    created_at=datetime.now().isoformat(),
                    hit_count=1,
                )
                return model
            except Exception:
                # Cache corruption - remove entry
                cache_path.unlink(missing_ok=True)

        return None

    def put(self, key: str, model: Any) -> None:
        """
        Store a model in the cache.

        Args:
            key: Cache key.
            model: Model to cache.
        """
        if not self.enabled:
            return

        # Store in memory cache
        self._memory_cache[key] = CacheEntry(
            cache_key=key,
            model=model,
            created_at=datetime.now().isoformat(),
        )

        # Store on disk
        cache_path = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(model, f)
        except Exception:
            # If serialization fails, just skip disk cache
            pass

        # Check size limits
        self._enforce_size_limit()

    def invalidate(self, key: str) -> bool:
        """
        Remove an entry from the cache.

        Args:
            key: Cache key.

        Returns:
            True if entry was found and removed.
        """
        removed = False

        if key in self._memory_cache:
            del self._memory_cache[key]
            removed = True

        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            cache_path.unlink()
            removed = True

        return removed

    def clear(self) -> None:
        """Clear all cached entries."""
        self._memory_cache.clear()

        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def _enforce_size_limit(self) -> None:
        """Remove oldest entries if cache exceeds size limit."""
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.pkl")
        )
        max_size_bytes = self.max_size_mb * 1024 * 1024

        if total_size <= max_size_bytes:
            return

        # Sort by modification time and remove oldest
        cache_files = sorted(
            self.cache_dir.glob("*.pkl"),
            key=lambda f: f.stat().st_mtime,
        )

        for cache_file in cache_files:
            if total_size <= max_size_bytes * 0.8:  # Free up to 80%
                break

            file_size = cache_file.stat().st_size
            cache_file.unlink()
            total_size -= file_size

            # Also remove from memory cache
            key = cache_file.stem
            self._memory_cache.pop(key, None)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.pkl"))
        disk_size = sum(f.stat().st_size for f in disk_files)

        return {
            "enabled": self.enabled,
            "memory_entries": len(self._memory_cache),
            "disk_entries": len(disk_files),
            "disk_size_mb": disk_size / (1024 * 1024),
            "total_hits": sum(e.hit_count for e in self._memory_cache.values()),
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"FitCache(entries={stats['memory_entries'] + stats['disk_entries']}, "
            f"size_mb={stats['disk_size_mb']:.1f})"
        )
