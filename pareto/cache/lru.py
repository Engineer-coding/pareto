"""
A bounded LRU (Least Recently Used) cache implemented on top of
`collections.OrderedDict`.

We roll our own instead of `functools.lru_cache` because:
    - We want explicit insertion order control (move_to_end on get)
    - We need iteration over entries (for semantic similarity lookup)
    - We track hit/miss counters for observability
    - Type-generic over K and V (decorator-based caches are function-bound)

Used by SemanticCache as the underlying storage for query → response
entries. Eviction policy: when size exceeds capacity, oldest entry
(least recently get'd or add'd) is dropped.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Iterator, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Bounded LRU cache with observability counters."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._data: OrderedDict[K, V] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ── basic ops ─────────────────────────────────────────────────────────
    def get(self, key: K) -> V | None:
        """Get a value, marking it as recently used. Tracks hit/miss."""
        if key in self._data:
            self._data.move_to_end(key)
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None

    def peek(self, key: K) -> V | None:
        """Get without affecting recency or hit/miss counters."""
        return self._data.get(key)

    def add(self, key: K, value: V) -> None:
        """Insert or update. Evicts oldest if over capacity."""
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
            return
        if len(self._data) >= self.capacity:
            self._data.popitem(last=False)  # evict oldest (LRU)
        self._data[key] = value

    def touch(self, key: K) -> bool:
        """Mark a key as recently used without changing its value.
        Returns True if key was present."""
        if key in self._data:
            self._data.move_to_end(key)
            return True
        return False

    def remove(self, key: K) -> V | None:
        """Remove and return a value. None if absent."""
        return self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()
        self._hits = 0
        self._misses = 0

    # ── iteration ─────────────────────────────────────────────────────────
    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: K) -> bool:
        return key in self._data
    
    def __bool__(self) -> bool:
        return True

    # ── observability ─────────────────────────────────────────────────────
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        """Snapshot of cache state. Plays well with pareto stats."""
        return {
            "size": len(self._data),
            "capacity": self.capacity,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    def __repr__(self) -> str:
        return (
            f"LRUCache(size={len(self)}, capacity={self.capacity}, "
            f"hit_rate={self.hit_rate:.2%})"
        )