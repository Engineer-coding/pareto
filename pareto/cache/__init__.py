"""Semantic cache — embedding-based query deduplication."""

from pareto.cache.lru import LRUCache
from pareto.cache.semantic_cache import CacheEntry, CacheHit, SemanticCache

__all__ = [
    "LRUCache",
    "CacheEntry",
    "CacheHit",
    "SemanticCache",
]