"""
SemanticCache — embedding-based query deduplication.

A query that's semantically similar to a previously-cached one returns
the cached response instantly, without invoking the LLM. The similarity
threshold is tunable; default 0.92 was chosen for the E5-multilingual
embedder based on empirical paraphrase distances:

    "What is GDPR?" ↔ "Explain GDPR"      → ~0.93  (hit at 0.92)
    "What is GDPR?" ↔ "GDPR nedir?"       → ~0.91  (miss at 0.92, EN↔TR drift)
    "What is GDPR?" ↔ "What is HIPAA?"    → ~0.75  (miss, clearly different)

Cache key is composite: (query_embedding, retriever, top_k, model).
Two queries with identical wording but different retrievers cannot share
a cache entry — they may have produced different answers grounded in
different chunks.

Invalidation strategy — chunk-level staleness:
    Each entry remembers which chunk_ids contributed to its answer.
    On lookup, we optionally verify those chunks still exist in the
    current vector store (via valid_chunk_ids). If any is gone — corpus
    changed — the entry is stale and skipped.

    This means corpus updates don't require manual cache clears. Week 1's
    deterministic chunk IDs (SHA-256 over (source, content)) make this
    trivial: same content → same id → stale check is fast.

Performance: lookup is O(N) over matching entries with NumPy-vectorized
cosine similarity. For N=1000 entries × dim=384, ~0.5 ms on CPU.
"""

from __future__ import annotations

import time
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np

from pareto.cache.lru import LRUCache




@dataclass
class CacheEntry:
    """One cached query result with provenance + observability metadata."""
    query: str
    embedding: np.ndarray          # 1-D float32, L2-normalized
    retriever: str                 # e.g. 'HybridRetriever'
    top_k: int
    model: str                     # e.g. 'ollama/llama3.2:3b'
    answer: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    generation_latency_ms: int
    chunks_used_ids: list[str]     # for staleness check
    citations: list[str]           # source filenames, for downstream display
    timestamp_unix: float          # when first cached
    access_count: int = 1          # how many times served from cache


@dataclass
class CacheHit:
    """Result of a successful cache lookup."""
    entry: CacheEntry
    similarity: float              # cosine sim, ≥ threshold


class SemanticCache:
    """Embedding-based query cache with cosine similarity threshold."""

    def __init__(
        self,
        capacity: int = 1000,
        threshold: float = 0.93,
    ):
        """
        Args:
            capacity: max cached entries (LRU eviction beyond this).
            threshold: cosine similarity for a cache hit. Default 0.93,
                chosen via Week 3 threshold sweep (scripts/tune_threshold.py):
                0.93 strictly dominates 0.92 (6 true / 5 false vs 5 true /
                6 false), preserves 24.4% hit rate, and most "false" hits
                return acceptable answers (similar questions share sources).
                For quality-critical use, 0.95 gives zero strict false hits
                at the cost of recall (catches 4/15 paraphrases).
        """
        if not 0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.threshold = threshold
        # Entries keyed by a monotonic integer id (we look up by similarity,
        # not by key — the int id is just an LRU bookkeeping primitive).
        self._entries: LRUCache[int, CacheEntry] = LRUCache(capacity)
        self._next_id = 0
        # Cache-level counters (separate from LRU's get/miss, because
        # *no entry exists* is different from *no similar entry above threshold*)
        self._semantic_hits = 0
        self._semantic_misses = 0

    # ── lookup ────────────────────────────────────────────────────────────
    def lookup(
        self,
        query_embedding: np.ndarray,
        retriever: str,
        top_k: int,
        model: str,
        valid_chunk_ids: set[str] | None = None,
    ) -> CacheHit | None:
        """
        Find the best matching cached entry.

        Args:
            query_embedding: 1-D L2-normalized float32 array (dim must match
                cached entries' embedder).
            retriever: name of the retriever (filter — exact match required).
            top_k: chunks retrieved (filter — exact match required).
            model: LLM model name (filter — exact match required).
            valid_chunk_ids: if provided, entries whose chunks_used_ids
                are not all in this set are treated as stale and skipped.

        Returns:
            CacheHit if best matching entry has similarity ≥ threshold AND
            is not stale. None otherwise.
        """
        if len(self._entries) == 0:
            self._semantic_misses += 1
            return None

        # Filter to entries that match the pipeline config exactly.
        # We use peek (no recency update) — only the winner gets touched.
        matching: list[tuple[int, CacheEntry]] = [
            (eid, entry)
            for eid, entry in self._entries.items()
            if entry.retriever == retriever
            and entry.top_k == top_k
            and entry.model == model
        ]

        if not matching:
            self._semantic_misses += 1
            return None

        # Vectorized cosine similarity. Both query and cached embeddings
        # are L2-normalized → dot product = cosine similarity.
        embs = np.stack([e.embedding for _, e in matching])  # (N, dim)
        q = query_embedding.astype(np.float32)
        sims = embs @ q  # (N,)

        # Sort by similarity descending; we may need to skip stale entries
        ranked_indices = np.argsort(-sims)  # best first

        for idx in ranked_indices:
            sim = float(sims[idx])
            if sim < self.threshold:
                break  # nothing else will pass either

            eid, entry = matching[int(idx)]

            # Staleness check
            if valid_chunk_ids is not None:
                if not all(cid in valid_chunk_ids for cid in entry.chunks_used_ids):
                    continue  # stale — try next candidate

            # Winner — mark as recently used and increment access
            self._entries.touch(eid)
            entry.access_count += 1
            self._semantic_hits += 1
            return CacheHit(entry=entry, similarity=sim)

        # No entry passed both threshold and staleness
        self._semantic_misses += 1
        return None

    # ── add ───────────────────────────────────────────────────────────────
    def add(
        self,
        query: str,
        query_embedding: np.ndarray,
        retriever: str,
        top_k: int,
        model: str,
        answer: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        generation_latency_ms: int = 0,
        chunks_used_ids: list[str] | None = None,
        citations: list[str] | None = None,
    ) -> int:
        """Store a new entry. Returns the assigned entry id."""
        entry = CacheEntry(
            query=query,
            embedding=query_embedding.astype(np.float32),
            retriever=retriever,
            top_k=top_k,
            model=model,
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            generation_latency_ms=generation_latency_ms,
            chunks_used_ids=list(chunks_used_ids or []),
            citations=list(citations or []),
            timestamp_unix=time.time(),
        )
        eid = self._next_id
        self._next_id += 1
        self._entries.add(eid, entry)
        return eid

    # ── housekeeping ──────────────────────────────────────────────────────
    def clear(self) -> None:
        self._entries.clear()
        self._semantic_hits = 0
        self._semantic_misses = 0

    def evict_stale(self, valid_chunk_ids: set[str]) -> int:
        """
        Proactively remove entries whose chunks are no longer in the store.
        Returns the number of entries evicted. Useful after corpus updates.
        """
        stale_ids: list[int] = [
            eid
            for eid, entry in self._entries.items()
            if not all(cid in valid_chunk_ids for cid in entry.chunks_used_ids)
        ]
        for eid in stale_ids:
            self._entries.remove(eid)
        return len(stale_ids)

    def __len__(self) -> int:
        return len(self._entries)
    
    def __bool__(self) -> bool:
        """Always truthy — instance presence ≠ emptiness."""
        return True

    # ── observability ─────────────────────────────────────────────────────
    @property
    def hit_rate(self) -> float:
        total = self._semantic_hits + self._semantic_misses
        return self._semantic_hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._entries),
            "capacity": self.capacity,
            "threshold": self.threshold,
            "hits": self._semantic_hits,
            "misses": self._semantic_misses,
            "hit_rate": self.hit_rate,
        }
    
    # ── persistence ───────────────────────────────────────────────────────
    SAVE_FORMAT_VERSION = 1

    def save(self, path: Path | str) -> None:
        """
        Atomic save to disk via pickle. Writes to <path>.tmp first, then
        renames — partial writes never leave a corrupted cache on disk.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": self.SAVE_FORMAT_VERSION,
            "capacity": self.capacity,
            "threshold": self.threshold,
            "entries": list(self._entries.items()),  # preserve LRU order
            "next_id": self._next_id,
            "hits": self._semantic_hits,
            "misses": self._semantic_misses,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic on POSIX, near-atomic on Windows

    @classmethod
    def load(cls, path: Path | str) -> "SemanticCache":
        """
        Load a previously saved cache. Returns a fresh empty cache if the
        file doesn't exist or has an incompatible format (graceful fallback).
        """
        path = Path(path)
        if not path.exists():
            return cls()

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            import sys
            print(
                f"[pareto-cache] could not load {path}: {e}. "
                f"Starting fresh.", file=sys.stderr,
            )
            return cls()

        if state.get("version") != cls.SAVE_FORMAT_VERSION:
            import sys
            print(
                f"[pareto-cache] {path} format v{state.get('version')} "
                f"!= expected v{cls.SAVE_FORMAT_VERSION}. Starting fresh.",
                file=sys.stderr,
            )
            return cls()

        cache = cls(
            capacity=state["capacity"],
            threshold=state["threshold"],
        )
        cache._next_id = state["next_id"]
        cache._semantic_hits = state["hits"]
        cache._semantic_misses = state["misses"]
        # Restore entries preserving LRU order
        for eid, entry in state["entries"]:
            cache._entries.add(eid, entry)
        return cache

    def __repr__(self) -> str:
        return (
            f"SemanticCache(size={len(self)}, "
            f"capacity={self.capacity}, "
            f"threshold={self.threshold}, "
            f"hit_rate={self.hit_rate:.2%})"
        )