"""
DenseRetriever — thin adapter wrapping Indexer (embedder + vector store)
to expose the unified retrieval interface used by BenchmarkRunner.

This keeps the runner agnostic about which retriever it's testing —
Week 5 (HNSW tuning), Week 7 (knowledge-graph augmented), and any
future retriever just need to implement `search(query, k) -> list[Hit]`
where each Hit has `.record` and `.score`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pareto.indexing.indexer import Indexer
    from pareto.indexing.models import SearchResult


class DenseRetriever:
    """Wraps Indexer (embedder + vector store) as a unified retriever."""

    def __init__(self, indexer: "Indexer"):
        self.indexer = indexer

    def search(self, query: str, k: int = 5) -> list["SearchResult"]:
        q_vec = self.indexer.embedder.encode_query(query)
        return self.indexer.store.search(q_vec, k=k)

    def __repr__(self) -> str:
        return f"DenseRetriever(store_size={self.indexer.store.size})"