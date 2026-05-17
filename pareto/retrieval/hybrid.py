"""
HybridRetriever — combine dense vector retrieval and BM25 via RRF.

Pareto's first multi-stage retriever. Two systems run in parallel, then
their ranked lists are fused with Reciprocal Rank Fusion (see rrf.py).

Workflow:
    1. Dense:  embed query → FAISS HNSW search → top-N candidates
    2. Sparse: tokenize query → BM25 score → top-N candidates
    3. Fuse: rank-based RRF, optionally weighted
    4. Return top-k fused hits with provenance (rank in each system)

Over-fetch parameter:
    We retrieve `k * fetch_multiplier` candidates from each system and
    fuse them. This is critical — RRF only sees the candidates each
    system surfaced; a relevant doc invisible to BOTH systems can't be
    rescued. fetch_multiplier=5 is a pragmatic default (k=5 → 25 candidates
    per system, 50 union, robust to single-system blind spots).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pareto.retrieval.rrf import reciprocal_rank_fusion

if TYPE_CHECKING:
    from pareto.indexing.indexer import Indexer
    from pareto.indexing.models import VectorRecord
    from pareto.retrieval.bm25 import BM25Ranker


@dataclass
class HybridHit:
    """
    A single fused result with provenance.

    `dense_rank` and `bm25_rank` are useful for debugging and observability:
    you can see whether a doc was found because of semantic similarity,
    keyword match, or both.
    """
    record: "VectorRecord"
    score: float                # the fused RRF score
    dense_rank: int | None      # 1-indexed rank in dense results, None if absent
    bm25_rank: int | None       # 1-indexed rank in BM25 results, None if absent

    @property
    def found_by_both(self) -> bool:
        return self.dense_rank is not None and self.bm25_rank is not None


class HybridRetriever:
    """Dense + BM25 + RRF fusion. Drop-in retrieval layer."""

    def __init__(
        self,
        indexer: "Indexer",
        bm25_ranker: "BM25Ranker",
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        fetch_multiplier: int = 5,
    ):
        """
        Args:
            indexer: a built Indexer (embedder + vector store).
            bm25_ranker: a built BM25Ranker on the same record set.
            rrf_k: RRF smoothing constant (default 60).
            dense_weight: weight for dense system in RRF (default 1.0).
            sparse_weight: weight for BM25 system in RRF (default 1.0).
            fetch_multiplier: each system fetches `k * fetch_multiplier`
                candidates before fusion.
        """
        self.indexer = indexer
        self.bm25 = bm25_ranker
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.fetch_multiplier = fetch_multiplier

        # Sanity check — both systems must index the same records
        store_size = self.indexer.store.size
        bm25_size = self.bm25.doc_count
        if store_size != bm25_size:
            raise ValueError(
                f"Indexer and BM25Ranker have different doc counts "
                f"({store_size} vs {bm25_size}). Are they built from the same records?"
            )

    def search(self, query: str, k: int = 5) -> list[HybridHit]:
        """
        Hybrid retrieval: dense + BM25, fused via RRF, top-k returned.

        Args:
            query: natural-language query string.
            k: number of final results to return.
        """
        fetch_k = k * self.fetch_multiplier

        # ── 1. Dense retrieval ──
        q_vec = self.indexer.embedder.encode_query(query)
        dense_results = self.indexer.store.search(q_vec, k=fetch_k)
        dense_chunk_ids = [r.record.chunk_id for r in dense_results]
        # rank_map for quick lookup later
        dense_rank_map = {cid: i + 1 for i, cid in enumerate(dense_chunk_ids)}

        # ── 2. BM25 retrieval ──
        bm25_results = self.bm25.search(query, k=fetch_k)
        bm25_chunk_ids = [h.record.chunk_id for h in bm25_results]
        bm25_rank_map = {cid: i + 1 for i, cid in enumerate(bm25_chunk_ids)}

        # ── 3. RRF fusion ──
        fused = reciprocal_rank_fusion(
            [dense_chunk_ids, bm25_chunk_ids],
            k=self.rrf_k,
            weights=[self.dense_weight, self.sparse_weight],
        )

        # ── 4. Build HybridHit results, top-k ──
        results: list[HybridHit] = []
        for chunk_id, fused_score in fused[:k]:
            record = self.indexer.store.get(chunk_id)
            if record is None:
                continue  # shouldn't happen if BM25 and store agree, but defensive
            results.append(HybridHit(
                record=record,
                score=fused_score,
                dense_rank=dense_rank_map.get(chunk_id),
                bm25_rank=bm25_rank_map.get(chunk_id),
            ))

        return results

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(docs={self.indexer.store.size}, "
            f"rrf_k={self.rrf_k}, "
            f"weights=(dense={self.dense_weight}, sparse={self.sparse_weight}), "
            f"fetch_multiplier={self.fetch_multiplier})"
        )