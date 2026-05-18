"""
RAG request / response data models.

RAGResponse wraps everything that happened during a single Q&A turn:
the question, the answer, the chunks retrieved, the LLM stats, and the
end-to-end latency breakdown. Week 2's observability layer ingests this
directly.

Design note — generic `retrieved` field:
    `retrieved` accepts any list of objects that expose `.record` (with
    `.content` and `.source` attributes) and `.score`. This includes
    SearchResult (dense), BM25Hit (sparse), HybridHit (RRF fusion), and
    any future retriever's hit type. We deliberately use `list[Any]` +
    arbitrary_types_allowed to keep the model extensible: each retriever
    can carry its own metadata (rrf_score, dense_rank, bm25_rank, cache_hit
    flags, etc.) without modifying RAGResponse.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RAGResponse(BaseModel):
    """The full record of a single RAG query."""

    # Allow non-Pydantic hit types (dataclass-based BM25Hit, HybridHit, ...)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── input ─────────────────────────────────────────────────────────────
    question: str

    # ── output ────────────────────────────────────────────────────────────
    answer: str

    # ── retrieval ─────────────────────────────────────────────────────────
    retrieved: list[Any] = Field(default_factory=list)
    """Top-k hits returned by the retriever. Each element must expose
    `.record` (with `.content`, `.source`) and `.score`. Concrete types in
    Week 2: SearchResult | BM25Hit | HybridHit."""

    retrieval_latency_ms: int = 0

    # ── generation ────────────────────────────────────────────────────────
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    generation_latency_ms: int = 0

    # ── end-to-end ────────────────────────────────────────────────────────
    total_latency_ms: int = 0

    # ── extras ────────────────────────────────────────────────────────────
    extra: dict[str, Any] = Field(default_factory=dict)
    """Slot for layer-specific data: cache_hit, router_tier, etc."""

    # ── helpers ───────────────────────────────────────────────────────────
    def citations(self) -> list[str]:
        """Unique source paths in retrieval order."""
        seen: list[str] = []
        for hit in self.retrieved:
            src = hit.record.source
            if src not in seen:
                seen.append(src)
        return seen

    def summary(self) -> str:
        return (
            f"RAGResponse(q={self.question[:40]!r}, "
            f"tokens={self.total_tokens}, "
            f"cost=${self.cost_usd:.5f}, "
            f"retrieval={self.retrieval_latency_ms}ms, "
            f"generation={self.generation_latency_ms}ms, "
            f"total={self.total_latency_ms}ms)"
        )