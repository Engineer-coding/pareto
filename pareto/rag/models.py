"""
RAG request / response data models.

RAGResponse wraps everything that happened during a single Q&A turn:
the question, the answer, the chunks retrieved, the LLM stats, and the
end-to-end latency breakdown. Week 2's observability layer ingests this
directly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from pareto.indexing.models import SearchResult


class RAGResponse(BaseModel):
    """The full record of a single RAG query."""

    # ── input ─────────────────────────────────────────────────────────────
    question: str

    # ── output ────────────────────────────────────────────────────────────
    answer: str

    # ── retrieval ─────────────────────────────────────────────────────────
    retrieved: list[SearchResult] = Field(default_factory=list)
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