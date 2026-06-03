"""
RoutedRAG — orchestrates QueryRouter + multiple retrievers over NaiveRAG.

Flow:
    Query → Router (pick retriever) → NaiveRAG (cache-aware, uses the
    picked retriever) → response, annotated with routing metadata.

The router runs BEFORE cache lookup because the cache key includes the
retriever — we must know which retriever was chosen to find the right
cache entry. The router is deterministic, so the same query always routes
to the same retriever, keeping cache lookups consistent.

This is a thin orchestration layer. NaiveRAG stays focused on the
retrieve→generate→cache pipeline; RoutedRAG only decides which retriever
to hand it per query.
"""

from __future__ import annotations

from pareto.rag.models import RAGResponse
from pareto.rag.naive_rag import NaiveRAG
from pareto.routing import QueryRouter


class RoutedRAG:
    """Router + retriever registry over a single NaiveRAG."""

    def __init__(
        self,
        retrievers: dict,           # {"dense": ..., "bm25": ..., "hybrid": ...}
        router: QueryRouter,
        llm=None,
        top_k: int = 5,
        cache=None,
        log_store=None,
        default_retriever: str = "hybrid",
    ):
        if default_retriever not in retrievers:
            raise ValueError(
                f"default_retriever '{default_retriever}' not in "
                f"retrievers {list(retrievers.keys())}"
            )
        self.retrievers = retrievers
        self.router = router
        self.default_retriever = default_retriever
        # Single NaiveRAG; we override its retriever per query.
        # Pass an indexer-bearing retriever so cache embedding works.
        self.rag = NaiveRAG(
            retriever=retrievers[default_retriever],
            llm=llm,
            top_k=top_k,
            cache=cache,
            log_store=log_store,
        )

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        decision = self.router.route(question)
        active = self.retrievers.get(decision.retriever)
        if active is None:
            # Router picked something we don't have; fall back to default.
            active = self.retrievers[self.default_retriever]

        response = self.rag.query(question, top_k=top_k, retriever=active)

        # Annotate with routing metadata (merge into existing extra)
        response.extra["route"] = decision.retriever
        response.extra["route_reason"] = decision.reason
        response.extra["model_tier"] = decision.model_tier
        return response

    def __repr__(self) -> str:
        return (
            f"RoutedRAG(retrievers={list(self.retrievers.keys())}, "
            f"default={self.default_retriever}, router={self.router})"
        )