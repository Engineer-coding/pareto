"""
RoutedRAG — orchestrates QueryRouter + multiple retrievers + model tiering
over NaiveRAG.

Flow:
    Query → Router → (retriever, model_tier) → NaiveRAG (cache-aware, uses
    the picked retriever AND picked model) → response + routing metadata.

Two LLM clients:
    - standard: the full model (e.g. llama3.2:3b) for complex queries
    - small:    a cheaper/faster model (e.g. llama3.2:1b) for short,
                factual queries the router deems simple

An optional reranker (Week 5) is passed down to the inner NaiveRAG, so
routed queries also get two-stage retrieve→rerank when enabled.

The router runs before cache lookup because the cache key includes both
retriever and model. Deterministic routing keeps cache lookups consistent.
If no small model is provided, small-tier queries fall back to standard.
"""

from __future__ import annotations

from pareto.rag.models import RAGResponse
from pareto.rag.naive_rag import NaiveRAG
from pareto.routing import QueryRouter


class RoutedRAG:
    """Router + retriever registry + model tiering over a single NaiveRAG."""

    def __init__(
        self,
        retrievers: dict,           # {"dense": ..., "bm25": ..., "hybrid": ...}
        router: QueryRouter,
        llm=None,                   # standard-tier LLM client
        llm_small=None,             # small-tier LLM client (optional)
        top_k: int = 5,
        cache=None,
        log_store=None,
        default_retriever: str = "hybrid",
        reranker=None,
        rerank_candidates: int = 20,
    ):
        if default_retriever not in retrievers:
            raise ValueError(
                f"default_retriever '{default_retriever}' not in "
                f"retrievers {list(retrievers.keys())}"
            )
        self.retrievers = retrievers
        self.router = router
        self.default_retriever = default_retriever
        self.llm_standard = llm
        # Fall back to standard if no small model — tiering becomes a no-op
        self.llm_small = llm_small if llm_small is not None else llm
        # RoutedRAG owns logging — route metadata is attached after the
        # inner NaiveRAG returns, so NaiveRAG must NOT log (route would be null).
        self.log_store = log_store

        # Single NaiveRAG; we override retriever + llm per query.
        # Reranker (if any) lives on the inner NaiveRAG and applies to all
        # routes uniformly.
        self.rag = NaiveRAG(
            retriever=retrievers[default_retriever],
            llm=llm,
            top_k=top_k,
            cache=cache,
            log_store=None,
            reranker=reranker,
            rerank_candidates=rerank_candidates,
        )

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        decision = self.router.route(question)

        active_retriever = self.retrievers.get(decision.retriever)
        if active_retriever is None:
            active_retriever = self.retrievers[self.default_retriever]

        active_llm = (
            self.llm_small if decision.model_tier == "small" else self.llm_standard
        )

        response = self.rag.query(
            question,
            top_k=top_k,
            retriever=active_retriever,
            llm=active_llm,
        )

        # Annotate with routing metadata (merge into existing extra)
        response.extra["route"] = decision.retriever
        response.extra["route_reason"] = decision.reason
        response.extra["model_tier"] = decision.model_tier

        # Log AFTER route metadata is attached (NaiveRAG didn't log)
        if self.log_store is not None:
            try:
                self.log_store.log(
                    response,
                    retriever=decision.retriever,
                    top_k=top_k or self.rag.top_k,
                )
            except Exception as e:
                import sys
                print(f"[pareto-routed] log failed: {e}", file=sys.stderr)

        return response

    def __repr__(self) -> str:
        small = getattr(self.llm_small, "model_name", "none")
        standard = getattr(self.llm_standard, "model_name", "none")
        reranker_name = type(self.rag.reranker).__name__ if self.rag.reranker is not None else "none"
        return (
            f"RoutedRAG(retrievers={list(self.retrievers.keys())}, "
            f"default={self.default_retriever}, "
            f"standard={standard}, small={small}, reranker={reranker_name})"
        )