"""
FastAPI HTTP server for Pareto.

Surface area:
    GET  /health              -- liveness + index stats + loaded layers
    POST /ask                 -- full RAG query -> grounded answer + sources +
                                 stats, with optional cache / router / rerank
    POST /search              -- retrieval-only (no LLM)

Shared components (index, BM25, retrievers, reranker, cache, log store) are
loaded ONCE at app startup and reused across requests. In particular the
semantic cache is app-level, so paraphrase queries across requests hit it.

The Next.js / static dev frontend talks to this server at http://localhost:8000.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pareto.generation import LiteLLMClient, LLMConfig
from pareto.indexing import Indexer
from pareto.rag import NaiveRAG


# -- request / response schemas --------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    model: str = "ollama/llama3.2:3b"
    retriever: str = "hybrid"           # dense | bm25 | hybrid
    use_router: bool = False            # adaptive router (overrides retriever)
    use_rerank: bool = False            # two-stage cross-encoder rerank
    rerank_threshold: float | None = None
    no_cache: bool = False              # bypass the semantic cache


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    domain: str | None = None  # filter by corpus subfolder name


class CitationItem(BaseModel):
    source: str
    score: float | None = None
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]
    stats: dict[str, Any]
    # Layer indicators -- let the UI show badges (cache / route / rerank)
    cache_hit: bool = False
    cache_similarity: float | None = None
    route: str | None = None
    route_reason: str | None = None
    model_tier: str | None = None
    reranked: bool = False


class SearchResponse(BaseModel):
    query: str
    hits: list[CitationItem]


# -- app factory -----------------------------------------------------------

def create_app(
    index_dir: str | Path = "benchmarks/results/index",
    enable_rerank: bool = True,
    cache_threshold: float = 0.92,
    small_model: str = "ollama/llama3.2:1b",
) -> FastAPI:
    """Build a configured FastAPI app. Loads index + layers once at startup."""
    app = FastAPI(
        title="Pareto API",
        version="0.0.7",
        description="Cost-optimized RAG infrastructure -- HTTP interface.",
    )

    # CORS: allow local dev frontends (Next.js 3000, static servers, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load the index once at boot
    idx_path = Path(index_dir)
    if not idx_path.exists():
        raise FileNotFoundError(
            f"No index at {idx_path}. Run `pareto index <corpus>` first."
        )

    print(f"[pareto-api] Loading index from {idx_path}...")
    indexer = Indexer.load(idx_path)
    print(f"[pareto-api] Index ready. {indexer.store.size} chunks loaded.")

    # Shared retrievers (built once). BM25 needs the full record set.
    from pareto.retrieval import DenseRetriever, BM25Ranker, HybridRetriever
    ranker = BM25Ranker()
    ranker.build_from_records(indexer.store.records)
    retrievers = {
        "dense": DenseRetriever(indexer),
        "bm25": ranker,
        "hybrid": HybridRetriever(indexer=indexer, bm25_ranker=ranker),
    }
    print("[pareto-api] Retrievers ready (dense / bm25 / hybrid).")

    # Shared reranker (optional, loaded once -- this is the expensive load)
    reranker = None
    if enable_rerank:
        from pareto.retrieval import CrossEncoderReranker
        print("[pareto-api] Loading cross-encoder reranker...")
        reranker = CrossEncoderReranker()
        print("[pareto-api] Reranker ready.")

    # App-level shared semantic cache (paraphrase hits persist across requests)
    from pareto.cache import SemanticCache
    cache = SemanticCache(threshold=cache_threshold)

    # Shared observability log store
    from pareto.observability import QueryLogStore
    log_store = QueryLogStore()

    print("[pareto-api] Ready.")

    # -- helpers ------------------------------------------------------------

    def _citations_from_response(rag_resp) -> list[CitationItem]:
        """Build citation items from a RAGResponse (handles cache-hit case)."""
        if rag_resp.extra.get("cache_hit"):
            # Cache hits don't carry retrieved objects, only cached source names
            return [
                CitationItem(source=s, score=None, preview="")
                for s in rag_resp.extra.get("cached_citations", [])
            ]
        return [
            CitationItem(
                source=Path(h.record.source).name,
                score=getattr(h, "score", None),
                preview=h.record.content[:240],
            )
            for h in rag_resp.retrieved
        ]

    def _build_rag(req: AskRequest):
        """Construct the right RAG pipeline for this request from shared parts."""
        active_cache = None if req.no_cache else cache
        active_reranker = reranker if req.use_rerank else None
        llm = LiteLLMClient(LLMConfig(model=req.model))

        if req.use_router:
            from pareto.routing import QueryRouter
            from pareto.rag import RoutedRAG
            return RoutedRAG(
                retrievers=retrievers,
                router=QueryRouter(),
                llm=llm,
                llm_small=LiteLLMClient(LLMConfig(model=small_model)),
                top_k=req.k,
                cache=active_cache,
                log_store=log_store,
                reranker=active_reranker,
            )

        retriever_obj = retrievers.get(req.retriever)
        if retriever_obj is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown retriever '{req.retriever}'. Use dense/bm25/hybrid.",
            )
        return NaiveRAG(
            retriever=retriever_obj,
            indexer=indexer,
            llm=llm,
            top_k=req.k,
            cache=active_cache,
            log_store=log_store,
            reranker=active_reranker,
            rerank_score_threshold=req.rerank_threshold,
        )

    # -- routes -------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "index_size": indexer.store.size,
            "embedding_dim": indexer.store.config.embedding_dim,
            "embedder": indexer.embedder.model_name,
            "reranker_loaded": reranker is not None,
            "cache_size": cache.stats().get("size", 0),
        }

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        q_vec = indexer.embedder.encode_query(req.query)
        if req.domain:
            filter_fn = lambda r: (
                f"\\{req.domain}\\" in r.source or f"/{req.domain}/" in r.source
            )
            hits = indexer.store.search(q_vec, k=req.k, filter_fn=filter_fn)
        else:
            hits = indexer.store.search(q_vec, k=req.k)
        return SearchResponse(
            query=req.query,
            hits=[
                CitationItem(
                    source=Path(h.record.source).name,
                    score=h.score,
                    preview=h.record.content[:240],
                )
                for h in hits
            ],
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        try:
            rag = _build_rag(req)
            rag_resp = rag.query(req.question)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"{type(e).__name__}: {e}"
            ) from e

        extra = rag_resp.extra or {}
        stats = {
            "model": rag_resp.model,
            "prompt_tokens": rag_resp.prompt_tokens,
            "completion_tokens": rag_resp.completion_tokens,
            "total_tokens": rag_resp.total_tokens,
            "cost_usd": rag_resp.cost_usd,
            "retrieval_latency_ms": rag_resp.retrieval_latency_ms,
            "generation_latency_ms": rag_resp.generation_latency_ms,
            "total_latency_ms": rag_resp.total_latency_ms,
        }
        # On a cache hit, surface what was saved (phantom cost) for the UI
        if extra.get("cache_hit"):
            stats["saved_latency_ms"] = extra.get("original_generation_latency_ms", 0)
            stats["saved_cost_usd"] = extra.get("original_cost_usd", 0)

        return AskResponse(
            question=req.question,
            answer=rag_resp.answer,
            citations=_citations_from_response(rag_resp),
            stats=stats,
            cache_hit=bool(extra.get("cache_hit")),
            cache_similarity=extra.get("cache_similarity"),
            route=extra.get("route"),
            route_reason=extra.get("route_reason"),
            model_tier=extra.get("model_tier"),
            reranked=bool(req.use_rerank and not extra.get("cache_hit")),
        )

    return app