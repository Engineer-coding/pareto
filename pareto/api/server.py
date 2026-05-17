"""
FastAPI HTTP server for Pareto.

Single small surface area for now:
    GET  /health              — liveness + index stats
    POST /ask                 — full RAG query → grounded answer + sources + stats
    POST /search              — retrieval-only (no LLM)

The Next.js dev frontend talks to this server at http://localhost:8000.
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


# ── request / response schemas ───────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    model: str = "ollama/llama3.2:3b"


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


class SearchResponse(BaseModel):
    query: str
    hits: list[CitationItem]


# ── app factory ──────────────────────────────────────────────────────────

def create_app(index_dir: str | Path = "benchmarks/results/index") -> FastAPI:
    """Build a configured FastAPI app. Loads the index once at startup."""
    app = FastAPI(
        title="Pareto API",
        version="0.0.1",
        description="Cost-optimized RAG infrastructure — HTTP interface.",
    )

    # CORS: allow the Next.js dev server (3000) and any local dev origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
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
    print(f"[pareto-api] Ready. {indexer.store.size} chunks loaded.")

    # ── routes ────────────────────────────────────────────────────────────

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "index_size": indexer.store.size,
            "embedding_dim": indexer.store.config.embedding_dim,
            "embedder": indexer.embedder.model_name,
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
            llm = LiteLLMClient(LLMConfig(model=req.model))
            rag = NaiveRAG(indexer=indexer, llm=llm, top_k=req.k)
            rag_resp = rag.query(req.question)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

        citations = [
            CitationItem(
                source=Path(h.record.source).name,
                score=h.score,
                preview=h.record.content[:240],
            )
            for h in rag_resp.retrieved
        ]

        return AskResponse(
            question=req.question,
            answer=rag_resp.answer,
            citations=citations,
            stats={
                "model": rag_resp.model,
                "prompt_tokens": rag_resp.prompt_tokens,
                "completion_tokens": rag_resp.completion_tokens,
                "total_tokens": rag_resp.total_tokens,
                "cost_usd": rag_resp.cost_usd,
                "retrieval_latency_ms": rag_resp.retrieval_latency_ms,
                "generation_latency_ms": rag_resp.generation_latency_ms,
                "total_latency_ms": rag_resp.total_latency_ms,
            },
        )

    return app