"""
NaiveRAG — the baseline pipeline.

Pure pipeline, no optimization:
    1. (Optional) Semantic cache lookup. Hit → instant response.
    2. Retrieve top-k candidates (via injected retriever).
    3. Concatenate retrieved chunks into a prompt.
    4. Single LLM call.
    5. (Optional) Cache the response keyed by query embedding.
    6. Wrap everything into a RAGResponse with latency / cost breakdown.

Retriever-agnostic since Week 2: any object with `search(query, k) -> list[Hit]`
works (DenseRetriever, BM25Ranker, HybridRetriever, future cached retrievers).
Cache-aware since Week 3: optional SemanticCache injected at construction.

Cache hits bypass retrieval and generation entirely — ~10-15ms total latency
vs. ~50000ms LLM call. The cost field is set to 0 for cache hits (no LLM
charge), with the original generation cost preserved in `extra` for
"savings" projection in observability.
"""

from __future__ import annotations

import time
from pathlib import Path

from pareto.generation.llm_client import BaseLLMClient, LiteLLMClient
from pareto.indexing.indexer import Indexer
from pareto.rag.models import RAGResponse
from pareto.rag.prompts import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    DEFAULT_RAG_USER_TEMPLATE,
    format_context_block,
)


class NaiveRAG:
    """Top-k retrieval + single LLM call. The baseline."""

    def __init__(
        self,
        indexer: Indexer | None = None,
        llm: BaseLLMClient | None = None,
        top_k: int = 5,
        retriever=None,
        system_prompt: str = DEFAULT_RAG_SYSTEM_PROMPT,
        user_template: str = DEFAULT_RAG_USER_TEMPLATE,
        max_context_chars: int = 6000,
        log_store=None,
        cache=None,
    ):
        """
        Args:
            indexer: built Indexer. Required if `retriever` is None.
            llm: LLM client. Defaults to LiteLLMClient (Ollama llama3.2:3b).
            top_k: number of chunks to retrieve.
            retriever: any retriever exposing `search(query, k) -> list[Hit]`.
                If None, builds a DenseRetriever from `indexer`.
            system_prompt: RAG system prompt template.
            user_template: User-side prompt template ({context}, {question}).
            max_context_chars: truncate concatenated context above this size.
            log_store: optional QueryLogStore for persistence.
            cache: optional SemanticCache for embedding-based dedup.
        """
        # Resolve retriever
        if retriever is None:
            if indexer is None:
                raise ValueError("NaiveRAG requires either `indexer` or `retriever`.")
            from pareto.retrieval.dense import DenseRetriever
            retriever = DenseRetriever(indexer)
        self.retriever = retriever

        # Keep indexer accessible — also needed for cache embeddings
        self.indexer = indexer if indexer is not None else getattr(
            retriever, "indexer", None
        )

        self.llm = llm or LiteLLMClient()
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_context_chars = max_context_chars
        self.log_store = log_store
        self.cache = cache

    # ── public API ────────────────────────────────────────────────────────
    def query(self, question: str, top_k: int | None = None, retriever=None, llm=None) -> RAGResponse:
        """
        Run the full RAG pipeline for a single question.

        Args:
            question: the user query.
            top_k: override the default retrieval depth.
            retriever: override the configured retriever for this call.
                Used by RoutedRAG to apply per-query routing decisions.
                Cache keys include the retriever name, so a deterministic
                router keeps cache lookups consistent.
        """
        k = top_k or self.top_k
        t0 = time.perf_counter()
        active_retriever = retriever if retriever is not None else self.retriever
        retriever_name = type(active_retriever).__name__
        active_llm = llm if llm is not None else self.llm

        # ── 1. Embed query (cache lookup + retrieval share it) ──
        query_embedding = None
        if self.indexer is not None and self.cache is not None:
            try:
                query_embedding = self.indexer.embedder.encode_query(question)
            except Exception as e:
                import sys
                print(f"[pareto-rag] cache embed failed: {e}", file=sys.stderr)
                query_embedding = None

        # ── 2. Cache lookup ──
        if self.cache is not None and query_embedding is not None:
            try:
                hit = self.cache.lookup(
                    query_embedding=query_embedding,
                    retriever=retriever_name,
                    top_k=k,
                    model=active_llm.model_name,
                    valid_chunk_ids=None,
                )
            except Exception as e:
                import sys
                print(f"[pareto-rag] cache lookup failed: {e}", file=sys.stderr)
                hit = None

            if hit is not None:
                return self._cache_hit_response(question, hit, k, retriever_name, t0)

        # ── 3. Retrieval (cache miss path) ──
        t_retr_start = time.perf_counter()
        hits = active_retriever.search(question, k=k)
        retrieval_latency_ms = int((time.perf_counter() - t_retr_start) * 1000)

        # ── 4. Prompt build ──
        context = self._build_context(hits)
        user_prompt = self.user_template.format(context=context, question=question)

        # ── 5. Generation ──
        t_gen_start = time.perf_counter()
        llm_resp = active_llm.generate(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        generation_latency_ms = int((time.perf_counter() - t_gen_start) * 1000)
        total_latency_ms = int((time.perf_counter() - t0) * 1000)

        # ── 6. Build response ──
        response = RAGResponse(
            question=question,
            answer=llm_resp.text,
            retrieved=hits,
            retrieval_latency_ms=retrieval_latency_ms,
            model=llm_resp.model,
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            total_tokens=llm_resp.total_tokens,
            cost_usd=llm_resp.cost_usd,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            extra={"cache_hit": False},
        )

        # ── 7. Cache write (best-effort) ──
        if self.cache is not None and query_embedding is not None:
            try:
                chunks_used = [
                    h.record.chunk_id
                    for h in hits
                    if hasattr(h, "record") and hasattr(h.record, "chunk_id")
                ]
                self.cache.add(
                    query=question,
                    query_embedding=query_embedding,
                    retriever=retriever_name,
                    top_k=k,
                    model=llm_resp.model,
                    answer=llm_resp.text,
                    prompt_tokens=llm_resp.prompt_tokens,
                    completion_tokens=llm_resp.completion_tokens,
                    cost_usd=llm_resp.cost_usd,
                    generation_latency_ms=generation_latency_ms,
                    chunks_used_ids=chunks_used,
                    citations=response.citations(),
                )
            except Exception as e:
                import sys
                print(f"[pareto-rag] cache write failed: {e}", file=sys.stderr)

        # ── 8. Query log write (best-effort) ──
        if self.log_store is not None:
            try:
                self.log_store.log(response, retriever=retriever_name, top_k=k)
            except Exception as e:
                import sys
                print(f"[pareto-rag] log_store.log() failed: {e}", file=sys.stderr)

        return response

    # ── helpers ───────────────────────────────────────────────────────────
    def _cache_hit_response(
        self,
        question: str,
        hit,
        k: int,
        retriever_name: str,
        t0: float,
    ) -> RAGResponse:
        """Build a RAGResponse from a SemanticCache hit. Fast path."""
        entry = hit.entry
        total_latency_ms = int((time.perf_counter() - t0) * 1000)

        response = RAGResponse(
            question=question,
            answer=entry.answer,
            retrieved=[],  # cache doesn't store retrieved objects, just chunk_ids
            retrieval_latency_ms=0,
            model=entry.model,
            prompt_tokens=entry.prompt_tokens,
            completion_tokens=entry.completion_tokens,
            total_tokens=entry.prompt_tokens + entry.completion_tokens,
            cost_usd=0.0,  # cache hit costs nothing
            generation_latency_ms=0,
            total_latency_ms=total_latency_ms,
            extra={
                "cache_hit": True,
                "cache_similarity": hit.similarity,
                "cached_original_query": entry.query,
                "cached_at_unix": entry.timestamp_unix,
                "cached_access_count": entry.access_count,
                "cached_citations": list(entry.citations),
                # "Phantom" cost — what we would have paid if not cached
                "original_cost_usd": entry.cost_usd,
                "original_generation_latency_ms": entry.generation_latency_ms,
            },
        )

        # Still log cache hits — observability needs to see them
        if self.log_store is not None:
            try:
                self.log_store.log(
                    response,
                    retriever=retriever_name,
                    top_k=k,
                )
            except Exception as e:
                import sys
                print(f"[pareto-rag] log_store.log() failed: {e}", file=sys.stderr)

        return response

    def _build_context(self, hits: list) -> str:
        """Render hits as numbered blocks, truncating to max_context_chars."""
        if not hits:
            return "(no relevant context found)"

        blocks: list[str] = []
        total = 0
        for i, hit in enumerate(hits, start=1):
            src_name = Path(hit.record.source).name
            block = format_context_block(i, hit.record.content, src_name)
            if total + len(block) > self.max_context_chars and blocks:
                break
            blocks.append(block)
            total += len(block)

        return "\n\n---\n\n".join(blocks)

    def __repr__(self) -> str:
        retriever_type = type(self.retriever).__name__
        cache_name = type(self.cache).__name__ if self.cache is not None else "None"
        return (
            f"NaiveRAG(retriever={retriever_type}, "
            f"top_k={self.top_k}, model={self.llm.model_name}, "
            f"cache={cache_name})"
        )