"""
NaiveRAG — the baseline pipeline.

Pure pipeline, no optimization:
    1. Retrieve top-k candidates (via injected retriever).
    2. Concatenate retrieved chunks into a prompt.
    3. Single LLM call.
    4. Wrap everything into a RAGResponse with latency / cost breakdown.

Retriever-agnostic since Week 2: any object with `search(query, k) -> list[Hit]`
works (DenseRetriever, BM25Ranker, HybridRetriever, future cached retrievers).
The retriever is the only knob that controls retrieval quality; the rest of
the pipeline (prompt build, LLM call) is identical across configurations.

This is intentionally the simplest possible RAG. Week 2+ layers (hybrid
retrieval, semantic cache, adaptive routing, knowledge graph) plug in via
the retriever interface. If a new retriever doesn't beat the dense baseline
on the benchmark, it doesn't earn its place.
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
    ):
        """
        Args:
            indexer: built Indexer. Required if `retriever` is None.
            llm: LLM client. Defaults to LiteLLMClient (Ollama llama3.2:3b).
            top_k: number of chunks to retrieve.
            retriever: any retriever exposing `search(query, k) -> list[Hit]`
                where each Hit has `.record` and `.score`. If None, builds
                a DenseRetriever from `indexer`.
            system_prompt: RAG system prompt template.
            user_template: User-side prompt template (must contain {context}, {question}).
            max_context_chars: truncate concatenated context above this size.
                Keeps prompts inside local-model windows and bounds cost.
        """
        # Resolve the retriever first — it's the new primary dependency
        if retriever is None:
            if indexer is None:
                raise ValueError(
                    "NaiveRAG requires either `indexer` or `retriever`."
                )
            from pareto.retrieval.dense import DenseRetriever
            retriever = DenseRetriever(indexer)

        self.retriever = retriever

        # Keep indexer accessible for backward compatibility and introspection
        # (e.g. BenchmarkRunner.end_to_end may inspect rag.indexer).
        # If the caller didn't pass one, try to find it on the retriever.
        self.indexer = indexer if indexer is not None else getattr(
            retriever, "indexer", None
        )

        self.llm = llm or LiteLLMClient()
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_context_chars = max_context_chars
        self.log_store = log_store

    # ── public API ────────────────────────────────────────────────────────
    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Run the full RAG pipeline for a single question."""
        k = top_k or self.top_k
        t0 = time.perf_counter()

        # Step 1: retrieval (delegated to the injected retriever)
        t_retr_start = time.perf_counter()
        hits = self.retriever.search(question, k=k)
        retrieval_latency_ms = int((time.perf_counter() - t_retr_start) * 1000)

        # Step 2: prompt build
        context = self._build_context(hits)
        user_prompt = self.user_template.format(context=context, question=question)

        # Step 3: generation
        t_gen_start = time.perf_counter()
        llm_resp = self.llm.generate(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        generation_latency_ms = int((time.perf_counter() - t_gen_start) * 1000)

        total_latency_ms = int((time.perf_counter() - t0) * 1000)


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
        )

        # Best-effort logging. Failures never break the query.
        if self.log_store is not None:
            try:
                self.log_store.log(
                    response,
                    retriever=type(self.retriever).__name__,
                    top_k=k,
                )
            except Exception as e:  # noqa: BLE001
                import sys
                print(
                    f"[pareto-rag] log_store.log() failed: {e}",
                    file=sys.stderr,
                )

        return response

    # ── helpers ───────────────────────────────────────────────────────────
    def _build_context(self, hits: list) -> str:
        """Render hits as numbered blocks, truncating to max_context_chars.

        Accepts any hit object with `.record` (which itself has `.content`
        and `.source`) — works for SearchResult, BM25Hit, HybridHit.
        """
        if not hits:
            return "(no relevant context found)"

        blocks: list[str] = []
        total = 0
        for i, hit in enumerate(hits, start=1):
            src_name = Path(hit.record.source).name
            block = format_context_block(i, hit.record.content, src_name)
            if total + len(block) > self.max_context_chars and blocks:
                # Stop if adding this block would exceed the cap
                # (but always include at least one block)
                break
            blocks.append(block)
            total += len(block)

        return "\n\n---\n\n".join(blocks)

    def __repr__(self) -> str:
        retriever_type = type(self.retriever).__name__
        return (
            f"NaiveRAG(retriever={retriever_type}, "
            f"top_k={self.top_k}, model={self.llm.model_name})"
        )