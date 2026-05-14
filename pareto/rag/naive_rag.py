"""
NaiveRAG — the baseline pipeline.

Pure pipeline, no optimization:
    1. Embed the query.
    2. Top-k vector search.
    3. Concatenate retrieved chunks into a prompt.
    4. Single LLM call.
    5. Wrap everything into a RAGResponse with latency / cost breakdown.

This is intentionally the simplest possible RAG. Week 2+ layers (hybrid
retrieval, semantic cache, adaptive routing, knowledge graph) will all
be benchmarked AGAINST this baseline. If they don't beat NaiveRAG on
some metric, they don't earn their place.
"""

from __future__ import annotations

import time
from pathlib import Path

from pareto.generation.llm_client import BaseLLMClient, LiteLLMClient
from pareto.indexing.indexer import Indexer
from pareto.indexing.models import SearchResult
from pareto.rag.models import RAGResponse
from pareto.rag.prompts import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    DEFAULT_RAG_USER_TEMPLATE,
    format_context_block,
)


class NaiveRAG:
    """Top-k vector retrieval + single LLM call. The baseline."""

    def __init__(
        self,
        indexer: Indexer,
        llm: BaseLLMClient | None = None,
        top_k: int = 5,
        system_prompt: str = DEFAULT_RAG_SYSTEM_PROMPT,
        user_template: str = DEFAULT_RAG_USER_TEMPLATE,
        max_context_chars: int = 6000,
    ):
        self.indexer = indexer
        self.llm = llm or LiteLLMClient()
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_context_chars = max_context_chars
        """Truncate the concatenated context if it exceeds this many chars.
        Keeps prompts within local-model context windows and bounds cost."""

    # ── public API ────────────────────────────────────────────────────────
    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Run the full RAG pipeline for a single question."""
        k = top_k or self.top_k
        t0 = time.perf_counter()

        # Step 1: retrieval
        t_retr_start = time.perf_counter()
        q_vec = self.indexer.embedder.encode_query(question)
        hits = self.indexer.store.search(q_vec, k=k)
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

        return RAGResponse(
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

    # ── helpers ───────────────────────────────────────────────────────────
    def _build_context(self, hits: list[SearchResult]) -> str:
        """Render hits as numbered blocks, truncating to max_context_chars."""
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