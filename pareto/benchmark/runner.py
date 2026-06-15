"""
BenchmarkRunner — execute a test set against a RAG system and produce a report.

Two modes:
    retrieval_only — embed + search only. Fast (seconds for 30 queries).
                     Used continuously during Week 2-5 retrieval optimization.
                     Since Week 6, an optional reranker can be applied here too
                     (two-stage retrieve-broad -> rerank-to-k) so reranking can
                     be measured without paying for LLM generation.

    end_to_end     — full RAG pipeline (retrieve + LLM). Slow on local CPU
                     (~30 minutes for 30 queries with llama3.2:3b). Used at
                     milestone reviews and demo prep.

The runner is system-agnostic: it accepts any object that exposes
`indexer` (with `.embedder` and `.store`) and optionally `rag` (a NaiveRAG
or any subclass). Future Week 2+ systems plug in here without changes.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from pareto.benchmark.metrics import (
    aggregate_answer,
    aggregate_by_domain,
    aggregate_retrieval,
    compute_answer_metrics,
    compute_retrieval_metrics,
)
from pareto.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
)
from pareto.benchmark.test_set import TestSet
from pareto.indexing.indexer import Indexer
from pareto.rag.naive_rag import NaiveRAG


class BenchmarkRunner:
    """Runs a TestSet against an Indexer (retrieval) or NaiveRAG (end-to-end)."""

    def __init__(
        self,
        indexer: Indexer,
        rag: NaiveRAG | None = None,
        retriever=None,  # any object with `search(query, k) -> list[Hit]`
        reranker=None,   # optional CrossEncoderReranker (retrieval mode)
        rerank_candidates: int = 20,
        rerank_score_threshold: float | None = None,
    ):
        self.indexer = indexer
        self.rag = rag
        # Default retriever: DenseRetriever wrapping the indexer
        if retriever is None:
            from pareto.retrieval.dense import DenseRetriever
            retriever = DenseRetriever(indexer)
        self.retriever = retriever
        # Reranker is used only in retrieval-only mode. In end-to-end mode the
        # reranker lives inside `rag` (NaiveRAG/RoutedRAG) and is applied there.
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.rerank_score_threshold = rerank_score_threshold

    # -- retrieval-only mode ------------------------------------------------
    def run_retrieval(
        self,
        test_set: TestSet,
        k: int = 5,
        system_name: str = "retrieval_only",
        limit: int | None = None,
        show_progress: bool = True,
    ) -> BenchmarkReport:
        """Run retrieval-only benchmark. No LLM calls."""
        return self._run(
            test_set=test_set, k=k, system_name=system_name,
            limit=limit, use_llm=False, show_progress=show_progress,
        )

    # -- end-to-end mode ----------------------------------------------------
    def run_end_to_end(
        self,
        test_set: TestSet,
        k: int = 5,
        system_name: str = "naive_rag",
        limit: int | None = None,
        show_progress: bool = True,
    ) -> BenchmarkReport:
        """Run full RAG pipeline benchmark (retrieve + LLM)."""
        if self.rag is None:
            raise ValueError(
                "End-to-end benchmark requires `rag` to be set on the runner."
            )
        return self._run(
            test_set=test_set, k=k, system_name=system_name,
            limit=limit, use_llm=True, show_progress=show_progress,
        )

    # -- internals ----------------------------------------------------------
    def _run(
        self,
        test_set: TestSet,
        k: int,
        system_name: str,
        limit: int | None,
        use_llm: bool,
        show_progress: bool,
    ) -> BenchmarkReport:
        queries = list(test_set)
        if limit:
            queries = queries[:limit]

        results: list[BenchmarkResult] = []

        if show_progress:
            console = Console()
            mode_label = "end-to-end" if use_llm else "retrieval-only"
            with Progress(
                TextColumn(f"[progress.description]Benchmark ({mode_label})"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("running", total=len(queries))
                for q in queries:
                    results.append(self._run_one(q, k, use_llm))
                    progress.update(task, advance=1)
        else:
            for q in queries:
                results.append(self._run_one(q, k, use_llm))

        return self._build_report(
            results=results, test_set=test_set, k=k,
            system_name=system_name, answer_evaluated=use_llm,
        )

    def _retrieve(self, query_text: str, k: int):
        """Retrieval-only retrieval, with optional two-stage reranking.

        Mirrors NaiveRAG's retrieval step exactly: if a reranker is set,
        fetch a broad candidate set then rerank down to top-k (with the same
        score_threshold semantics); otherwise retrieve top-k directly.
        """
        if self.reranker is not None:
            candidates = self.retriever.search(query_text, k=self.rerank_candidates)
            return self.reranker.rerank(
                query_text, candidates, top_k=k,
                score_threshold=self.rerank_score_threshold,
            )
        return self.retriever.search(query_text, k=k)

    def _run_one(self, query, k: int, use_llm: bool) -> BenchmarkResult:
        """Execute one query through retrieval (and optionally generation)."""
        t0 = time.perf_counter()

        if use_llm:
            # Full RAG path -- single call covers both retrieval and generation
            rag_response = self.rag.query(query.query, top_k=k)
            retrieved_sources = [
                Path(h.record.source).name for h in rag_response.retrieved
            ]
            retrieval = compute_retrieval_metrics(
                query, retrieved_sources, rag_response.retrieval_latency_ms,
            )
            answer_metrics = compute_answer_metrics(
                query,
                answer=rag_response.answer,
                prompt_tokens=rag_response.prompt_tokens,
                completion_tokens=rag_response.completion_tokens,
                cost_usd=rag_response.cost_usd,
                generation_latency_ms=rag_response.generation_latency_ms,
            )
            answer = rag_response.answer
        else:
            # Retrieval-only path -- injected retriever (Dense/BM25/Hybrid/...)
            # with optional two-stage reranking (Week 6).
            t_retr = time.perf_counter()
            hits = self._retrieve(query.query, k=k)
            retrieval_latency_ms = int((time.perf_counter() - t_retr) * 1000)

            retrieved_sources = [Path(h.record.source).name for h in hits]
            retrieval = compute_retrieval_metrics(
                query, retrieved_sources, retrieval_latency_ms,
            )
            answer_metrics = None
            answer = None

        total_latency_ms = int((time.perf_counter() - t0) * 1000)

        return BenchmarkResult(
            query_id=query.id,
            query=query.query,
            domain=query.domain,
            difficulty=query.difficulty,
            type=query.type,
            retrieved_sources=retrieved_sources,
            answer=answer,
            retrieval=retrieval,
            answer_metrics=answer_metrics,
            total_latency_ms=total_latency_ms,
        )

    def _build_report(
        self,
        results: list[BenchmarkResult],
        test_set: TestSet,
        k: int,
        system_name: str,
        answer_evaluated: bool,
    ) -> BenchmarkReport:
        retr_agg = aggregate_retrieval(results)
        by_domain = aggregate_by_domain(results)

        report = BenchmarkReport(
            system_name=system_name,
            test_set_name=test_set.name,
            num_queries=len(results),
            k=k,
            avg_hit_at_k=retr_agg["avg_hit_at_k"],
            avg_precision_at_k=retr_agg["avg_precision_at_k"],
            avg_recall_at_k=retr_agg["avg_recall_at_k"],
            avg_mrr=retr_agg["avg_mrr"],
            avg_retrieval_latency_ms=retr_agg["avg_retrieval_latency_ms"],
            by_domain=by_domain,
            results=results,
            answer_evaluated=answer_evaluated,
        )

        if answer_evaluated:
            ans_agg = aggregate_answer(results)
            report.avg_keyword_coverage = ans_agg["avg_keyword_coverage"]
            report.avg_prompt_tokens = ans_agg["avg_prompt_tokens"]
            report.avg_completion_tokens = ans_agg["avg_completion_tokens"]
            report.avg_cost_usd = ans_agg["avg_cost_usd"]
            report.total_cost_usd = ans_agg["total_cost_usd"]
            report.avg_generation_latency_ms = ans_agg["avg_generation_latency_ms"]
            report.extra["refusal_accuracy"] = ans_agg["refusal_accuracy"]

        return report


# -- persistence ------------------------------------------------------------

def _to_json(report) -> str:
    """Serialize any report type to a JSON string.

    Handles, in order of preference:
      - pydantic v2 models      (.model_dump_json)
      - pydantic v1 models      (.json)
      - dataclasses             (asdict)
      - plain objects           (__dict__)
      - anything else           (json.dumps with default=str fallback)

    This keeps save_report agnostic to the report type, so both
    BenchmarkReport (pydantic) and CorpusChunkingReport (dataclass) work.
    """
    # pydantic v2
    model_dump_json = getattr(report, "model_dump_json", None)
    if callable(model_dump_json):
        return model_dump_json(indent=2)

    # pydantic v1
    json_method = getattr(report, "json", None)
    if callable(json_method):
        return json_method(indent=2)

    # dataclass instance
    if dataclasses.is_dataclass(report) and not isinstance(report, type):
        return json.dumps(
            dataclasses.asdict(report), indent=2, ensure_ascii=False, default=str
        )

    # plain object with attributes
    if hasattr(report, "__dict__"):
        return json.dumps(
            vars(report), indent=2, ensure_ascii=False, default=str
        )

    # last-resort: let json figure it out, stringifying unknowns
    return json.dumps(report, indent=2, ensure_ascii=False, default=str)


def save_report(report, path: str | Path) -> Path:
    """Persist a report as JSON. Returns the written path.

    Accepts any report type (BenchmarkReport, CorpusChunkingReport, ...).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_to_json(report), encoding="utf-8")
    return p


def load_report(path: str | Path) -> BenchmarkReport:
    """Inverse of save_report for BenchmarkReport. Useful for comparing runs.

    Note: only reconstructs BenchmarkReport. Other report types are written
    by save_report but not read back here (not needed downstream).
    """
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    return BenchmarkReport(**raw)