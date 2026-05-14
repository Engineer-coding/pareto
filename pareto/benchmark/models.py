"""
Benchmark data models.

A test set is a list of BenchmarkQuery items.
Running a system against a test set produces BenchmarkResult items,
aggregated into a BenchmarkReport.

Ground truth is intentionally light: a list of expected sources (filename
substrings) and a list of expected keywords. This is pragmatic — building
"perfect" chunk-level labels is months of work; source/keyword labels can
be authored by one engineer in an afternoon and still drive meaningful
retrieval evals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QueryType(str, Enum):
    FACTUAL = "factual"        # "What is X?"
    MULTI_HOP = "multi_hop"    # "What is X's relation to Y?"
    SUMMARY = "summary"        # "Summarize the key points of X."
    COMPARISON = "comparison"  # "Compare X and Y."
    NO_ANSWER = "no_answer"    # ground truth: corpus has no answer


class BenchmarkQuery(BaseModel):
    """A single test case."""

    id: str
    """Stable identifier like 'legal-001'."""

    query: str
    """The natural-language query."""

    domain: str
    """Used for filtering and per-domain reports (legal/finance/health)."""

    difficulty: QueryDifficulty = QueryDifficulty.MEDIUM
    type: QueryType = QueryType.FACTUAL

    expected_sources: list[str] = Field(default_factory=list)
    """Filename substrings that should appear among retrieved chunks.
    Substring match keeps the schema robust against full-path differences."""

    expected_keywords: list[str] = Field(default_factory=list)
    """Keywords expected in the generated answer (lowercase substring match)."""

    notes: str | None = None
    """Optional human note about why this query is in the suite."""


class RetrievalMetrics(BaseModel):
    """Per-query retrieval metrics."""

    hit_at_k: bool = False
    """Did at least one retrieved chunk match expected_sources?"""

    precision_at_k: float = 0.0
    """Fraction of top-k results matching expected_sources."""

    recall_at_k: float = 0.0
    """Fraction of expected_sources covered by top-k results."""

    mrr: float = 0.0
    """Reciprocal rank of the FIRST matching result (1/rank). 0 if none."""

    retrieval_latency_ms: int = 0


class AnswerMetrics(BaseModel):
    """Per-query end-to-end answer metrics (only when LLM was used)."""

    keyword_coverage: float = 0.0
    """Fraction of expected_keywords found (case-insensitive) in the answer."""

    answer_length_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    generation_latency_ms: int = 0


class BenchmarkResult(BaseModel):
    """The result of running ONE BenchmarkQuery against a RAG system."""

    query_id: str
    query: str
    domain: str
    difficulty: QueryDifficulty
    type: QueryType

    retrieved_sources: list[str] = Field(default_factory=list)
    """Filenames of chunks returned (for inspection)."""

    answer: str | None = None
    """LLM answer, if generation was run."""

    retrieval: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    answer_metrics: AnswerMetrics | None = None
    """Present only if end-to-end benchmark ran."""

    total_latency_ms: int = 0


class BenchmarkReport(BaseModel):
    """Aggregate report across a full test set run."""

    system_name: str
    """Identifier: 'naive_rag', 'naive_rag+hybrid', etc."""

    test_set_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    num_queries: int
    k: int
    """Retrieval depth used during this run."""

    # ── retrieval averages ────────────────────────────────────────────────
    avg_hit_at_k: float = 0.0
    avg_precision_at_k: float = 0.0
    avg_recall_at_k: float = 0.0
    avg_mrr: float = 0.0
    avg_retrieval_latency_ms: float = 0.0

    # ── answer averages (optional) ────────────────────────────────────────
    answer_evaluated: bool = False
    avg_keyword_coverage: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_cost_usd: float = 0.0
    avg_generation_latency_ms: float = 0.0
    total_cost_usd: float = 0.0

    # ── per-domain breakdown ──────────────────────────────────────────────
    by_domain: dict[str, dict[str, float]] = Field(default_factory=dict)

    # ── full per-query results ────────────────────────────────────────────
    results: list[BenchmarkResult] = Field(default_factory=list)

    extra: dict[str, Any] = Field(default_factory=dict)

    def summary_line(self) -> str:
        return (
            f"{self.system_name}@{self.k}: "
            f"hit={self.avg_hit_at_k:.2%}  "
            f"P@{self.k}={self.avg_precision_at_k:.3f}  "
            f"R@{self.k}={self.avg_recall_at_k:.3f}  "
            f"MRR={self.avg_mrr:.3f}  "
            f"lat={self.avg_retrieval_latency_ms:.0f}ms"
        )