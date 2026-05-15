"""Benchmark suite: test sets, metrics, runners, reports."""

from pareto.benchmark.metrics import (
    aggregate_answer,
    aggregate_by_domain,
    aggregate_retrieval,
    compute_answer_metrics,
    compute_retrieval_metrics,
)
from pareto.benchmark.models import (
    AnswerMetrics,
    BenchmarkQuery,
    BenchmarkReport,
    BenchmarkResult,
    QueryDifficulty,
    QueryType,
    RetrievalMetrics,
)
from pareto.benchmark.test_set import TestSet

__all__ = [
    "BenchmarkQuery",
    "BenchmarkResult",
    "BenchmarkReport",
    "QueryDifficulty",
    "QueryType",
    "RetrievalMetrics",
    "AnswerMetrics",
    "TestSet",
    "compute_retrieval_metrics",
    "compute_answer_metrics",
    "aggregate_retrieval",
    "aggregate_answer",
    "aggregate_by_domain",
]