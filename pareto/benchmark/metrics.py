"""
Retrieval and answer-quality metrics.

All metrics are pure functions of a BenchmarkQuery and the system's
response — no global state, easy to test, easy to vectorize later.

Retrieval metrics implemented:
    * hit@k         — at least one retrieved source matches ground truth
    * precision@k   — fraction of top-k that match
    * recall@k      — fraction of expected sources found in top-k
    * MRR           — reciprocal rank of the FIRST match (0 if none)

Answer metric implemented:
    * keyword_coverage — fraction of expected keywords present in the answer
    * correctly_refused — for NO_ANSWER queries: did the answer admit it?

Important nuance: NO_ANSWER queries have empty expected_sources, so their
retrieval metrics are NOT MEANINGFUL and we record zeros without flagging
them as failures. Aggregation explicitly excludes NO_ANSWER queries from
retrieval averages.
"""

from __future__ import annotations

from pareto.benchmark.models import (
    AnswerMetrics,
    BenchmarkQuery,
    BenchmarkResult,
    QueryType,
    RetrievalMetrics,
)


# ── retrieval metrics ────────────────────────────────────────────────────

def _source_matches_expected(retrieved_source: str, expected_substrings: list[str]) -> bool:
    """Substring match, case-insensitive. Handles both 'gdpr_full_text.html' and 'gdpr_full_text'."""
    src_lower = retrieved_source.lower()
    return any(exp.lower() in src_lower for exp in expected_substrings)


def compute_retrieval_metrics(
    query: BenchmarkQuery,
    retrieved_sources: list[str],
    retrieval_latency_ms: int = 0,
) -> RetrievalMetrics:
    """
    Compute retrieval metrics for one query.

    For NO_ANSWER queries, retrieval metrics return zeros — these queries
    are excluded from retrieval averages elsewhere. We DON'T penalize a
    NO_ANSWER question for retrieving plausible-but-irrelevant chunks,
    because the failure mode for NO_ANSWER is the LLM hallucinating, not
    the retriever pulling chunks.
    """
    if not query.expected_sources:
        return RetrievalMetrics(retrieval_latency_ms=retrieval_latency_ms)

    k = len(retrieved_sources)
    if k == 0:
        return RetrievalMetrics(retrieval_latency_ms=retrieval_latency_ms)

    # Per-position match flags
    matches = [
        _source_matches_expected(src, query.expected_sources)
        for src in retrieved_sources
    ]

    # hit@k: at least one positive
    hit = any(matches)

    # precision@k: positives / k
    precision = sum(matches) / k

    # recall@k: distinct expected sources covered / |expected|
    found_expected: set[str] = set()
    for src in retrieved_sources:
        for exp in query.expected_sources:
            if exp.lower() in src.lower():
                found_expected.add(exp.lower())
    recall = len(found_expected) / len(query.expected_sources)

    # MRR: 1/rank of FIRST positive (rank is 1-indexed)
    mrr = 0.0
    for i, matched in enumerate(matches):
        if matched:
            mrr = 1.0 / (i + 1)
            break

    return RetrievalMetrics(
        hit_at_k=hit,
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=mrr,
        retrieval_latency_ms=retrieval_latency_ms,
    )


# ── answer metrics ───────────────────────────────────────────────────────

# Phrases that indicate the system honestly refused to answer.
# Used to detect "correctly refused" for NO_ANSWER queries.
_REFUSAL_INDICATORS = (
    "cannot answer",
    "can't answer",
    "do not have",
    "don't have",
    "no information",
    "not contain",
    "not enough",
    "not provided",
    "not mentioned",
    "i don't know",
    "i do not know",
    "unable to",
    "unknown",
    "no record",
    "not in the context",
    "context does not",
)


def _is_refusal(answer: str) -> bool:
    """Heuristic: did the answer admit it couldn't answer?"""
    answer_lower = answer.lower()
    return any(indicator in answer_lower for indicator in _REFUSAL_INDICATORS)


def compute_answer_metrics(
    query: BenchmarkQuery,
    answer: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    generation_latency_ms: int = 0,
) -> AnswerMetrics:
    """
    Compute answer-quality metrics for one query.

    For NO_ANSWER queries, keyword_coverage is interpreted differently:
    we record whether the answer is a refusal (`extra.correctly_refused`).
    """
    answer_lower = answer.lower()

    # Keyword coverage: lowercase substring matches.
    if query.expected_keywords:
        found = sum(1 for kw in query.expected_keywords if kw.lower() in answer_lower)
        coverage = found / len(query.expected_keywords)
    else:
        coverage = 0.0

    metrics = AnswerMetrics(
        keyword_coverage=coverage,
        answer_length_chars=len(answer),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        generation_latency_ms=generation_latency_ms,
    )

    # Tag NO_ANSWER queries with whether they correctly refused.
    # We pack this into the keyword_coverage slot as 1.0 / 0.0 for NO_ANSWER
    # queries so aggregation can compute a "refusal accuracy" cleanly.
    if query.type == QueryType.NO_ANSWER:
        metrics.keyword_coverage = 1.0 if _is_refusal(answer) else 0.0

    return metrics


# ── aggregation ──────────────────────────────────────────────────────────

def aggregate_retrieval(
    results: list[BenchmarkResult],
) -> dict[str, float]:
    """
    Average retrieval metrics across results.

    NO_ANSWER queries are excluded because their retrieval metrics are
    zero by construction (no expected_sources) and would distort the mean.
    """
    relevant = [r for r in results if r.type != QueryType.NO_ANSWER]
    n = len(relevant)
    if n == 0:
        return {
            "avg_hit_at_k": 0.0,
            "avg_precision_at_k": 0.0,
            "avg_recall_at_k": 0.0,
            "avg_mrr": 0.0,
            "avg_retrieval_latency_ms": 0.0,
        }

    return {
        "avg_hit_at_k": sum(1 for r in relevant if r.retrieval.hit_at_k) / n,
        "avg_precision_at_k": sum(r.retrieval.precision_at_k for r in relevant) / n,
        "avg_recall_at_k": sum(r.retrieval.recall_at_k for r in relevant) / n,
        "avg_mrr": sum(r.retrieval.mrr for r in relevant) / n,
        "avg_retrieval_latency_ms": sum(r.retrieval.retrieval_latency_ms for r in relevant) / n,
    }


def aggregate_answer(
    results: list[BenchmarkResult],
) -> dict[str, float]:
    """
    Average answer metrics across results that have an LLM answer.

    For NO_ANSWER queries, keyword_coverage actually encodes "correctly
    refused" (0.0 or 1.0), reported separately as `refusal_accuracy`.
    """
    with_answer = [r for r in results if r.answer_metrics is not None]
    n = len(with_answer)
    if n == 0:
        return {
            "avg_keyword_coverage": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_cost_usd": 0.0,
            "avg_generation_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "refusal_accuracy": 0.0,
        }

    answerable = [r for r in with_answer if r.type != QueryType.NO_ANSWER]
    refusable = [r for r in with_answer if r.type == QueryType.NO_ANSWER]

    avg_kw = (
        sum(r.answer_metrics.keyword_coverage for r in answerable) / len(answerable)
        if answerable else 0.0
    )
    refusal_acc = (
        sum(r.answer_metrics.keyword_coverage for r in refusable) / len(refusable)
        if refusable else 0.0
    )

    return {
        "avg_keyword_coverage": avg_kw,
        "avg_prompt_tokens": sum(r.answer_metrics.prompt_tokens for r in with_answer) / n,
        "avg_completion_tokens": sum(r.answer_metrics.completion_tokens for r in with_answer) / n,
        "avg_cost_usd": sum(r.answer_metrics.cost_usd for r in with_answer) / n,
        "total_cost_usd": sum(r.answer_metrics.cost_usd for r in with_answer),
        "avg_generation_latency_ms": (
            sum(r.answer_metrics.generation_latency_ms for r in with_answer) / n
        ),
        "refusal_accuracy": refusal_acc,
    }


def aggregate_by_domain(results: list[BenchmarkResult]) -> dict[str, dict[str, float]]:
    """Per-domain retrieval averages. Useful for spotting weak domains."""
    by_domain: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        if r.type == QueryType.NO_ANSWER:
            continue
        by_domain.setdefault(r.domain, []).append(r)

    out: dict[str, dict[str, float]] = {}
    for domain, rs in by_domain.items():
        n = len(rs)
        out[domain] = {
            "n": float(n),
            "hit_at_k": sum(1 for r in rs if r.retrieval.hit_at_k) / n,
            "precision_at_k": sum(r.retrieval.precision_at_k for r in rs) / n,
            "recall_at_k": sum(r.retrieval.recall_at_k for r in rs) / n,
            "mrr": sum(r.retrieval.mrr for r in rs) / n,
        }
    return out