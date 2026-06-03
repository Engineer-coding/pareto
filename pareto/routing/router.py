"""
QueryRouter — rule-based adaptive routing.

Consumes QuerySignals, produces a RouteDecision: which retriever, which
model tier, and WHY (every decision carries a human-readable reason).

Design philosophy:
- Explainable: each route has a reason string, surfaced in observability.
- Conservative: when uncertain, fall to hybrid (the strongest retriever).
- Grounded in measurement: rules encode Week 2-3 findings, not guesses.
    * NO_ANSWER → BM25  (Week 2: BM25 refusal accuracy 100% vs 75%)
    * Turkish   → Hybrid (BM25 multilingual layer essential)
    * Specific  → Hybrid (exact-term precision)
    * default   → Dense  (semantic, fast enough)

Rule-based, not ML: explainable, testable, sub-ms. A learned router is a
Week 5+ option if rules prove insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

from pareto.routing.signals import QuerySignals, extract_signals


@dataclass
class RouteDecision:
    """The router's verdict for one query."""
    retriever: str          # "dense" | "bm25" | "hybrid"
    model_tier: str         # "small" | "standard"
    reason: str             # which rule fired
    signals: QuerySignals   # for debugging / observability


class QueryRouter:
    """Rule-based query router."""

    def __init__(
        self,
        no_answer_threshold: float = 0.5,
        small_model_max_tokens: int = 8,
    ):
        self.no_answer_threshold = no_answer_threshold
        self.small_model_max_tokens = small_model_max_tokens

    def route(self, query: str) -> RouteDecision:
        sig = extract_signals(query)
        retriever, reason = self._choose_retriever(sig)
        model_tier = self._choose_model_tier(sig)
        return RouteDecision(
            retriever=retriever,
            model_tier=model_tier,
            reason=reason,
            signals=sig,
        )

    def _choose_retriever(self, sig: QuerySignals) -> tuple[str, str]:
        # Rule 1: NO_ANSWER risk → BM25 (refusal safety, Week 2 finding)
        if sig.no_answer_score >= self.no_answer_threshold:
            return "bm25", "no_answer_signal"
        # Rule 2: Turkish → Hybrid (multilingual BM25 layer)
        if sig.language == "tr":
            return "hybrid", "turkish_query"
        # Rule 3: Specific terms (acronyms/numbers) → Hybrid (keyword precision)
        if sig.is_specific:
            return "hybrid", "specific_terms"
        # Rule 4 (default): Dense (semantic, fast)
        return "dense", "default_semantic"

    def _choose_model_tier(self, sig: QuerySignals) -> str:
        # Short factual queries → small model; everything else → standard
        if (
            sig.query_type == "factual"
            and sig.length_tokens <= self.small_model_max_tokens
            and sig.no_answer_score < self.no_answer_threshold
        ):
            return "small"
        return "standard"

    def __repr__(self) -> str:
        return (
            f"QueryRouter(no_answer_threshold={self.no_answer_threshold}, "
            f"small_model_max_tokens={self.small_model_max_tokens})"
        )