"""
Query signal extraction for the adaptive router.

Pure feature extraction — no routing decisions here. The router consumes
these signals and applies rules. Separation makes both testable in
isolation and lets a future ML router consume the same QuerySignals.

All heuristics are deliberately simple and dependency-free (no langdetect,
no spaCy). Explainable and fast (sub-ms). Week 5+ can swap in heavier NLP
if the rule-based approach proves insufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Turkish-specific characters (strong language signal)
_TURKISH_CHARS = set("şğıüöçŞĞİÜÖÇ")

# Common Turkish function words (for queries without special chars, e.g. "GDPR nedir")
_TURKISH_WORDS = {
    "nedir", "ne", "nasıl", "kaç", "hangi", "nerede", "niçin", "neden",
    "için", "ile", "mi", "mı", "mu", "mü", "midir", "var", "yok", "nelerdir",
    "kapsamı", "belirtileri", "anlamına", "açıkla", "açıklayın", "söyle",
}

# Temporal / real-time keywords → likely NO_ANSWER (corpus is static)
_TEMPORAL_KEYWORDS = {
    "today", "current", "currently", "now", "latest", "recent", "real-time",
    "realtime", "this week", "this month", "this year", "live", "up-to-date",
    "stock price", "share price", "exchange rate", "projected", "forecast",
}

# Year patterns suggesting real-time / future data the static corpus lacks
_FUTURE_YEAR = re.compile(r"\b(202[5-9]|20[3-9]\d)\b")

# Summary intent
_SUMMARY_KEYWORDS = {
    "summarize", "summary", "overview", "main goals", "main objectives",
    "key points", "özetle", "özet",
}

# Yes/no intent (sentence starters)
_YESNO_STARTERS = {"does", "is", "are", "can", "do", "did", "will", "has", "have"}

# Acronym pattern (2+ uppercase): GDPR, LCR, NSFR, KVKK, EU, III
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")

# Number / article-reference: "Article 17", "130", "Q4", "18"
_NUMBER = re.compile(r"\b\d+\b|\bArticle\s+\d+\b|\bQ[1-4]\b")


@dataclass
class QuerySignals:
    """Extracted features for routing. Pure data, no decisions."""
    query: str
    length_tokens: int
    length_chars: int
    language: str               # "tr" | "en"
    has_temporal: bool          # real-time keywords → NO_ANSWER risk
    has_acronym: bool           # GDPR, LCR → BM25 keyword precision helps
    has_number: bool            # Article 17, Q4 → specific
    query_type: str             # "factual" | "summary" | "yes_no"
    no_answer_score: float      # 0..1 heuristic

    @property
    def is_specific(self) -> bool:
        """Has exact tokens (acronyms/numbers) that BM25 matches well."""
        return self.has_acronym or self.has_number


def detect_language(query: str) -> str:
    """Cheap Turkish vs English detector. No external deps."""
    if any(ch in _TURKISH_CHARS for ch in query):
        return "tr"
    words = {w.strip("?.,!").lower() for w in query.split()}
    if words & _TURKISH_WORDS:
        return "tr"
    return "en"


def _detect_query_type(query_lower: str, first_word: str) -> str:
    if any(kw in query_lower for kw in _SUMMARY_KEYWORDS):
        return "summary"
    if first_word in _YESNO_STARTERS:
        return "yes_no"
    return "factual"


def _no_answer_score(query: str, query_lower: str) -> float:
    """
    Heuristic 0..1 score for 'corpus probably can't answer this'.
    Static corpus → real-time/future requests are likely NO_ANSWER.
    """
    score = 0.0
    if any(kw in query_lower for kw in _TEMPORAL_KEYWORDS):
        score += 0.6
    if _FUTURE_YEAR.search(query):
        score += 0.4
    return min(score, 1.0)


def extract_signals(query: str) -> QuerySignals:
    """Extract all routing features from a raw query string."""
    q = query.strip()
    q_lower = q.lower()
    tokens = q.split()
    first_word = tokens[0].lower().strip("?.,!") if tokens else ""

    return QuerySignals(
        query=q,
        length_tokens=len(tokens),
        length_chars=len(q),
        language=detect_language(q),
        has_temporal=any(kw in q_lower for kw in _TEMPORAL_KEYWORDS),
        has_acronym=bool(_ACRONYM.search(q)),
        has_number=bool(_NUMBER.search(q)),
        query_type=_detect_query_type(q_lower, first_word),
        no_answer_score=_no_answer_score(q, q_lower),
    )