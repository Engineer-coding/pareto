"""
Lightweight tokenization for BM25.

Design:
    * Unicode-aware (handles Turkish, accented chars correctly)
    * Lowercase normalization (NFKC fold, then casefold)
    * Stopword filtering (English + Turkish minimal lists, no external deps)
    * Numeric tokens kept (years, regulation numbers like "GDPR 25" matter)
    * No stemming — E5 dense embeddings already capture morphology;
      BM25's job is exact-keyword recall, not fuzzy matching

A real production tokenizer would add: language detection, language-specific
stemming (Snowball for English, Zemberek for Turkish), domain-specific
keep-lists. We document those as future hooks but keep the MVP minimal.
"""

from __future__ import annotations

import re
import unicodedata


# Unicode-aware word boundary (handles Turkish, accents)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# Minimal stopword sets — chosen for high frequency, low information content.
# Both lists kept short on purpose; over-aggressive stopword removal hurts
# retrieval (e.g. "to be or not to be" → empty after stopword removal).
STOPWORDS_EN: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "from", "by", "for", "with",
    "as", "that", "this", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "them", "his", "her",
    "do", "does", "did", "have", "has", "had",
    "not", "no", "so", "such", "than",
})

STOPWORDS_TR: frozenset[str] = frozenset({
    "ve", "veya", "ile", "için", "ama", "fakat", "ancak",
    "bir", "bu", "şu", "o", "bunu", "şunu", "onu",
    "ne", "ki", "de", "da", "mi", "mı", "mu", "mü",
    "olan", "olarak", "olur", "olabilir", "değil",
    "her", "hiç", "tüm", "bütün", "bazı",
    "var", "yok", "gibi", "kadar",
})

STOPWORDS: frozenset[str] = STOPWORDS_EN | STOPWORDS_TR


def tokenize(text: str, min_length: int = 2) -> list[str]:
    """Lowercase + unicode-normalize + split + drop stopwords/short tokens.

    Pure function — no global state, deterministic, fast.
    """
    if not text:
        return []

    # NFKC normalizes "ﬁ" → "fi", "ı" stays, full-width chars → ASCII, etc.
    # casefold is more aggressive than lower() (handles 'İ' → 'i̇', 'ß' → 'ss').
    text = unicodedata.normalize("NFKC", text).casefold()

    tokens = _TOKEN_RE.findall(text)
    return [
        t for t in tokens
        if len(t) >= min_length and t not in STOPWORDS
    ]