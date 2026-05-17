"""
Classic inverted index data structure.

A textbook inverted index: a mapping from each term to a posting list,
where each posting records (doc_id, term_frequency). This is the
foundation of BM25 scoring.

Memory footprint is dominated by the term vocabulary; for 380 chunks
of mixed-language content we expect ~10-20k unique terms, ~200KB RAM.
Save/load uses JSON for portability; for >1M chunks we'd switch to
a binary format (msgpack, pickle, or LMDB).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


class InvertedIndex:
    """
    Maps term -> {doc_id: term_frequency}.
    Tracks per-document length and global average for BM25 normalization.
    """

    def __init__(self):
        # term -> {doc_id: tf}
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        # doc_id -> total token count
        self.doc_lengths: dict[int, int] = {}
        # cached aggregate stats — computed by finalize()
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        """Add one document to the index. Caller ensures unique doc_ids."""
        if doc_id in self.doc_lengths:
            raise ValueError(f"doc_id {doc_id} already indexed")

        # Counter handles term frequency in O(len(tokens))
        tf_map = Counter(tokens)
        self.doc_lengths[doc_id] = sum(tf_map.values())

        for term, tf in tf_map.items():
            self.postings[term][doc_id] = tf

    def finalize(self) -> None:
        """Compute aggregate statistics. Call after all add_document() calls."""
        self.doc_count = len(self.doc_lengths)
        if self.doc_count == 0:
            self.avg_doc_length = 0.0
        else:
            total = sum(self.doc_lengths.values())
            self.avg_doc_length = total / self.doc_count

    # ── stats / introspection ─────────────────────────────────────────────
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def document_frequency(self, term: str) -> int:
        """How many documents contain this term?"""
        return len(self.postings.get(term, {}))

    def get_postings(self, term: str) -> dict[int, int] | None:
        """Return {doc_id: tf} for term, or None if term not in vocabulary."""
        return self.postings.get(term)

    def __repr__(self) -> str:
        return (
            f"InvertedIndex(docs={self.doc_count}, "
            f"vocab={self.vocabulary_size()}, "
            f"avg_doc_len={self.avg_doc_length:.1f})"
        )

    # ── persistence ───────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        # Convert int keys to strings for JSON; we'll reverse on load
        return {
            "postings": {
                term: {str(d): tf for d, tf in posting.items()}
                for term, posting in self.postings.items()
            },
            "doc_lengths": {str(d): l for d, l in self.doc_lengths.items()},
            "avg_doc_length": self.avg_doc_length,
            "doc_count": self.doc_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InvertedIndex":
        idx = cls()
        idx.postings = defaultdict(
            dict,
            {
                term: {int(d): int(tf) for d, tf in posting.items()}
                for term, posting in data["postings"].items()
            },
        )
        idx.doc_lengths = {int(d): int(l) for d, l in data["doc_lengths"].items()}
        idx.avg_doc_length = float(data["avg_doc_length"])
        idx.doc_count = int(data["doc_count"])
        return idx

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)