"""
BM25Okapi ranking on top of an inverted index.

Formula (Robertson et al., 1995):

    score(q, d) = Σ IDF(qᵢ) · (f(qᵢ, d) · (k1 + 1)) / (f(qᵢ, d) + k1 · (1 - b + b · |d| / avgdl))

    IDF(qᵢ) = log((N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5) + 1)

Where:
    f(qᵢ, d)  : term frequency of qᵢ in document d
    |d|       : document length (token count)
    avgdl     : average document length across corpus
    N         : total documents
    n(qᵢ)     : number of documents containing qᵢ (document frequency)
    k1        : term frequency saturation parameter (default 1.5)
    b         : length normalization parameter (default 0.75)

Notes on the IDF variant:
    We use the "+1" smoothed BM25 IDF (rank_bm25's default, also Lucene's),
    which keeps scores non-negative even for common terms. Pure Robertson
    IDF can go negative for terms appearing in >50% of docs.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from pareto.retrieval.inverted_index import InvertedIndex
from pareto.retrieval.tokenizer import tokenize


@dataclass
class BM25Config:
    k1: float = 1.5      # term frequency saturation
    b: float = 0.75      # length normalization weight
    min_token_length: int = 2


@dataclass
class BM25Hit:
    """A single search result. Mirrors the shape of vector store hits for
    drop-in compatibility in HybridRetriever (Çarşamba)."""
    doc_id: int
    score: float
    record: "ChunkRecord" = field(default=None)  # type: ignore  # late binding


class BM25Ranker:
    """
    BM25 retrieval over a corpus of chunks.

    Lifecycle:
        ranker = BM25Ranker()
        ranker.build_from_records(records)   # one-time
        hits = ranker.search("query", k=5)   # many times
        ranker.save("path/to/bm25/")
    """

    def __init__(self, config: BM25Config | None = None):
        self.config = config or BM25Config()
        self.index = InvertedIndex()
        # Parallel array — doc_id -> ChunkRecord (same indexing as VectorStore)
        self._records: list = []
        self._built: bool = False

    # ── build ─────────────────────────────────────────────────────────────
    def build_from_records(self, records: list) -> None:
        """Tokenize each record's content, build the inverted index.

        `records` must be a list of ChunkRecord-shaped objects with a
        `.content` string attribute. Doc IDs are positional (index in list).
        """
        if self._built:
            raise RuntimeError("BM25Ranker.build_from_records called twice")

        for doc_id, rec in enumerate(records):
            tokens = tokenize(rec.content, min_length=self.config.min_token_length)
            if not tokens:
                # Empty doc still gets a slot (preserves doc_id alignment)
                self.index.doc_lengths[doc_id] = 0
                self._records.append(rec)
                continue
            self.index.add_document(doc_id, tokens)
            self._records.append(rec)

        self.index.finalize()
        self._built = True

    # ── search ────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[BM25Hit]:
        """Return top-k hits ranked by BM25 score (descending)."""
        if not self._built:
            raise RuntimeError("Call build_from_records() before search()")

        query_tokens = tokenize(query, min_length=self.config.min_token_length)
        if not query_tokens:
            return []

        N = self.index.doc_count
        if N == 0:
            return []

        avgdl = self.index.avg_doc_length
        k1 = self.config.k1
        b = self.config.b

        # Accumulate per-document scores across all query terms
        scores: dict[int, float] = {}

        for term in query_tokens:
            postings = self.index.get_postings(term)
            if not postings:
                continue  # term not in corpus, contributes 0

            df = len(postings)
            # Smoothed BM25 IDF (Lucene variant)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in postings.items():
                doc_len = self.index.doc_lengths.get(doc_id, 0)
                # Length normalization factor
                norm = 1.0 - b + b * (doc_len / avgdl if avgdl > 0 else 1.0)
                # BM25 term contribution
                tf_component = (tf * (k1 + 1)) / (tf + k1 * norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_component

        # Sort by score, take top-k
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]

        return [
            BM25Hit(doc_id=doc_id, score=score, record=self._records[doc_id])
            for doc_id, score in top
        ]

    # ── stats ─────────────────────────────────────────────────────────────
    @property
    def doc_count(self) -> int:
        return self.index.doc_count

    @property
    def vocabulary_size(self) -> int:
        return self.index.vocabulary_size()

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, directory: str | Path) -> Path:
        """
        Persist to a directory:
            bm25/
              ├── index.json   — inverted index
              └── records.pkl  — parallel ChunkRecord array

        Pickle for records is pragmatic; they contain Pydantic models with
        arbitrary nested structures. We don't ship pickle files between
        machines without trust — within one project they're fine.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.index.save(d / "index.json")
        with open(d / "records.pkl", "wb") as f:
            pickle.dump(self._records, f)
        # Persist config too
        (d / "config.json").write_text(
            f'{{"k1": {self.config.k1}, "b": {self.config.b}}}',
            encoding="utf-8",
        )
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Ranker":
        d = Path(directory)
        import json
        config_data = json.loads((d / "config.json").read_text(encoding="utf-8"))
        ranker = cls(config=BM25Config(**config_data))
        ranker.index = InvertedIndex.load(d / "index.json")
        with open(d / "records.pkl", "rb") as f:
            ranker._records = pickle.load(f)
        ranker._built = True
        return ranker

    def __repr__(self) -> str:
        return (
            f"BM25Ranker(docs={self.doc_count}, "
            f"vocab={self.vocabulary_size}, "
            f"k1={self.config.k1}, b={self.config.b}, "
            f"built={self._built})"
        )