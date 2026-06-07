"""
Cross-encoder reranker for two-stage retrieval.

Stage 1 (bi-encoder retrieval) returns a broad candidate set fast.
Stage 2 (this) re-scores each (query, candidate) pair jointly with a
cross-encoder — slower but more accurate than bi-encoder cosine, because
the model sees query and document together.

Score thresholding (Week 5): rerank() can drop candidates below a score
cutoff. Two effects with one mechanism:
  1. Garbage filtering — irrelevant candidates (negative scores) never
     reach the LLM context.
  2. NO_ANSWER signal — if NO candidate clears the threshold, the corpus
     has nothing relevant; the empty result is a post-retrieval NO_ANSWER
     signal the pre-retrieval router (Week 4) could not produce.

Multilingual model (mmarco-mMiniLMv2) so Turkish queries rerank correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RerankedHit:
    """A retrieval candidate after cross-encoder reranking."""
    record: Any              # ChunkRecord (same object retriever hits carry)
    rerank_score: float      # cross-encoder score (higher = more relevant)
    original_score: float    # the stage-1 retriever score
    original_rank: int       # 0-based position before reranking

    @property
    def score(self) -> float:
        """Alias so RerankedHit is drop-in compatible with retriever hits."""
        return self.rerank_score


class CrossEncoderReranker:
    """Stage-2 reranker. Re-scores (query, candidate) pairs jointly."""

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Low-level: score each (query, text) pair. Returns raw scores."""
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        hits: list,
        top_k: int = 5,
        score_threshold: float | None = None,
        min_keep: int = 1,
    ) -> list[RerankedHit]:
        """
        Re-score retriever hits and return top_k reranked.

        If score_threshold is set, candidates below it are dropped — BUT at
        least `min_keep` hits are always returned (the top scorers), so a
        hard-but-answerable query is never emptied. This makes thresholding
        a garbage filter, not an automatic NO_ANSWER mechanism: Week 5
        benchmarking showed rerank score alone cannot separate NO_ANSWER
        from hard-but-answerable queries (they overlap: a normal query
        scored -4.36 while a NO_ANSWER query scored -2.73). The LLM still
        makes the final refusal call from the (cleaned) context.

        Each hit must expose `.record` (with `.content`) and `.score`.
        """
        if not hits:
            return []
        texts = [h.record.content for h in hits]
        scores = self.score_pairs(query, texts)

        scored = [
            (hit, score, rank)
            for rank, (hit, score) in enumerate(zip(hits, scores))
        ]
        scored.sort(key=lambda x: -x[1])

        if score_threshold is not None:
            kept = [s for s in scored if s[1] >= score_threshold]
            if len(kept) < min_keep:
                kept = scored[:min_keep]  # never empty an answerable query
            scored = kept

        return [
            RerankedHit(
                record=hit.record,
                rerank_score=score,
                original_score=hit.score,
                original_rank=rank,
            )
            for hit, score, rank in scored[:top_k]
        ]

    def max_score(self, query: str, hits: list) -> float:
        """
        Highest rerank score among hits — the post-retrieval NO_ANSWER
        signal. Low max → no candidate matches → likely NO_ANSWER.
        """
        if not hits:
            return float("-inf")
        texts = [h.record.content for h in hits]
        return max(self.score_pairs(query, texts))

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.model_name!r})"