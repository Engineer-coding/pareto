"""
Reciprocal Rank Fusion (RRF) — combine multiple ranked retrieval lists
into a single ranking.

Reference:
    Cormack, Clarke, Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods."
    SIGIR Conf. on Information Retrieval.

Why RRF over linear score combination?
    BM25 scores are unbounded (typically 0 to 30+). Dense embedding cosine
    scores are bounded (-1 to 1). A naive `α · bm25 + (1-α) · dense` would
    let BM25 dominate. RRF works on RANKS, not SCORES, so it's invariant
    to each system's score distribution.

Default k = 60 from the original TREC paper. Larger k smooths rank
contributions (each position contributes more equally); smaller k makes
top-ranked items dominate. We expose it for tuning in Week 5.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists into one fused ranking.

    Args:
        ranked_lists: List of ranked lists. Each list contains doc IDs in
            descending order of relevance (position 0 = top-1).
        k: Smoothing constant. Default 60 per the TREC paper.
        weights: Optional per-system weights. Length must match
            len(ranked_lists). Defaults to uniform weight 1.0 for all.

    Returns:
        List of (doc_id, fused_score), sorted by fused_score descending.

    Properties:
        - Distribution-invariant: each system's score range is irrelevant.
        - Documents in multiple lists get boosted (additive contributions).
        - Documents in only one list still contribute (1/(k+rank)) but less.
        - Tie-breaking is by Python's stable sort (insertion order in dict).
    """
    if not ranked_lists:
        return []

    n_systems = len(ranked_lists)

    if weights is None:
        weights = [1.0] * n_systems
    elif len(weights) != n_systems:
        raise ValueError(
            f"weights length {len(weights)} != ranked_lists length {n_systems}"
        )

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    fused_scores: dict[str, float] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked_list, start=1):
            contribution = weight / (k + rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + contribution

    # Sort by fused score descending
    return sorted(fused_scores.items(), key=lambda x: -x[1])