"""
Rerank threshold tuning — LLM-free (but runs the cross-encoder).

For every test query, compute the max rerank score over the top-N
candidates. Then sweep thresholds, measuring:
  - NO_ANSWER recall: is_no_answer AND max < threshold  (caught)
  - false positives:  normal query AND max < threshold  (wrongly flagged)

The goal: a threshold that catches all NO_ANSWER queries with zero false
positives — closing the legal-010 gap the Week 4 router could not.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pareto.indexing import Indexer
from pareto.retrieval import HybridRetriever, BM25Ranker, CrossEncoderReranker

TEST_SET = Path("benchmarks/queries/queries.yaml")
CANDIDATES = 20
THRESHOLDS = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5]


def load_queries(path: Path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else data.get("queries", [])


def main():
    print("Loading index + retrievers + reranker...")
    indexer = Indexer.load("benchmarks/results/index")
    ranker = BM25Ranker(); ranker.build_from_records(indexer.store.records)
    hybrid = HybridRetriever(indexer=indexer, bm25_ranker=ranker)
    reranker = CrossEncoderReranker()

    queries = load_queries(TEST_SET)
    na_total = sum(1 for q in queries if q.get("type") == "no_answer")
    print(f"Scoring {len(queries)} queries ({na_total} NO_ANSWER)...\n")

    # Compute max rerank score per query once
    rows = []  # (id, is_na, max_score)
    for q in queries:
        cands = hybrid.search(q["query"], k=CANDIDATES)
        scores = reranker.score_pairs(q["query"], [c.record.content for c in cands])
        mx = max(scores) if scores else float("-inf")
        rows.append((q.get("id", "?"), q.get("type") == "no_answer", mx))

    # Sweep thresholds
    print(f"{'threshold':>10} | {'NA recall':>11} | {'false pos':>10}")
    print("-" * 40)
    for thr in THRESHOLDS:
        caught = sum(1 for _, is_na, mx in rows if is_na and mx < thr)
        fp = sum(1 for _, is_na, mx in rows if not is_na and mx < thr)
        print(f"{thr:>10.1f} | {caught:>4}/{na_total:<6} | {fp:>10}")

    # Show NO_ANSWER queries + their max scores
    print("\n=== NO_ANSWER queries (max rerank score) ===")
    for qid, is_na, mx in rows:
        if is_na:
            print(f"  {qid:<14} max={mx:+.2f}")

    # Show any normal query with a suspiciously low max (potential FP)
    print("\n=== Normal queries with max < 0 (false-positive risk) ===")
    risky = [(qid, mx) for qid, is_na, mx in rows if not is_na and mx < 0]
    if risky:
        for qid, mx in sorted(risky, key=lambda x: x[1]):
            print(f"  {qid:<14} max={mx:+.2f}")
    else:
        print("  (none — all normal queries score >= 0)")


if __name__ == "__main__":
    main()