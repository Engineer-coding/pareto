"""
Router threshold tuning — LLM-free.

Replays the test set through the router at several no_answer_thresholds,
measuring NO_ANSWER recall (caught → bm25) vs false positives (a normal
query wrongly sent to bm25). No LLM calls; runs in seconds.

Week 4 finding: at the default 0.50, only 2/4 NO_ANSWER queries reach bm25
(health-010 scores 0.40, just under). This script checks whether lowering
the threshold catches more NO_ANSWER queries without creating false
positives on normal queries.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pareto.routing import QueryRouter

TEST_SET = Path("benchmarks/queries/queries.yaml")
THRESHOLDS = [0.50, 0.45, 0.40, 0.35, 0.30]


def load_queries(path: Path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else data.get("queries", [])


def main():
    queries = load_queries(TEST_SET)
    na_total = sum(1 for q in queries if q.get("type") == "no_answer")
    print(f"Loaded {len(queries)} queries ({na_total} NO_ANSWER)\n")

    print(f"{'threshold':>10} | {'NA recall':>11} | {'false pos':>10}")
    print("-" * 40)
    for thr in THRESHOLDS:
        router = QueryRouter(no_answer_threshold=thr)
        caught = fp = 0
        for q in queries:
            d = router.route(q["query"])
            is_na = q.get("type") == "no_answer"
            bm25 = d.retriever == "bm25"
            if is_na and bm25:
                caught += 1
            elif not is_na and bm25:
                fp += 1
        print(f"{thr:>10.2f} | {caught:>4}/{na_total:<6} | {fp:>10}")

    # Detail: queries newly routed to bm25 when going 0.50 -> 0.40
    print("\n=== Routed to bm25 at 0.40 but NOT at 0.50 ===")
    r50 = QueryRouter(no_answer_threshold=0.50)
    r40 = QueryRouter(no_answer_threshold=0.40)
    any_change = False
    for q in queries:
        d50 = r50.route(q["query"])
        d40 = r40.route(q["query"])
        if d40.retriever == "bm25" and d50.retriever != "bm25":
            any_change = True
            tag = "NO_ANSWER" if q.get("type") == "no_answer" else "normal!!!"
            print(f"  [{tag:>9}] {q['id']}: score={d40.signals.no_answer_score:.2f}")
            print(f"             {q['query']}")
    if not any_change:
        print("  (none)")


if __name__ == "__main__":
    main()