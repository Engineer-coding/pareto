"""
Offline threshold tuning for the semantic cache.

Cache hit/miss is purely embedding-similarity vs threshold — no LLM call
needed. So we embed all 45 queries once, then sweep thresholds in seconds,
classifying each hit as TRUE (correct paraphrase match) or FALSE (wrong
entry) using ground-truth paraphrase mappings from the test set's notes.

This gives the precise false-hit rate that the LLM-in-the-loop benchmark
(Thursday B) could only show indirectly via keyword-coverage collapse.

Run: python scripts/tune_threshold.py
"""

import re

import numpy as np

from pareto.benchmark.test_set import TestSet
from pareto.indexing import SentenceTransformerEmbedder


def build_paraphrase_map(queries) -> dict[str, str]:
    """Parse 'paraphrase of <id>' from each dupe's notes field."""
    para_map: dict[str, str] = {}
    for q in queries:
        if not q.id.startswith("dupe"):
            continue
        note = q.notes or ""
        m = re.search(r"paraphrase of ([\w-]+)", note, re.IGNORECASE)
        if m:
            para_map[q.id] = m.group(1)
    return para_map


def simulate(queries, embeddings, threshold, para_map):
    """
    Replay queries in order against a growing cache.
    Returns (hits, true_hits, false_hits, false_detail, cache_size).
    """
    cache: list[tuple[str, np.ndarray]] = []  # (query_id, embedding)
    hits = true_hits = false_hits = 0
    false_detail: list[tuple[str, str, float]] = []  # (query, matched, sim)

    for q in queries:
        q_emb = embeddings[q.id]
        best_sim, best_id = -1.0, None
        for cached_id, cached_emb in cache:
            sim = float(q_emb @ cached_emb)
            if sim > best_sim:
                best_sim, best_id = sim, cached_id

        if best_sim >= threshold:
            hits += 1
            expected = para_map.get(q.id)  # None for originals
            if expected is not None and best_id == expected:
                true_hits += 1
            else:
                false_hits += 1
                false_detail.append((q.id, best_id, best_sim))
            # hit → cache returns existing entry, nothing added
        else:
            cache.append((q.id, q_emb))  # miss → cache it

    return hits, true_hits, false_hits, false_detail, len(cache)


def main():
    ts = TestSet.from_yaml("benchmarks/queries/queries_with_dupes.yaml")
    queries = ts.queries
    para_map = build_paraphrase_map(queries)

    print(f"Loaded {len(queries)} queries, {len(para_map)} paraphrase mappings")
    print("Embedding all queries (once)...")
    emb = SentenceTransformerEmbedder()
    embeddings = {q.id: emb.encode_query(q.query) for q in queries}

    n_paraphrases = len(para_map)
    thresholds = [0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]

    print(f"\n{'thr':>5} | {'hits':>4} | {'true':>4} | {'false':>5} | "
          f"{'recall':>6} | {'precision':>9} | {'hit_rate':>8}")
    print("-" * 62)

    results = []
    for t in thresholds:
        hits, true_h, false_h, fdetail, size = simulate(
            queries, embeddings, t, para_map
        )
        recall = true_h / n_paraphrases if n_paraphrases else 0.0  # of 15 paraphrases caught
        precision = true_h / hits if hits else 0.0                 # of hits, how many correct
        hit_rate = hits / len(queries)
        results.append((t, hits, true_h, false_h, recall, precision, hit_rate, fdetail))
        print(f"{t:>5} | {hits:>4} | {true_h:>4} | {false_h:>5} | "
              f"{recall:>6.1%} | {precision:>9.1%} | {hit_rate:>8.1%}")

    # Show false hits at the conservative end (threshold where false first hits 0)
    print("\n=== False hit detail (per threshold) ===")
    for t, hits, true_h, false_h, recall, precision, hit_rate, fdetail in results:
        if fdetail:
            print(f"\nthreshold={t} — {len(fdetail)} false hits:")
            for qid, matched, sim in fdetail[:8]:
                print(f"  {qid} → matched {matched} (sim={sim:.4f})")
            if len(fdetail) > 8:
                print(f"  ... and {len(fdetail) - 8} more")
        else:
            print(f"\nthreshold={t} — ZERO false hits ✓")

    # Recommendation
    print("\n=== Recommendation ===")
    zero_false = [r for r in results if r[3] == 0]
    if zero_false:
        # Among zero-false-hit thresholds, pick the one with highest recall
        best = max(zero_false, key=lambda r: r[4])
        print(f"Lowest threshold with ZERO false hits: {best[0]}")
        print(f"  → catches {best[2]}/{n_paraphrases} paraphrases ({best[4]:.1%} recall)")
        print(f"  → overall hit rate {best[6]:.1%}")
    else:
        print("No threshold eliminates all false hits in this sweep.")
        print("Consider the precision/recall tradeoff manually.")


if __name__ == "__main__":
    main()