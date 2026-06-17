"""
Warm the API's in-memory semantic cache before a demo.

The FastAPI server keeps its SemanticCache in memory (per-process). After
`pareto serve` starts, the cache is empty, so the first time each query is
asked it pays the full LLM cost (~50s on CPU). This script pre-runs the demo
queries once so that, during the live demo, paraphrases hit the cache and
return in ~50ms.

Usage (with the server already running -- `pareto serve` in another terminal):
    python scripts/warm_cache.py
    python scripts/warm_cache.py --url http://localhost:8000

Stdlib only (urllib) -- no extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import urllib.error


# Queries that will be demonstrated live. We warm the *originals* here; in the
# demo we ask paraphrases of these, which then hit the cache instantly.
WARM_QUERIES: list[dict] = [
    # Legal -- the headline cache demo (real GDPR content)
    {"question": "What rights do data subjects have under GDPR?"},
    {"question": "What is GDPR and when did it take effect?"},
    # Finance -- shown without router so the 3b answer stays clean
    {"question": "What is Basel III?"},
    # Health
    {"question": "What is considered high blood pressure?"},
]


def ask(url: str, body: dict, timeout: int = 180) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/ask",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/health", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm the Pareto API cache.")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running API.")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    # Connectivity check
    try:
        h = health(url)
    except urllib.error.URLError as e:
        print(f"[warm] Cannot reach {url}. Is `pareto serve` running? ({e})")
        raise SystemExit(1)

    print(f"[warm] Server up: {h.get('index_size')} chunks, "
          f"cache_size={h.get('cache_size')}")
    print(f"[warm] Warming {len(WARM_QUERIES)} queries "
          f"(each pays full LLM cost once -- this is slow, be patient)...\n")

    t0 = time.perf_counter()
    for i, body in enumerate(WARM_QUERIES, 1):
        q = body["question"]
        t = time.perf_counter()
        try:
            r = ask(url, body)
        except Exception as e:
            print(f"  [{i}/{len(WARM_QUERIES)}] FAILED: {q[:50]}... ({e})")
            continue
        dt = time.perf_counter() - t
        cached = r.get("cache_hit")
        tag = "cache hit" if cached else "warmed"
        print(f"  [{i}/{len(WARM_QUERIES)}] {tag} in {dt:5.1f}s : {q[:55]}")

    total = time.perf_counter() - t0
    h2 = health(url)
    print(f"\n[warm] Done in {total:.0f}s. cache_size now {h2.get('cache_size')}.")
    print("[warm] In the demo, ask PARAPHRASES of these to get instant cache hits.")


if __name__ == "__main__":
    main()