"""
Bulk corpus fetcher for Pareto.

Uses the Wikipedia category API to pull many articles per domain, in both
English and Turkish, and saves each as an HTML file under
benchmarks/corpus/<domain>/, matching the existing corpus layout.

Strategy:
  - For each domain, list several categories (EN + TR).
  - For each category, ask the API for its member PAGES (not sub-categories).
  - For each page, fetch clean article HTML via the REST API.
  - Save to corpus/<domain>/<lang>_<slug>.html, skipping ones already present.

Usage (from repo root):
    python scripts/fetch_corpus.py
    python scripts/fetch_corpus.py --per-category 40 --out benchmarks/corpus

Tune CATEGORIES below to control size and topic mix. Be polite: the script
rate-limits itself. Stdlib only (urllib) — no extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

# --- Which categories to pull, per domain and language --------------------
# Category titles must exist on the respective Wikipedia. Adjust freely.
CATEGORIES: dict[str, dict[str, list[str]]] = {
    "legal": {
        "en": [
            "Data protection",
            "Privacy law",
            "Intellectual property law",
            "Information privacy",
        ],
        "tr": [
            "Hukuk",
            "Kişisel veriler",
        ],
    },
    "finance": {
        "en": [
            "Financial regulation",
            "Banking",
            "Monetary policy",
            "Financial risk",
        ],
        "tr": [
            "Finans",
            "Para politikası",
        ],
    },
    "health": {
        "en": [
            "Public health",
            "Nutrition",
            "Cardiovascular diseases",
            "Endocrine diseases",
        ],
        "tr": [
            "Sağlık",
            "Hastalıklar",
        ],
    },
}

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
WIKI_REST_HTML = "https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}"
UA = "ParetoCorpusFetcher/1.0 (graduation project; contact: latif)"


def _get(url: str, params: dict | None = None, timeout: int = 30) -> str:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def category_pages(lang: str, category: str, limit: int) -> list[str]:
    """Return up to `limit` article titles that are members of a category."""
    titles: list[str] = []
    cmcontinue = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}" if lang == "en" else f"Kategori:{category}",
            "cmtype": "page",           # pages only, skip sub-categories
            "cmlimit": min(limit - len(titles), 50),
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        try:
            data = json.loads(_get(WIKI_API.format(lang=lang), params))
        except Exception as e:
            print(f"    ! category '{category}' ({lang}) failed: {e}")
            break
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(0.3)
    return titles[:limit]


def fetch_html(lang: str, title: str) -> str | None:
    """Fetch clean article HTML for a page title."""
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    try:
        return _get(WIKI_REST_HTML.format(lang=lang, title=t))
    except Exception as e:
        print(f"    ! fetch '{title}' ({lang}) failed: {e}")
        return None


def slugify(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9ğüşıöç]+", "_", s)
    return s.strip("_")[:80]


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk-fetch Wikipedia corpus.")
    ap.add_argument("--out", default="benchmarks/corpus", help="corpus root")
    ap.add_argument("--per-category", type=int, default=30,
                    help="max pages per category")
    args = ap.parse_args()

    total_new = 0
    total_skip = 0
    for domain, langs in CATEGORIES.items():
        ddir = os.path.join(args.out, domain)
        os.makedirs(ddir, exist_ok=True)
        print(f"\n=== {domain} ===")
        seen_titles: set[str] = set()
        for lang, cats in langs.items():
            for cat in cats:
                print(f"  [{lang}] Category:{cat}")
                titles = category_pages(lang, cat, args.per_category)
                print(f"      {len(titles)} pages")
                for title in titles:
                    key = f"{lang}:{title}"
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    fname = f"{lang}_{slugify(title)}.html"
                    fpath = os.path.join(ddir, fname)
                    if os.path.exists(fpath):
                        total_skip += 1
                        continue
                    html = fetch_html(lang, title)
                    if not html or len(html) < 500:
                        continue
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(html)
                    total_new += 1
                    if total_new % 10 == 0:
                        print(f"      ... {total_new} saved")
                    time.sleep(0.4)  # be polite

    print(f"\nDone. {total_new} new documents saved, {total_skip} skipped "
          f"(already present).")
    print("Next: re-index with  pareto index benchmarks/corpus "
          "--output benchmarks/results/index")


if __name__ == "__main__":
    main()