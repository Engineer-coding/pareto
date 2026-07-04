<div align="center">

# Pareto

**Cost-optimized, open-source RAG infrastructure.**
*80% of the quality, 20% of the cost.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Week 3 of 8](https://img.shields.io/badge/status-week%203%20of%208-orange.svg)]()

</div>

---

Most RAG frameworks help you *build* RAG.
Pareto helps you *afford* it.

Production RAG systems are expensive: high LLM call volume, large context windows, repeated queries hitting the LLM unnecessarily, and noisy retrieval. Pareto attacks every one of these problems with opinionated, **measurable** engineering.

Every optimization layer in Pareto must beat the naive baseline on a published benchmark — or it doesn't ship. And every optimization must preserve answer quality, not just cut cost.

## Why Pareto

LangChain, LlamaIndex, Haystack and friends make it *easy* to assemble a RAG. None of them make it *cheap to run in production*. The bills only show up after launch — and by then the architecture is locked in.

Pareto inverts the priority. Cost is a first-class concern in every layer:

- **Hybrid retrieval** — BM25 + dense vectors via Reciprocal Rank Fusion (Week 2 ✓)
- **Hierarchical chunking** — tree-based parent-child retrieval (Week 1 ✓)
- **HNSW vector index** — tuned for throughput, runtime ef_search adjustable (Week 1 ✓)
- **Built-in cost & latency observability** — SQLite query log + `pareto stats` (Week 2 ✓)
- **Semantic caching** — semantically similar queries reuse cached responses, ~1264x faster, $0 (Week 3 ✓)
- **Adaptive query routing** — simple queries don't need the expensive model (Week 4)
- **Lightweight knowledge graph** — entity-based augmentation for multi-hop queries (Week 7)

Every layer can be benchmarked independently. The repo includes the test set, the metrics framework, and persisted baseline reports for every milestone.

## Current Baseline

### Retrieval (30 queries, k=5)

| Metric | Dense | BM25 | **Hybrid (default)** |
| :--- | ---: | ---: | ---: |
| hit@5 | 100.00% | 100.00% | 100.00% |
| **MRR** | 0.981 | 0.981 | **1.000** |
| avg retrieval latency | 29 ms | **0 ms** | 32 ms |

### Semantic Cache (45-query test set, 30 original + 15 paraphrase)

| Threshold | Hit Rate | Keyword Coverage | Verdict |
| :--- | ---: | ---: | :--- |
| 0.92 (default) | 24.4% | 0.878 | Quality preserved ✓ |
| 0.85 | 71.1% | 0.634 | Too aggressive — false hits |

Cache hit latency **~76 ms** vs **~96,000 ms** for a fresh LLM call on CPU — a **1264x speedup**, at **$0** cost.

## Semantic Cache

A query that's semantically similar to a previously-answered one returns the cached response instantly — no LLM call, no cost. The match is by embedding cosine similarity, not exact text, so paraphrases hit too.

```bash
pareto ask "What is GDPR?"        # cold: ~96s, full LLM
pareto ask "What is GDPR?"        # hit:  ~76ms, $0  ⚡
pareto ask "Explain GDPR" --cache-threshold 0.90   # paraphrase hit ⚡
```

**Threshold is a quality/cost tradeoff, not a free dial.** Lowering it raises hit rate but introduces *false hits* — semantically-near but distinct queries returning the wrong answer. Pareto reports the **quality-adjusted** hit rate: keyword coverage is the canary, and it collapses (0.878 → 0.634) when the threshold drops too far. We don't chase a bigger number by sacrificing correctness.

**Cache invalidation is automatic.** Each entry remembers which chunks produced its answer (Week 1's deterministic chunk IDs). When the corpus changes, stale entries are skipped — no manual cache clear needed.

**Cost projection** (GPT-4o, 100K queries/day, conservative 24.4% hit):

| Provider | Annual (no cache) | Saved by cache |
| :--- | ---: | ---: |
| GPT-4o | $93,075 | **$22,710** |
| Claude Sonnet 4.6 | $122,640 | **$29,924** |

Full breakdown in [`docs/COST_PROJECTION.md`](./docs/COST_PROJECTION.md).

## Quickstart

```bash
git clone https://github.com/Engineer-coding/pareto.git
cd pareto
uv venv --python 3.10
.venv/Scripts/Activate.ps1     # Windows
# source .venv/bin/activate    # Linux/macOS
uv pip install -e .

ollama pull llama3.2:3b        # optional, $0-cost local LLM

pareto index ./benchmarks/corpus
pareto ask "What are the key principles of GDPR?"

# Cache in action
pareto ask "What is GDPR?"                          # cold
pareto ask "What is GDPR?"                           # cache hit ⚡
pareto ask "Explain GDPR" --cache-threshold 0.90    # paraphrase hit
pareto ask "What is GDPR?" --no-cache                # bypass cache

# Different retrievers
pareto ask "GDPR Article 17?" --retriever bm25

# Observability
pareto stats                    # overall + cache hit rate + savings
pareto stats --by cache_hit     # hit vs miss latency breakdown

# Benchmark
pareto benchmark --retriever hybrid --mode retrieval --k 5
pareto benchmark --test-set benchmarks/queries/queries_with_dupes.yaml \
    --mode end_to_end --cache-threshold 0.92
```

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Ingestion  │───▶│   Chunking   │───▶│   Indexing   │
│   5 formats  │    │  Hierarchical│    │  FAISS HNSW  │
│   Pydantic   │    │  Tree-based  │    │  + BM25 inv. │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                        ┌───────────────────────┘
                        ▼
                ┌────────────────┐
   query  ────▶ │ Semantic Cache │ ──(hit, ~76ms, $0)──▶ response ⚡
                └───────┬────────┘
                        │ (miss)
                        ▼
             ┌────────────────────┐
             │   Retrieval Layer  │
             │  Dense/BM25/Hybrid │
             │      (via RRF)     │
             └──────────┬─────────┘
                        ▼
             ┌────────────────────┐
             │     Generation     │
             │  LiteLLM wrapper   │
             └──────────┬─────────┘
                        │
                        ├──▶ Cache write (with chunk provenance)
                        ▼
             ┌────────────────────┐
             │   Observability    │
             │  SQLite query log  │
             │  cache hit rate +  │
             │  savings projection│
             └────────────────────┘

    ◀── Benchmark Suite (P@k, R@k, MRR, refusal acc, cache hit rate) ──▶
    ◀── Multilingual (English + Turkish, 100+ via E5) ──▶
```

## CLI

| Command | What it does |
| :--- | :--- |
| `pareto ingest <dir>` | Load supported documents from a directory |
| `pareto chunk <file>` | Build and visualize a chunk tree |
| `pareto index <dir>` | Ingest + chunk + embed + persist a vector index |
| `pareto search "<query>"` | Semantic search over a saved index |
| `pareto ask "<question>"` | Full RAG with cache (`--retriever`, `--no-cache`, `--cache-threshold`) |
| `pareto stats` | Cost, latency, cache hit rate, savings projection |
| `pareto serve` | Start the FastAPI HTTP server |
| `pareto benchmark` | Run the test set, emit a JSON report (`--test-set`, `--cache-threshold`) |

### `pareto stats` Examples

```bash
pareto stats                       # overall + cache section + savings
pareto stats --by retriever        # hit rate per retriever
pareto stats --by cache_hit        # hit vs miss latency (488x gap)
pareto stats --since 24h           # last 24 hours
pareto stats --slow 30000          # slow queries (cache hits excluded)
```

## Roadmap

| Week | Milestone | Status |
| :---: | :--- | :---: |
| 1 | Scaffolding, hierarchical chunking, naive baseline, benchmark suite | ✅ |
| 2 | Hybrid retrieval (BM25 + dense), observability | ✅ |
| 3 | Semantic cache (LRU + embedding similarity) | ✅ |
| 4 | Adaptive query router | 🚧 |
| 5 | HNSW tuning, cross-encoder reranking | ⏳ |
| 6 | MCP server + client integration | ⏳ |
| 7 | Light knowledge graph (entity extraction + linking) | ⏳ |
| 8 | Vertical demo apps, documentation, public demo | ⏳ |

## Design Principles

1. **Data-driven engineering.** Every optimization layer must beat the previous benchmark on at least one metric or it doesn't ship.
2. **Quality is non-negotiable.** A cost cut that degrades answers isn't a win. Cache reports a quality-adjusted hit rate; false hits are counted as regressions, not savings.
3. **Idempotent everything.** Document and chunk IDs are deterministic. This powers free re-indexing *and* automatic cache invalidation.
4. **Pragmatic ground truth.** Source-level + keyword-level labels in YAML.
5. **Model-agnostic by default.** Local-first (Ollama); LiteLLM swaps to OpenAI/Anthropic/Cohere with one config change.
6. **Cost is a first-class field.** Every call carries token, cost, and latency metadata, persisted to SQLite. Phantom cost (what cache hits *didn't* pay) is tracked for savings projection.
7. **Retriever-agnostic pipeline.** NaiveRAG accepts any `search(query, k) -> list[Hit]` implementation.
8. **Failure-safe everything.** Telemetry never breaks a request; cache failures fall through to the LLM; cache loading degrades gracefully on corruption.

## Project Structure

```
pareto/
├── ingestion/       # 5-format document readers
├── chunking/        # Hierarchical chunker with tree visualization
├── indexing/        # Embedder + FAISS HNSW + indexer pipeline
├── retrieval/       # Dense, BM25, RRF, HybridRetriever
├── cache/           # LRUCache + SemanticCache (embedding similarity)
├── rag/             # NaiveRAG pipeline (retriever- and cache-aware)
├── generation/      # LiteLLM client wrapper with cost tracking
├── benchmark/       # Test sets, metrics, runner, reports
├── observability/   # SQLite query log + cache stats + savings
├── api/             # FastAPI HTTP server
└── cli.py           # 8 commands

benchmarks/
├── corpus/          # 13 sample documents (legal/finance/health, EN+TR)
├── queries/         # 30-query test set + 45-query cache test set
└── results/         # Baseline JSON reports

docs/
├── RETRO_WEEK_1.md
├── RETRO_WEEK_2.md
├── RETRO_WEEK_3.md
└── COST_PROJECTION.md
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file.

## Acknowledgements

Built as part of the Istanbul Medeniyet University TeknoKampus AI specialization program.

Sample corpus includes public-domain and openly-licensed sources: GDPR full text ([gdpr-info.eu](https://gdpr-info.eu)), Basel III summary ([bis.org](https://www.bis.org)), CDC/WHO health factsheets, MIT and Apache 2.0 license texts, and KVKK + Hipertansiyon articles from [tr.wikipedia.org](https://tr.wikipedia.org) (CC-BY-SA). Synthetic legal, earnings, and clinical documents for testing.

If Pareto helps you ship cheaper RAG, [open an issue](https://github.com/Engineer-coding/pareto/issues) and tell me what you'd improve.
