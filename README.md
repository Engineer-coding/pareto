<div align="center">

# Pareto

**Cost-optimized, open-source RAG infrastructure.**
*80% of the quality, 20% of the cost.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Week 2 of 8](https://img.shields.io/badge/status-week%202%20of%208-orange.svg)]()

</div>

---

Most RAG frameworks help you *build* RAG.
Pareto helps you *afford* it.

Production RAG systems are expensive: high LLM call volume, large context windows, repeated queries hitting the LLM unnecessarily, and noisy retrieval. Pareto attacks every one of these problems with opinionated, **measurable** engineering.

Every optimization layer in Pareto must beat the naive baseline on a published benchmark — or it doesn't ship.

## Why Pareto

LangChain, LlamaIndex, Haystack and friends make it *easy* to assemble a RAG. None of them make it *cheap to run in production*. The bills only show up after launch — and by then the architecture is locked in.

Pareto inverts the priority. Cost is a first-class concern in every layer:

- **Hybrid retrieval** — BM25 + dense vectors via Reciprocal Rank Fusion (Week 2 ✓)
- **Hierarchical chunking** — tree-based parent-child retrieval; smaller indexed chunks, broader context on demand (Week 1 ✓)
- **HNSW vector index** — tuned for throughput, runtime ef_search adjustable (Week 1 ✓)
- **Built-in cost & latency observability** — SQLite query log + `pareto stats` (Week 2 ✓)
- **Semantic caching** — semantically similar queries reuse cached responses (Week 3)
- **Adaptive query routing** — simple queries don't need the expensive model (Week 4)
- **Lightweight knowledge graph** — entity-based augmentation for multi-hop queries (Week 7)

Every layer can be benchmarked independently. The repo includes the test set, the metrics framework, and persisted baseline reports for every milestone.

## Current Baseline — Week 2

The numbers below are real, reproducible with `pareto benchmark`, and committed under `benchmarks/results/`.

### Retrieval-only Benchmark (30 queries, k=5)

| Metric | Dense (Week 1) | BM25 | **Hybrid (default)** |
| :--- | ---: | ---: | ---: |
| hit@5 | 100.00% | 100.00% | 100.00% |
| **MRR** | 0.981 | 0.981 | **1.000** |
| precision@5 | **0.800** | 0.697 | 0.762 |
| recall@5 | 1.000 | 1.000 | 1.000 |
| avg retrieval latency | 29 ms | **0 ms** | 32 ms |

**Hybrid wins on MRR = 1.000** — the correct source always ranks first. This matters: the LLM attends to the top of the prompt most strongly, so MRR translates directly into answer quality.

> The dense baseline already achieves perfect recall on a small test set. Hybrid's real advantage shows up in queries with rare keywords, regulation numbers ("GDPR Article 17"), and Turkish queries where BM25's exact-match power complements dense semantic search.

**Test set:** 30 hand-authored queries across legal, finance, health domains (3 NO_ANSWER for hallucination detection).
**Corpus:** 13 documents (English + Turkish), 381 leaf chunks after hierarchical chunking.
**Embedder:** `intfloat/multilingual-e5-small` (384-dim, 100+ languages, MIT).
**LLM:** `ollama/llama3.2:3b` (local, $0 / query).

## Quickstart

```bash
# Clone and install
git clone https://github.com/Engineer-coding/pareto.git
cd pareto
uv venv --python 3.10
.venv/Scripts/Activate.ps1     # Windows
# source .venv/bin/activate    # Linux/macOS
uv pip install -e .

# (Optional) pull a local LLM for $0-cost generation
ollama pull llama3.2:3b

# Walk a corpus, build an index, ask questions
pareto index ./benchmarks/corpus
pareto search "What is GDPR?"
pareto ask "What are the key principles of GDPR?"

# Try different retrievers
pareto ask "GDPR Article 17?" --retriever dense
pareto ask "GDPR Article 17?" --retriever bm25
pareto ask "GDPR Article 17?" --retriever hybrid   # default

# Check what you spent
pareto stats
pareto stats --by retriever
pareto stats --slow 30000

# Benchmark
pareto benchmark --retriever hybrid --mode retrieval --k 5
```

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Ingestion  │───▶│   Chunking   │───▶│   Indexing   │
│   5 formats  │    │  Hierarchical│    │  FAISS HNSW  │
│   Pydantic   │    │  Tree-based  │    │  + BM25 inv. │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                                                ▼
                                     ┌────────────────────┐
                                     │   Retrieval Layer  │
                                     │ ┌────────────────┐ │
                                     │ │ Dense (E5)     │ │
                                     │ ├────────────────┤ │
                                     │ │ BM25 (custom)  │ │
                                     │ ├────────────────┤ │
                                     │ │ Hybrid (RRF)   │◀┼─── default
                                     │ └────────────────┘ │
                                     └──────────┬─────────┘
                                                │
                                                ▼
                                     ┌────────────────────┐
                                     │     Generation     │
                                     │  LiteLLM wrapper   │
                                     │  Ollama / OpenAI   │
                                     │  Anthropic / ...   │
                                     └──────────┬─────────┘
                                                │
                                                ▼
                                     ┌────────────────────┐
                                     │   Observability    │
                                     │  SQLite query log  │
                                     │  pareto stats CLI  │
                                     └────────────────────┘

         ◀─── Benchmark Suite (P@k, R@k, MRR, refusal accuracy, e2e) ───▶
         ◀─── Multilingual Support (English + Turkish, 100+ via E5)    ───▶
```

## CLI

| Command | What it does |
| :--- | :--- |
| `pareto ingest <dir>` | Load supported documents from a directory |
| `pareto chunk <file>` | Build and visualize a chunk tree for one document |
| `pareto chunk-corpus <dir>` | Batch-chunk a corpus, emit a JSON report |
| `pareto index <dir>` | Ingest + chunk + embed + persist a vector index |
| `pareto search "<query>"` | Semantic search over a saved index |
| `pareto ask "<question>" [--retriever]` | Ask a question against the index (full RAG) |
| `pareto stats` | Query log statistics: cost, latency, breakdowns |
| `pareto serve` | Start the FastAPI HTTP server |
| `pareto benchmark` | Run the test set against the system, emit a JSON report |

### `pareto stats` Examples

```bash
pareto stats                       # overall summary + last 10 queries
pareto stats --last 50             # last 50 queries
pareto stats --since 24h           # last 24 hours
pareto stats --by retriever        # group by retriever (count, cost, latency)
pareto stats --by model            # group by LLM model
pareto stats --slow 30000          # queries slower than 30 seconds
pareto stats --show-questions      # include question text in tables
```

## Roadmap

| Week | Milestone | Status |
| :---: | :--- | :---: |
| 1 | Scaffolding, hierarchical chunking, naive baseline, benchmark suite | ✅ |
| 2 | Hybrid retrieval (BM25 + dense), observability dashboard | ✅ |
| 3 | Semantic cache (LRU + embedding-based) | 🚧 |
| 4 | Adaptive query router | ⏳ |
| 5 | HNSW tuning, cross-encoder reranking | ⏳ |
| 6 | MCP server + client integration | ⏳ |
| 7 | Light knowledge graph (entity extraction + linking) | ⏳ |
| 8 | Vertical demo apps, documentation, public demo | ⏳ |

## Design Principles

1. **Data-driven engineering.** Every optimization layer must beat the previous benchmark on at least one metric or it doesn't ship.
2. **Pragmatic ground truth.** Source-level + keyword-level labels in YAML. Authorable in an afternoon, robust enough to drive meaningful evals.
3. **Idempotent everything.** Document and chunk IDs are deterministic. Re-running a pipeline on unchanged data is free.
4. **Model-agnostic by default.** Local-first (Ollama) for the development loop; LiteLLM wrapper lets you swap to OpenAI/Anthropic/Cohere with a single config change.
5. **Cost is a first-class field.** Every LLM call carries token, cost, and latency metadata. Production cost reports are not bolted on later — they're emitted from day one and persisted to SQLite.
6. **Retriever-agnostic pipeline.** NaiveRAG accepts any `search(query, k) -> list[Hit]` implementation. Strategy pattern, drop-in retrievers from Week 4 onwards.
7. **Failure-safe observability.** Telemetry never breaks the user's request. Two-layer try/except, stderr fallback.

## Project Structure

```
pareto/
├── ingestion/       # 5-format document readers (PDF, DOCX, MD, HTML, TXT)
├── chunking/        # Hierarchical chunker with tree visualization
├── indexing/        # Embedder + FAISS HNSW vector store + indexer pipeline
├── retrieval/       # Dense, BM25, RRF, HybridRetriever
├── rag/             # NaiveRAG pipeline (retriever-agnostic)
├── generation/      # LiteLLM client wrapper with cost tracking
├── benchmark/       # Test sets, metrics, runner, reports
├── observability/   # SQLite query log + aggregation
├── api/             # FastAPI HTTP server
└── cli.py           # 9 commands

benchmarks/
├── corpus/          # 13 sample documents (legal/finance/health, EN+TR)
├── queries/         # 30-query YAML test set
└── results/         # Baseline JSON reports

frontend/            # Next.js chat UI

docs/
├── RETRO_WEEK_1.md  # Week 1 retrospective
└── RETRO_WEEK_2.md  # Week 2 retrospective
```

## Development

```bash
# Run smoke tests
python -m pytest tests/

# Format
ruff format .

# Lint
ruff check .

# Compare retrievers side-by-side
python scripts/compare_retrievers.py
```

## License

MIT — see [LICENSE](./LICENSE).

## Acknowledgements

Built as part of the Istanbul Medeniyet University TeknoKampus AI specialization program.

Sample corpus includes:
- **GDPR** full text from [gdpr-info.eu](https://gdpr-info.eu) (public domain)
- **Basel III** summary from [bis.org](https://www.bis.org/bcbs/basel3.htm) (public)
- **CDC Hypertension** factsheet (public)
- **WHO Diabetes** factsheet (public)
- **MIT License** and **Apache 2.0 License** texts (public)
- **KVKK** (Turkish data protection law) from [tr.wikipedia.org](https://tr.wikipedia.org) (CC-BY-SA)
- **Hipertansiyon** article from [tr.wikipedia.org](https://tr.wikipedia.org) (CC-BY-SA)
- Synthetic legal, earnings, and clinical documents for chunker testing.

If Pareto helps you ship cheaper RAG, [open an issue](https://github.com/Engineer-coding/pareto/issues) and tell me what you'd improve.