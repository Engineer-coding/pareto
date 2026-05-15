<div align="center">

# Pareto

**Cost-optimized, open-source RAG infrastructure.**
*80% of the quality, 20% of the cost.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Week 1 of 8](https://img.shields.io/badge/status-week%201%20of%208-orange.svg)]()

</div>

---

Most RAG frameworks help you *build* RAG.
Pareto helps you *afford* it.

Production RAG systems are expensive: high LLM call volume, large context windows, repeated queries hitting the LLM unnecessarily, and noisy retrieval. Pareto attacks every one of these problems with opinionated, **measurable** engineering.

Every optimization layer in Pareto must beat the naive baseline on a published benchmark — or it doesn't ship.

## Why Pareto

LangChain, LlamaIndex, Haystack and friends make it *easy* to assemble a RAG. None of them make it *cheap to run in production*. The bills only show up after launch — and by then the architecture is locked in.

Pareto inverts the priority. Cost is a first-class concern in every layer:

- **Adaptive query routing** — simple queries don't need the expensive model
- **Semantic caching** — semantically similar queries reuse cached responses
- **Hybrid retrieval** — BM25 + dense vectors via Reciprocal Rank Fusion
- **Hierarchical chunking** — tree-based parent-child retrieval; smaller indexed chunks, broader context on demand
- **HNSW vector index** — tuned for throughput, runtime ef_search adjustable
- **Lightweight knowledge graph** — entity-based augmentation for multi-hop queries
- **Built-in cost & latency observability** — no external tool needed

Every layer can be benchmarked independently. The repo includes the test set, the metrics framework, and persisted baseline reports for every milestone.

## Week 1 Baseline (Naive RAG)

The numbers below are real, reproducible with `pareto benchmark`, and committed under `benchmarks/results/`. They are the bar every Week 2+ optimization must clear.

| Metric | k=3 | k=5 | k=10 |
| :--- | ---: | ---: | ---: |
| hit@k | **96.15%** | 96.15% | 96.15% |
| precision@k | **0.821** | 0.754 | 0.604 |
| recall@k | 0.962 | 0.962 | 0.962 |
| MRR | **0.962** | 0.962 | 0.962 |
| avg retrieval latency | 23 ms | 22 ms | 23 ms |

**Test set:** 30 hand-authored queries across legal, finance, health domains.
**Corpus:** 10 documents, 380 leaf chunks after hierarchical chunking.
**Embedder:** `intfloat/multilingual-e5-small` (384-dim, 100+ languages).
**LLM:** `ollama/llama3.2:3b` (local, $0 / query).

> Recall and MRR saturate at k=3; increasing k only dilutes precision and inflates prompt tokens. This insight drives Week 4's adaptive query router.

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

# Benchmark
pareto benchmark --mode retrieval --k 5
```

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Ingestion  │───▶│   Chunking   │───▶│   Indexing   │
│   5 formats  │    │  Hierarchical│    │  FAISS HNSW  │
│   Pydantic   │    │  Tree-based  │    │  Idempotent  │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                                                ▼
                                       ┌──────────────────┐
                                       │     Retrieval    │
                                       │  Vector (+ BM25  │
                                       │   from Week 2)   │
                                       └────────┬─────────┘
                                                │
                                                ▼
                                       ┌──────────────────┐
                                       │    Generation    │
                                       │ LiteLLM wrapper  │
                                       │ Ollama / OpenAI  │
                                       │ Anthropic / ...  │
                                       └──────────────────┘

                  ◀────  Benchmark Suite (P@k, R@k, MRR, refusal accuracy)  ────▶
                  ◀────  Observability (cost + latency per call, built-in)  ────▶
```

## CLI

| Command | What it does |
| :--- | :--- |
| `pareto ingest <dir>` | Load supported documents from a directory |
| `pareto chunk <file>` | Build and visualize a chunk tree for one document |
| `pareto chunk-corpus <dir>` | Batch-chunk a corpus, emit a JSON report |
| `pareto index <dir>` | Ingest + chunk + embed + persist a vector index |
| `pareto search "<query>"` | Semantic search over a saved index |
| `pareto ask "<question>"` | Ask a question against the index (full RAG) |
| `pareto benchmark` | Run the test set against the system, emit a JSON report |

## Roadmap

| Week | Milestone | Status |
| :---: | :--- | :---: |
| 1 | Scaffolding, hierarchical chunking, naive baseline, benchmark suite | ✅ |
| 2 | Hybrid retrieval (BM25 + dense), observability dashboard | 🚧 |
| 3 | Semantic cache (LRU + embedding-based) | ⏳ |
| 4 | Adaptive query router | ⏳ |
| 5 | HNSW tuning, cross-encoder reranking | ⏳ |
| 6 | MCP server + client integration | ⏳ |
| 7 | Light knowledge graph (entity extraction + linking) | ⏳ |
| 8 | Vertical demo apps, documentation, public demo | ⏳ |

## Design Principles

1. **Data-driven engineering.** Every optimization layer must beat the previous benchmark on at least one metric or it doesn't ship.
2. **Pragmatic ground truth.** Source-level + keyword-level labels in YAML. Authorable in an afternoon, robust enough to drive meaningful evals.
3. **Idempotent everything.** Document and chunk IDs are deterministic. Re-running a pipeline on unchanged data is free — embeddings, cache entries, observability traces all key on stable IDs.
4. **Model-agnostic by default.** Local-first (Ollama) for the development loop; LiteLLM wrapper lets you swap to OpenAI/Anthropic/Cohere with a single config change.
5. **Cost is a first-class field.** Every LLM call carries token, cost, and latency metadata. Production cost reports are not bolted on later — they're emitted from day one.

## Development

```bash
# Run smoke tests
python -m pytest tests/

# Format
ruff format .

# Lint
ruff check .
```

## License

MIT — see [LICENSE](./LICENSE).

## Acknowledgements

Built as part of the Istanbul Medeniyet University TeknoKampus AI specialization program.

If Pareto helps you ship cheaper RAG, [open an issue](https://github.com/Engineer-coding/pareto/issues) and tell me what you'd improve.