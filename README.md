# Pareto

> **Cost-optimized, open-source RAG infrastructure.**  
> 80% of the quality, 20% of the cost.

**Active development — not production-ready yet.**

---

## What is Pareto?

Most RAG frameworks help you *build* RAG.  
Pareto helps you *afford* RAG.

Modern RAG systems in production are expensive: high LLM costs, slow latency, repeated queries hitting the LLM unnecessarily, and noisy retrieval. Pareto attacks these problems with opinionated, measurable engineering.

### Core ideas

- **Adaptive query routing** — simple queries don't need the expensive model
- **Semantic caching** — semantically similar queries reuse cached responses
- **Hybrid retrieval** — BM25 + dense vectors with Reciprocal Rank Fusion
- **Hierarchical chunking** — tree-based parent-child context retrieval
- **HNSW vector indexing** — tuned for throughput
- **Lightweight knowledge graph** — entity-based augmentation for multi-hop queries
- **Built-in cost & latency observability** — no external tool needed

---

## Goal Metrics (vs naive RAG baseline)

| Metric | Target |
| :--- | :--- |
| LLM cost reduction | ~60% |
| p50 latency reduction | ~5x |
| Throughput (queries/sec) | ~5x |
| Multi-hop accuracy uplift | +30 points |

> Real benchmark results will be published as the project progresses. See `benchmarks/`.

---

## Status & Roadmap

| Week | Milestone |
| :---: | :--- |
| 1 | Scaffolding, hierarchical chunking, naive baseline, benchmark suite |
| 2 | Hybrid retrieval (BM25 + dense), observability |
| 3 | Semantic cache (LRU + embedding-based) |
| 4 | Adaptive query router |
| 5 | HNSW tuning, cross-encoder reranking |
| 6 | MCP integration (server + client) |
| 7 | Light knowledge graph |
| 8 | Demo, vertical examples, documentation |

---

## Quickstart

> _Coming soon. Will be `docker compose up`._

---

## License

MIT — see [LICENSE](./LICENSE).