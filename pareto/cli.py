"""
Pareto command-line interface.

Today this is a tiny demo of the ingestion layer. As the project grows we'll
register more sub-commands here (index, query, benchmark, serve, ...).

Usage:
    pareto ingest path/to/corpus
    pareto ingest path/to/corpus --domain legal
    pareto ingest path/to/corpus --limit 10
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pareto.ingestion import load_directory

from pareto.chunking import HierarchicalChunker, render_graphviz, render_rich_tree, chunk_directory, save_report
from pareto.ingestion import read_file
from pareto.indexing import Indexer, SentenceTransformerEmbedder
from pareto.rag import NaiveRAG

from pareto.benchmark import (
    BenchmarkRunner,
    TestSet,
    save_report,
)




app = typer.Typer(
    name="pareto",
    help="Cost-optimized RAG infrastructure — CLI.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def _main() -> None:
    """Pareto CLI root. Use a subcommand below."""
    pass

@app.command()
def ingest(
    root: Path = typer.Argument(..., exists=True, file_okay=False, help="Corpus directory."),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Stop after N documents."),
    show_extra: bool = typer.Option(False, "--show-extra", help="Print per-document extra metadata."),
) -> None:
    
    """Walk a corpus directory, parse every supported file, and summarize."""
    console.print(f"[bold]Loading documents from:[/bold] {root}")
    docs, failures = load_directory(root)

    if limit:
        docs = docs[:limit]

    # ── summary table ────────────────────────────────────────────────────
    table = Table(title="Loaded Documents", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Format", style="cyan")
    table.add_column("Length", justify="right")
    table.add_column("Hints", justify="right")
    table.add_column("Title", style="green")
    table.add_column("Source", style="dim")

    for i, d in enumerate(docs, start=1):
        table.add_row(
            str(i),
            d.format.value,
            f"{d.length:,}",
            str(len(d.structural_hints)),
            (d.title or "(none)")[:50],
            str(Path(d.source).name)[:40],
        )
    console.print(table)

    # ── aggregate stats ──────────────────────────────────────────────────
    by_format: dict[str, int] = {}
    total_chars = 0
    total_hints = 0
    for d in docs:
        by_format[d.format.value] = by_format.get(d.format.value, 0) + 1
        total_chars += d.length
        total_hints += len(d.structural_hints)

    console.print()
    console.print(f"[bold]Total documents:[/bold] {len(docs)}")
    console.print(f"[bold]By format:[/bold] {by_format}")
    console.print(f"[bold]Total characters:[/bold] {total_chars:,}")
    console.print(f"[bold]Total structural hints:[/bold] {total_hints}")

    if failures:
        console.print()
        console.print(f"[red]Failures: {len(failures)}[/red]")
        for path, err in failures[:5]:
            console.print(f"  • {path.name}: {type(err).__name__}: {err}")
        if len(failures) > 5:
            console.print(f"  … and {len(failures) - 5} more")

    if show_extra and docs:
        console.print()
        console.print("[bold]Extra metadata (first 3 documents):[/bold]")
        for d in docs[:3]:
            console.print(f"  • {Path(d.source).name}: {d.extra}")


@app.command()
def chunk(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Document file to chunk."),
    image: Path | None = typer.Option(
        None, "--image", "-o",
        help="If given, also export a graphviz image to this path (without extension).",
    ),
    image_format: str = typer.Option(
        "png", "--format", "-f", help="Graphviz output format: png, svg, pdf.",
    ),
) -> None:
    """Parse one document, build its chunk tree, and print/render the structure."""
    console.print(f"[bold]Reading:[/bold] {path}")
    doc = read_file(path)
    console.print(f"[dim]→ {doc.short_repr()}[/dim]")

    chunker = HierarchicalChunker()
    tree = chunker.chunk(doc)

    console.print()
    console.print(
        f"[bold]Tree:[/bold] {tree.num_nodes} nodes, "
        f"{tree.num_leaves} leaves, depth {tree.depth()}"
    )
    console.print()

    render_rich_tree(tree, console=console)

    if image is not None:
        try:
            out = render_graphviz(tree, image, format=image_format)
            console.print(f"\n[green]Image written:[/green] {out}")
        except (ImportError, RuntimeError) as e:
            console.print(f"\n[red]Image export failed:[/red] {e}")

@app.command("chunk-corpus")
def chunk_corpus(
    root: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Corpus directory to process."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Where to save the JSON report. Default: benchmarks/results/chunking_report.json",
    ),
    show_failures: bool = typer.Option(
        False, "--show-failures", help="Print every failed source path.",
    ),
) -> None:
    """Chunk every supported document under a corpus directory and emit a JSON report."""
    console.print(f"[bold]Chunking corpus:[/bold] {root}")
    _, report = chunk_directory(root)

    # ── summary table ────────────────────────────────────────────────────
    table = Table(title="Per-Document Chunking Stats", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Format", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Chars", justify="right")
    table.add_column("Nodes", justify="right")
    table.add_column("Leaves", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("Avg leaf", justify="right")

    for i, d in enumerate(report.per_document, start=1):
        table.add_row(
            str(i),
            d.format,
            (d.title or Path(d.source).stem)[:40],
            f"{d.content_length:,}",
            str(d.num_nodes),
            str(d.num_leaves),
            str(d.depth),
            f"{d.avg_leaf_length:.0f}",
        )
    console.print(table)

    # ── aggregates ───────────────────────────────────────────────────────
    console.print()
    console.print(f"[bold]Documents processed:[/bold] {report.num_documents}")
    console.print(f"[bold]Total characters:[/bold] {report.total_chars:,}")
    console.print(f"[bold]Total chunk-tree nodes:[/bold] {report.total_nodes}")
    console.print(f"[bold]Total leaves (indexable chunks):[/bold] {report.total_leaves}")
    console.print(f"[bold]Format distribution:[/bold] {report.formats}")

    if report.failures:
        console.print()
        console.print(f"[red]Failures: {report.num_failed}[/red]")
        if show_failures:
            for f in report.failures:
                console.print(f"  • {f['source']}: {f['error']}")
        else:
            console.print("[dim]  (use --show-failures to list them)[/dim]")

    # ── persist JSON report ──────────────────────────────────────────────
    out_path = output or Path("benchmarks/results/chunking_report.json")
    saved = save_report(report, out_path)
    console.print()
    console.print(f"[green]Report written:[/green] {saved}")

@app.command()
def index(
    corpus: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Corpus directory."
    ),
    output: Path = typer.Option(
        Path("benchmarks/results/index"), "--output", "-o",
        help="Where to save the index.",
    ),
    incremental: bool = typer.Option(
        False, "--incremental",
        help="If an index already exists at --output, append to it (skip duplicates).",
    ),
) -> None:
    """Ingest, chunk, embed, and persist a corpus as a vector index."""
    from pareto.chunking import chunk_directory

    console.print(f"[bold]Ingesting + chunking:[/bold] {corpus}")
    results, report = chunk_directory(corpus)
    console.print(
        f"  → [green]{report.num_documents}[/green] docs, "
        f"[green]{report.total_leaves}[/green] leaves"
    )
    if report.failures:
        console.print(f"  → [yellow]{report.num_failed} failures (use chunk-corpus --show-failures for detail)[/yellow]")

    # Either load existing or build fresh
    if incremental and (output / "index.faiss").exists():
        console.print(f"[bold]Loading existing index:[/bold] {output}")
        indexer = Indexer.load(output)
        console.print(f"  → existing size: {indexer.store.size}")
    else:
        console.print(f"[bold]Building fresh index[/bold] (embedder: e5-multilingual-small)")
        indexer = Indexer()

    docs = [d for d, _ in results]
    trees = [t for _, t in results]

    console.print()
    stats = indexer.index_chunk_trees(trees, documents=docs, show_progress=True)
    console.print()
    console.print(
        f"[bold]Indexed:[/bold] {stats.num_chunks_indexed}  "
        f"[dim]| skipped: {stats.num_chunks_skipped}  filtered: {stats.num_chunks_filtered}[/dim]"
    )
    console.print(f"[bold]Total chars embedded:[/bold] {stats.total_chars_indexed:,}")
    console.print(f"[bold]Store size:[/bold] {indexer.store.size}")
    console.print(f"[bold]Format mix:[/bold] {stats.formats}")

    saved = indexer.save(output)
    console.print(f"\n[green]Saved to:[/green] {saved}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query."),
    index_dir: Path = typer.Option(
        Path("benchmarks/results/index"), "--index", "-i",
        help="Directory of a saved index.",
    ),
    k: int = typer.Option(5, "--k", "-k", help="Number of results."),
    domain: str | None = typer.Option(
        None, "--domain", "-d",
        help="Optional: restrict to a corpus subfolder name (legal/finance/health).",
    ),
    show_full: bool = typer.Option(
        False, "--full", help="Show full chunk content (default: truncated preview)."
    ),
) -> None:
    """Semantic search over a previously-built index."""
    if not index_dir.exists():
        console.print(f"[red]No index at {index_dir}.[/red] Run `pareto index <corpus>` first.")
        raise typer.Exit(1)

    console.print(f"[bold]Loading index:[/bold] {index_dir}")
    indexer = Indexer.load(index_dir)
    console.print(f"  → {indexer.store.size} chunks, dim {indexer.store.config.embedding_dim}")

    q_vec = indexer.embedder.encode_query(query)

    if domain:
        # Match the domain via source path (e.g. ...\corpus\legal\foo.pdf)
        filter_fn = lambda r: f"\\{domain}\\" in r.source or f"/{domain}/" in r.source
        hits = indexer.store.search(q_vec, k=k, filter_fn=filter_fn)
    else:
        hits = indexer.store.search(q_vec, k=k)

    console.print()
    console.print(f"[bold]Query:[/bold] {query}")
    if domain:
        console.print(f"[dim]Filter: domain={domain}[/dim]")
    console.print(f"[bold]Top {len(hits)} hits:[/bold]\n")

    for i, h in enumerate(hits, 1):
        src_name = Path(h.record.source).name
        console.print(f"  [bold cyan]{i}.[/bold cyan] [yellow]score={h.score:.3f}[/yellow]  [dim]{src_name}[/dim]")
        body = h.record.content if show_full else h.record.content[:200].replace("\n", " ")
        suffix = "" if show_full or len(h.record.content) <= 200 else "…"
        console.print(f"     {body}{suffix}\n")

    if not hits:
        console.print("[yellow]No matches.[/yellow]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the index."),
    index_dir: Path = typer.Option(
        Path("benchmarks/results/index"),
        "--index-dir", "-i",
        help="Directory of a saved Pareto index.",
    ),
    model: str = typer.Option(
        "ollama/llama3.2:3b", "--model", "-m",
        help="LiteLLM model identifier.",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    retriever: str = typer.Option(
        "hybrid", "--retriever", "-r",
        help="Retriever mode: 'dense', 'bm25', or 'hybrid' (default).",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache",
        help="Bypass the semantic cache for this query.",
    ),
    cache_threshold: float = typer.Option(
        0.92, "--cache-threshold",
        help="Cosine similarity threshold for cache hits.",
    ),
    cache_capacity: int = typer.Option(
        1000, "--cache-capacity",
        help="Maximum number of cached entries.",
    ),
    cache_path: Path = typer.Option(
        Path("benchmarks/results/cache.pkl"),
        "--cache-path",
        help="Where to persist the cache across CLI calls.",
    ),
) -> None:
    """Ask a question against a saved Pareto index (full RAG pipeline)."""
    from pareto.cache import SemanticCache
    from pareto.generation.llm_client import LiteLLMClient, LLMConfig
    from pareto.indexing import Indexer
    from pareto.observability import QueryLogStore
    from pareto.rag import NaiveRAG

    if not index_dir.exists():
        console.print(f"[red]Index not found at {index_dir}[/red]")
        raise typer.Exit(1)

    # Load index
    console.print(f"Loading index from {index_dir}...")
    indexer = Indexer.load(index_dir)
    console.print(f"  → {indexer.store.size} chunks")

    # Build retriever
    if retriever == "dense":
        from pareto.retrieval import DenseRetriever
        retriever_obj = DenseRetriever(indexer)
    elif retriever == "bm25":
        from pareto.retrieval import BM25Ranker
        ranker = BM25Ranker()
        ranker.build_from_records(indexer.store.records)
        retriever_obj = ranker
    elif retriever == "hybrid":
        from pareto.retrieval import BM25Ranker, HybridRetriever
        ranker = BM25Ranker()
        ranker.build_from_records(indexer.store.records)
        retriever_obj = HybridRetriever(indexer=indexer, bm25_ranker=ranker)
    else:
        console.print(f"[red]Unknown retriever:[/red] {retriever}")
        raise typer.Exit(1)

    # Load or create cache (unless --no-cache)
    cache = None
    if not no_cache:
        if cache_path.exists():
            cache = SemanticCache.load(cache_path)
            # Override threshold/capacity at runtime if user passed flags
            cache.threshold = cache_threshold
        else:
            cache = SemanticCache(
                capacity=cache_capacity,
                threshold=cache_threshold,
            )

    # Build LLM + log store + RAG
    log_store = QueryLogStore()
    llm = LiteLLMClient(LLMConfig(model=model))
    rag = NaiveRAG(
        retriever=retriever_obj,
        llm=llm,
        top_k=top_k,
        log_store=log_store,
        cache=cache,
    )

    # Run the query
    console.print(f"Q: [bold]{question}[/bold]")
    console.print(f"Thinking with {model}...")
    response = rag.query(question)

    # Display cache status FIRST if hit (most exciting line)
    if response.extra.get("cache_hit"):
        sim = response.extra.get("cache_similarity", 0)
        orig_query = response.extra.get("cached_original_query", "")
        console.print(
            f"\n[bold green]⚡ Cache HIT[/bold green] "
            f"(similarity={sim:.4f})"
        )
        if orig_query != question:
            console.print(
                f"[dim]   Reusing cached answer from: \"{orig_query}\"[/dim]"
            )

    # Display answer
    console.print(f"\n[bold]Answer:[/bold]")
    console.print(response.answer)

    # Sources
    citations = response.citations() or response.extra.get("cached_citations", [])
    if citations:
        console.print(f"\n[bold]Sources:[/bold]")
        for i, src in enumerate(citations, 1):
            console.print(f"  [{i}] {src}")

    # Stats footer
    if response.extra.get("cache_hit"):
        orig_lat = response.extra.get("original_generation_latency_ms", 0)
        orig_cost = response.extra.get("original_cost_usd", 0)
        console.print(
            f"\n[dim]Stats: cached response | "
            f"total={response.total_latency_ms}ms | "
            f"saved ~{orig_lat:,}ms + ${orig_cost:.5f}[/dim]"
        )
    else:
        console.print(
            f"\n[dim]Stats: {response.total_tokens} tokens "
            f"(prompt={response.prompt_tokens}, completion={response.completion_tokens}) "
            f"| retrieval={response.retrieval_latency_ms}ms "
            f"| generation={response.generation_latency_ms}ms "
            f"| total={response.total_latency_ms}ms "
            f"| cost=${response.cost_usd:.5f}[/dim]"
        )

    # Save cache for next CLI call
    if cache is not None:
        try:
            cache.save(cache_path)
        except Exception as e:
            console.print(f"[yellow]Warning: cache save failed: {e}[/yellow]")

@app.command()
def benchmark(
    test_set_path: Path = typer.Option(
        Path("benchmarks/queries/queries.yaml"),
        "--test-set", "-t",
        help="Path to the YAML test set.",
    ),
    index_dir: Path = typer.Option(
        Path("benchmarks/results/index"), "--index", "-i",
        help="Directory of a saved index.",
    ),
    mode: str = typer.Option(
        "retrieval", "--mode", "-M",
        help="Benchmark mode: 'retrieval' (fast) or 'end_to_end' (slow, runs LLM).",
    ),
    k: int = typer.Option(5, "--k", "-k", help="Retrieval depth."),
    limit: int | None = typer.Option(
        None, "--limit", "-n",
        help="Run only the first N queries (useful for end-to-end smoke tests).",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Where to save the JSON report. Default name based on mode/k.",
    ),
    system_name: str | None = typer.Option(
        None, "--name",
        help="System identifier in the report. Default: 'naive_<mode>_k<k>'.",
    ),
    model: str = typer.Option(
        "ollama/llama3.2:3b", "--model",
        help="LLM model (end_to_end mode only).",
    ),

    retriever: str = typer.Option(
        "hybrid",
        "--retriever", "-r",
        help="Retriever mode: 'dense', 'bm25', or 'hybrid' (default).",
    ),
) -> None:
    """Run a benchmark of a Pareto system against the test set."""
    if not index_dir.exists():
        console.print(f"[red]No index at {index_dir}.[/red] Run `pareto index <corpus>` first.")
        raise typer.Exit(1)
    if mode not in ("retrieval", "end_to_end"):
        console.print(f"[red]Unknown mode: {mode}.[/red] Use 'retrieval' or 'end_to_end'.")
        raise typer.Exit(2)

    # Load components
    console.print(f"[dim]Loading index from {index_dir}...[/dim]")
    from pareto.generation import LiteLLMClient, LLMConfig
    from pareto.indexing import Indexer
    indexer = Indexer.load(index_dir)
    test_set = TestSet.from_yaml(test_set_path)
    console.print(
        f"[dim]  → {indexer.store.size} chunks, {len(test_set)} queries[/dim]"
    )

   # Build the requested retriever
    if retriever == "dense":
        from pareto.retrieval import DenseRetriever
        retriever_obj = DenseRetriever(indexer)
    elif retriever == "bm25":
        from pareto.retrieval import BM25Ranker
        ranker = BM25Ranker()
        ranker.build_from_records(indexer.store.records)
        retriever_obj = ranker
    elif retriever == "hybrid":
        from pareto.retrieval import BM25Ranker, HybridRetriever
        ranker = BM25Ranker()
        ranker.build_from_records(indexer.store.records)
        retriever_obj = HybridRetriever(indexer=indexer, bm25_ranker=ranker)
    else:
        console.print(f"[red]Unknown retriever:[/red] {retriever}")
        raise typer.Exit(1)

    # Build the requested retriever
    if retriever == "dense":
        from pareto.retrieval import DenseRetriever
        retriever_obj = DenseRetriever(indexer)
    elif retriever == "bm25":
        from pareto.retrieval import BM25Ranker
        ranker = BM25Ranker()
        ranker.build_from_records(indexer.store.records)
        retriever_obj = ranker
    elif retriever == "hybrid":
        from pareto.retrieval import BM25Ranker, HybridRetriever
        ranker = BM25Ranker()
        ranker.build_from_records(indexer.store.records)
        retriever_obj = HybridRetriever(indexer=indexer, bm25_ranker=ranker)
    else:
        console.print(f"[red]Unknown retriever:[/red] {retriever}")
        raise typer.Exit(1)

    # Runner setup
    rag = None
    if mode == "end_to_end":
        rag = NaiveRAG(
            retriever=retriever_obj,
            llm=LiteLLMClient(LLMConfig(model=model)),
            top_k=k,
        )
        if limit is None:
            console.print(
                "[yellow]Warning:[/yellow] end-to-end mode without --limit will run "
                f"all {len(test_set)} queries through the LLM. This can take 15-30 minutes on CPU."
            )
    runner = BenchmarkRunner(indexer=indexer, rag=rag, retriever=retriever_obj)

    name = system_name or f"naive_{mode}_k{k}"
    console.print(f"\n[bold]Running benchmark:[/bold] {name} (mode={mode}, retriever={retriever}, k={k})\n")

    if mode == "retrieval":
        report = runner.run_retrieval(test_set, k=k, system_name=name, limit=limit)
    else:
        report = runner.run_end_to_end(test_set, k=k, system_name=name, limit=limit)

    # ── pretty-print summary ─────────────────────────────────────────────
    console.print()
    console.print(f"[bold cyan]System:[/bold cyan] {report.system_name}")
    console.print(
        f"[bold cyan]Test set:[/bold cyan] {report.test_set_name}  "
        f"N={report.num_queries}  k={report.k}"
    )

    table = Table(title="Retrieval Metrics", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("hit@k", f"{report.avg_hit_at_k:.2%}")
    table.add_row("precision@k", f"{report.avg_precision_at_k:.3f}")
    table.add_row("recall@k", f"{report.avg_recall_at_k:.3f}")
    table.add_row("MRR", f"{report.avg_mrr:.3f}")
    table.add_row("avg latency (ms)", f"{report.avg_retrieval_latency_ms:.1f}")
    console.print(table)

    # Per-domain
    dt = Table(title="By Domain", show_header=True)
    dt.add_column("Domain")
    dt.add_column("N", justify="right")
    dt.add_column("hit@k", justify="right")
    dt.add_column("P@k", justify="right")
    dt.add_column("R@k", justify="right")
    dt.add_column("MRR", justify="right")
    for domain, m in report.by_domain.items():
        dt.add_row(
            domain,
            str(int(m["n"])),
            f"{m['hit_at_k']:.2%}",
            f"{m['precision_at_k']:.3f}",
            f"{m['recall_at_k']:.3f}",
            f"{m['mrr']:.3f}",
        )
    console.print(dt)

    # Answer metrics if end-to-end
    if report.answer_evaluated:
        at = Table(title="Answer Metrics", show_header=True)
        at.add_column("Metric")
        at.add_column("Value", justify="right")
        at.add_row("keyword coverage", f"{report.avg_keyword_coverage:.3f}")
        at.add_row("refusal accuracy (NO_ANSWER)", f"{report.extra.get('refusal_accuracy', 0.0):.2%}")
        at.add_row("avg prompt tokens", f"{report.avg_prompt_tokens:.0f}")
        at.add_row("avg completion tokens", f"{report.avg_completion_tokens:.0f}")
        at.add_row("avg generation latency (ms)", f"{report.avg_generation_latency_ms:.0f}")
        at.add_row("avg cost / query (USD)", f"${report.avg_cost_usd:.5f}")
        at.add_row("total cost (USD)", f"${report.total_cost_usd:.5f}")
        console.print(at)

    # Misses
    misses = [
        r for r in report.results
        if r.type.value != "no_answer" and not r.retrieval.hit_at_k
    ]
    if misses:
        console.print(f"\n[yellow]Retrieval misses ({len(misses)}):[/yellow]")
        for r in misses[:5]:
            console.print(f"  ✗ [{r.query_id}] {r.query}")
            console.print(f"    [dim]retrieved: {r.retrieved_sources}[/dim]")
        if len(misses) > 5:
            console.print(f"  ... and {len(misses) - 5} more")

    # Save
    if output is None:
        output = Path(f"benchmarks/results/{name}.json")
    saved = save_report(report, output)
    console.print(f"\n[green]Report saved:[/green] {saved}")

@app.command()
def serve(
    index_dir: Path = typer.Option(
        Path("benchmarks/results/index"), "--index", "-i",
        help="Directory of a saved index.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
) -> None:
    """Start the Pareto HTTP API server."""
    import uvicorn

    from pareto.api.server import create_app

    if not index_dir.exists():
        console.print(
            f"[red]No index at {index_dir}.[/red] Run `pareto index <corpus>` first."
        )
        raise typer.Exit(1)

    api_app = create_app(index_dir=index_dir)
    console.print(f"[green]Pareto API:[/green] http://{host}:{port}")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    uvicorn.run(api_app, host=host, port=port, log_level="info")


@app.command()
def stats(
    db_path: Path = typer.Option(
        Path("benchmarks/results/pareto.db"), "--db",
        help="Path to the Pareto query log database.",
    ),
    last: int | None = typer.Option(
        None, "--last", "-n",
        help="Show the last N queries in a detail table.",
    ),
    since: str | None = typer.Option(
        None, "--since", "-s",
        help="Filter to queries since a duration: '30m', '24h', '7d', '2w'.",
    ),
    by: str | None = typer.Option(
        None, "--by",
        help="Group aggregate by: 'model' or 'retriever'.",
    ),
    slow: int | None = typer.Option(
        None, "--slow",
        help="Show queries slower than this many milliseconds (default 30000 if flag given).",
    ),
    show_questions: bool = typer.Option(
        False, "--show-questions",
        help="Include the question text column in detail tables.",
    ),
) -> None:
    """Show query statistics from the Pareto log database."""
    from datetime import datetime, timedelta, timezone

    from pareto.observability import QueryLogConfig, QueryLogStore

    if not db_path.exists():
        console.print(
            f"[yellow]No query log found at {db_path}.[/yellow]\n"
            f"[dim]Run a few `pareto ask` queries first to populate it.[/dim]"
        )
        raise typer.Exit(1)

    store = QueryLogStore(QueryLogConfig(db_path=db_path))

    # Parse --since
    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.now(timezone.utc) - _parse_duration(since)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(2)

    # Flag priority: slow > by > last > overall
    if slow is not None:
        _stats_show_slow(store, threshold_ms=slow if slow > 0 else 30000)
        return

    if by is not None:
        _stats_show_grouped(store, group_by=by, since_dt=since_dt)
        return

    if last is not None:
        _stats_show_recent(store, n=last, show_questions=show_questions)
        return

    # Default: overall summary + last 10
    _stats_show_overall(store, since_dt=since_dt)
    console.print()
    _stats_show_recent(store, n=10, show_questions=show_questions)


# ── stats helpers ────────────────────────────────────────────────────────

def _parse_duration(s: str):
    """Parse '30m', '24h', '7d', '2w' into a timedelta."""
    from datetime import timedelta
    s = s.lower().strip()
    if not s:
        raise ValueError("Empty duration.")
    unit = s[-1]
    try:
        n = int(s[:-1])
    except ValueError as e:
        raise ValueError(f"Bad duration {s!r}. Use '30m', '24h', '7d', or '2w'.") from e
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)
    if unit == "w":
        return timedelta(weeks=n)
    raise ValueError(f"Unknown duration unit in {s!r}. Use m/h/d/w.")


def _stats_show_overall(store, since_dt) -> None:
    agg = store.aggregate(since=since_dt)
    overall = agg.get("overall", {}) if agg else {}
    if not overall or (overall.get("count") or 0) == 0:
        window = f"since {since_dt.isoformat()}" if since_dt else "all time"
        console.print(f"[dim]No queries logged ({window}).[/dim]")
        return

    title = "Overall stats"
    if since_dt:
        title += f" (since {since_dt.strftime('%Y-%m-%d %H:%M')} UTC)"

    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total queries", f"{overall['count']:,}")
    table.add_row("Total tokens", f"{(overall.get('total_tokens') or 0):,}")
    table.add_row("Total cost (USD)", f"${(overall.get('total_cost_usd') or 0):.5f}")
    table.add_row("Avg cost / query", f"${(overall.get('avg_cost_usd') or 0):.5f}")
    table.add_row("Avg retrieval latency", f"{(overall.get('avg_retrieval_latency_ms') or 0):.0f} ms")
    table.add_row("Avg generation latency", f"{(overall.get('avg_generation_latency_ms') or 0):.0f} ms")
    table.add_row("Avg total latency", f"{(overall.get('avg_total_latency_ms') or 0):.0f} ms")
    table.add_row("Max total latency", f"{(overall.get('max_total_latency_ms') or 0):,} ms")
    # Cache section
    cache_hits = overall.get("cache_hits") or 0
    cache_rate = overall.get("cache_hit_rate") or 0
    avg_sim = overall.get("avg_cache_similarity")
    table.add_row("[bold cyan]Cache hits[/bold cyan]",
                  f"[bold cyan]{cache_hits} ({cache_rate:.1%})[/bold cyan]")
    if avg_sim is not None:
        table.add_row("Avg cache similarity", f"{avg_sim:.4f}")
    console.print(table)

    # Savings table (only if there are hits)
    if cache_hits > 0:
        savings = store.total_savings(since=since_dt)
        save_tbl = Table(title="Cache Savings (phantom cost)")
        save_tbl.add_column("Metric")
        save_tbl.add_column("Value", justify="right")
        save_tbl.add_row("Cache hits served", f"{savings['hits']:,}")
        save_tbl.add_row("Cost saved (USD)", f"${savings['saved_cost_usd']:.5f}")
        save_tbl.add_row("Latency saved", f"{savings['saved_latency_ms']:,} ms")
        save_tbl.add_row("≈ time equivalent", f"{savings['saved_latency_ms']/1000:.1f} s")
        console.print()
        console.print(save_tbl)


def _stats_show_grouped(store, group_by: str, since_dt) -> None:
    if group_by not in ("model", "retriever", "cache_hit"):
        console.print(
            f"[red]--by must be 'model', 'retriever', or 'cache_hit', got {group_by!r}[/red]"
        )
        raise typer.Exit(2)

    agg = store.aggregate(since=since_dt, group_by=group_by)
    if not agg:
        console.print("[dim]No queries to aggregate.[/dim]")
        return

    title = f"Stats by {group_by}"
    if since_dt:
        title += f" (since {since_dt.strftime('%Y-%m-%d %H:%M')} UTC)"

    table = Table(title=title)
    table.add_column(group_by.capitalize())
    table.add_column("count", justify="right")
    table.add_column("cache hits", justify="right")
    table.add_column("hit rate", justify="right")
    table.add_column("total cost", justify="right")
    table.add_column("avg gen ms", justify="right")
    table.add_column("avg total ms", justify="right")

    sorted_groups = sorted(agg.items(), key=lambda kv: -(kv[1].get("count") or 0))
    for group_name, stats in sorted_groups:
        hit_rate = stats.get("cache_hit_rate") or 0
        cache_hits = stats.get("cache_hits") or 0
        hit_rate_str = f"[bold cyan]{hit_rate:.1%}[/bold cyan]" if hit_rate > 0 else f"{hit_rate:.1%}"
        table.add_row(
            str(group_name),
            f"{stats['count']:,}",
            f"{cache_hits:,}",
            hit_rate_str,
            f"${(stats.get('total_cost_usd') or 0):.5f}",
            f"{(stats.get('avg_generation_latency_ms') or 0):.0f}",
            f"{(stats.get('avg_total_latency_ms') or 0):.0f}",
        )
    console.print(table)


def _stats_show_recent(store, n: int, show_questions: bool) -> None:
    rows = store.last_n(n)
    if not rows:
        console.print("[dim]No queries logged.[/dim]")
        return

    table = Table(title=f"Last {len(rows)} queries")
    table.add_column("id", justify="right")
    table.add_column("timestamp")
    table.add_column("retriever")
    table.add_column("model")
    if show_questions:
        table.add_column("question")
    table.add_column("tokens", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("cache", justify="center")
    table.add_column("total ms", justify="right")

    for r in rows:
        ts_short = (r["timestamp"] or "")[:16].replace("T", " ")
        row_items = [
            str(r["id"]),
            ts_short,
            r["retriever"] or "—",
            (r["model"] or "—").replace("ollama/", ""),
        ]
        if show_questions:
            q = (r["question"] or "")[:55]
            if len(r["question"] or "") > 55:
                q += "..."
            row_items.append(q)
        cache_marker = "⚡" if r.get("cache_hit") else "—"
        row_items.extend([
            f"{(r['total_tokens'] or 0):,}",
            f"${(r['cost_usd'] or 0):.5f}",
            cache_marker,
            f"{(r['total_latency_ms'] or 0):,}",
        ])
        table.add_row(*row_items)
    console.print(table)


def _stats_show_slow(store, threshold_ms: int) -> None:
    rows = store.slow_queries(threshold_ms=threshold_ms, limit=20)
    if not rows:
        console.print(f"[dim]No queries above {threshold_ms} ms.[/dim]")
        return

    table = Table(title=f"Slow queries (≥ {threshold_ms:,} ms)")
    table.add_column("id", justify="right")
    table.add_column("question")
    table.add_column("retriever")
    table.add_column("model")
    table.add_column("retr ms", justify="right")
    table.add_column("gen ms", justify="right")
    table.add_column("total ms", justify="right", style="red")

    for r in rows:
        q = (r["question"] or "")[:55]
        if len(r["question"] or "") > 55:
            q += "..."
        table.add_row(
            str(r["id"]),
            q,
            r["retriever"] or "—",
            (r["model"] or "—").replace("ollama/", ""),
            f"{(r['retrieval_latency_ms'] or 0):,}",
            f"{(r['generation_latency_ms'] or 0):,}",
            f"{(r['total_latency_ms'] or 0):,}",
        )
    console.print(table)


if __name__ == "__main__":
    app()