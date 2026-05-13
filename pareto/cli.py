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


if __name__ == "__main__":
    app()