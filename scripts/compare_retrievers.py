"""
Three-way retriever comparison: Dense vs BM25 vs Hybrid.

Runs the same TestSet against all three retrieval strategies and emits a
side-by-side report. The output JSONs are committed as historical baseline.

Usage:
    python scripts/compare_retrievers.py
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from pareto.benchmark import BenchmarkRunner, TestSet, save_report
from pareto.indexing import Indexer
from pareto.retrieval import BM25Ranker, DenseRetriever, HybridRetriever


def main() -> None:
    console = Console()

    # ── 1. Load shared resources ──
    console.print("[dim]Loading index + test set...[/dim]")
    indexer = Indexer.load("benchmarks/results/index")
    test_set = TestSet.from_yaml("benchmarks/queries/queries.yaml")
    console.print(
        f"[dim]  → {indexer.store.size} chunks, {len(test_set)} queries[/dim]"
    )

    # ── 2. Build the three retrievers ──
    console.print("[dim]Building retrievers...[/dim]")
    dense_ret = DenseRetriever(indexer)
    bm25 = BM25Ranker()
    bm25.build_from_records(indexer.store.records)
    hybrid_ret = HybridRetriever(indexer=indexer, bm25_ranker=bm25)
    console.print(
        f"[dim]  → dense, BM25 (vocab={bm25.vocabulary_size}), "
        f"hybrid (rrf_k={hybrid_ret.rrf_k})[/dim]"
    )

    # ── 3. Run each retriever ──
    k = 5
    runs: list[tuple[str, "BenchmarkReport"]] = []  # noqa: F821
    for name, retriever in [
        ("dense_only", dense_ret),
        ("bm25_only", bm25),
        ("hybrid_rrf", hybrid_ret),
    ]:
        console.print(f"\n[bold]Running [cyan]{name}[/cyan] at k={k}...[/bold]")
        runner = BenchmarkRunner(indexer=indexer, retriever=retriever)
        report = runner.run_retrieval(test_set, k=k, system_name=name)
        console.print(f"  → {report.summary_line()}")

        out_path = save_report(
            report, f"benchmarks/results/compare_{name}_k{k}.json"
        )
        runs.append((name, report))

    # ── 4. Pretty side-by-side comparison ──
    console.print()
    table = Table(title=f"Three-Way Retrieval Comparison (k={k}, N={runs[0][1].num_queries})")
    table.add_column("Retriever")
    table.add_column("hit@k", justify="right")
    table.add_column("precision@k", justify="right")
    table.add_column("recall@k", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("latency (ms)", justify="right")

    for name, report in runs:
        table.add_row(
            name,
            f"{report.avg_hit_at_k:.2%}",
            f"{report.avg_precision_at_k:.3f}",
            f"{report.avg_recall_at_k:.3f}",
            f"{report.avg_mrr:.3f}",
            f"{report.avg_retrieval_latency_ms:.1f}",
        )
    console.print(table)

    # ── 5. Per-domain hit@k comparison ──
    dt = Table(title="hit@k by domain")
    dt.add_column("Domain")
    for name, _ in runs:
        dt.add_column(name, justify="right")

    domains = list(runs[0][1].by_domain.keys())
    for domain in domains:
        row = [domain]
        for _, report in runs:
            row.append(f"{report.by_domain[domain]['hit_at_k']:.2%}")
        dt.add_row(*row)
    console.print(dt)

    # ── 6. Quick winner analysis ──
    console.print("\n[bold yellow]Quick analysis:[/bold yellow]")
    dense = runs[0][1]
    bm25_r = runs[1][1]
    hybrid = runs[2][1]

    def fmt_delta(a: float, b: float) -> str:
        d = a - b
        return f"[green]+{d:.4f}[/green]" if d > 0 else (
            f"[red]{d:.4f}[/red]" if d < 0 else f"[dim]={d:.4f}[/dim]"
        )

    console.print(
        f"  hit@k:  hybrid vs dense = {fmt_delta(hybrid.avg_hit_at_k, dense.avg_hit_at_k)}, "
        f"hybrid vs bm25 = {fmt_delta(hybrid.avg_hit_at_k, bm25_r.avg_hit_at_k)}"
    )
    console.print(
        f"  MRR:    hybrid vs dense = {fmt_delta(hybrid.avg_mrr, dense.avg_mrr)}, "
        f"hybrid vs bm25 = {fmt_delta(hybrid.avg_mrr, bm25_r.avg_mrr)}"
    )
    console.print(
        f"  P@k:    hybrid vs dense = {fmt_delta(hybrid.avg_precision_at_k, dense.avg_precision_at_k)}, "
        f"hybrid vs bm25 = {fmt_delta(hybrid.avg_precision_at_k, bm25_r.avg_precision_at_k)}"
    )


if __name__ == "__main__":
    main() 