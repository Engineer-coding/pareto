"""
Batch chunking: walk a corpus directory, chunk every document, collect stats.

This is the bridge between Tuesday's ingestion layer and Wednesday's chunker.
It also produces the JSON manifest that Week 2's benchmark suite will consume.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pareto.chunking.chunker import HierarchicalChunker
from pareto.chunking.config import ChunkerConfig
from pareto.chunking.models import ChunkTree
from pareto.ingestion.loader import load_directory
from pareto.ingestion.models import Document


@dataclass
class DocumentChunkingStats:
    """Per-document stats — one row of the manifest."""

    document_id: str
    source: str
    format: str
    title: str | None
    content_length: int
    num_nodes: int
    num_leaves: int
    depth: int
    avg_leaf_length: float
    min_leaf_length: int
    max_leaf_length: int


@dataclass
class CorpusChunkingReport:
    """Aggregate stats across the whole batch run."""

    root: str
    num_documents: int
    num_failed: int
    total_chars: int
    total_nodes: int
    total_leaves: int
    formats: dict[str, int] = field(default_factory=dict)
    per_document: list[DocumentChunkingStats] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "num_documents": self.num_documents,
            "num_failed": self.num_failed,
            "total_chars": self.total_chars,
            "total_nodes": self.total_nodes,
            "total_leaves": self.total_leaves,
            "formats": self.formats,
            "per_document": [asdict(d) for d in self.per_document],
            "failures": self.failures,
        }


def _stats_for(doc: Document, tree: ChunkTree) -> DocumentChunkingStats:
    """Compute single-document stats from a Document/ChunkTree pair."""
    leaves = tree.leaves()
    leaf_lengths = [leaf.length for leaf in leaves if leaf.length > 0]

    return DocumentChunkingStats(
        document_id=doc.id,
        source=doc.source,
        format=doc.format.value,
        title=doc.title,
        content_length=doc.length,
        num_nodes=tree.num_nodes,
        num_leaves=tree.num_leaves,
        depth=tree.depth(),
        avg_leaf_length=(sum(leaf_lengths) / len(leaf_lengths)) if leaf_lengths else 0.0,
        min_leaf_length=min(leaf_lengths) if leaf_lengths else 0,
        max_leaf_length=max(leaf_lengths) if leaf_lengths else 0,
    )


def chunk_directory(
    root: str | Path,
    config: ChunkerConfig | None = None,
) -> tuple[list[tuple[Document, ChunkTree]], CorpusChunkingReport]:
    """
    Load every supported document under `root`, chunk each, return results + report.

    Args:
        root: corpus directory.
        config: optional chunker config (defaults to ChunkerConfig()).

    Returns:
        (results, report) where:
            results = list of (Document, ChunkTree) pairs for every success.
            report = aggregate stats with per-document detail and failures.
    """
    root_path = Path(root)
    docs, ingestion_failures = load_directory(root_path)

    chunker = HierarchicalChunker(config=config)
    results: list[tuple[Document, ChunkTree]] = []
    per_doc_stats: list[DocumentChunkingStats] = []
    chunking_failures: list[dict[str, str]] = []
    format_counter: Counter[str] = Counter()
    total_chars = 0
    total_nodes = 0
    total_leaves = 0

    for doc in docs:
        try:
            tree = chunker.chunk(doc)
        except Exception as e:
            chunking_failures.append(
                {"source": doc.source, "error": f"{type(e).__name__}: {e}"}
            )
            continue

        results.append((doc, tree))
        stats = _stats_for(doc, tree)
        per_doc_stats.append(stats)

        format_counter[doc.format.value] += 1
        total_chars += doc.length
        total_nodes += tree.num_nodes
        total_leaves += tree.num_leaves

    # Translate ingestion failures into the report's uniform schema
    failures = chunking_failures + [
        {"source": str(p), "error": f"{type(e).__name__}: {e}"}
        for p, e in ingestion_failures
    ]

    report = CorpusChunkingReport(
        root=str(root_path.resolve()),
        num_documents=len(results),
        num_failed=len(failures),
        total_chars=total_chars,
        total_nodes=total_nodes,
        total_leaves=total_leaves,
        formats=dict(format_counter),
        per_document=per_doc_stats,
        failures=failures,
    )
    return results, report


def save_report(report: CorpusChunkingReport, path: str | Path) -> Path:
    """Persist the report as JSON. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path