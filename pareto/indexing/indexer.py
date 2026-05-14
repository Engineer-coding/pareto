"""
Indexer — the bridge between chunking and vector retrieval.

Takes ChunkTrees (with optional source Documents for richer metadata),
embeds their leaves, and writes them to a VectorStore.

Designed to be incremental and re-runnable: chunks already present in
the store are skipped, so re-running on an updated corpus only embeds
the new chunks. This is what keeps the embedding bill predictable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from pareto.chunking.models import ChunkNode, ChunkTree, NodeKind
from pareto.indexing.embedder import BaseEmbedder, SentenceTransformerEmbedder
from pareto.indexing.models import VectorRecord
from pareto.indexing.vector_store import VectorStore, VectorStoreConfig
from pareto.ingestion.models import Document


@dataclass
class IndexingStats:
    """Per-run statistics. Useful for benchmarks and demo-day numbers."""

    num_trees: int = 0
    num_leaves_seen: int = 0
    num_chunks_indexed: int = 0
    num_chunks_skipped: int = 0  # already in store
    num_chunks_filtered: int = 0  # below min length, etc.
    total_chars_indexed: int = 0
    embedding_batches: int = 0
    formats: dict[str, int] = field(default_factory=dict)


class Indexer:
    """High-level pipeline: ChunkTrees → embeddings → VectorStore."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        store: VectorStore | None = None,
        min_chunk_chars: int = 30,
        batch_size: int = 32,
    ):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.store = store or VectorStore(
            VectorStoreConfig(
                embedding_dim=self.embedder.embedding_dim,
                embedder_name=self.embedder.model_name,
            )
        )
        self.min_chunk_chars = min_chunk_chars
        self.batch_size = batch_size

    # ── primary API ───────────────────────────────────────────────────────
    def index_chunk_trees(
        self,
        chunk_trees: list[ChunkTree],
        documents: list[Document] | None = None,
        show_progress: bool = True,
    ) -> IndexingStats:
        """
        Index the leaf chunks from every tree.

        Args:
            chunk_trees: trees produced by HierarchicalChunker.
            documents: optional list of source Documents; used to enrich
                VectorRecords with `source` and document format. Looked up
                by `document_id`.
            show_progress: print a rich progress bar.

        Returns:
            IndexingStats summarizing the run.
        """
        doc_index: dict[str, Document] = (
            {d.id: d for d in documents} if documents else {}
        )

        # Step 1: collect candidate (record, content) pairs across all trees,
        # skipping anything already in the store or below the size threshold.
        candidates: list[tuple[VectorRecord, str]] = []
        stats = IndexingStats(num_trees=len(chunk_trees))

        for tree in chunk_trees:
            doc = doc_index.get(tree.document_id)
            for leaf in tree.leaves():
                stats.num_leaves_seen += 1

                # Filter: chunk must have real content
                if len(leaf.content.strip()) < self.min_chunk_chars:
                    stats.num_chunks_filtered += 1
                    continue

                # Skip duplicates (already indexed)
                if leaf.id in self.store:
                    stats.num_chunks_skipped += 1
                    continue

                record = self._build_record(leaf, doc)
                candidates.append((record, leaf.content))

                fmt = doc.format.value if doc else "unknown"
                stats.formats[fmt] = stats.formats.get(fmt, 0) + 1
                stats.total_chars_indexed += len(leaf.content)

        if not candidates:
            return stats  # nothing new to index

        # Step 2: encode in batches and add to the store
        records = [c[0] for c in candidates]
        texts = [c[1] for c in candidates]

        if show_progress:
            self._encode_with_progress(records, texts, stats)
        else:
            vectors = self.embedder.encode_passages(texts, batch_size=self.batch_size)
            self.store.add(records, vectors)
            stats.embedding_batches = max(1, (len(texts) + self.batch_size - 1) // self.batch_size)
            stats.num_chunks_indexed = len(records)

        return stats

    # ── helpers ───────────────────────────────────────────────────────────
    def _build_record(self, leaf: ChunkNode, doc: Document | None) -> VectorRecord:
        """Translate a ChunkNode (+ source Document) into a VectorRecord."""
        if doc is not None:
            source = doc.source
            extra = {"format": doc.format.value, "title": doc.title}
        else:
            source = "unknown"
            extra = {}

        return VectorRecord(
            chunk_id=leaf.id,
            document_id=leaf.document_id,
            parent_id=leaf.parent_id,
            content=leaf.content,
            source=source,
            level=leaf.level,
            kind=leaf.kind.value,
            start_char=leaf.start_char,
            end_char=leaf.end_char,
            extra=extra,
        )

    def _encode_with_progress(
        self,
        records: list[VectorRecord],
        texts: list[str],
        stats: IndexingStats,
    ) -> None:
        """Encode + add in chunks of `batch_size` with a progress bar."""
        console = Console()
        total = len(texts)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding chunks", total=total)

            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                batch_records = records[start:end]
                batch_texts = texts[start:end]

                vectors = self.embedder.encode_passages(
                    batch_texts, batch_size=self.batch_size
                )
                self.store.add(batch_records, vectors)

                stats.embedding_batches += 1
                stats.num_chunks_indexed += len(batch_records)
                progress.update(task, advance=len(batch_records))

    # ── persistence convenience ───────────────────────────────────────────
    def save(self, directory: str | Path) -> Path:
        """Save the underlying store. Default index location is the same path."""
        return self.store.save(directory)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        embedder: BaseEmbedder | None = None,
    ) -> "Indexer":
        """
        Load a previously-saved index. The embedder is NOT persisted (just its
        name in the config); pass a compatible embedder or accept the default.
        """
        store = VectorStore.load(directory)
        if embedder is None:
            embedder = SentenceTransformerEmbedder(
                model_name=store.config.embedder_name
                or SentenceTransformerEmbedder.DEFAULT_MODEL
            )
        if embedder.embedding_dim != store.config.embedding_dim:
            raise ValueError(
                f"Embedder dim {embedder.embedding_dim} mismatches store dim "
                f"{store.config.embedding_dim}. Are you using a different model?"
            )
        return cls(embedder=embedder, store=store)