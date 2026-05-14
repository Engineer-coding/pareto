"""
Data models for the vector store.

VectorRecord  — the metadata we keep alongside each embedded chunk.
SearchResult  — a single hit returned by a vector search.

We deliberately keep VectorRecord small: only the fields a downstream
retriever or LLM prompt actually needs. Heavier per-document data stays
in the Document/ChunkTree layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """Per-chunk metadata stored next to each vector in the index."""

    chunk_id: str
    document_id: str
    parent_id: str | None = None
    content: str
    source: str
    level: int = 0
    kind: str = "paragraph"
    start_char: int = 0
    end_char: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """One hit from VectorStore.search()."""

    record: VectorRecord
    score: float
    """Cosine similarity in [-1, 1] for normalized vectors. Higher is better."""