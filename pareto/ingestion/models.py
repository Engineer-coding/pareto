"""
Document data model — the core "currency" of the Pareto ingestion pipeline.

Every reader produces Documents.
Every downstream component (chunker, indexer, retriever, etc.) consumes Documents.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator


class DocumentFormat(str, Enum):
    """Supported source formats."""

    PDF = "pdf"
    DOCX = "docx"
    MD = "md"
    HTML = "html"
    TXT = "txt"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, path: str | Path) -> "DocumentFormat":
        """Infer format from a file extension."""
        ext = Path(path).suffix.lower().lstrip(".")
        try:
            return cls(ext)
        except ValueError:
            return cls.UNKNOWN


class StructuralHint(BaseModel):
    """
    A structural marker discovered during ingestion.

    These are passed to the chunker as hints for hierarchical splitting.
    Examples: a heading at char offset 240, level 2; a page break at offset 5120.
    """

    kind: str  # "heading" | "page_break" | "list_start" | "table" | ...
    level: int = 0  # for headings: 1 = H1, 2 = H2, ...
    start_char: int
    end_char: int | None = None
    text: str | None = None  # for headings, the heading text


class Document(BaseModel):
    """
    A single document ingested into the Pareto pipeline.

    This is the input to chunking and the unit of source-attribution in retrieval.
    """

    # ── identity ───────────────────────────────────────────────────────────
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Stable unique identifier. Auto-generated UUID4 unless explicitly set."""

    # ── content ────────────────────────────────────────────────────────────
    content: str
    """The full text body of the document."""

    # ── source ─────────────────────────────────────────────────────────────
    source: str
    """Origin: a file path, URL, or any locator describing where this came from."""

    format: DocumentFormat = DocumentFormat.UNKNOWN
    """The original format of the source."""

    title: str | None = None
    """Optional title (e.g. PDF metadata, <title>, first H1, or filename fallback)."""

    # ── structure ──────────────────────────────────────────────────────────
    structural_hints: list[StructuralHint] = Field(default_factory=list)
    """Headings, page breaks, etc. — used by the chunker for tree construction."""

    # ── time ───────────────────────────────────────────────────────────────
    created_at: datetime | None = None
    """When the source document was originally created (if known from metadata)."""

    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """When Pareto ingested this document."""

    # ── extensibility ──────────────────────────────────────────────────────
    extra: dict[str, Any] = Field(default_factory=dict)
    """
    Domain-specific or reader-specific metadata.
    Examples: {"page_count": 14, "language": "en", "legal_jurisdiction": "TR"}
    """

    # ── validators ─────────────────────────────────────────────────────────
    @field_validator("content", mode="before")
    @classmethod
    def _strip_bom_and_normalize(cls, v: str) -> str:
        """
        Defensive cleanup applied to every Document's content:
          * remove leading UTF-8/UTF-16 BOM if present (common on Windows files)
          * normalize stray carriage returns

        This runs before any other field logic, so all downstream consumers
        (chunker, indexer, embeddings) see clean text.
        """
        if not isinstance(v, str) or not v:
            return v
        # Strip BOMs (UTF-8 EF BB BF decodes to \ufeff)
        v = v.lstrip("\ufeff")
        # Normalize Windows-style line endings to Unix
        v = v.replace("\r\n", "\n").replace("\r", "\n")
        return v

    # ── computed ───────────────────────────────────────────────────────────
    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """SHA-256 of the content. Used for change detection and cache invalidation."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def length(self) -> int:
        """Number of characters in the content."""
        return len(self.content)

    # ── helpers ────────────────────────────────────────────────────────────
    def short_repr(self) -> str:
        """One-line summary for logging."""
        title_part = self.title or "(untitled)"
        return f"Document(id={self.id[:8]}, format={self.format.value}, len={self.length}, title={title_part!r})"