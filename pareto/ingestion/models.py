"""
Document data model — the core "currency" of the Pareto ingestion pipeline.

Every reader produces Documents.
Every downstream component (chunker, indexer, retriever, etc.) consumes Documents.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


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
    level: int = 0
    start_char: int
    end_char: int | None = None
    text: str | None = None


class Document(BaseModel):
    """
    A single document ingested into the Pareto pipeline.

    Identity is DETERMINISTIC: the same (source, content) pair always produces
    the same id. This is what makes re-indexing idempotent — if a document is
    unchanged, its id is unchanged, and downstream cache layers (vector store,
    semantic cache, observability traces) all see a stable key.
    """

    # ── identity ───────────────────────────────────────────────────────────
    id: str | None = None
    """
    Stable deterministic identifier derived from (source, content).
    If left as None, it is computed automatically after construction.
    Callers may also pass an explicit id (e.g. for legacy data).
    """

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

    # ── time ───────────────────────────────────────────────────────────────
    created_at: datetime | None = None

    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── extensibility ──────────────────────────────────────────────────────
    extra: dict[str, Any] = Field(default_factory=dict)

    # ── validators ─────────────────────────────────────────────────────────
    @field_validator("content", mode="before")
    @classmethod
    def _strip_bom_and_normalize(cls, v: str) -> str:
        """Defensive cleanup: strip UTF-8 BOM and normalize line endings."""
        if not isinstance(v, str) or not v:
            return v
        v = v.lstrip("\ufeff")
        v = v.replace("\r\n", "\n").replace("\r", "\n")
        return v

    @model_validator(mode="after")
    def _assign_deterministic_id(self) -> "Document":
        """Derive a stable id from (source, content) if the caller didn't provide one."""
        if not self.id:
            self.id = self.compute_id(self.source, self.content)
        return self

    @staticmethod
    def compute_id(source: str, content: str) -> str:
        """
        Compute a deterministic id from source + content.

        Truncated SHA-256 (first 32 hex chars) — gives 128 bits of entropy,
        more than enough to avoid collisions in any realistic corpus,
        while keeping ids short and human-glanceable.
        """
        raw = f"{source}\x00{content}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    # ── computed ───────────────────────────────────────────────────────────
    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA-256 of the content. Used for cache invalidation."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @computed_field
    @property
    def length(self) -> int:
        return len(self.content)

    # ── helpers ────────────────────────────────────────────────────────────
    def short_repr(self) -> str:
        title_part = self.title or "(untitled)"
        id_part = (self.id or "")[:8]
        return f"Document(id={id_part}, format={self.format.value}, len={self.length}, title={title_part!r})"