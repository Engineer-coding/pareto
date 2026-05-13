"""
Chunker configuration.

Every parameter is exposed here so we can tune chunking strategy per
domain or per benchmark run without touching the algorithm code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkerConfig(BaseModel):
    """Tunable parameters for HierarchicalChunker."""

    # ── size targets (characters) ─────────────────────────────────────────
    min_chunk_size: int = Field(default=200, ge=50)
    """Don't create leaf chunks smaller than this when avoidable."""

    target_chunk_size: int = Field(default=600, ge=100)
    """Preferred leaf chunk size. The splitter aims for this."""

    max_chunk_size: int = Field(default=1000, ge=200)
    """Hard ceiling. A leaf larger than this WILL be split."""

    # ── overlap ───────────────────────────────────────────────────────────
    chunk_overlap: int = Field(default=80, ge=0)
    """Char overlap between consecutive sibling leaves. Helps preserve context."""

    # ── splitting hierarchy ───────────────────────────────────────────────
    paragraph_separators: tuple[str, ...] = (
        "\n\n\n",
        "\n\n",
        "\n",
    )
    """Tried in order from coarsest to finest when splitting big text blocks."""

    sentence_separators: tuple[str, ...] = (". ", "! ", "? ", "; ")
    """If paragraph splits fail to bring chunks under max_chunk_size, fall back to these."""

    # ── behavior ──────────────────────────────────────────────────────────
    keep_empty_sections: bool = False
    """If False, sections containing no text after splitting are pruned."""