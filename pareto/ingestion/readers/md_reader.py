"""Markdown reader. Extracts content and structural headings as hints."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from markdown_it import MarkdownIt

from pareto.ingestion.base import BaseReader
from pareto.ingestion.models import Document, DocumentFormat, StructuralHint


class MDReader(BaseReader):
    """
    Reads Markdown files.

    Beyond raw text, it walks the markdown-it token stream to record every
    heading as a StructuralHint with its level and char offset. The chunker
    later uses these hints to build the document tree.
    """

    format = DocumentFormat.MD
    extensions = ("md", "markdown")

    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark")

    def read(self, path: str | Path) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Markdown file not found: {p}")

        content = p.read_text(encoding="utf-8")
        hints = self._extract_heading_hints(content)
        title = self._infer_title(hints, fallback=p.stem)

        stat = p.stat()
        return Document(
            content=content,
            source=str(p.resolve()),
            format=self.format,
            title=title,
            structural_hints=hints,
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            extra={"byte_size": stat.st_size, "heading_count": len(hints)},
        )

    # ── internals ─────────────────────────────────────────────────────────

    def _extract_heading_hints(self, content: str) -> list[StructuralHint]:
        """
        Walk the markdown-it token stream and emit a StructuralHint per heading.

        We map each heading back to its char offset in the original source
        by tracking lines, because markdown-it exposes (start_line, end_line)
        per token but not char positions directly.
        """
        # Pre-compute char offset of each line start
        line_starts = [0]
        for i, ch in enumerate(content):
            if ch == "\n":
                line_starts.append(i + 1)

        tokens = self._parser.parse(content)
        hints: list[StructuralHint] = []

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open":
                level = int(tok.tag[1:])  # "h2" → 2
                start_line, end_line = tok.map if tok.map else (0, 0)
                start_char = line_starts[start_line] if start_line < len(line_starts) else 0
                # Find the inline token (the heading text) that follows
                heading_text = None
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    heading_text = tokens[i + 1].content.strip()
                # end_char: start of the line after the heading block
                end_char = (
                    line_starts[end_line]
                    if end_line < len(line_starts)
                    else len(content)
                )
                hints.append(
                    StructuralHint(
                        kind="heading",
                        level=level,
                        start_char=start_char,
                        end_char=end_char,
                        text=heading_text,
                    )
                )
            i += 1

        return hints

    @staticmethod
    def _infer_title(hints: list[StructuralHint], fallback: str) -> str:
        """Title is the first H1 if present, else the filename stem."""
        for h in hints:
            if h.kind == "heading" and h.level == 1 and h.text:
                return h.text
        return fallback