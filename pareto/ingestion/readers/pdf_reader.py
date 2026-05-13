"""
PDF reader using pypdf.

Limitations (known and accepted for MVP):
    - No OCR — scanned PDFs without an embedded text layer produce empty text.
    - No layout analysis — multi-column PDFs may interleave columns.
    - No image/table extraction.

For higher-fidelity ingestion we may swap in PyMuPDF, pdfplumber, or Docling in V2.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from pareto.ingestion.base import BaseReader
from pareto.ingestion.models import Document, DocumentFormat, StructuralHint


class PDFReader(BaseReader):
    """Reads PDF files into plain text, page by page."""

    format = DocumentFormat.PDF
    extensions = ("pdf",)

    def read(self, path: str | Path) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {p}")

        reader = PdfReader(str(p))

        # Extract per-page text and record page-break hints with char offsets
        page_texts: list[str] = []
        hints: list[StructuralHint] = []
        cursor = 0
        for page_idx, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                # Some PDFs have pages that raise during extraction; skip gracefully
                txt = ""
            page_texts.append(txt)
            # Record a page_break hint at the START of each page (after the first)
            if page_idx > 0:
                hints.append(
                    StructuralHint(
                        kind="page_break",
                        level=0,
                        start_char=cursor,
                        text=f"page {page_idx + 1}",
                    )
                )
            cursor += len(txt) + 2  # +2 for the "\n\n" join below

        content = "\n\n".join(page_texts)

        # Pull title from PDF metadata if present, else filename
        meta_title = None
        try:
            if reader.metadata and reader.metadata.title:
                meta_title = str(reader.metadata.title).strip()
        except Exception:
            pass
        title = meta_title or p.stem

        # Optional metadata
        page_count = len(reader.pages)
        try:
            author = str(reader.metadata.author).strip() if reader.metadata and reader.metadata.author else None
        except Exception:
            author = None

        stat = p.stat()
        return Document(
            content=content,
            source=str(p.resolve()),
            format=self.format,
            title=title,
            structural_hints=hints,
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            extra={
                "page_count": page_count,
                "author": author,
                "byte_size": stat.st_size,
                "is_likely_scanned": len(content.strip()) < 50 and page_count > 0,
            },
        )
    