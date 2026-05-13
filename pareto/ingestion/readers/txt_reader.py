"""Plain-text reader. The simplest possible reader, used as a baseline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pareto.ingestion.base import BaseReader
from pareto.ingestion.models import Document, DocumentFormat


class TXTReader(BaseReader):
    """Reads plain UTF-8 text files."""

    format = DocumentFormat.TXT
    extensions = ("txt",)

    def read(self, path: str | Path) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"TXT file not found: {p}")

        # Try UTF-8 first, fall back to latin-1 for legacy files
        try:
            content = p.read_text(encoding="utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = p.read_text(encoding="latin-1")
            encoding = "latin-1"

        stat = p.stat()
        return Document(
            content=content,
            source=str(p.resolve()),
            format=self.format,
            title=p.stem,
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            extra={
                "encoding": encoding,
                "byte_size": stat.st_size,
            },
        )