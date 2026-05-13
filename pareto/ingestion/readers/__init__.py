"""Reader implementations and a registry to dispatch by extension."""

from __future__ import annotations

from pathlib import Path

from pareto.ingestion.base import BaseReader
from pareto.ingestion.models import Document, DocumentFormat
from pareto.ingestion.readers.docx_reader import DOCXReader
from pareto.ingestion.readers.html_reader import HTMLReader
from pareto.ingestion.readers.md_reader import MDReader
from pareto.ingestion.readers.pdf_reader import PDFReader
from pareto.ingestion.readers.txt_reader import TXTReader

#: Default registry: maps extension → reader class.
DEFAULT_READERS: dict[str, type[BaseReader]] = {
    "pdf": PDFReader,
    "docx": DOCXReader,
    "md": MDReader,
    "markdown": MDReader,
    "html": HTMLReader,
    "htm": HTMLReader,
    "txt": TXTReader,
}


def get_reader_for(path: str | Path) -> BaseReader:
    """
    Return an instantiated reader appropriate for the given file path.

    Raises:
        ValueError: if no reader is registered for the file's extension.
    """
    ext = Path(path).suffix.lower().lstrip(".")
    reader_cls = DEFAULT_READERS.get(ext)
    if reader_cls is None:
        raise ValueError(
            f"No reader registered for extension '.{ext}'. "
            f"Supported: {sorted(DEFAULT_READERS)}"
        )
    return reader_cls()


def read_file(path: str | Path) -> Document:
    """Convenience: select the right reader and read the file in one call."""
    return get_reader_for(path).read(path)


__all__ = [
    "DOCXReader",
    "HTMLReader",
    "MDReader",
    "PDFReader",
    "TXTReader",
    "DEFAULT_READERS",
    "get_reader_for",
    "read_file",
]