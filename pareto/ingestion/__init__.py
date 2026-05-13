"""Document ingestion: readers for PDF, DOCX, MD, HTML, TXT → Document."""

from pareto.ingestion.base import BaseReader
from pareto.ingestion.models import Document, DocumentFormat, StructuralHint
from pareto.ingestion.readers import (
    DEFAULT_READERS,
    get_reader_for,
    read_file,
)

__all__ = [
    "BaseReader",
    "Document",
    "DocumentFormat",
    "StructuralHint",
    "DEFAULT_READERS",
    "get_reader_for",
    "read_file",
]