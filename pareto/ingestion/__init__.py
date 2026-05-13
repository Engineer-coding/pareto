"""Document ingestion: readers for PDF, DOCX, MD, HTML, TXT → Document."""

from pareto.ingestion.base import BaseReader
from pareto.ingestion.models import Document, DocumentFormat, StructuralHint

__all__ = [
    "BaseReader",
    "Document",
    "DocumentFormat",
    "StructuralHint",
]