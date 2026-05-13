"""
Abstract base class for document readers.

Each reader (PDFReader, DOCXReader, etc.) implements the same interface:
given a file path, produce one or more Documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pareto.ingestion.models import Document, DocumentFormat


class BaseReader(ABC):
    """
    Common contract for all document readers.

    Subclasses implement `read()` to convert a source file into a Document.
    A single source file always produces exactly one Document
    (multi-document sources like ZIP archives are handled at a higher level).
    """

    #: The format this reader handles. Subclasses must set this.
    format: DocumentFormat = DocumentFormat.UNKNOWN

    #: File extensions this reader accepts (lowercase, without dot).
    extensions: tuple[str, ...] = ()

    def can_read(self, path: str | Path) -> bool:
        """Return True if this reader can handle the given file by extension."""
        return Path(path).suffix.lower().lstrip(".") in self.extensions

    @abstractmethod
    def read(self, path: str | Path) -> Document:
        """
        Read the file at `path` and return a Document.

        Implementations should:
            - Extract textual content
            - Populate `source`, `format`, `title` if available
            - Populate `structural_hints` when the format exposes structure
            - Populate `extra` with format-specific metadata
            - Raise `IOError` or a subclass on read failure
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(format={self.format.value})"