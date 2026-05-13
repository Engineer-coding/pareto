"""
High-level convenience: load all supported documents from a directory tree.

This is what the rest of the pipeline (chunker, indexer, benchmarks) consumes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from pareto.ingestion.models import Document
from pareto.ingestion.readers import DEFAULT_READERS, get_reader_for

logger = logging.getLogger(__name__)


def discover_files(
    root: str | Path,
    extensions: Iterable[str] | None = None,
    recursive: bool = True,
) -> list[Path]:
    """
    Walk `root` and return all files with a supported (or specified) extension.

    Args:
        root: directory to scan.
        extensions: iterable of extensions (without dot, lowercase) to include.
            Defaults to all extensions registered in DEFAULT_READERS.
        recursive: walk subdirectories if True.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    allowed = set(e.lower() for e in (extensions or DEFAULT_READERS.keys()))
    pattern = "**/*" if recursive else "*"

    files: list[Path] = []
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower().lstrip(".") in allowed:
            files.append(p)

    return sorted(files)


def load_directory(
    root: str | Path,
    extensions: Iterable[str] | None = None,
    recursive: bool = True,
    skip_failures: bool = True,
) -> tuple[list[Document], list[tuple[Path, Exception]]]:
    """
    Load every supported file under `root` into Documents.

    Returns:
        (documents, failures) where failures is a list of (path, exception) for
        files that could not be read. If skip_failures is False, the first
        failure raises.
    """
    files = discover_files(root, extensions=extensions, recursive=recursive)
    documents: list[Document] = []
    failures: list[tuple[Path, Exception]] = []

    for path in files:
        try:
            reader = get_reader_for(path)
            documents.append(reader.read(path))
        except Exception as e:
            failures.append((path, e))
            logger.warning("Failed to read %s: %s", path, e)
            if not skip_failures:
                raise

    return documents, failures