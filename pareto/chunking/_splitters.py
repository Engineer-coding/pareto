"""
Low-level text splitting helpers used by the chunker.

These work on raw strings and know nothing about ChunkTree or NodeKind.
Keeping them separate makes them trivially testable and reusable.
"""

from __future__ import annotations


def split_by_separator(text: str, separator: str, keep_separator: bool = True) -> list[str]:
    """
    Split `text` on `separator`. If keep_separator is True, the separator stays
    attached to the *end* of the preceding piece, which preserves char offsets
    when we concatenate later.
    """
    if not separator:
        return [text]
    parts = text.split(separator)
    if not keep_separator or len(parts) == 1:
        return [p for p in parts if p]
    rebuilt: list[str] = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            rebuilt.append(p + separator)
        else:
            if p:
                rebuilt.append(p)
    return rebuilt


def recursive_split(
    text: str,
    separators: list[str],
    max_size: int,
) -> list[str]:
    """
    Recursively try each separator in order until every piece is <= max_size.

    The classic "RecursiveCharacterTextSplitter" idea: prefer to break on
    paragraph boundaries; if that's not enough, break on lines; if still
    not enough, break on sentence boundaries; finally fall back to hard
    character cuts.
    """
    if len(text) <= max_size or not separators:
        # Either small enough already, or we've exhausted all separators
        if len(text) <= max_size:
            return [text]
        # Hard cut: walk char-by-char
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

    separator, *rest = separators
    pieces = split_by_separator(text, separator, keep_separator=True)

    out: list[str] = []
    for piece in pieces:
        if len(piece) <= max_size:
            out.append(piece)
        else:
            out.extend(recursive_split(piece, rest, max_size))
    return out


def merge_small_pieces(
    pieces: list[str],
    target_size: int,
    overlap: int = 0,
) -> list[str]:
    """
    Greedily merge consecutive pieces until each merged group is near `target_size`.

    This compensates for over-fragmentation by the recursive splitter:
    if a paragraph has many short sentences, we'd rather a few balanced
    chunks than many tiny ones.

    With overlap > 0, each merged chunk reuses the tail of the previous one
    to preserve cross-chunk context.
    """
    if not pieces:
        return []

    merged: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue
        if len(current) + len(piece) <= target_size:
            current = current + piece
        else:
            merged.append(current)
            # Build the next chunk with optional overlap from the previous tail
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + piece
            else:
                current = piece

    if current:
        merged.append(current)
    return merged