"""HTML reader. Strips boilerplate (scripts, styles, nav) and extracts clean text."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from pareto.ingestion.base import BaseReader
from pareto.ingestion.models import Document, DocumentFormat, StructuralHint


# Tags whose contents are never useful as document content
_DROP_TAGS = ("script", "style", "noscript", "iframe", "svg", "canvas")

# Tags that are usually layout chrome, not content
_BOILERPLATE_TAGS = ("nav", "header", "footer", "aside")


class HTMLReader(BaseReader):
    """
    Reads HTML files into clean text.

    Strategy:
        1. Drop tags that never carry content (script/style/...).
        2. Drop common boilerplate (nav/header/footer/aside) unless main content is missing.
        3. Prefer the contents of <main> or <article> if present.
        4. Extract heading hints (h1-h6) with char offsets in the cleaned text.
    """

    format = DocumentFormat.HTML
    extensions = ("html", "htm")

    def read(self, path: str | Path) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"HTML file not found: {p}")

        raw = p.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")

        # Step 1: drop tags that never carry content
        for tag in soup(_DROP_TAGS):
            tag.decompose()

        # Step 2: prefer main/article if available
        main = soup.find("main") or soup.find("article")
        root = main if main else soup

        # Step 3: if we're falling back to the full body, drop boilerplate
        if root is soup:
            for tag in root(_BOILERPLATE_TAGS):
                tag.decompose()

        # Step 4: extract heading hints AND clean text together
        # We collect text and headings in document order
        content_parts: list[str] = []
        hints: list[StructuralHint] = []
        cursor = 0  # current char offset in the assembled content

        for element in root.descendants:
            if not getattr(element, "name", None) and isinstance(element, str):
                # NavigableString — bare text
                text = str(element).strip()
                if text:
                    content_parts.append(text)
                    cursor += len(text) + 1  # +1 for the joining newline
                continue

            tag_name = getattr(element, "name", None)
            if tag_name and tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                heading_text = element.get_text(strip=True)
                if heading_text:
                    level = int(tag_name[1])
                    hints.append(
                        StructuralHint(
                            kind="heading",
                            level=level,
                            start_char=cursor,
                            end_char=cursor + len(heading_text),
                            text=heading_text,
                        )
                    )

        content = "\n".join(content_parts)
        title = self._extract_title(soup, hints, fallback=p.stem)

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

    @staticmethod
    def _extract_title(soup: BeautifulSoup, hints: list[StructuralHint], fallback: str) -> str:
        """Prefer <title>, then first H1, then filename."""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        for h in hints:
            if h.level == 1 and h.text:
                return h.text
        return fallback