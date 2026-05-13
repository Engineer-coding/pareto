"""
HierarchicalChunker — turns a Document into a ChunkTree.

Algorithm (high level):

    1. Build a SECTION tree from the document's structural_hints (headings).
       Heading H1 -> child of root. Heading H2 -> child of nearest H1. Etc.
       Plain page_break hints (PDF) become flat children of root if no
       headings are present.

    2. For EVERY section (not just leaves), extract its intro span — the
       text between its heading and its first child section (or the whole
       body if no child sections exist) — and split that span into one or
       more PARAGRAPH leaves using recursive_split + greedy merging.

    3. Prune sections whose subtree has no textual content.

    4. Sort each parent's children by start_char so the tree reflects
       document order (intro paragraphs before nested sections).

Design choices:
    * Char offsets are authoritative. Hints carry start_char from the
      reader; we never re-discover structure here.
    * The algorithm is deterministic and pure — same Document + same config
      always produce the same tree. This is critical for caching and benchmarks.
"""

from __future__ import annotations

import uuid

from pareto.chunking._splitters import merge_small_pieces, recursive_split
from pareto.chunking.config import ChunkerConfig
from pareto.chunking.models import ChunkNode, ChunkTree, NodeKind
from pareto.ingestion.models import Document, StructuralHint


class HierarchicalChunker:
    """Chunks documents into a hierarchical tree of sections and paragraphs."""

    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()

    # ── public API ────────────────────────────────────────────────────────

    def chunk(self, document: Document) -> ChunkTree:
        """Build and return the ChunkTree for one Document."""
        root_id = str(uuid.uuid4())
        tree = ChunkTree(document_id=document.id, root_id=root_id)

        root = ChunkNode(
            id=root_id,
            document_id=document.id,
            kind=NodeKind.ROOT,
            title=document.title,
            start_char=0,
            end_char=len(document.content),
        )
        tree.add(root)

        # Step 1: build the section skeleton from heading hints
        headings = [h for h in document.structural_hints if h.kind == "heading"]
        if headings:
            self._build_section_tree(tree, root_id, document, headings)
        else:
            # No headings: the whole document is a single "section" child of root
            self._add_flat_section(tree, root_id, document)

        # Step 2: populate EVERY section with its intro paragraphs.
        # A section's intro is the text between its heading and its first
        # child section (or the whole body if no child sections exist).
        # This is what preserves prose that appears before nested subsections.
        all_sections = [
            n for n in tree.nodes.values() if n.kind == NodeKind.SECTION
        ]
        for section in all_sections:
            self._populate_section_with_paragraphs(tree, section, document)

        # Step 3: optionally prune empty sections
        if not self.config.keep_empty_sections:
            self._prune_empty_sections(tree)

        # Step 4: sort each parent's children by start_char so the document
        # flows in order (intro paragraphs before nested sections).
        self._sort_children_by_position(tree)

        return tree

    # ── step 1: section skeleton ──────────────────────────────────────────

    def _build_section_tree(
        self,
        tree: ChunkTree,
        root_id: str,
        document: Document,
        headings: list[StructuralHint],
    ) -> None:
        """
        Construct a section subtree under root using a stack to track the
        currently-open headings at each level. Classic outline-parsing pattern.
        """
        # Stack entries: (level, node_id). We always know the current parent
        # by popping until we find a strictly-lower-level ancestor.
        stack: list[tuple[int, str]] = [(0, root_id)]

        # Sort headings by position to be defensive (readers should already)
        headings = sorted(headings, key=lambda h: h.start_char)

        for idx, h in enumerate(headings):
            # Find the parent: the topmost stack entry with level < h.level
            while stack and stack[-1][0] >= h.level:
                stack.pop()
            parent_id = stack[-1][1] if stack else root_id

            # The section spans from the heading's start to the start of the
            # next heading (or end of document)
            next_start = (
                headings[idx + 1].start_char
                if idx + 1 < len(headings)
                else len(document.content)
            )

            section = ChunkNode(
                id=str(uuid.uuid4()),
                document_id=document.id,
                parent_id=parent_id,
                kind=NodeKind.SECTION,
                level=h.level,
                title=h.text,
                start_char=h.start_char,
                end_char=next_start,
            )
            tree.add(section)
            stack.append((h.level, section.id))

    def _add_flat_section(
        self,
        tree: ChunkTree,
        root_id: str,
        document: Document,
    ) -> None:
        """When the document has no headings, the body is one big section."""
        if not document.content.strip():
            return
        section = ChunkNode(
            id=str(uuid.uuid4()),
            document_id=document.id,
            parent_id=root_id,
            kind=NodeKind.SECTION,
            level=1,
            title=document.title,
            start_char=0,
            end_char=len(document.content),
        )
        tree.add(section)

    # ── step 2: populate leaves ───────────────────────────────────────────

    def _populate_section_with_paragraphs(
        self,
        tree: ChunkTree,
        section: ChunkNode,
        document: Document,
    ) -> None:
        """
        Add paragraph leaves to `section`, covering only its intro span
        (from after the heading line up to the first child section, or to
        section.end_char if no child sections exist).
        """
        body_start = section.start_char
        # Skip past the heading line itself when we have one
        if section.title:
            heading_line_end = document.content.find("\n", body_start)
            if heading_line_end != -1 and heading_line_end < section.end_char:
                body_start = heading_line_end + 1

        # Determine where this section's intro ends: at the first child
        # section, or at the section's own end if it has none.
        child_sections = [
            tree.get(cid) for cid in section.children_ids
            if tree.get(cid).kind == NodeKind.SECTION
        ]
        if child_sections:
            body_end = min(s.start_char for s in child_sections)
        else:
            body_end = section.end_char

        body = document.content[body_start:body_end].strip()
        if not body:
            return

        # Step 2a: recursive split down to <= max_chunk_size
        pieces = recursive_split(
            body,
            list(self.config.paragraph_separators)
            + list(self.config.sentence_separators)
            + [""],  # "" triggers the hard-cut fallback
            self.config.max_chunk_size,
        )

        # Step 2b: greedy merge towards target_chunk_size with overlap
        merged = merge_small_pieces(
            pieces,
            target_size=self.config.target_chunk_size,
            overlap=self.config.chunk_overlap,
        )

        # Step 2c: emit one PARAGRAPH node per merged chunk
        # We track char offsets to keep position info accurate (approximate
        # because of overlap, but good enough for citations).
        cursor = body_start
        for chunk_text in merged:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            # Best-effort positional anchor (overlap makes this approximate)
            start = document.content.find(chunk_text, cursor)
            if start == -1:
                start = cursor
            end = start + len(chunk_text)
            cursor = end

            leaf = ChunkNode(
                id=str(uuid.uuid4()),
                document_id=document.id,
                parent_id=section.id,
                kind=NodeKind.PARAGRAPH,
                content=chunk_text,
                start_char=start,
                end_char=end,
            )
            tree.add(leaf)

    # ── step 3: pruning ───────────────────────────────────────────────────

    def _prune_empty_sections(self, tree: ChunkTree) -> None:
        """Remove section nodes whose subtree contains no paragraph content."""
        to_remove: set[str] = set()
        for node in tree.nodes.values():
            if node.kind != NodeKind.SECTION:
                continue
            descendants = tree.descendants(node.id)
            if not descendants:
                to_remove.add(node.id)
                continue
            has_text = any(d.content.strip() for d in descendants)
            if not has_text:
                to_remove.add(node.id)

        for node_id in to_remove:
            node = tree.nodes.pop(node_id, None)
            if node and node.parent_id:
                parent = tree.nodes.get(node.parent_id)
                if parent and node_id in parent.children_ids:
                    parent.children_ids.remove(node_id)

    # ── step 4: ordering ──────────────────────────────────────────────────

    def _sort_children_by_position(self, tree: ChunkTree) -> None:
        """
        Order each node's children by start_char so that intro paragraphs
        appear before their following child sections (document order).
        """
        for node in tree.nodes.values():
            if len(node.children_ids) > 1:
                node.children_ids.sort(key=lambda cid: tree.get(cid).start_char)