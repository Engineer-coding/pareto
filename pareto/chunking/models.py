"""
Tree-based chunk representation for hierarchical retrieval.

A document is represented as a tree:
    DocumentRoot
    ├── Section (H1)
    │   ├── Subsection (H2)
    │   │   ├── Paragraph
    │   │   └── Paragraph
    │   └── Subsection (H2)
    └── Section (H1)

Leaves are the small, embeddable chunks.
Internal nodes carry the broader context that wraps them.

At retrieval time we match on leaves (precision) and optionally expand to
parents (context). This is the "small-to-big" retrieval pattern.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeKind(str, Enum):
    """The role a node plays in the chunk tree."""

    ROOT = "root"          # the document itself
    SECTION = "section"    # a heading block (any level)
    PARAGRAPH = "paragraph"  # a leaf chunk of text
    PAGE = "page"          # PDF page boundary (when no headings exist)


class ChunkNode(BaseModel):
    """A node in the hierarchical chunk tree."""

    # ── identity ──────────────────────────────────────────────────────────
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    """The id of the source Document this node belongs to."""

    # ── tree links ────────────────────────────────────────────────────────
    parent_id: str | None = None
    """The id of the parent node. None for root."""

    children_ids: list[str] = Field(default_factory=list)
    """Ids of direct children, in document order."""

    # ── content ───────────────────────────────────────────────────────────
    kind: NodeKind
    level: int = 0
    """For sections, the heading level (1 = H1). For other kinds, 0."""

    content: str = ""
    """The text body of this node. For internal nodes, may be empty (or a title)."""

    title: str | None = None
    """Heading text for sections; None for paragraphs."""

    # ── position in source ────────────────────────────────────────────────
    start_char: int = 0
    end_char: int = 0
    """Character offsets in the original Document.content."""

    # ── extensibility ─────────────────────────────────────────────────────
    extra: dict[str, Any] = Field(default_factory=dict)
    """Reader-/chunker-specific metadata (e.g. page number, token count)."""

    # ── helpers ───────────────────────────────────────────────────────────
    @property
    def length(self) -> int:
        return len(self.content)

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    def short_repr(self) -> str:
        label = self.title or self.content[:40].replace("\n", " ")
        return f"{self.kind.value}(L{self.level}, len={self.length}, {label!r})"


class ChunkTree(BaseModel):
    """
    A complete chunk tree for one Document.

    Nodes are stored in a flat dict keyed by id. Parent/child links use ids
    rather than direct references — this keeps the structure JSON-serializable
    and easy to persist (which we'll need for the vector store).
    """

    document_id: str
    root_id: str
    nodes: dict[str, ChunkNode] = Field(default_factory=dict)

    # ── construction ──────────────────────────────────────────────────────
    def add(self, node: ChunkNode) -> None:
        """Add a node and wire it into its parent's children list."""
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self.nodes[node.id] = node
        if node.parent_id is not None:
            parent = self.nodes.get(node.parent_id)
            if parent is None:
                raise ValueError(f"Parent {node.parent_id} not in tree yet")
            parent.children_ids.append(node.id)

    # ── access ────────────────────────────────────────────────────────────
    @property
    def root(self) -> ChunkNode:
        return self.nodes[self.root_id]

    def get(self, node_id: str) -> ChunkNode:
        return self.nodes[node_id]

    def children(self, node_id: str) -> list[ChunkNode]:
        return [self.nodes[cid] for cid in self.nodes[node_id].children_ids]

    def leaves(self) -> list[ChunkNode]:
        """All leaf nodes in document order."""
        result: list[ChunkNode] = []
        self._collect_leaves(self.root_id, result)
        return result

    def _collect_leaves(self, node_id: str, out: list[ChunkNode]) -> None:
        node = self.nodes[node_id]
        if node.is_leaf:
            out.append(node)
            return
        for child_id in node.children_ids:
            self._collect_leaves(child_id, out)

    def ancestors(self, node_id: str) -> list[ChunkNode]:
        """
        Ancestors from immediate parent up to the root, in that order.

        Used by parent-context expansion: given a matching leaf, walk up to
        find the broader context to feed the LLM.
        """
        path: list[ChunkNode] = []
        current = self.nodes[node_id]
        while current.parent_id is not None:
            current = self.nodes[current.parent_id]
            path.append(current)
        return path

    def descendants(self, node_id: str) -> list[ChunkNode]:
        """All transitive descendants in DFS pre-order."""
        out: list[ChunkNode] = []
        self._collect_descendants(node_id, out)
        return out

    def _collect_descendants(self, node_id: str, out: list[ChunkNode]) -> None:
        for child_id in self.nodes[node_id].children_ids:
            out.append(self.nodes[child_id])
            self._collect_descendants(child_id, out)

    # ── stats ─────────────────────────────────────────────────────────────
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_leaves(self) -> int:
        return len(self.leaves())

    def depth(self) -> int:
        """Maximum depth from root (root depth = 0)."""
        return self._depth(self.root_id, 0)

    def _depth(self, node_id: str, current: int) -> int:
        node = self.nodes[node_id]
        if node.is_leaf:
            return current
        return max(self._depth(cid, current + 1) for cid in node.children_ids)