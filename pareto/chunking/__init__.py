"""Hierarchical chunking: turn a Document into a ChunkTree."""

from pareto.chunking.chunker import HierarchicalChunker
from pareto.chunking.config import ChunkerConfig
from pareto.chunking.models import ChunkNode, ChunkTree, NodeKind
from pareto.chunking.visualize import render_graphviz, render_rich_tree

__all__ = [
    "ChunkNode",
    "ChunkTree",
    "NodeKind",
    "ChunkerConfig",
    "HierarchicalChunker",
    "render_rich_tree",
    "render_graphviz",
]