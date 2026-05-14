"""Indexing: embeddings + vector store + pipeline."""

from pareto.indexing.embedder import BaseEmbedder, SentenceTransformerEmbedder
from pareto.indexing.indexer import Indexer, IndexingStats
from pareto.indexing.models import SearchResult, VectorRecord
from pareto.indexing.vector_store import VectorStore, VectorStoreConfig

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "VectorRecord",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
    "Indexer",
    "IndexingStats",
]