"""RAG orchestration layer: retrieval + generation pipelines."""

from pareto.rag.models import RAGResponse
from pareto.rag.naive_rag import NaiveRAG
from pareto.rag.routed_rag import RoutedRAG
from pareto.rag.prompts import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    DEFAULT_RAG_USER_TEMPLATE,
)

__all__ = [
    "NaiveRAG",
    "RAGResponse",
    "DEFAULT_RAG_SYSTEM_PROMPT",
    "DEFAULT_RAG_USER_TEMPLATE",
    "RoutedRAG",
]