"""
Prompt templates for RAG.

Kept as plain strings rather than a templating library: small enough to
read at a glance, easy to A/B test in benchmarks.
"""

from __future__ import annotations

# Default RAG prompt. Designed to:
#   * Encourage grounded answers (use only the context).
#   * Allow honest "I don't know" rather than hallucination.
#   * Encourage citation markers [1], [2] tied to context indices.
DEFAULT_RAG_SYSTEM_PROMPT = """You are a precise assistant. Answer the user's question using ONLY the context provided below. \
If the context does not contain enough information to answer the question, say so plainly — do not invent facts.

Cite sources with bracketed numbers like [1], [2] that match the numbered context blocks.
Keep answers concise: 1-3 short paragraphs unless the question requires more detail."""

DEFAULT_RAG_USER_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def format_context_block(index: int, text: str, source: str) -> str:
    """Render one numbered context block. Source is the short filename, not full path."""
    return f"[{index}] (source: {source})\n{text}"