"""
Data models for the generation layer.

LLMResponse  — uniform response shape across providers (Ollama, OpenAI, ...).
LLMConfig    — per-call config: model name, sampling params, endpoint, timeout.

Cost and latency are first-class fields, not afterthoughts. Every LLM call
in Pareto carries enough metadata for the observability layer to answer:
"how much did this request cost, and how long did it take?"
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class LLMConfig:
    """Settings for a single LLM call. Kept as a dataclass for easy override."""

    model: str = "ollama/llama3.2:3b"
    """LiteLLM model identifier. 'ollama/<name>', 'openai/<name>', 'anthropic/<name>', etc."""

    api_base: str | None = "http://localhost:11434"
    """Base URL for self-hosted providers (Ollama default). None for cloud."""

    temperature: float = 0.0
    """0.0 → deterministic. Critical for RAG: same query should give same answer."""

    max_tokens: int = 1024
    """Max completion length. RAG answers rarely need more than 500."""

    timeout: int = 120
    """Seconds before the call is aborted. Local 3B models can be slow."""


class LLMResponse(BaseModel):
    """Uniform response from any provider."""

    text: str
    """The generated completion text."""

    model: str
    """Model identifier used. May differ from request if a fallback was hit."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    cost_usd: float = 0.0
    """Estimated cost in USD. 0.0 for local/self-hosted models."""

    latency_ms: int = 0
    """Wall-clock time for the call."""

    finish_reason: str | None = None
    """'stop', 'length', 'tool_calls', etc."""

    extra: dict = Field(default_factory=dict)
    """Provider-specific extras (e.g. {'cached': True} from a future cache layer)."""

    def short_repr(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"LLMResponse(model={self.model!r}, tokens={self.total_tokens}, "
            f"cost=${self.cost_usd:.5f}, latency={self.latency_ms}ms, "
            f"text={preview!r}...)"
        )


class LLMError(Exception):
    """Raised when an LLM call fails. Wraps the underlying provider error."""