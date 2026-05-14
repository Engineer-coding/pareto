"""
LLM client layer — uniform interface, multiple providers.

BaseLLMClient is the abstract contract.
LiteLLMClient wraps LiteLLM, which itself supports 100+ providers (Ollama,
OpenAI, Anthropic, Cohere, ...) via a single completion() call. This is the
right level of abstraction for Pareto: we get model swap-ability for free
without writing a different SDK call per provider.

Every successful call returns an LLMResponse with cost & latency populated,
so Week 2's observability layer can record every dollar spent and every
millisecond burnt.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from pareto.generation.models import LLMConfig, LLMError, LLMResponse

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract LLM client. Subclasses implement `.generate()` per provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model."""

    @abstractmethod
    def generate(
        self,
        prompt: str | list[dict[str, str]],
        **overrides,
    ) -> LLMResponse:
        """
        Run a completion.

        Args:
            prompt: either a raw user string (becomes a single user message)
                or an OpenAI-style messages list.
            **overrides: per-call overrides for LLMConfig fields
                (e.g. temperature=0.3, max_tokens=200).
        """


class LiteLLMClient(BaseLLMClient):
    """
    LiteLLM-backed client. Works with Ollama (default), OpenAI, Anthropic, etc.

    Default config is Ollama + llama3.2:3b, matching Pareto's
    "local-first, cost-zero" Week 1 stance.
    """

    def __init__(self, config: LLMConfig | None = None):
        # Heavy import deferred
        import litellm  # noqa: F401

        self._config = config or LLMConfig()

    # ── BaseLLMClient API ─────────────────────────────────────────────────
    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def config(self) -> LLMConfig:
        return self._config

    def generate(
        self,
        prompt: str | list[dict[str, str]],
        **overrides,
    ) -> LLMResponse:
        import litellm

        messages = self._normalize_messages(prompt)
        cfg = self._merged_config(**overrides)

        start = time.perf_counter()
        try:
            raw = litellm.completion(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                api_base=cfg.api_base,
                timeout=cfg.timeout,
            )
        except Exception as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e

        latency_ms = int((time.perf_counter() - start) * 1000)

        # Extract fields defensively — providers vary in what they return
        text = ""
        finish_reason = None
        try:
            choice = raw.choices[0]
            text = (choice.message.content or "").strip()
            finish_reason = getattr(choice, "finish_reason", None)
        except (AttributeError, IndexError) as e:
            raise LLMError(f"Malformed LLM response: {e}") from e

        # Token usage (may be absent on some local providers)
        usage = getattr(raw, "usage", None) or {}
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens)

        # Cost estimate. LiteLLM has pricing tables for hosted providers;
        # for local (Ollama) it returns 0.0 — which is the correct answer.
        try:
            cost_usd = float(litellm.completion_cost(completion_response=raw) or 0.0)
        except Exception:
            cost_usd = 0.0

        return LLMResponse(
            text=text,
            model=cfg.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_messages(
        prompt: str | list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            return prompt
        raise TypeError(f"prompt must be str or list, got {type(prompt).__name__}")

    def _merged_config(self, **overrides) -> LLMConfig:
        """Return a copy of self._config with the given fields overridden."""
        from dataclasses import replace
        return replace(self._config, **overrides)

    def __repr__(self) -> str:
        return f"LiteLLMClient(model={self._config.model!r})"