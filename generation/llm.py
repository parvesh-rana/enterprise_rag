"""LLMClient abstract interface and a factory keyed by `settings.llm_provider`.

The contract is deliberately tiny: one method, `complete(messages) -> Completion`.
Streaming is a Phase 5 concern (the API will stream tokens to the client).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from core.config import get_settings

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    """LLM response. Token counts are best-effort (provider-dependent)."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(ABC):
    """One method, one job. Providers implement this and nothing else."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Build the configured client. Cached so we don't reinit per request."""
    provider = get_settings().llm_provider
    if provider == "nvidia":
        from generation.providers.nvidia_client import NvidiaClient

        return NvidiaClient()
    if provider == "anthropic":
        from generation.providers.anthropic_client import AnthropicClient

        return AnthropicClient()
    if provider == "ollama":
        from generation.providers.ollama_client import OllamaClient

        return OllamaClient()
    raise ValueError(f"Unknown llm_provider: {provider!r}")


def reset_llm_cache() -> None:
    """Test helper: drop the cached client (e.g. after switching providers)."""
    get_llm_client.cache_clear()
