"""Ollama client. Reuses the OpenAI SDK because Ollama exposes a compatible API."""

from __future__ import annotations

from openai import OpenAI

from core.config import get_settings
from generation.llm import Completion, LLMClient, Message


class OllamaClient(LLMClient):
    def __init__(self) -> None:
        s = get_settings()
        # Ollama doesn't require a key; the SDK still wants one set.
        self._client = OpenAI(base_url=s.ollama_base_url, api_key="ollama")
        self._model = s.ollama_model
        self._default_temperature = s.llm_temperature
        self._default_max_tokens = s.llm_max_tokens

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return Completion(
            text=(choice.message.content or "").strip(),
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
