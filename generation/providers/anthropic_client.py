"""Anthropic client. The Messages API has a separate `system` parameter, so
system messages are extracted and concatenated."""

from __future__ import annotations

from anthropic import Anthropic

from core.config import get_settings
from generation.llm import Completion, LLMClient, Message


class AnthropicClient(LLMClient):
    def __init__(self) -> None:
        s = get_settings()
        if not s.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is empty. Set it in .env or switch LLM_PROVIDER."
            )
        self._client = Anthropic(api_key=s.anthropic_api_key)
        self._model = s.anthropic_model
        self._default_temperature = s.llm_temperature
        self._default_max_tokens = s.llm_max_tokens

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        system_parts = [m.content for m in messages if m.role == "system"]
        chat = [m for m in messages if m.role != "system"]

        resp = self._client.messages.create(
            model=self._model,
            system="\n\n".join(system_parts) if system_parts else None,  # type: ignore[arg-type]
            messages=[{"role": m.role, "content": m.content} for m in chat],
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return Completion(
            text=text.strip(),
            model=self._model,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
