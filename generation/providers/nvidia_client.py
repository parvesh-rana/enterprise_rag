"""NVIDIA NIM client (default provider).

NIM speaks the OpenAI Chat Completions wire format, so the official `openai`
SDK works against it by overriding `base_url`. This is exactly what
view_code.txt demonstrated; we wrap it behind the LLMClient interface.
"""

from __future__ import annotations

from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletion

from core.config import get_settings
from generation.llm import Completion, LLMClient, Message


class NvidiaClient(LLMClient):
    def __init__(self) -> None:
        s = get_settings()
        if not s.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is empty. Set it in .env or switch LLM_PROVIDER.")
        self._client = OpenAI(base_url=s.nvidia_base_url, api_key=s.nvidia_api_key)
        self._model = s.llm_model
        self._default_temperature = s.llm_temperature
        self._default_max_tokens = s.llm_max_tokens

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        # The OpenAI SDK uses overly strict TypedDicts for messages; cast to Any
        # since our Message.role is already constrained to the supported literals.
        payload = cast(Any, [{"role": m.role, "content": m.content} for m in messages])
        resp: ChatCompletion = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
            top_p=0.95,
            stream=False,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return Completion(
            text=(choice.message.content or "").strip(),
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
