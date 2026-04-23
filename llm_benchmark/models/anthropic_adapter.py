from __future__ import annotations

from typing import Sequence

from llm_benchmark.messages import ChatMessage
from llm_benchmark.models.base import ModelAdapter


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        base_url: str | None = None,
    ) -> None:
        super().__init__(provider="anthropic", model_name=model_name)
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("Missing dependency 'anthropic'. Install requirements first.") from exc

        client_kwargs: dict[str, str] = {"api_key": api_key}
        if base_url and base_url.strip():
            client_kwargs["base_url"] = self._normalize_base_url(base_url.strip())
        self.client = Anthropic(**client_kwargs)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        system_chunks = [message.content for message in messages if message.role == "system"]
        anthropic_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"user", "assistant"}
        ]

        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": "Start the speaking task."}]

        kwargs = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if system_chunks:
            kwargs["system"] = "\n\n".join(system_chunks)

        response = self.client.messages.create(**kwargs)
        text_blocks = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                text_blocks.append(block.text)

        if text_blocks:
            return "\n".join(text_blocks).strip()

        return "[No response text returned by Anthropic API]"

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """
        Anthropic SDK appends `/v1/messages` internally.
        Some gateway URLs are provided as `.../v1`, which would become `/v1/v1/messages`.
        """
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[:-3]
        return normalized
