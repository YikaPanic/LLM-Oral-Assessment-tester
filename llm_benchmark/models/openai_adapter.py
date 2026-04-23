from __future__ import annotations

from typing import Sequence

from llm_benchmark.messages import ChatMessage
from llm_benchmark.models.base import ModelAdapter


class OpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        timeout_s: int = 120,
    ) -> None:
        super().__init__(provider="openai", model_name=model_name)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Missing dependency 'openai'. Install requirements first.") from exc

        self.client = OpenAI(api_key=api_key, timeout=timeout_s)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        input_messages = []
        for message in messages:
            # In Responses API history, assistant content must be "output_text".
            content_type = "output_text" if message.role == "assistant" else "input_text"
            input_messages.append(
                {
                    "role": message.role,
                    "content": [{"type": content_type, "text": message.content}],
                }
            )

        response = self.client.responses.create(
            model=self.model_name,
            input=input_messages,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        output_text = getattr(response, "output_text", "")
        if output_text:
            return output_text.strip()

        fallback_parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", "") != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") in {"output_text", "text"}:
                    text = getattr(content, "text", "")
                    if text:
                        fallback_parts.append(text)

        if fallback_parts:
            return "\n".join(fallback_parts).strip()

        return "[No response text returned by OpenAI API]"
