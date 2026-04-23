from __future__ import annotations

from typing import Sequence

import requests

from llm_benchmark.messages import ChatMessage
from llm_benchmark.models.base import ModelAdapter


class GeminiAdapter(ModelAdapter):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        timeout_s: int = 120,
    ) -> None:
        super().__init__(provider="gemini", model_name=model_name)
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        system_messages = [message.content for message in messages if message.role == "system"]
        contents = []

        for message in messages:
            if message.role not in {"user", "assistant"}:
                continue

            role = "user" if message.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": message.content}]})

        if not contents:
            contents.append(
                {"role": "user", "parts": [{"text": "Start the speaking task."}]}
            )

        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if system_messages:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_messages)}]
            }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model_name}:generateContent"
        )
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout_s,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text.strip()}"
            )

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "[No response text returned by Gemini API]"

        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if part.get("text")]
        if texts:
            return "\n".join(texts).strip()

        return "[No response text returned by Gemini API]"
