from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from llm_benchmark.messages import ChatMessage


class ModelAdapter(ABC):
    def __init__(self, provider: str, model_name: str) -> None:
        self.provider = provider
        self.model_name = model_name

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model_name}"

    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage]) -> str:
        """Generate a single assistant response for the provided conversation."""
