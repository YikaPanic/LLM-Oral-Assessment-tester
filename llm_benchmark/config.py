from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    openai_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openai_model: str = "gpt-5.4-nano"
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    anthropic_model: str = "claude-haiku-4.5"
    temperature: float = 0.7
    max_output_tokens: int = 700

    @staticmethod
    def from_file(path: str | Path) -> "AppConfig":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config file must contain a top-level mapping")

        def pick(key: str, *env_names: str, default: Any = "") -> Any:
            for env_name in env_names:
                value = os.getenv(env_name)
                if value:
                    return value
            value = raw.get(key, default)
            return default if value is None else value

        return AppConfig(
            openai_api_key=str(pick("openai_api_key", "OPENAI_API_KEY", default="")).strip(),
            gemini_api_key=str(
                pick("gemini_api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY", default="")
            ).strip(),
            anthropic_api_key=str(
                pick("anthropic_api_key", "ANTHROPIC_API_KEY", default="")
            ).strip(),
            openai_model=str(pick("openai_model", "OPENAI_MODEL", default="gpt-5.4-nano")),
            gemini_model=str(
                pick("gemini_model", "GEMINI_MODEL", default="gemini-3.1-flash-lite-preview")
            ),
            anthropic_model=str(
                pick("anthropic_model", "ANTHROPIC_MODEL", default="claude-haiku-4.5")
            ),
            temperature=float(pick("temperature", "LLM_TEMPERATURE", default=0.7)),
            max_output_tokens=int(pick("max_output_tokens", "LLM_MAX_OUTPUT_TOKENS", default=700)),
        )
