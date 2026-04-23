from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


class ConversationLogger:
    def __init__(self, logs_dir: Path, model_label: str, task_id: str) -> None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_sanitize_filename(model_label)}.txt"
        self.path = logs_dir / filename

        if not self.path.exists():
            self.path.write_text(
                f"Task: {task_id}\n"
                f"Model: {model_label}\n"
                f"Started: {datetime.now().isoformat(timespec='seconds')}\n"
                + ("=" * 72)
                + "\n",
                encoding="utf-8",
            )

    def write(self, role: str, text: str, turn: int | str) -> None:
        entry = (
            f"[Turn {turn}] {role}\n"
            f"{text.strip()}\n"
            + ("-" * 72)
            + "\n"
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
