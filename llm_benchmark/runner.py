from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from llm_benchmark.logging_utils import ConversationLogger
from llm_benchmark.messages import ChatMessage
from llm_benchmark.models.base import ModelAdapter
from llm_benchmark.tasks import DEFAULT_ASSESSMENT_PROMPT, SpeakingTask


class BenchmarkRunner:
    def __init__(self, models: list[ModelAdapter], logs_root: Path) -> None:
        if not models:
            raise ValueError("At least one model adapter is required.")

        self.models = models
        self.logs_root = logs_root

    def run(self, task: SpeakingTask, max_turns: int | None = None) -> None:
        total_turns = max_turns if max_turns is not None else task.max_turns
        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.logs_root / f"{session_stamp}_{task.task_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        loggers = {
            model.label: ConversationLogger(session_dir, model.label, task.task_id)
            for model in self.models
        }

        histories: dict[str, list[ChatMessage]] = {
            model.label: [
                ChatMessage(role="system", content=DEFAULT_ASSESSMENT_PROMPT),
                ChatMessage(role="system", content=task.system_prompt),
            ]
            for model in self.models
        }

        print(f"\nTask: {task.title}")
        print(task.description)
        print(f"Max turns: {total_turns}")
        print("Type /quit to end early.\n")

        start_signal = ChatMessage(role="user", content=task.starter_instruction)
        for model in self.models:
            histories[model.label].append(start_signal)
            loggers[model.label].write("system-default-prompt", DEFAULT_ASSESSMENT_PROMPT, turn="setup")
            loggers[model.label].write("system-task-prompt", task.system_prompt, turn="setup")
            loggers[model.label].write("system-bootstrap", task.starter_instruction, turn="0")

        print("Examiner opening turn (parallel model generation)...\n")
        opening = self._generate_parallel(histories)
        self._record_and_print(opening, histories, loggers, turn=0)

        for turn in range(1, total_turns + 1):
            user_text = input(f"Candidate (turn {turn})> ").strip()
            if user_text.lower() in {"/quit", "quit", "exit"}:
                print("\nSession ended by user.")
                break
            if not user_text:
                print("Please enter a response or /quit.")
                continue

            for model in self.models:
                histories[model.label].append(ChatMessage(role="user", content=user_text))
                loggers[model.label].write("candidate", user_text, turn=turn)

            model_replies = self._generate_parallel(histories)
            self._record_and_print(model_replies, histories, loggers, turn=turn)

        print(f"\nLogs saved to: {session_dir}")

    def _generate_parallel(
        self, histories: dict[str, list[ChatMessage]]
    ) -> dict[str, str]:
        responses: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=len(self.models)) as pool:
            futures = {
                pool.submit(model.generate, list(histories[model.label])): model.label
                for model in self.models
            }

            for future in as_completed(futures):
                model_label = futures[future]
                try:
                    responses[model_label] = future.result().strip()
                except Exception as exc:  # noqa: BLE001
                    responses[model_label] = f"[ERROR] {exc}"

        return responses

    def _record_and_print(
        self,
        replies: dict[str, str],
        histories: dict[str, list[ChatMessage]],
        loggers: dict[str, ConversationLogger],
        turn: int,
    ) -> None:
        for model in self.models:
            label = model.label
            reply = replies.get(label, "[No response captured]")
            histories[label].append(ChatMessage(role="assistant", content=reply))
            loggers[label].write("examiner", reply, turn=turn)

            print(f"[{label}]\n{reply}\n")
