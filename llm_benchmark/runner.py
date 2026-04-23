from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from llm_benchmark.logging_utils import ConversationLogger
from llm_benchmark.messages import ChatMessage
from llm_benchmark.models.base import ModelAdapter
from llm_benchmark.tasks import DEFAULT_ASSESSMENT_PROMPT, SpeakingTask

END_TASK_SIGNAL = "[[END_TASK]]"
END_TASK_SIGNAL_PATTERN = re.compile(re.escape(END_TASK_SIGNAL))


class BenchmarkRunner:
    def __init__(self, models: list[ModelAdapter], logs_root: Path) -> None:
        if not models:
            raise ValueError("At least one model adapter is required.")

        self.models = models
        self.logs_root = logs_root

    def run(
        self,
        task: SpeakingTask,
        max_turns: int | None = None,
    ) -> None:
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
        active_models: dict[str, ModelAdapter] = {model.label: model for model in self.models}

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
        current_models = list(active_models.values())
        opening = self._generate_parallel(histories, current_models)
        ended = self._record_and_print(opening, histories, loggers, turn=0, models=current_models)
        for label in ended:
            active_models.pop(label, None)
        if not active_models:
            print("All models signaled task completion.")
            print(f"\nLogs saved to: {session_dir}")
            return

        turn = 1
        while turn <= total_turns and active_models:
            user_text = input(f"Candidate (turn {turn})> ").strip()
            if user_text.lower() in {"/quit", "quit", "exit"}:
                print("\nSession ended by user.")
                break
            if not user_text:
                print("Please enter a response or /quit.")
                continue

            current_models = list(active_models.values())
            for model in current_models:
                histories[model.label].append(ChatMessage(role="user", content=user_text))
                loggers[model.label].write("candidate", user_text, turn=turn)

            model_replies = self._generate_parallel(histories, current_models)
            ended = self._record_and_print(
                model_replies,
                histories,
                loggers,
                turn=turn,
                models=current_models,
            )
            for label in ended:
                active_models.pop(label, None)
            if not active_models:
                print("All models signaled task completion.")
                break
            turn += 1

        print(f"\nLogs saved to: {session_dir}")

    def _generate_parallel(
        self, histories: dict[str, list[ChatMessage]], models: list[ModelAdapter]
    ) -> dict[str, str]:
        responses: dict[str, str] = {}
        if not models:
            return responses

        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {
                pool.submit(model.generate, list(histories[model.label])): model.label
                for model in models
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
        models: list[ModelAdapter],
    ) -> set[str]:
        ended_models: set[str] = set()
        for model in models:
            label = model.label
            raw_reply = replies.get(label, "[No response captured]")
            reply, signaled_end, ignored_end_signal = self._extract_end_signal(raw_reply)
            histories[label].append(ChatMessage(role="assistant", content=reply))
            loggers[label].write("examiner", reply, turn=turn)
            if signaled_end:
                loggers[label].write(
                    "system-session-end",
                    f"Model emitted end signal: {END_TASK_SIGNAL}",
                    turn=turn,
                )
                ended_models.add(label)
            elif ignored_end_signal:
                loggers[label].write(
                    "system-session-note",
                    "Ignored END_TASK signal because reply still contains a question.",
                    turn=turn,
                )

            print(f"[{label}]\n{reply}\n")
            if signaled_end:
                print(f"[{label}] signaled task completion.\n")
        return ended_models

    @staticmethod
    def _extract_end_signal(text: str) -> tuple[str, bool, bool]:
        raw_signal = bool(END_TASK_SIGNAL_PATTERN.search(text))
        cleaned_text = END_TASK_SIGNAL_PATTERN.sub("", text).strip()
        if raw_signal and not cleaned_text:
            cleaned_text = "[Task completed]"
        accepted_signal = raw_signal and not BenchmarkRunner._is_question_turn(cleaned_text)
        ignored_signal = raw_signal and not accepted_signal
        return cleaned_text, accepted_signal, ignored_signal

    @staticmethod
    def _is_question_turn(text: str) -> bool:
        return "?" in text or "？" in text
