from __future__ import annotations

import argparse
from pathlib import Path

from llm_benchmark.config import AppConfig
from llm_benchmark.models import AnthropicAdapter, GeminiAdapter, ModelAdapter, OpenAIAdapter
from llm_benchmark.runner import BenchmarkRunner
from llm_benchmark.tasks import get_task, list_tasks

SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "gemini", "anthropic")


def _build_adapters(config: AppConfig, requested: set[str]) -> list[ModelAdapter]:
    adapters: list[ModelAdapter] = []

    if "openai" in requested:
        if not config.openai_api_key:
            raise ValueError("Missing OpenAI API key. Set it in config.yaml or OPENAI_API_KEY.")
        adapters.append(
            OpenAIAdapter(
                api_key=config.openai_api_key,
                model_name=config.openai_model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            )
        )

    if "gemini" in requested:
        if not config.gemini_api_key:
            raise ValueError("Missing Gemini API key. Set it in config.yaml or GEMINI_API_KEY.")
        adapters.append(
            GeminiAdapter(
                api_key=config.gemini_api_key,
                model_name=config.gemini_model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            )
        )

    if "anthropic" in requested:
        if not config.anthropic_api_key:
            raise ValueError(
                "Missing Anthropic API key. Set it in config.yaml or ANTHROPIC_API_KEY."
            )
        adapters.append(
            AnthropicAdapter(
                api_key=config.anthropic_api_key,
                model_name=config.anthropic_model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            )
        )

    return adapters


def _parse_requested_models(raw: str) -> set[str]:
    requested = {chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()}
    allowed = set(SUPPORTED_PROVIDERS)

    unknown = sorted(requested - allowed)
    if unknown:
        unknown_csv = ", ".join(unknown)
        raise ValueError(f"Unknown model provider(s): {unknown_csv}")

    if not requested:
        raise ValueError("At least one model provider must be selected.")

    return requested


def _configured_providers(config: AppConfig) -> set[str]:
    configured: set[str] = set()
    if config.openai_api_key:
        configured.add("openai")
    if config.gemini_api_key:
        configured.add("gemini")
    if config.anthropic_api_key:
        configured.add("anthropic")
    return configured


def _select_models_interactively(config: AppConfig) -> set[str]:
    configured_providers = _configured_providers(config)
    if not configured_providers:
        raise ValueError(
            "No API keys configured. Please set at least one key in config.yaml "
            "or environment variables before running."
        )

    options = [
        ("openai", "OpenAI GPT-5.4 nano", bool(config.openai_api_key)),
        ("gemini", "Google Gemini 3.1 Flash Lite Preview", bool(config.gemini_api_key)),
        ("anthropic", "Anthropic Claude Haiku 4.5", bool(config.anthropic_api_key)),
    ]

    print("\nChoose model provider(s) for this test:\n")
    for idx, (_, label, configured) in enumerate(options, start=1):
        key_status = "key configured" if configured else "key missing"
        print(f"{idx}. {label} [{key_status}]")
    print("A. All configured models")

    while True:
        raw = input("\nModel selection (e.g. 1,3 or A)> ").strip().lower()
        if not raw:
            print("Please select at least one model.")
            continue

        if raw in {"a", "all"}:
            selected = set(configured_providers)
            if not selected:
                print("No configured models found. Add API keys in config.yaml first.")
                continue
            return selected

        selected: set[str] = set()
        invalid = False
        for chunk in raw.split(","):
            value = chunk.strip()
            if not value.isdigit():
                invalid = True
                break
            index = int(value)
            if index < 1 or index > len(options):
                invalid = True
                break
            selected.add(options[index - 1][0])

        if invalid or not selected:
            print("Invalid selection. Use numbers like 1,2 or A.")
            continue

        unconfigured = sorted(selected - configured_providers)
        if unconfigured:
            print(
                "Selected model(s) missing API keys: "
                + ", ".join(unconfigured)
                + ". Configure keys first."
            )
            continue

        return selected


def _print_tasks() -> None:
    print("Available DELA speaking tasks:\n")
    for task in list_tasks():
        print(f"- {task.task_id}: {task.title}")
        print(f"  {task.description}")


def _select_task_interactively() -> str:
    tasks = list_tasks()
    print("\nChoose a task:\n")
    for idx, task in enumerate(tasks, start=1):
        print(f"{idx}. {task.task_id} - {task.title}")

    while True:
        choice = input("\nTask number> ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        index = int(choice)
        if 1 <= index <= len(tasks):
            return tasks[index - 1].task_id

        print("Number out of range.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple LLM APIs on DELA speaking-task simulations.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--task",
        help="Task ID to run. If omitted, an interactive selector is shown.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available predefined tasks and exit.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated providers: openai, gemini, anthropic. "
            "If omitted, you will choose models interactively."
        ),
    )
    parser.add_argument("--logs-dir", default="logs", help="Directory for TXT conversation logs")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override the task's default maximum turns",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_tasks:
        _print_tasks()
        return 0

    try:
        config = AppConfig.from_file(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 2

    try:
        requested_models = (
            _parse_requested_models(args.models)
            if args.models
            else _select_models_interactively(config)
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130

    try:
        adapters = _build_adapters(config, requested_models)
    except (ImportError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 2

    if not adapters:
        print("No adapters were initialized. Check --models and API keys.")
        return 2

    try:
        task_id = args.task or _select_task_interactively()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130

    try:
        task = get_task(task_id)
    except ValueError as exc:
        print(exc)
        return 2

    runner = BenchmarkRunner(models=adapters, logs_root=Path(args.logs_dir))
    try:
        runner.run(task=task, max_turns=args.max_turns)
    except KeyboardInterrupt:
        print("\nSession cancelled by user.")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
