# LLM DELA Speaking Benchmark CLI

A CLI tool to benchmark multiple LLM APIs in a structured DELA-style speaking-task simulation.

## Features

- Predefined DELA speaking tasks
- Unified model adapter interface: `response = model.generate(messages)`
- Parallel API calls across providers each turn
- Human-in-the-loop, turn-by-turn conversation
- Separate TXT transcript per model and session
- Configurable API keys and model IDs via `config.yaml` or environment variables

## Supported Providers

- OpenAI (`gpt-5.4-nano` default)
- Google Gemini (`gemini-3.1-flash-lite-preview` default)
- Anthropic (`claude-haiku-4.5` default)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create local config from template, then fill your API keys:

```bash
cp config.example.yaml config.yaml
```

You can also use environment variables.
If you use a proxy/gateway for Anthropic, set `anthropic_base_url` (or `ANTHROPIC_BASE_URL`).

## Usage

List tasks:

```bash
python main.py --list-tasks
```

Run with interactive task selection:

```bash
python main.py
```
When `--models` is omitted, the CLI asks you to choose model providers for this test run.

Run a specific task and providers:

```bash
python main.py --task dela_task1_project_partner_roleplay --models openai,gemini
```

Set custom max turns:

```bash
python main.py --max-turns 5
```

Logs are written under `logs/<timestamp>_<task_id>/` with one `.txt` file per model.

## Environment Variable Overrides

- `OPENAI_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `OPENAI_MODEL`
- `GEMINI_MODEL`
- `ANTHROPIC_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_OUTPUT_TOKENS`
