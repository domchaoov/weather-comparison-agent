# Weather Comparison Agent

A small demo AI agent that compares the weather between two places. It calls a
local [Ollama](https://ollama.com) model, which uses tool calling to:

1. Look up the weather for each place (`get_weather`) — backed by a hard-coded
   dataset in [weather_agent/tools.py](weather_agent/tools.py).
2. Calculate the temperature difference (`calculate_temperature_difference`).
3. Combine the results into a final natural-language answer.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a running local Ollama instance
with a tool-calling-capable model pulled (e.g. `ollama pull llama3.2`).

```bash
uv sync
```

## Usage

```bash
uv run main.py "What's the difference in weather between London and Cairo?"
```

Or run with the built-in default question:

```bash
uv run main.py
```

Options:

- `--model <name>` — Ollama model to use (default: `gemma4:latest`; small
  models like `llama3.2:1b` are often too weak to reliably chain multiple
  tool calls).
- `--quiet` — suppress the tool-call logging and just print the final answer.

Known locations: London, New York, Tokyo, Sydney, Cairo, Moscow, Nairobi,
Reykjavik.
