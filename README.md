# Weather Comparison Agent

A small demo AI agent that compares the weather between two places. It calls
a model via [Ollama](https://ollama.com) (local) or [OpenRouter](https://openrouter.ai)
(hosted), which uses tool calling to:

1. Look up the weather for each place (`get_weather`) — backed by a hard-coded
   dataset in [weather_agent/tools.py](weather_agent/tools.py).
2. Calculate the temperature difference (`calculate_temperature_difference`).
3. Combine the results into a final natural-language answer.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Pick a backend:

- **Ollama (default)** — a running local Ollama instance with a
  tool-calling-capable model pulled (e.g. `ollama pull llama3.2`).
- **OpenRouter** — set `OPENROUTER_API_KEY` in `.env` (get a key at
  [openrouter.ai/keys](https://openrouter.ai/keys)).

## Usage

```bash
uv run main.py "What's the difference in weather between London and Cairo?"
```

Or run with the built-in default question:

```bash
uv run main.py
```

Use OpenRouter instead of local Ollama:

```bash
uv run main.py --provider openrouter --model openai/gpt-4o-mini "Compare London and Tokyo"
```

Options:

- `--provider <ollama|openrouter>` — chat backend to use (default: `ollama`).
- `--model <name>` — model to use (default depends on `--provider`:
  `gemma4:latest` for Ollama, `openai/gpt-4o-mini` for OpenRouter; small
  models like `llama3.2:1b` are often too weak to reliably chain multiple
  tool calls).
- `--quiet` — suppress the tool-call logging and just print the final answer.

Known locations: London, New York, Tokyo, Sydney, Cairo, Moscow, Nairobi,
Reykjavik.
