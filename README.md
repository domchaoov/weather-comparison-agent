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

## Stress testing

```bash
uv run scripts/stress_test.py
```

Fires a large batch of single-turn prompts (happy path, edge cases,
out-of-scope, and adversarial/prompt-injection attempts) and reports how the
agent handles each one — see [scripts/stress_test.py](scripts/stress_test.py)
for the full case list and options (`--provider`, `--model`, `--category`,
`--output`).

```bash
uv run scripts/multi_turn_stress_test.py
```

A smaller script that runs a few multi-turn conversations — each with 2 or
more distinct weather comparison questions, including follow-ups that refer
back to an earlier turn — against the agent. See
[scripts/multi_turn_stress_test.py](scripts/multi_turn_stress_test.py).

## Tracing

The agent is instrumented with [Overmind](https://overmindlab.ai) via
`overmind.init()` and `@entry_point` in
[weather_agent/agent.py](weather_agent/agent.py); set `OVERMIND_API_KEY` in
`.env` to export traces. `run()` accepts an optional `conversation_id` (and a
`history` list to thread message context across calls) — passing the same
`conversation_id` on multiple `run()` calls tags their spans so Overmind
groups those turns into a single session in the platform, instead of showing
them as unrelated one-off traces. `scripts/multi_turn_stress_test.py`
demonstrates this by generating one `conversation_id` per conversation.
