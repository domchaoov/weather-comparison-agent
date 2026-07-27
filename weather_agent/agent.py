"""A small tool-calling agent that talks to a local Ollama model or OpenRouter."""

from __future__ import annotations

from overmind import entry_point, init

init(
    agent_name="Weather Comparison Assistant",
    service_name="weather-comparison-agent",
    providers=["openai"],
)

from .providers import DEFAULT_PROVIDER, PROVIDERS
from .tools import TOOL_IMPLEMENTATIONS, TOOL_SCHEMAS

DEFAULT_MODEL = PROVIDERS[DEFAULT_PROVIDER].default_model

SYSTEM_PROMPT = (
    "You are a weather comparison assistant. When asked to compare the weather "
    "between two or more places, always call the get_weather tool for each place "
    "to find its temperature and conditions, then call the "
    "calculate_temperature_difference tool to work out the numeric difference "
    "before giving your final answer. Never guess a temperature or difference "
    "yourself - always use the tools. Once you have all the tool results you "
    "need, reply with a concise final answer in plain English."
)

MAX_TOOL_ROUNDS = 6


def _call_tool(name: str, arguments: dict) -> dict:
    implementation = TOOL_IMPLEMENTATIONS.get(name)
    if implementation is None:
        return {"error": f"Unknown tool '{name}'"}
    return implementation(**arguments)


@entry_point("Weather Comparison Assistant")
def run(
    user_message: str,
    model: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    verbose: bool = True,
) -> str:
    """Run the agent loop for a single user request and return the final answer."""
    try:
        chat_provider = PROVIDERS[provider]
    except KeyError:
        raise ValueError(f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDERS)}") from None

    model = model or chat_provider.default_model

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        message = chat_provider.chat(model=model, messages=messages, tools=TOOL_SCHEMAS)
        messages.append(message)

        tool_calls = chat_provider.extract_tool_calls(message)
        if not tool_calls:
            return message.get("content") or ""

        for tool_call in tool_calls:
            if verbose:
                print(f"  -> calling tool: {tool_call.name}({tool_call.arguments})")

            result = _call_tool(tool_call.name, tool_call.arguments)

            if verbose:
                print(f"  <- tool result: {result}")

            messages.append(chat_provider.build_tool_message(tool_call, result))

    return "Sorry, I couldn't finish comparing the weather within the tool-call budget."
