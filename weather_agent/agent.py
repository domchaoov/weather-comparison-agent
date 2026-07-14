"""A small tool-calling agent that talks to a local Ollama model."""

from __future__ import annotations

import json

from ollama import chat
from overmind import entry_point

from .tools import TOOL_IMPLEMENTATIONS, TOOL_SCHEMAS

DEFAULT_MODEL = "gemma4:latest"

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


@entry_point("Weather Comparison Agent")
def run(user_message: str, model: str = DEFAULT_MODEL, verbose: bool = True) -> str:
    """Run the agent loop for a single user request and return the final answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = chat(model=model, messages=messages, tools=TOOL_SCHEMAS)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message.get("content", "")

        for tool_call in tool_calls:
            fn = tool_call["function"]
            name = fn["name"]
            arguments = fn["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if verbose:
                print(f"  -> calling tool: {name}({arguments})")

            result = _call_tool(name, arguments)

            if verbose:
                print(f"  <- tool result: {result}")

            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                }
            )

    return "Sorry, I couldn't finish comparing the weather within the tool-call budget."
