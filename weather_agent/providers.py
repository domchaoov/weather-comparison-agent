"""Chat backends the agent can talk to: local Ollama or OpenRouter's hosted API.

Both speak OpenAI-style tool schemas (see tools.TOOL_SCHEMAS), but the two
clients disagree on how tool calls and tool results are shaped on the wire -
Ollama is happy to echo its own dicts back untouched, while OpenAI-compatible
APIs require tool_call_id to thread a result back to its call and want
arguments JSON-encoded. Each Provider below owns that formatting so
agent.run()'s loop can stay backend-agnostic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str | None
    name: str
    arguments: dict[str, Any]


def _parse_tool_calls(raw_tool_calls: list[dict] | None) -> list[ToolCall]:
    calls = []
    for call in raw_tool_calls or []:
        fn = call["function"]
        arguments = fn["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append(ToolCall(id=call.get("id"), name=fn["name"], arguments=arguments))
    return calls


class Provider(Protocol):
    default_model: str

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> dict: ...
    def extract_tool_calls(self, message: dict) -> list[ToolCall]: ...
    def build_tool_message(self, tool_call: ToolCall, result: dict) -> dict: ...


class OllamaProvider:
    """Talks to a local Ollama server."""

    default_model = "gemma4:latest"

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> dict:
        from ollama import chat

        response = chat(model=model, messages=messages, tools=tools)
        return response["message"]

    def extract_tool_calls(self, message: dict) -> list[ToolCall]:
        return _parse_tool_calls(message.get("tool_calls"))

    def build_tool_message(self, tool_call: ToolCall, result: dict) -> dict:
        return {"role": "tool", "content": json.dumps(result)}


class OpenRouterProvider:
    """Talks to OpenRouter's OpenAI-compatible API. Requires OPENROUTER_API_KEY."""

    default_model = "openai/gpt-4o-mini"

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set. Add it to .env or your environment "
                    "to use --provider openrouter. Get a key at https://openrouter.ai/keys"
                )
            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={"X-Title": "Weather Comparison Agent"},
            )
        return self._client

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> dict:
        client = self._get_client()
        response = client.chat.completions.create(model=model, messages=messages, tools=tools)
        message = response.choices[0].message
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (message.tool_calls or [])
        ]
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls or None,
        }

    def extract_tool_calls(self, message: dict) -> list[ToolCall]:
        return _parse_tool_calls(message.get("tool_calls"))

    def build_tool_message(self, tool_call: ToolCall, result: dict) -> dict:
        return {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}


PROVIDERS: dict[str, Provider] = {
    "ollama": OllamaProvider(),
    "openrouter": OpenRouterProvider(),
}

DEFAULT_PROVIDER = "ollama"
