"""Chat backends the agent can talk to: OpenAI's API or OpenRouter's hosted API.

Both speak the same OpenAI tool-calling wire format (see tools.TOOL_SCHEMAS),
requiring tool_call_id to thread a result back to its call and JSON-encoded
arguments. Each Provider below owns that formatting so agent.run()'s loop can
stay backend-agnostic.
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


class _OpenAICompatibleProvider:
    """Base for providers that speak the OpenAI chat completions API."""

    default_model: str
    base_url: str | None = None
    api_key_env: str
    extra_headers: dict[str, str] | None = None
    signup_url: str

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.api_key_env} is not set. Add it to .env or your environment "
                    f"to use this provider. Get a key at {self.signup_url}"
                )
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=api_key, default_headers=self.extra_headers)
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


class OpenAIProvider(_OpenAICompatibleProvider):
    """Talks directly to OpenAI's API. Requires OPENAI_API_KEY."""

    default_model = "gpt-4o-mini"
    api_key_env = "OPENAI_API_KEY"
    signup_url = "https://platform.openai.com/api-keys"


class OpenRouterProvider(_OpenAICompatibleProvider):
    """Talks to OpenRouter's OpenAI-compatible API. Requires OPENROUTER_API_KEY."""

    default_model = "openai/gpt-4o-mini"
    base_url = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"
    extra_headers = {"X-Title": "Weather Comparison Agent"}
    signup_url = "https://openrouter.ai/keys"


PROVIDERS: dict[str, Provider] = {
    "openai": OpenAIProvider(),
    "openrouter": OpenRouterProvider(),
}

DEFAULT_PROVIDER = "openai"
