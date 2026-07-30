"""Fire a handful of multi-turn conversations at the weather comparison agent.

Unlike scripts/stress_test.py (single-turn, adversarial-focused), this script
exercises multi-turn sessions: each conversation reuses one conversation_id
and one message history across 2+ distinct weather comparison questions, so
the agent can resolve follow-ups ("what about X instead?") and Overmind groups
the turns into a single session trace rather than unrelated one-offs.

Usage:
    uv run scripts/multi_turn_stress_test.py
    uv run scripts/multi_turn_stress_test.py --provider openrouter --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from weather_agent.agent import run  # noqa: E402
from weather_agent.providers import DEFAULT_PROVIDER, PROVIDERS  # noqa: E402


@dataclass
class Conversation:
    name: str
    turns: list[str]


CONVERSATIONS: list[Conversation] = [
    Conversation(
        "follow_up_city_swap",
        [
            "What's the weather difference between London and Tokyo?",
            "What about Cairo instead of Tokyo?",
        ],
    ),
    Conversation(
        "three_turn_chain",
        [
            "Compare the weather in Nairobi and Moscow.",
            "Now compare Nairobi to Sydney.",
            "Which of the cities I've asked about so far is warmest?",
        ],
    ),
    Conversation(
        "unrelated_comparisons_same_session",
        [
            "Is it colder in Reykjavik or Moscow?",
            "Separately, compare the weather between New York and Cairo.",
        ],
    ),
]


def run_conversation(conversation: Conversation, model: str, provider: str, verbose: bool) -> None:
    conversation_id = f"{conversation.name}-{uuid.uuid4().hex[:8]}"
    history: list[dict] = []

    print(f"\n{'=' * 72}\nConversation: {conversation.name}  (conversation_id={conversation_id})\n{'=' * 72}")

    for turn_num, prompt in enumerate(conversation.turns, 1):
        print(f"\n[turn {turn_num}] Q: {prompt}")
        answer = run(
            prompt,
            model=model,
            provider=provider,
            verbose=verbose,
            conversation_id=conversation_id,
            history=history,
        )
        print(f"[turn {turn_num}] A: {answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT_PROVIDER, help=f"Chat backend to test (default: {DEFAULT_PROVIDER})")
    parser.add_argument("--model", default=None, help="Model to test (default depends on --provider)")
    parser.add_argument("--quiet", action="store_true", help="Hide tool-call logging.")
    args = parser.parse_args()

    model = args.model or PROVIDERS[args.provider].default_model

    print(f"Running {len(CONVERSATIONS)} multi-turn conversations against provider={args.provider!r} model={model!r}")

    for conversation in CONVERSATIONS:
        run_conversation(conversation, model=model, provider=args.provider, verbose=not args.quiet)


if __name__ == "__main__":
    main()
