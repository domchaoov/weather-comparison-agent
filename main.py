import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from overmind import init

init(service_name="Weather Comparison Agent", providers=[])

from weather_agent.agent import run
from weather_agent.providers import DEFAULT_PROVIDER, PROVIDERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the weather between two places using OpenAI or OpenRouter.")
    parser.add_argument(
        "question",
        nargs="*",
        help="e.g. 'What's the difference in weather between London and Tokyo?'",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default=DEFAULT_PROVIDER,
        help=f"Chat backend to use (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument("--model", default=None, help="Model to use (default depends on --provider)")
    parser.add_argument("--quiet", action="store_true", help="Hide tool-call logging.")
    args = parser.parse_args()

    question = " ".join(args.question) if args.question else "What's the difference in weather between London and Tokyo?"

    print(f"Q: {question}\n")
    answer = run(question, model=args.model, provider=args.provider, verbose=not args.quiet)
    print(f"\nA: {answer}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surfaced to the user, e.g. missing API key
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
