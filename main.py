import argparse
import sys

from weather_agent.agent import DEFAULT_MODEL, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the weather between two places using a local Ollama model.")
    parser.add_argument(
        "question",
        nargs="*",
        help="e.g. 'What's the difference in weather between London and Tokyo?'",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--quiet", action="store_true", help="Hide tool-call logging.")
    args = parser.parse_args()

    question = " ".join(args.question) if args.question else "What's the difference in weather between London and Tokyo?"

    print(f"Q: {question}\n")
    answer = run(question, model=args.model, verbose=not args.quiet)
    print(f"\nA: {answer}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surfaced to the user, e.g. Ollama not running
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
