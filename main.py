"""panya-manee — NT Grade 3 Local LLM Arena for Thai language benchmarking."""

import argparse
import sys


def cmd_run(args):
    from benchmark_runner import run_benchmark, check_ollama

    if not args.dry_run and not check_ollama():
        print("Error: Ollama is not running at localhost:11434", file=sys.stderr)
        print("Start it with: ollama serve", file=sys.stderr)
        sys.exit(1)

    subjects = [s.strip() for s in args.subjects.split(",")]
    think = None if args.no_think or args.think is None else args.think
    run_benchmark(args.model, subjects, args.run_id, args.dry_run, think=think)


def cmd_summarize(args):
    from summarize_results import summarize

    summarize(run_id=args.run_id)


def main():
    parser = argparse.ArgumentParser(
        prog="panya-manee",
        description="NT Grade 3 Local LLM Arena — benchmark Thai language understanding with Ollama",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run benchmark against an Ollama model")
    p_run.add_argument("--model", required=True, help="Ollama model name (e.g. qwen3:0.6b)")
    p_run.add_argument("--subjects", default="thai,math", help="Comma-separated subjects (default: thai,math)")
    p_run.add_argument("--run-id", default="run001", help="Run identifier (default: run001)")
    p_run.add_argument("--dry-run", action="store_true", help="Skip API calls, use fake answers")
    p_run.add_argument("--think", type=int, default=None, metavar="TOKENS",
                       help="Enable thinking mode with token budget (e.g. --think 4096)")
    p_run.add_argument("--no-think", action="store_true",
                       help="Explicitly disable thinking (default behavior)")

    # summarize
    p_sum = sub.add_parser("summarize", help="Summarize benchmark results")
    p_sum.add_argument("--run-id", default=None, help="Filter to specific run ID (default: all)")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "summarize":
        cmd_summarize(args)


if __name__ == "__main__":
    main()
