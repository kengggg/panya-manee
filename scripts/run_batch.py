#!/usr/bin/env python3
"""
Run benchmark batch: multiple models x N runs, serially.

Thin wrapper around main.py run — loops over models and run counts,
generating deterministic run IDs in the format: {batch_id}-{slug}-r{NN}

Usage:
  python scripts/run_batch.py --batch-id mini-r10-20260415 --models "qwen3:0.6b,gemma3:1b" --runs-per-model 10
  python scripts/run_batch.py --batch-id test --models "qwen3:0.6b" --runs-per-model 1 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from summarize_batch import generate_repeat_summary


def slugify(model_id: str) -> str:
    """Convert model ID to a filename-safe slug (e.g. 'gemma4:e4b' -> 'gemma4-e4b')."""
    return model_id.replace(":", "-").replace("/", "-")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark batch: multiple models x N runs")
    parser.add_argument("--batch-id", required=True, help="Batch identifier (e.g. mini-r10-20260415)")
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--runs-per-model", type=int, default=1, help="Number of runs per model")
    parser.add_argument("--subjects", default="thai,math", help="Comma-separated subjects")
    parser.add_argument("--think", type=int, default=None, help="Think mode token budget (omit to disable)")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, use fake answers")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERROR: No models specified", file=sys.stderr)
        sys.exit(1)

    total_runs = len(models) * args.runs_per_model
    print(f"Batch '{args.batch_id}': {len(models)} model(s) x {args.runs_per_model} run(s) = {total_runs} total")
    if args.dry_run:
        print("DRY RUN — no API calls\n")

    completed = 0
    for model in models:
        slug = slugify(model)
        for run_num in range(1, args.runs_per_model + 1):
            run_id = f"{args.batch_id}-{slug}-r{run_num:02d}"
            completed += 1

            print(f"\n{'='*60}")
            print(f"[{completed}/{total_runs}] {model} — run {run_num}/{args.runs_per_model}")
            print(f"  run_id: {run_id}")
            print(f"{'='*60}")

            cmd = [
                sys.executable, str(PROJECT_ROOT / "main.py"), "run",
                "--model", model,
                "--subjects", args.subjects,
                "--run-id", run_id,
            ]
            if args.think is not None:
                cmd += ["--think", str(args.think)]
            if args.dry_run:
                cmd += ["--dry-run"]

            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print(f"\nERROR: Run failed (exit {result.returncode})", file=sys.stderr)
                print(f"  model: {model}", file=sys.stderr)
                print(f"  run_id: {run_id}", file=sys.stderr)
                sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Batch complete: {completed}/{total_runs} runs finished")
    print(f"Results in: {PROJECT_ROOT / 'benchmark_responses'}/")
    print(f"{'='*60}")

    # Generate repeat_summary for downstream snapshot pipeline
    if args.dry_run:
        print("\nDry run — skipping repeat summary (no JSONL artifacts produced)")
    else:
        print(f"\nGenerating repeat summary...")
        generate_repeat_summary(args.batch_id, PROJECT_ROOT / "benchmark_responses")


if __name__ == "__main__":
    main()
