#!/usr/bin/env python3
from __future__ import annotations

"""
Generate repeat_summary_{batch_id}.json from raw JSONL response files.

This bridges the gap between run_batch.py (which produces raw JSONL files)
and build_snapshot.py (which expects a repeat_summary for file discovery,
canonical run selection, and the transparency bundle).

Usage:
  python scripts/summarize_batch.py --batch-id mini-r10-20260415
  python scripts/summarize_batch.py --batch-id mini-r10-20260415 --responses-dir ./benchmark_responses
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"
MODELS_CONFIG = PROJECT_ROOT / "registry" / "models.json"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model_sizes() -> dict[str, float]:
    """Load model_id -> loaded_size_gb_approx from registry/models.json."""
    if not MODELS_CONFIG.exists():
        return {}
    with open(MODELS_CONFIG, encoding="utf-8") as f:
        data = json.load(f)

    sizes: dict[str, float] = {}
    for model in data.get("models", []):
        size = model.get("loaded_size_gb_approx", 0.0)
        sizes[model["model_id"]] = size
        for alias in model.get("aliases") or []:
            sizes[alias] = size

        model_id = model["model_id"]
        if model_id.endswith(":latest"):
            sizes[model_id[:-7]] = size

        for alias in model.get("aliases") or []:
            if alias.endswith(":latest"):
                sizes[alias[:-7]] = size
    return sizes


def slugify(model_id: str) -> str:
    """Derive a short alias from model_id, matching run_batch.py convention."""
    # Strip provider prefix (e.g. "scb10x/typhoon2.1-gemma3-4b:latest" -> "typhoon2.1-gemma3-4b")
    name = model_id
    if "/" in name:
        name = name.split("/", 1)[1]
    # Strip :tag suffix
    if ":" in name:
        name = name.split(":", 1)[0]
    # Take first segment before hyphen-with-digit as alias (e.g. "typhoon2.1-gemma3-4b" -> "typhoon2")
    # But keep it simple: just use the slug
    return name.replace(":", "-").replace("/", "-")


def extract_run_index(run_id: str, batch_id: str) -> int:
    """Extract the run index (NN) from a run_id like '{batch_id}-{slug}-rNN'."""
    suffix = run_id[len(batch_id):] if run_id.startswith(batch_id) else run_id
    match = re.search(r"-r(\d+)$", suffix)
    return int(match.group(1)) if match else 0


def summarize_run(rows: list[dict], run_id: str, run_index: int,
                  output_file: str, model_id: str, alias: str,
                  memory_gb: float) -> dict:
    """Compute per-run statistics from response rows."""
    total = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct", False))
    parseable = sum(1 for r in rows if r.get("is_parseable", False))

    thai = [r for r in rows if r.get("subject") == "thai"]
    math = [r for r in rows if r.get("subject") == "math"]
    thai_correct = sum(1 for r in thai if r.get("is_correct", False))
    math_correct = sum(1 for r in math if r.get("is_correct", False))

    total_latency_ms = sum(r.get("latency_ms", 0) for r in rows)
    latency_s = round(total_latency_ms / 1000, 3) if total_latency_ms else 0.0

    total_eval_ms = sum(r.get("eval_duration_ms", 0) for r in rows)
    total_prompt_ms = sum(r.get("prompt_eval_duration_ms", 0) for r in rows)
    total_eval_tokens = sum(r.get("eval_tokens", 0) for r in rows)
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in rows)

    # Score: 3 points per correct answer (matches existing convention)
    score = correct * 3
    max_score = total * 3

    questions_per_min = (total / latency_s * 60) if latency_s > 0 else 0.0
    correct_per_min = (correct / latency_s * 60) if latency_s > 0 else 0.0
    score_per_min = (score / latency_s * 60) if latency_s > 0 else 0.0
    correct_per_gb_min = (correct_per_min / memory_gb) if memory_gb > 0 else 0.0
    sec_per_correct = (latency_s / correct) if correct > 0 else 0.0

    return {
        "questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "thai_correct": thai_correct,
        "thai_total": len(thai),
        "thai_accuracy": thai_correct / len(thai) if thai else 0.0,
        "math_correct": math_correct,
        "math_total": len(math),
        "math_accuracy": math_correct / len(math) if math else 0.0,
        "parseable": parseable,
        "parse_rate": parseable / total if total else 0.0,
        "latency_s": latency_s,
        "eval_s": round(total_eval_ms / 1000, 3),
        "prompt_s": round(total_prompt_ms / 1000, 3),
        "eval_tokens": total_eval_tokens,
        "prompt_tokens": total_prompt_tokens,
        "score": score,
        "max_score": max_score,
        "model_id": model_id,
        "alias": alias,
        "run_index": run_index,
        "run_id": run_id,
        "output_file": str(output_file),
        "memory_gb": memory_gb,
        "questions_per_min": questions_per_min,
        "correct_per_min": correct_per_min,
        "score_per_min": score_per_min,
        "correct_per_gb_min": correct_per_gb_min,
        "sec_per_correct": sec_per_correct,
    }


def summarize_model(model_id: str, run_summaries: list[dict], memory_gb: float) -> dict:
    """Compute per-model aggregate statistics across runs."""
    num_runs = len(run_summaries)
    corrects = [r["correct"] for r in run_summaries]
    accuracies = [r["accuracy"] for r in run_summaries]
    thai_accs = [r["thai_accuracy"] for r in run_summaries]
    math_accs = [r["math_accuracy"] for r in run_summaries]
    times = [r["latency_s"] for r in run_summaries]
    correct_per_mins = [r["correct_per_min"] for r in run_summaries]
    correct_per_gb_mins = [r["correct_per_gb_min"] for r in run_summaries]

    return {
        "model_id": model_id,
        "memory_gb": memory_gb,
        "runs": num_runs,
        "mean_correct": round(statistics.mean(corrects), 4) if corrects else 0,
        "min_correct": min(corrects) if corrects else 0,
        "max_correct": max(corrects) if corrects else 0,
        "mean_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
        "stdev_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "mean_thai_accuracy": statistics.mean(thai_accs) if thai_accs else 0.0,
        "mean_math_accuracy": statistics.mean(math_accs) if math_accs else 0.0,
        "mean_time_s": round(statistics.mean(times), 4) if times else 0.0,
        "median_time_s": round(statistics.median(times), 4) if times else 0.0,
        "mean_correct_per_min": statistics.mean(correct_per_mins) if correct_per_mins else 0.0,
        "mean_correct_per_gb_min": statistics.mean(correct_per_gb_mins) if correct_per_gb_mins else 0.0,
    }


def generate_repeat_summary(batch_id: str, responses_dir: Optional[Path] = None) -> dict:
    """Generate a repeat_summary dict from raw JSONL files for a batch.

    Returns the summary dict (also writes it to disk).
    """
    responses_dir = responses_dir or DEFAULT_RESPONSES_DIR

    # Discover JSONL files for this batch
    pattern = f"responses_{batch_id}-*.jsonl"
    files = sorted(responses_dir.glob(pattern))
    if not files:
        print(f"ERROR: No response files found matching {responses_dir / pattern}", file=sys.stderr)
        sys.exit(1)

    # Load model sizes from registry
    model_sizes = load_model_sizes()

    # Group files by model_id, extract run metadata
    model_runs: dict[str, list[tuple[int, Path, list[dict]]]] = defaultdict(list)
    earliest_ts = float("inf")
    latest_ts = 0.0

    for fpath in files:
        rows = load_jsonl(fpath)
        if not rows:
            continue
        model_id = rows[0]["model_id"]
        run_id = rows[0].get("run_id", fpath.stem)
        run_index = extract_run_index(run_id, batch_id)

        # Approximate batch timing from file modification times
        mtime = fpath.stat().st_mtime
        earliest_ts = min(earliest_ts, mtime - sum(r.get("latency_ms", 0) for r in rows) / 1000)
        latest_ts = max(latest_ts, mtime)

        model_runs[model_id].append((run_index, fpath, rows, run_id))

    # Sort runs within each model by run_index
    for model_id in model_runs:
        model_runs[model_id].sort(key=lambda x: x[0])

    # Build per-run summaries
    all_run_summaries = []
    model_summaries = []

    for model_id in sorted(model_runs.keys()):
        memory_gb = model_sizes.get(model_id, 0.0)
        alias = slugify(model_id)
        run_sums = []

        for run_index, fpath, rows, run_id in model_runs[model_id]:
            rs = summarize_run(
                rows, run_id, run_index,
                str(fpath.resolve()), model_id, alias, memory_gb,
            )
            run_sums.append(rs)
            all_run_summaries.append(rs)

        model_summaries.append(summarize_model(model_id, run_sums, memory_gb))

    # Sort models by mean_correct descending (matches existing convention)
    model_summaries.sort(key=lambda m: -m["mean_correct"])

    summary = {
        "batch_id": batch_id,
        "started_at_epoch": earliest_ts if earliest_ts != float("inf") else time.time(),
        "finished_at_epoch": latest_ts if latest_ts > 0 else time.time(),
        "models": model_summaries,
        "runs": all_run_summaries,
    }

    # Write to disk
    out_path = responses_dir / f"repeat_summary_{batch_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote repeat summary: {out_path}")
    print(f"  {len(model_summaries)} model(s), {len(all_run_summaries)} run(s)")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate repeat_summary_{batch_id}.json from raw JSONL response files"
    )
    parser.add_argument("--batch-id", required=True, help="Batch identifier")
    parser.add_argument("--responses-dir", type=Path, default=None,
                        help="Responses directory (default: benchmark_responses/)")
    args = parser.parse_args()

    generate_repeat_summary(args.batch_id, args.responses_dir)


if __name__ == "__main__":
    main()
