#!/usr/bin/env python3
"""
Check batch completeness after a benchmark run.

Verifies:
- Expected JSONL file count matches models × runs_per_model
- No truncated or malformed JSONL files
- Every JSONL has the expected item count (93 for text_only_core)
- repeat_summary exists and model/run counts are consistent

Usage:
  python scripts/check_batch_completeness.py --batch-id ntp3-screen-r3-20260411 --models "gemma4:e2b,gemma4:e4b" --runs-per-model 3
  python scripts/check_batch_completeness.py --batch-id mini-r10-20260409  # infer from repeat_summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"

# Expected item count per run for text_only_core (Thai 60 + Math 33)
EXPECTED_ITEMS_PER_RUN = 93

# Required fields in each JSONL row
REQUIRED_ROW_FIELDS = {
    "model_id", "run_id", "question_id", "is_correct", "is_parseable",
    "parsed_answer", "correct_answer", "raw_output", "subject",
}


def slugify(model_id: str) -> str:
    """Match run_batch.py slug convention."""
    return model_id.replace(":", "-").replace("/", "-")


def load_jsonl_safe(path: Path) -> tuple[list[dict], list[str]]:
    """Load JSONL, returning (rows, errors). Never raises on bad data."""
    rows: list[dict] = []
    errors: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    rows.append(row)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path.name}:{line_num}: malformed JSON: {exc}")
    except OSError as exc:
        errors.append(f"{path.name}: cannot read file: {exc}")
    return rows, errors


def check_batch_completeness(
    batch_id: str,
    models: list[str] | None = None,
    runs_per_model: int | None = None,
    responses_dir: Path | None = None,
    expected_items: int = EXPECTED_ITEMS_PER_RUN,
) -> dict:
    """Check batch completeness. Returns a result dict with status and details.

    If models/runs_per_model are not provided, they are inferred from
    the repeat_summary file.
    """
    responses_dir = responses_dir or DEFAULT_RESPONSES_DIR
    errors: list[str] = []
    warnings: list[str] = []
    file_details: list[dict] = []

    # Discover JSONL files for this batch
    pattern = f"responses_{batch_id}-*.jsonl"
    found_files = sorted(responses_dir.glob(pattern))

    # Load repeat_summary if available
    summary_path = responses_dir / f"repeat_summary_{batch_id}.json"
    summary: dict | None = None
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"Cannot parse repeat_summary: {exc}")

    # Infer models and runs_per_model from summary if not provided
    if summary and models is None:
        models = sorted({m["model_id"] for m in summary.get("models", [])})
    if summary and runs_per_model is None:
        run_counts = [m["runs"] for m in summary.get("models", [])]
        if run_counts and len(set(run_counts)) == 1:
            runs_per_model = run_counts[0]
        elif run_counts:
            warnings.append(f"Inconsistent run counts across models: {run_counts}")
            runs_per_model = max(run_counts)

    # Build a run_id → model_id map from summary if available, for reliable file matching
    summary_run_ids: dict[str, str] = {}
    if summary:
        for run in summary.get("runs", []):
            summary_run_ids[run["run_id"]] = run["model_id"]

    # Check expected file count
    if models and runs_per_model:
        expected_count = len(models) * runs_per_model
        if len(found_files) != expected_count:
            errors.append(
                f"Expected {expected_count} JSONL files "
                f"({len(models)} models × {runs_per_model} runs), "
                f"found {len(found_files)}"
            )

        # Check each model has exactly runs_per_model files
        if summary_run_ids:
            # Use summary run_ids for reliable matching (slug conventions may vary)
            for model_id in models:
                model_run_count = sum(
                    1 for rid, mid in summary_run_ids.items()
                    if mid == model_id
                )
                if model_run_count != runs_per_model:
                    errors.append(
                        f"{model_id}: expected {runs_per_model} runs in summary, "
                        f"found {model_run_count}"
                    )
        else:
            # Fallback: match by slug convention (run_batch.py style)
            for model_id in models:
                slug = slugify(model_id)
                model_files = [f for f in found_files if f"-{slug}-r" in f.name]
                if len(model_files) != runs_per_model:
                    errors.append(
                        f"{model_id}: expected {runs_per_model} files, "
                        f"found {len(model_files)}"
                    )

    if not found_files:
        errors.append(f"No JSONL files found matching {pattern}")

    # Validate each file
    for fpath in found_files:
        rows, parse_errors = load_jsonl_safe(fpath)
        detail: dict = {
            "file": fpath.name,
            "rows": len(rows),
            "parse_errors": len(parse_errors),
            "valid": True,
        }

        if parse_errors:
            detail["valid"] = False
            errors.extend(parse_errors)

        # Check row count
        if len(rows) != expected_items:
            detail["valid"] = False
            errors.append(
                f"{fpath.name}: expected {expected_items} rows, got {len(rows)}"
            )

        # Check required fields
        for idx, row in enumerate(rows):
            missing = REQUIRED_ROW_FIELDS - row.keys()
            if missing:
                detail["valid"] = False
                errors.append(f"{fpath.name}:row {idx}: missing fields: {missing}")
                break  # One per file is enough

        # Check consistent model_id and run_id within file
        if rows:
            model_ids = {r.get("model_id") for r in rows}
            run_ids = {r.get("run_id") for r in rows}
            if len(model_ids) > 1:
                detail["valid"] = False
                errors.append(f"{fpath.name}: mixed model_ids: {model_ids}")
            if len(run_ids) > 1:
                detail["valid"] = False
                errors.append(f"{fpath.name}: mixed run_ids: {run_ids}")

        file_details.append(detail)

    # Cross-check with repeat_summary
    if summary:
        summary_models = sorted(m["model_id"] for m in summary.get("models", []))
        summary_runs = len(summary.get("runs", []))

        # Model count check
        if models:
            file_models = sorted(set(models))
            if file_models != summary_models:
                errors.append(
                    f"Model mismatch: files have {file_models}, "
                    f"summary has {summary_models}"
                )

        # Run count check
        if len(found_files) != summary_runs:
            errors.append(
                f"Run count mismatch: {len(found_files)} JSONL files, "
                f"summary has {summary_runs} runs"
            )

        # Per-model run count in summary — use run_ids from summary for matching
        for m in summary.get("models", []):
            model_id = m["model_id"]
            expected_run_count = m["runs"]
            actual_run_count = sum(
                1 for run in summary.get("runs", [])
                if run["model_id"] == model_id
            )
            if actual_run_count != expected_run_count:
                errors.append(
                    f"{model_id}: summary model says {expected_run_count} runs, "
                    f"but summary runs list has {actual_run_count}"
                )
    else:
        warnings.append("No repeat_summary found; skipped cross-validation")

    status = "fail" if errors else "pass"
    result = {
        "batch_id": batch_id,
        "status": status,
        "total_files": len(found_files),
        "expected_items_per_run": expected_items,
        "file_details": file_details,
        "errors": errors,
        "warnings": warnings,
    }

    # Print report
    print(f"Batch completeness: {batch_id}")
    print(f"  Status: {status}")
    print(f"  Files: {len(found_files)}")
    valid_count = sum(1 for d in file_details if d["valid"])
    print(f"  Valid: {valid_count}/{len(file_details)}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    if warnings:
        print(f"  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Check batch completeness")
    parser.add_argument("--batch-id", required=True, help="Batch identifier")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model IDs (inferred from summary if omitted)")
    parser.add_argument("--runs-per-model", type=int, default=None,
                        help="Expected runs per model (inferred from summary if omitted)")
    parser.add_argument("--responses-dir", type=Path, default=None,
                        help="Responses directory (default: benchmark_responses/)")
    parser.add_argument("--expected-items", type=int, default=EXPECTED_ITEMS_PER_RUN,
                        help=f"Expected items per run (default: {EXPECTED_ITEMS_PER_RUN})")
    args = parser.parse_args()

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    result = check_batch_completeness(
        args.batch_id,
        models=models,
        runs_per_model=args.runs_per_model,
        responses_dir=args.responses_dir,
        expected_items=args.expected_items,
    )
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
