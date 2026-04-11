#!/usr/bin/env python3
"""
Deterministic verification gate for canonical+shadow publication batches.

Validates that a publication batch contains exactly one canonical run and one
shadow run per model, then compares parsed answers item-by-item. This is the
guardrail that allows a single canonical run to be publishable without letting
an ordinary 1-run hobby batch count as a publication candidate.

Usage:
  python scripts/verification_gate.py --batch-id ntp3-vr1-20260411
  python scripts/verification_gate.py --batch-id ntp3-vr1-20260411 \
    --screening-batch-id ntp3-screen-r3-20260411 \
    --expected-models "gemma4:e2b,qwen3:8b" \
    --json-out benchmark_responses/verification_report_ntp3-vr1-20260411.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"
DEFAULT_PROTOCOL = "canonical_shadow_v1"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_output_path(run: dict, responses_dir: Path) -> Path:
    fpath = Path(run["output_file"])
    if fpath.exists():
        return fpath
    alt = responses_dir / fpath.name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Run output file not found: {run['output_file']}")


def item_key(row: dict) -> tuple:
    return (
        row.get("exam_id"),
        row.get("year_buddhist"),
        row.get("subject"),
        row.get("question_id"),
        row.get("eval_split"),
        row.get("prompt_version"),
    )


def compare_runs(canonical_rows: list[dict], shadow_rows: list[dict]) -> tuple[int, list[dict]]:
    canon_map = {item_key(r): r for r in canonical_rows}
    shadow_map = {item_key(r): r for r in shadow_rows}
    all_keys = sorted(set(canon_map) | set(shadow_map), key=str)

    divergent_items: list[dict] = []
    matches = 0
    for key in all_keys:
        canon = canon_map.get(key)
        shadow = shadow_map.get(key)

        if canon is None:
            divergent_items.append({
                "item_key": list(key),
                "reason": "missing_in_canonical",
            })
            continue
        if shadow is None:
            divergent_items.append({
                "item_key": list(key),
                "reason": "missing_in_shadow",
            })
            continue

        canon_answer = canon.get("parsed_answer")
        shadow_answer = shadow.get("parsed_answer")
        canon_parseable = canon.get("is_parseable")
        shadow_parseable = shadow.get("is_parseable")

        if canon_answer == shadow_answer and canon_parseable == shadow_parseable:
            matches += 1
            continue

        divergent_items.append({
            "item_key": list(key),
            "reason": "parsed_answer_mismatch",
            "canonical": {
                "parsed_answer": canon_answer,
                "is_parseable": canon_parseable,
            },
            "shadow": {
                "parsed_answer": shadow_answer,
                "is_parseable": shadow_parseable,
            },
        })

    return matches, divergent_items


def verify_publication_batch(
    batch_id: str,
    responses_dir: Path | None = None,
    screening_batch_id: str | None = None,
    expected_models: list[str] | None = None,
    expected_runs: int = 2,
    canonical_run_index: int = 1,
    shadow_run_index: int = 2,
    protocol: str = DEFAULT_PROTOCOL,
) -> dict:
    responses_dir = responses_dir or DEFAULT_RESPONSES_DIR
    summary_path = responses_dir / f"repeat_summary_{batch_id}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"repeat_summary not found: {summary_path}")

    summary = load_json(summary_path)
    runs_by_model: dict[str, list[dict]] = {}
    for model in summary.get("models", []):
        runs = [r for r in summary.get("runs", []) if r.get("model_id") == model["model_id"]]
        runs.sort(key=lambda r: r.get("run_index", 0))
        runs_by_model[model["model_id"]] = runs

    summary_models = sorted(runs_by_model.keys())
    failures: list[str] = []

    if expected_models is not None:
        expected_sorted = sorted(expected_models)
        if expected_sorted != summary_models:
            failures.append(
                f"model set mismatch: expected {expected_sorted}, got {summary_models}"
            )

    model_reports: list[dict] = []
    all_deterministic = True

    for model_id in summary_models:
        runs = runs_by_model[model_id]
        reasons: list[str] = []

        if len(runs) != expected_runs:
            reasons.append(f"expected exactly {expected_runs} runs, got {len(runs)}")

        canonical = next((r for r in runs if r.get("run_index") == canonical_run_index), None)
        shadow = next((r for r in runs if r.get("run_index") == shadow_run_index), None)

        if canonical is None:
            reasons.append(f"missing canonical run_index={canonical_run_index}")
        if shadow is None:
            reasons.append(f"missing shadow run_index={shadow_run_index}")

        matching_items = 0
        total_items = 0
        divergent_items: list[dict] = []

        if not reasons and canonical and shadow:
            canonical_rows = load_jsonl(resolve_output_path(canonical, responses_dir))
            shadow_rows = load_jsonl(resolve_output_path(shadow, responses_dir))
            total_items = max(len(canonical_rows), len(shadow_rows))
            matching_items, divergent_items = compare_runs(canonical_rows, shadow_rows)
            if divergent_items:
                reasons.append(f"{len(divergent_items)} divergent item(s)")

        deterministic = not reasons
        if not deterministic:
            all_deterministic = False

        model_reports.append({
            "model_id": model_id,
            "canonical_run_id": canonical.get("run_id") if canonical else None,
            "shadow_run_id": shadow.get("run_id") if shadow else None,
            "total_items": total_items,
            "matching_items": matching_items,
            "divergent_count": len(divergent_items),
            "divergent_items": divergent_items,
            "deterministic": deterministic,
            "failure_reasons": reasons or None,
        })

    status = "pass" if not failures and all_deterministic else "fail"
    return {
        "batch_id": batch_id,
        "screening_batch_id": screening_batch_id,
        "protocol": protocol,
        "expected_runs_per_model": expected_runs,
        "canonical_run_index": canonical_run_index,
        "shadow_run_index": shadow_run_index,
        "expected_models": sorted(expected_models) if expected_models is not None else None,
        "summary_models": summary_models,
        "status": status,
        "all_deterministic": all_deterministic,
        "batch_failures": failures or None,
        "models": model_reports,
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic canonical+shadow verification gate")
    parser.add_argument("--batch-id", required=True, help="Verified publication batch ID")
    parser.add_argument("--responses-dir", type=Path, default=None,
                        help="Responses directory (default: benchmark_responses/)")
    parser.add_argument("--screening-batch-id", default=None,
                        help="Source screening batch ID")
    parser.add_argument("--expected-models", default=None,
                        help="Comma-separated expected model IDs")
    parser.add_argument("--expected-runs", type=int, default=2,
                        help="Expected runs per model (default: 2)")
    parser.add_argument("--canonical-run-index", type=int, default=1,
                        help="Canonical run index (default: 1)")
    parser.add_argument("--shadow-run-index", type=int, default=2,
                        help="Shadow run index (default: 2)")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Write verification report JSON")
    args = parser.parse_args()

    expected_models = None
    if args.expected_models:
        expected_models = [m.strip() for m in args.expected_models.split(",") if m.strip()]

    result = verify_publication_batch(
        batch_id=args.batch_id,
        responses_dir=args.responses_dir,
        screening_batch_id=args.screening_batch_id,
        expected_models=expected_models,
        expected_runs=args.expected_runs,
        canonical_run_index=args.canonical_run_index,
        shadow_run_index=args.shadow_run_index,
    )

    print(f"Verification gate: {args.batch_id}")
    print(f"  Status: {result['status']}")
    print(f"  Models: {len(result['models'])}")
    print(f"  All deterministic: {result['all_deterministic']}")
    if result.get("batch_failures"):
        for failure in result["batch_failures"]:
            print(f"  Batch failure: {failure}")

    for model in result["models"]:
        verdict = "ok" if model["deterministic"] else "fail"
        print(
            f"  - {model['model_id']}: {verdict} "
            f"({model['matching_items']}/{model['total_items']} items matched)"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"  Report written: {args.json_out}")

    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
