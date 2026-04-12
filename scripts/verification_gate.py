#!/usr/bin/env python3
"""
Deterministic verification gate for canonical+shadow publication batches.

Runs the publication decision inside the verified batch itself:
  1. require exactly one canonical run and one shadow run per model
  2. compare parsed answers item-by-item for determinism
  3. apply post-hoc publication thresholds to the canonical run
  4. optionally verify current model digests against preflight

This replaces the old screening -> publication contract with a single verified
batch that filters losers after deterministic verification.

Usage:
  python scripts/verification_gate.py --batch-id ntp3-vr1-20260411
  python scripts/verification_gate.py --batch-id ntp3-vr1-20260411 \
    --expected-models "gemma4:e2b,qwen3:8b" \
    --preflight preflight_records/preflight_20260411.json \
    --json-out benchmark_responses/verification_report_ntp3-vr1-20260411.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from screening_gate import (
    MIN_MEAN_ACCURACY,
    MIN_PARSE_RATE,
    list_current_ollama_digests,
    resolve_digest_match,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"
DEFAULT_PROTOCOL = "canonical_shadow_v1"
DEFAULT_PUBLICATION_MODE = "verified_posthoc_gate_v1"


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


def load_preflight_digests(preflight_path: Path | None) -> dict[str, str]:
    if preflight_path is None:
        return {}
    with open(preflight_path, encoding="utf-8") as f:
        preflight = json.load(f)
    digests: dict[str, str] = {}
    for entry in preflight.get("model_inventory", []):
        model_id = entry.get("model_id")
        digest = entry.get("digest")
        if model_id and digest:
            digests[model_id] = digest
    return digests


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
    preflight_path: Path | None = None,
    expected_models: list[str] | None = None,
    expected_runs: int = 2,
    canonical_run_index: int = 1,
    shadow_run_index: int = 2,
    protocol: str = DEFAULT_PROTOCOL,
    min_parse_rate: float = MIN_PARSE_RATE,
    min_accuracy: float = MIN_MEAN_ACCURACY,
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
    preflight_digests = load_preflight_digests(preflight_path)
    current_digests = list_current_ollama_digests() if preflight_digests else {}

    if expected_models is not None:
        expected_sorted = sorted(expected_models)
        if expected_sorted != summary_models:
            failures.append(
                f"model set mismatch: expected {expected_sorted}, got {summary_models}"
            )

    model_reports: list[dict] = []
    all_deterministic = True
    publishable_models: list[str] = []

    for model_id in summary_models:
        runs = runs_by_model[model_id]
        reasons: list[str] = []
        exclusion_reasons: list[str] = []

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
        canonical_accuracy = canonical.get("accuracy") if canonical else None
        canonical_parse_rate = canonical.get("parse_rate") if canonical else None
        digest_match = None

        if not reasons and canonical and shadow:
            canonical_rows = load_jsonl(resolve_output_path(canonical, responses_dir))
            shadow_rows = load_jsonl(resolve_output_path(shadow, responses_dir))
            total_items = max(len(canonical_rows), len(shadow_rows))
            matching_items, divergent_items = compare_runs(canonical_rows, shadow_rows)
            if divergent_items:
                reasons.append(f"{len(divergent_items)} divergent item(s)")

        if preflight_digests:
            expected_digest = preflight_digests.get(model_id)
            if expected_digest is None:
                exclusion_reasons.append("model not in preflight record")
                digest_match = False
            else:
                digest_match = resolve_digest_match(model_id, current_digests, expected_digest)
                if digest_match is False:
                    exclusion_reasons.append("current digest does not match preflight record")
                elif digest_match is None:
                    exclusion_reasons.append("digest not verified against current Ollama inventory")

        deterministic = not reasons
        if not deterministic:
            all_deterministic = False
            exclusion_reasons.extend(reasons)

        parse_threshold_pass = (
            canonical_parse_rate is not None and round(canonical_parse_rate, 6) >= min_parse_rate
        )
        if canonical_parse_rate is None:
            exclusion_reasons.append("missing canonical parse_rate")
        elif not parse_threshold_pass:
            exclusion_reasons.append(
                f"canonical parse_rate {canonical_parse_rate:.4f} < {min_parse_rate}"
            )

        accuracy_threshold_pass = (
            canonical_accuracy is not None and canonical_accuracy > min_accuracy
        )
        if canonical_accuracy is None:
            exclusion_reasons.append("missing canonical accuracy")
        elif not accuracy_threshold_pass:
            exclusion_reasons.append(
                f"canonical accuracy {canonical_accuracy:.4f} <= {min_accuracy}"
            )

        digest_threshold_pass = digest_match is not False
        publishable = deterministic and parse_threshold_pass and accuracy_threshold_pass and digest_threshold_pass
        if publishable:
            publishable_models.append(model_id)

        model_reports.append({
            "model_id": model_id,
            "canonical_run_id": canonical.get("run_id") if canonical else None,
            "shadow_run_id": shadow.get("run_id") if shadow else None,
            "canonical_accuracy": canonical_accuracy,
            "canonical_parse_rate": canonical_parse_rate,
            "total_items": total_items,
            "matching_items": matching_items,
            "divergent_count": len(divergent_items),
            "divergent_items": divergent_items,
            "deterministic": deterministic,
            "digest_checked": digest_match,
            "thresholds": {
                "parse_rate_pass": parse_threshold_pass,
                "accuracy_pass": accuracy_threshold_pass,
                "digest_pass": digest_threshold_pass,
            },
            "publishable": publishable,
            "exclusion_reasons": exclusion_reasons or None,
            "failure_reasons": reasons or None,
        })

    if not publishable_models:
        failures.append("no publishable models survived verified post-hoc gate")

    status = "pass" if not failures and publishable_models else "fail"
    return {
        "batch_id": batch_id,
        "publication_mode": DEFAULT_PUBLICATION_MODE,
        "preflight_record": str(preflight_path) if preflight_path else None,
        "protocol": protocol,
        "expected_runs_per_model": expected_runs,
        "canonical_run_index": canonical_run_index,
        "shadow_run_index": shadow_run_index,
        "thresholds": {
            "parse_rate_gte": min_parse_rate,
            "accuracy_gt": min_accuracy,
        },
        "expected_models": sorted(expected_models) if expected_models is not None else None,
        "summary_models": summary_models,
        "status": status,
        "all_deterministic": all_deterministic,
        "publishable_count": len(publishable_models),
        "publishable_models": sorted(publishable_models),
        "batch_failures": failures or None,
        "models": model_reports,
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic canonical+shadow verification gate")
    parser.add_argument("--batch-id", required=True, help="Verified publication batch ID")
    parser.add_argument("--responses-dir", type=Path, default=None,
                        help="Responses directory (default: benchmark_responses/)")
    parser.add_argument("--preflight", type=Path, default=None,
                        help="Preflight record JSON for digest verification")
    parser.add_argument("--expected-models", default=None,
                        help="Comma-separated expected model IDs")
    parser.add_argument("--expected-runs", type=int, default=2,
                        help="Expected runs per model (default: 2)")
    parser.add_argument("--canonical-run-index", type=int, default=1,
                        help="Canonical run index (default: 1)")
    parser.add_argument("--shadow-run-index", type=int, default=2,
                        help="Shadow run index (default: 2)")
    parser.add_argument("--min-parse-rate", type=float, default=MIN_PARSE_RATE,
                        help=f"Minimum canonical parse rate (default: {MIN_PARSE_RATE})")
    parser.add_argument("--min-accuracy", type=float, default=MIN_MEAN_ACCURACY,
                        help=f"Minimum canonical accuracy, strict greater-than (default: {MIN_MEAN_ACCURACY})")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Write verification report JSON")
    args = parser.parse_args()

    expected_models = None
    if args.expected_models:
        expected_models = [m.strip() for m in args.expected_models.split(",") if m.strip()]

    result = verify_publication_batch(
        batch_id=args.batch_id,
        responses_dir=args.responses_dir,
        preflight_path=args.preflight,
        expected_models=expected_models,
        expected_runs=args.expected_runs,
        canonical_run_index=args.canonical_run_index,
        shadow_run_index=args.shadow_run_index,
        min_parse_rate=args.min_parse_rate,
        min_accuracy=args.min_accuracy,
    )

    print(f"Verification gate: {args.batch_id}")
    print(f"  Status: {result['status']}")
    print(f"  Models: {len(result['models'])}")
    print(f"  All deterministic: {result['all_deterministic']}")
    print(f"  Publishable models: {result['publishable_count']}")
    if result.get("batch_failures"):
        for failure in result["batch_failures"]:
            print(f"  Batch failure: {failure}")

    for model in result["models"]:
        verdict = "publish" if model["publishable"] else "exclude"
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
