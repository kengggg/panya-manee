#!/usr/bin/env python3
"""
Deterministic screening gate for model promotion.

Reads a repeat_summary from a screening batch, applies hard promotion rules,
and outputs:
  1. A promoted model CSV (stdout or file) for direct use in publication batch
  2. A screening appendix JSON with per-model verdicts

Promotion rules (from full_benchmark_execution_checklist.md):
  - parse_rate >= 95%
  - mean accuracy > 25%
  - no crash / OOM / hard failure (proxied by: model has expected run count)
  - model digest matches preflight record (optional, checked if preflight provided)

Usage:
  python scripts/screening_gate.py --batch-id ntp3-screen-r3-20260411
  python scripts/screening_gate.py --batch-id ntp3-screen-r3-20260411 --preflight preflight_records/preflight_20260411.json
  python scripts/screening_gate.py --batch-id ntp3-screen-r3-20260411 --csv-out promoted.csv --appendix-out screening_appendix.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# NOTE: verification_gate.py imports the digest helpers and thresholds below.
# Keep those shared utilities stable until they are moved into a dedicated
# common module.

# Hard promotion thresholds
MIN_PARSE_RATE = 0.95
MIN_MEAN_ACCURACY = 0.25

# Model tier classification for the appendix.
# Ordered longest-match-first to avoid prefix collisions (e.g. qwen3 vs qwen3.5).
TIER_RULES: list[tuple[str, str]] = [
    # Thai-specific (check before international — typhoon contains qwen/llama substrings)
    ("typhoon2.1", "thai-specific"),
    ("typhoon2.5", "thai-specific"),
    ("llama3.1-typhoon2", "thai-specific"),
    # Core international
    ("gemma4", "core-international"),
    ("qwen3.5", "core-international"),
    ("qwen2.5", "core-international"),
    ("qwen3", "core-international"),
    ("llama3.1", "core-international"),
    ("mistral", "core-international"),
    ("phi4-mini", "core-international"),
    # Extended international
    ("phi3", "extended-international"),
    ("falcon3", "extended-international"),
    ("olmo2", "extended-international"),
    ("yi", "extended-international"),
    ("granite3.3", "extended-international"),
]


def classify_tier(model_id: str) -> str:
    """Classify a model into a tier based on its family/name."""
    name = model_id.lower()
    # Strip provider prefix
    if "/" in name:
        name = name.split("/", 1)[1]
    for key, tier in TIER_RULES:
        if key.lower() in name:
            return tier
    return "unknown"


def compute_mean_parse_rate(runs: list[dict], model_id: str) -> float:
    """Compute mean parse_rate across runs for a given model."""
    model_runs = [r for r in runs if r["model_id"] == model_id]
    if not model_runs:
        return 0.0
    return sum(r.get("parse_rate", 0.0) for r in model_runs) / len(model_runs)


def list_current_ollama_digests() -> dict[str, str]:
    """Return current Ollama model digests keyed by model name.

    Returns an empty dict if Ollama is unreachable.
    """
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return {}

    result: dict[str, str] = {}
    for model in data.get("models", []):
        name = model.get("name")
        digest = model.get("digest")
        if name and digest:
            result[name] = digest
    return result


def resolve_digest_match(model_id: str, current_digests: dict[str, str], expected_digest: str) -> bool | None:
    """Compare a model's current digest with the preflight digest.

    Returns True/False when a comparison is possible, otherwise None.
    """
    if not current_digests:
        return None

    candidates = [model_id]
    if not model_id.endswith(":latest"):
        candidates.append(f"{model_id}:latest")
    if model_id.endswith(":latest"):
        candidates.append(model_id[:-7])

    for candidate in candidates:
        current = current_digests.get(candidate)
        if current is not None:
            return current == expected_digest
    return False


def apply_screening_gate(
    batch_id: str,
    responses_dir: Path | None = None,
    preflight_path: Path | None = None,
    expected_runs: int | None = None,
) -> dict:
    """Apply promotion rules to a screening batch.

    Returns a dict with:
      - promoted: list of promoted model_ids (sorted)
      - appendix: list of per-model verdict dicts
      - csv: promoted model CSV string
    """
    responses_dir = responses_dir or DEFAULT_RESPONSES_DIR
    summary_path = responses_dir / f"repeat_summary_{batch_id}.json"

    if not summary_path.exists():
        print(f"ERROR: repeat_summary not found: {summary_path}", file=sys.stderr)
        sys.exit(1)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    # Load preflight digests if provided
    preflight_digests: dict[str, str] = {}
    if preflight_path and preflight_path.exists():
        with open(preflight_path, encoding="utf-8") as f:
            preflight = json.load(f)
        for entry in preflight.get("model_inventory", []):
            if entry.get("digest"):
                preflight_digests[entry["model_id"]] = entry["digest"]

    current_digests = list_current_ollama_digests() if preflight_digests else {}

    # Determine expected runs per model
    if expected_runs is None:
        run_counts = [m["runs"] for m in summary.get("models", [])]
        if run_counts:
            expected_runs = max(run_counts)
        else:
            expected_runs = 0

    runs = summary.get("runs", [])
    appendix: list[dict] = []
    promoted: list[str] = []

    for model in summary.get("models", []):
        model_id = model["model_id"]
        mean_accuracy = model.get("mean_accuracy", 0.0)
        mean_parse_rate = compute_mean_parse_rate(runs, model_id)
        num_runs = model.get("runs", 0)
        tier = classify_tier(model_id)

        reasons: list[str] = []
        status = "promoted"

        # Gate: parse rate (round to 6 decimal places to avoid float artefacts)
        if round(mean_parse_rate, 6) < MIN_PARSE_RATE:
            status = "not promoted"
            reasons.append(
                f"parse_rate {mean_parse_rate:.4f} < {MIN_PARSE_RATE}"
            )

        # Gate: mean accuracy
        if mean_accuracy <= MIN_MEAN_ACCURACY:
            status = "not promoted"
            reasons.append(
                f"mean_accuracy {mean_accuracy:.4f} <= {MIN_MEAN_ACCURACY}"
            )

        # Gate: complete runs (no crash/OOM)
        if num_runs < expected_runs:
            status = "not promoted"
            reasons.append(
                f"incomplete runs: {num_runs}/{expected_runs}"
            )

        # Gate: digest match (if preflight provided)
        digest_match = None
        if preflight_digests:
            expected_digest = preflight_digests.get(model_id)
            if expected_digest is None:
                status = "not promoted"
                reasons.append("model not in preflight record")
                digest_match = False
            else:
                digest_match = resolve_digest_match(model_id, current_digests, expected_digest)
                if digest_match is False:
                    status = "not promoted"
                    reasons.append("current digest does not match preflight record")
                elif digest_match is None:
                    reasons.append("digest not verified against current Ollama inventory")

        verdict = {
            "model_id": model_id,
            "tier": tier,
            "runs": num_runs,
            "mean_accuracy": round(mean_accuracy, 6),
            "mean_parse_rate": round(mean_parse_rate, 6),
            "mean_thai_accuracy": round(model.get("mean_thai_accuracy", 0.0), 6),
            "mean_math_accuracy": round(model.get("mean_math_accuracy", 0.0), 6),
            "status": status,
            "exclusion_reasons": reasons if reasons else None,
            "digest_checked": digest_match,
        }
        appendix.append(verdict)

        if status == "promoted":
            promoted.append(model_id)

    # Sort promoted list for determinism
    promoted.sort()

    # Build CSV string
    csv_buf = io.StringIO()
    csv_buf.write(",".join(promoted))
    csv_string = csv_buf.getvalue()

    result = {
        "batch_id": batch_id,
        "expected_runs_per_model": expected_runs,
        "total_models": len(summary.get("models", [])),
        "promoted_count": len(promoted),
        "promoted": promoted,
        "appendix": appendix,
        "csv": csv_string,
    }

    return result


def format_appendix_markdown(appendix: list[dict]) -> str:
    """Format the screening appendix as markdown table."""
    lines = [
        "| Model | Tier | Runs | Mean Accuracy | Parse Rate | Status | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in appendix:
        reason = "; ".join(v["exclusion_reasons"]) if v["exclusion_reasons"] else "-"
        lines.append(
            f"| {v['model_id']} | {v['tier']} | {v['runs']} | "
            f"{v['mean_accuracy']:.4f} | {v['mean_parse_rate']:.4f} | "
            f"{v['status']} | {reason} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Deterministic screening gate")
    parser.add_argument("--batch-id", required=True, help="Screening batch identifier")
    parser.add_argument("--responses-dir", type=Path, default=None,
                        help="Responses directory (default: benchmark_responses/)")
    parser.add_argument("--preflight", type=Path, default=None,
                        help="Preflight record JSON for digest verification")
    parser.add_argument("--expected-runs", type=int, default=None,
                        help="Expected runs per model (inferred from summary if omitted)")
    parser.add_argument("--csv-out", type=Path, default=None,
                        help="Write promoted model CSV to file")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Write full screening decision JSON to file")
    parser.add_argument("--appendix-out", type=Path, default=None,
                        help="Write screening appendix JSON array to file")
    parser.add_argument("--markdown", action="store_true",
                        help="Print appendix as markdown table")
    args = parser.parse_args()

    result = apply_screening_gate(
        args.batch_id,
        responses_dir=args.responses_dir,
        preflight_path=args.preflight,
        expected_runs=args.expected_runs,
    )

    # Print summary
    print(f"Screening gate: {args.batch_id}")
    print(f"  Models: {result['total_models']}")
    print(f"  Promoted: {result['promoted_count']}")
    print(f"  Promoted list: {result['csv']}")

    # Write CSV
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        args.csv_out.write_text(result["csv"] + "\n", encoding="utf-8")
        print(f"  CSV written: {args.csv_out}")
    else:
        print(f"\nPromoted models (CSV):\n{result['csv']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"  Decision JSON written: {args.json_out}")

    # Write appendix
    if args.appendix_out:
        args.appendix_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.appendix_out, "w", encoding="utf-8") as f:
            json.dump(result["appendix"], f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"  Appendix written: {args.appendix_out}")

    # Markdown table
    if args.markdown:
        print(f"\n{format_appendix_markdown(result['appendix'])}")

    sys.exit(0)


if __name__ == "__main__":
    main()
