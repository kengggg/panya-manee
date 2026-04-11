#!/usr/bin/env python3
"""
Deterministic promotion-to-verified-publication wrapper.

Reads screening gate output, validates the promotion decision, and emits
the exact inputs needed to dispatch the publication batch. No guesswork,
no manual copy-paste.

Outputs:
  1. Promoted model CSV (for direct use as --models in benchmark-run)
  2. Publication batch metadata JSON (batch_id, models, runs, contract)
  3. Optional shell-sourceable env file for workflow dispatch

The verified publication batch_id is derived deterministically:
  screening batch_id: ntp3-screen-r3-YYYYMMDD
  publication batch_id: ntp3-vr1-YYYYMMDD (same date suffix)

Usage:
  python scripts/prepare_publication_batch.py --screening-batch-id ntp3-screen-r3-20260411
  python scripts/prepare_publication_batch.py --screening-batch-id ntp3-screen-r3-20260411 --pub-runs 2 --env-out pub.env
  python scripts/prepare_publication_batch.py --screening-json screening_decision.json --pub-batch-id ntp3-vr1-20260412
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from screening_gate import apply_screening_gate
from verify_batch_inputs import load_compatibility

DEFAULT_RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"
DEFAULT_PUB_RUNS = 2
DEFAULT_VERIFICATION_PROTOCOL = "canonical_shadow_v1"


def derive_pub_batch_id(screening_batch_id: str, pub_runs: int) -> str:
    """Derive publication batch_id from screening batch_id.

    ntp3-screen-r3-20260411 -> ntp3-vr1-20260411  (for the canonical+shadow path)
    Falls back to ntp3-pub-r{N}-{screening_batch_id} if pattern doesn't match.
    """
    m = re.match(r"^(ntp3)-screen-r\d+-([\d]+)$", screening_batch_id)
    if m:
        if pub_runs == DEFAULT_PUB_RUNS:
            return f"{m.group(1)}-vr1-{m.group(2)}"
        return f"{m.group(1)}-pub-r{pub_runs}-{m.group(2)}"
    # Fallback: append to screening batch_id
    return f"ntp3-pub-r{pub_runs}-{screening_batch_id}"


def prepare_publication_batch(
    screening_batch_id: str | None = None,
    screening_json_path: Path | None = None,
    responses_dir: Path | None = None,
    preflight_path: Path | None = None,
    pub_batch_id: str | None = None,
    pub_runs: int = DEFAULT_PUB_RUNS,
) -> dict:
    """Prepare publication batch from screening results.

    Either screening_batch_id or screening_json_path must be provided.
    If screening_json_path is given, reads the pre-computed decision.
    Otherwise, runs the screening gate fresh.

    Returns a dict with:
      - pub_batch_id: publication batch identifier
      - screening_batch_id: source screening batch
      - promoted_models: sorted list of promoted model IDs
      - promoted_csv: comma-separated string
      - pub_runs: runs per model for verified publication
      - contract: V1 compatibility values
      - status: "ready" or "no_models_promoted"
    """
    # Get screening decision
    if screening_json_path and screening_json_path.exists():
        with open(screening_json_path, encoding="utf-8") as f:
            screening = json.load(f)
        screening_batch_id = screening.get("batch_id", screening_batch_id or "unknown")
    elif screening_batch_id:
        screening = apply_screening_gate(
            screening_batch_id,
            responses_dir=responses_dir or DEFAULT_RESPONSES_DIR,
            preflight_path=preflight_path,
        )
    else:
        print("ERROR: Provide --screening-batch-id or --screening-json", file=sys.stderr)
        sys.exit(1)

    promoted = screening.get("promoted", [])

    # Derive publication batch_id
    if pub_batch_id is None:
        pub_batch_id = derive_pub_batch_id(screening_batch_id, pub_runs)

    # Load contract for metadata
    try:
        compat = load_compatibility()
        contract = compat.get("v1_expected_values", {})
    except (FileNotFoundError, json.JSONDecodeError):
        contract = {}

    promoted_csv = ",".join(promoted)
    status = "ready" if promoted else "no_models_promoted"

    result = {
        "pub_batch_id": pub_batch_id,
        "screening_batch_id": screening_batch_id,
        "promoted_models": promoted,
        "promoted_csv": promoted_csv,
        "promoted_count": len(promoted),
        "pub_runs": pub_runs,
        "total_expected_runs": len(promoted) * pub_runs,
        "verification_protocol": DEFAULT_VERIFICATION_PROTOCOL,
        "canonical_run_index": 1,
        "shadow_run_index": 2,
        "contract": contract,
        "status": status,
    }

    # Print summary
    print(f"Publication batch preparation")
    print(f"  Screening batch: {screening_batch_id}")
    print(f"  Publication batch: {pub_batch_id}")
    print(f"  Promoted models: {len(promoted)}")
    print(f"  Runs per model: {pub_runs}")
    print(f"  Total runs: {len(promoted) * pub_runs}")
    print(f"  Status: {status}")
    if promoted:
        print(f"\n  Models CSV:\n  {promoted_csv}")
    else:
        print("\n  WARNING: No models promoted — publication batch cannot proceed")

    return result


def write_env_file(result: dict, path: Path) -> None:
    """Write shell-sourceable env file for workflow dispatch."""
    lines = [
        f'PUB_BATCH_ID="{result["pub_batch_id"]}"',
        f'PROMOTED_MODELS="{result["promoted_csv"]}"',
        f'PUB_RUNS="{result["pub_runs"]}"',
        f'SCREENING_BATCH_ID="{result["screening_batch_id"]}"',
        f'VERIFICATION_PROTOCOL="{result["verification_protocol"]}"',
        f'CANONICAL_RUN_INDEX="{result["canonical_run_index"]}"',
        f'SHADOW_RUN_INDEX="{result["shadow_run_index"]}"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare publication batch from screening results"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--screening-batch-id",
                       help="Screening batch ID (runs screening gate fresh)")
    group.add_argument("--screening-json", type=Path,
                       help="Pre-computed screening decision JSON file")

    parser.add_argument("--responses-dir", type=Path, default=None,
                        help="Responses directory (default: benchmark_responses/)")
    parser.add_argument("--preflight", type=Path, default=None,
                        help="Preflight record JSON for digest verification")
    parser.add_argument("--pub-batch-id", default=None,
                        help="Override publication batch ID")
    parser.add_argument("--pub-runs", type=int, default=DEFAULT_PUB_RUNS,
                        help=f"Runs per model for publication (default: {DEFAULT_PUB_RUNS})")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Write publication batch metadata JSON")
    parser.add_argument("--csv-out", type=Path, default=None,
                        help="Write promoted model CSV to file")
    parser.add_argument("--env-out", type=Path, default=None,
                        help="Write shell-sourceable env file")
    args = parser.parse_args()

    result = prepare_publication_batch(
        screening_batch_id=args.screening_batch_id,
        screening_json_path=args.screening_json,
        responses_dir=args.responses_dir,
        preflight_path=args.preflight,
        pub_batch_id=args.pub_batch_id,
        pub_runs=args.pub_runs,
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\n  Metadata written: {args.json_out}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        args.csv_out.write_text(result["promoted_csv"] + "\n", encoding="utf-8")
        print(f"  CSV written: {args.csv_out}")

    if args.env_out:
        write_env_file(result, args.env_out)
        print(f"  Env file written: {args.env_out}")

    sys.exit(0 if result["status"] == "ready" else 1)


if __name__ == "__main__":
    main()
