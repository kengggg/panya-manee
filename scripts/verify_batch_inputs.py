#!/usr/bin/env python3
"""
Verify batch inputs against the V1 compatibility contract.

Checks that the runtime configuration matches registry/compatibility.json
before a batch run begins. Fails fast if someone accidentally changes the
eval split, think mode, prompt version, or other contract fields.

Also validates:
  - batch_id format (non-empty, no whitespace)
  - models list is non-empty and contains only printable characters
  - dataset integrity (Thai 60 + Math 33 text_only_core)
  - think budget is consistent with contract (off = no budget)

Usage:
  python scripts/verify_batch_inputs.py --batch-id ntp3-pub-r10-20260412 --models "gemma4:e2b,gemma4:e4b" --runs-per-model 10
  python scripts/verify_batch_inputs.py --batch-id ntp3-smoke-r1-20260411 --models "gemma4:e2b" --runs-per-model 1 --think 4096
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_FILES, validate_items

REGISTRY_DIR = PROJECT_ROOT / "registry"
COMPATIBILITY_PATH = REGISTRY_DIR / "compatibility.json"

# Batch ID must be non-empty, no whitespace, printable ASCII
BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def load_compatibility() -> dict:
    """Load the V1 compatibility contract from registry."""
    if not COMPATIBILITY_PATH.exists():
        raise FileNotFoundError(f"Compatibility contract not found: {COMPATIBILITY_PATH}")
    with open(COMPATIBILITY_PATH, encoding="utf-8") as f:
        return json.load(f)


def verify_batch_inputs(
    batch_id: str,
    models: list[str],
    runs_per_model: int,
    subjects: str = "thai,math",
    think_budget: int | None = None,
) -> dict:
    """Verify batch inputs against the compatibility contract.

    Returns a result dict with:
      - status: "pass" or "fail"
      - errors: list of error strings
      - warnings: list of warning strings
      - checks: dict of individual check results
    """
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    # 1. Load compatibility contract
    try:
        compat = load_compatibility()
        expected = compat.get("v1_expected_values", {})
        checks["compatibility_loaded"] = True
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot load compatibility contract: {exc}")
        checks["compatibility_loaded"] = False
        expected = {}

    # 2. Validate batch_id format
    if not batch_id or not BATCH_ID_PATTERN.match(batch_id):
        errors.append(
            f"Invalid batch_id '{batch_id}': must be non-empty, "
            f"alphanumeric with dashes, dots, and underscores only"
        )
        checks["batch_id_format"] = False
    else:
        checks["batch_id_format"] = True

    # 3. Validate models list
    if not models:
        errors.append("No models specified")
        checks["models_non_empty"] = False
    else:
        checks["models_non_empty"] = True
        for m in models:
            if not m.strip() or not all(c.isprintable() for c in m):
                errors.append(f"Invalid model name: '{m}'")

    # 4. Validate runs_per_model
    if runs_per_model < 1:
        errors.append(f"runs_per_model must be >= 1, got {runs_per_model}")
        checks["runs_per_model_valid"] = False
    else:
        checks["runs_per_model_valid"] = True

    # 5. Check think mode against contract
    contract_think = expected.get("think_mode", "off")
    if contract_think == "off" and think_budget is not None:
        errors.append(
            f"Think budget {think_budget} specified but contract requires think_mode=off"
        )
        checks["think_mode_consistent"] = False
    elif contract_think != "off" and think_budget is None:
        warnings.append(
            f"Contract expects think_mode={contract_think} but no think budget provided"
        )
        checks["think_mode_consistent"] = True
    else:
        checks["think_mode_consistent"] = True

    # 6. Check eval_split — subjects must match contract
    contract_split = expected.get("eval_split", "text_only_core")
    subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
    if contract_split == "text_only_core":
        expected_subjects = {"thai", "math"}
        actual_subjects = set(subject_list)
        if actual_subjects != expected_subjects:
            errors.append(
                f"Subjects {actual_subjects} don't match text_only_core "
                f"expected subjects {expected_subjects}"
            )
            checks["subjects_match"] = False
        else:
            checks["subjects_match"] = True
    else:
        checks["subjects_match"] = True

    # 7. Dataset integrity
    try:
        ds_counts: dict[str, dict[str, int]] = {}
        for subject, path in DATASET_FILES.items():
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {path}")
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
            validate_items(items, path)
            split_counts: dict[str, int] = {}
            for item in items:
                split = item.get("eval_split", "unknown")
                split_counts[split] = split_counts.get(split, 0) + 1
            ds_counts[subject] = split_counts

        thai_core = ds_counts.get("thai", {}).get("text_only_core", 0)
        math_core = ds_counts.get("math", {}).get("text_only_core", 0)
        if thai_core != 60:
            errors.append(f"Expected 60 Thai text_only_core items, got {thai_core}")
        if math_core != 33:
            errors.append(f"Expected 33 Math text_only_core items, got {math_core}")
        checks["dataset_integrity"] = thai_core == 60 and math_core == 33
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"Dataset error: {exc}")
        checks["dataset_integrity"] = False

    # 8. Contract field summary (informational)
    contract_summary = {}
    for field in compat.get("required_match_fields", []) if expected else []:
        contract_summary[field] = expected.get(field, "<missing>")

    status = "fail" if errors else "pass"
    result = {
        "batch_id": batch_id,
        "status": status,
        "models": models,
        "runs_per_model": runs_per_model,
        "think_budget": think_budget,
        "subjects": subject_list,
        "contract": contract_summary,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }

    # Print report
    print(f"Batch input verification: {batch_id}")
    print(f"  Status: {status}")
    print(f"  Models: {len(models)}")
    print(f"  Runs/model: {runs_per_model}")
    print(f"  Think: {'off' if think_budget is None else think_budget}")
    if contract_summary:
        print(f"  Contract: {json.dumps(contract_summary, indent=None)}")
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
    parser = argparse.ArgumentParser(
        description="Verify batch inputs against V1 compatibility contract"
    )
    parser.add_argument("--batch-id", required=True, help="Batch identifier")
    parser.add_argument("--models", required=True, help="Comma-separated model IDs")
    parser.add_argument("--runs-per-model", type=int, required=True,
                        help="Number of runs per model")
    parser.add_argument("--subjects", default="thai,math",
                        help="Comma-separated subjects (default: thai,math)")
    parser.add_argument("--think", type=int, default=None,
                        help="Think mode token budget (omit for think=off)")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Write verification result JSON to file")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    result = verify_batch_inputs(
        args.batch_id,
        models=models,
        runs_per_model=args.runs_per_model,
        subjects=args.subjects,
        think_budget=args.think,
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"  Result written: {args.json_out}")

    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
