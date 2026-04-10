#!/usr/bin/env python3
"""
Preflight check for benchmark execution.

Verifies Ollama health, dataset integrity, and model availability.
Records model inventory with digests to a machine-readable JSON artifact.

Usage:
  python scripts/preflight.py --models "gemma4:e2b,gemma4:e4b"
  python scripts/preflight.py --models "gemma4:e2b" --pull
  python scripts/preflight.py --models "gemma4:e2b" --offline   # skip Ollama checks
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, DATASET_FILES, validate_items

OLLAMA_BASE = "http://localhost:11434"
PREFLIGHT_DIR = PROJECT_ROOT / "preflight_records"


def check_ollama_health() -> bool:
    """Return True if Ollama API is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def list_ollama_models() -> dict[str, dict]:
    """Return {name: {digest, size, ...}} for all locally available models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return {}
    result = {}
    for m in data.get("models", []):
        result[m["name"]] = {
            "digest": m.get("digest", ""),
            "size": m.get("size", 0),
            "modified_at": m.get("modified_at", ""),
        }
    return result


def pull_model(model_id: str) -> bool:
    """Pull a model via Ollama API. Returns True on success."""
    try:
        payload = json.dumps({"name": model_id, "stream": False}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def verify_datasets() -> dict:
    """Validate datasets and return item counts per subject and eval_split."""
    counts: dict[str, dict[str, int]] = {}
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
        counts[subject] = split_counts
    return counts


def resolve_model_name(requested: str, available: dict[str, dict]) -> str | None:
    """Find the best match for a requested model in available models.

    Ollama normalises tags — 'gemma4:e2b' may appear as 'gemma4:e2b' or
    with ':latest' appended.  Return the exact available name or None.
    """
    if requested in available:
        return requested
    # Try with :latest suffix
    if not requested.endswith(":latest") and f"{requested}:latest" in available:
        return f"{requested}:latest"
    # Try stripping :latest
    if requested.endswith(":latest") and requested[:-7] in available:
        return requested[:-7]
    return None


def run_preflight(
    models: list[str],
    *,
    do_pull: bool = False,
    offline: bool = False,
    output_path: Path | None = None,
) -> dict:
    """Run full preflight check. Returns the preflight record dict.

    Raises SystemExit on hard failures.
    """
    errors: list[str] = []
    warnings: list[str] = []
    record: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_models": models,
        "ollama_healthy": None,
        "dataset_counts": {},
        "model_inventory": [],
        "errors": errors,
        "warnings": warnings,
        "status": "pending",
    }

    # 1. Dataset verification (always runs, even offline)
    try:
        record["dataset_counts"] = verify_datasets()
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"Dataset error: {exc}")

    # Check text_only_core counts
    ds = record["dataset_counts"]
    thai_core = ds.get("thai", {}).get("text_only_core", 0)
    math_core = ds.get("math", {}).get("text_only_core", 0)
    if thai_core != 60:
        errors.append(f"Expected 60 Thai text_only_core items, got {thai_core}")
    if math_core != 33:
        errors.append(f"Expected 33 Math text_only_core items, got {math_core}")

    # 2. Ollama health
    if offline:
        record["ollama_healthy"] = None
        warnings.append("Offline mode: skipped Ollama checks")
    else:
        healthy = check_ollama_health()
        record["ollama_healthy"] = healthy
        if not healthy:
            errors.append("Ollama is not reachable at localhost:11434")

    # 3. Model availability and digests
    available = {} if offline else list_ollama_models()

    for model_id in models:
        entry: dict = {
            "model_id": model_id,
            "available": False,
            "pulled": False,
            "digest": None,
            "size": None,
        }

        if offline:
            entry["available"] = None
            warnings.append(f"Offline mode: skipped availability check for {model_id}")
        else:
            resolved = resolve_model_name(model_id, available)
            if resolved:
                entry["available"] = True
                entry["digest"] = available[resolved]["digest"]
                entry["size"] = available[resolved]["size"]
            elif do_pull:
                print(f"Pulling {model_id}...")
                if pull_model(model_id):
                    entry["pulled"] = True
                    # Refresh available models to get digest
                    available = list_ollama_models()
                    resolved = resolve_model_name(model_id, available)
                    if resolved:
                        entry["available"] = True
                        entry["digest"] = available[resolved]["digest"]
                        entry["size"] = available[resolved]["size"]
                    else:
                        errors.append(f"Pulled {model_id} but cannot find it in model list")
                else:
                    errors.append(f"Failed to pull {model_id}")
            else:
                errors.append(f"Model not available: {model_id} (use --pull to auto-pull)")

        record["model_inventory"].append(entry)

    # Final status
    record["status"] = "fail" if errors else "pass"

    # Write artifact
    if output_path is None:
        PREFLIGHT_DIR.mkdir(exist_ok=True)
        date_str = time.strftime("%Y%m%d")
        output_path = PREFLIGHT_DIR / f"preflight_{date_str}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Preflight record: {output_path}")
    print(f"  Status: {record['status']}")
    print(f"  Models: {len(record['model_inventory'])}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    if warnings:
        print(f"  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    return record


def main():
    parser = argparse.ArgumentParser(description="Preflight check for benchmark execution")
    parser.add_argument("--models", required=True, help="Comma-separated model IDs")
    parser.add_argument("--pull", action="store_true", help="Auto-pull missing models")
    parser.add_argument("--offline", action="store_true",
                        help="Skip Ollama checks (dataset validation only)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: preflight_records/preflight_YYYYMMDD.json)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERROR: No models specified", file=sys.stderr)
        sys.exit(1)

    record = run_preflight(models, do_pull=args.pull, offline=args.offline, output_path=args.output)
    sys.exit(0 if record["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
