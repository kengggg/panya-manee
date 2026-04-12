#!/usr/bin/env python3
"""Stage a deterministic benchmark batch bundle for cross-workflow artifact handoff.

The staged directory is intentionally shaped so that GitHub artifact extraction can
restore files back under `benchmark_responses/` in a later workflow.

Usage:
  python scripts/ci/stage_batch_artifact.py \
    --batch-id ntp3-vr1-20260412 \
    --artifact-name benchmark-verified-ntp3-vr1-20260412 \
    --output-dir artifact_bundles/ntp3-vr1-20260412
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"
DEFAULT_BUNDLES_DIR = PROJECT_ROOT / "artifact_bundles"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src: Path, dest_root: Path) -> dict:
    relative_path = Path("benchmark_responses") / src.name
    dest = dest_root / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {
        "path": str(relative_path),
        "name": src.name,
        "sha256": sha256_file(dest),
        "size_bytes": dest.stat().st_size,
    }


def stage_batch_artifact(
    batch_id: str,
    artifact_name: str,
    responses_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    responses_dir = responses_dir or DEFAULT_RESPONSES_DIR
    output_dir = output_dir or (DEFAULT_BUNDLES_DIR / batch_id)
    output_dir = output_dir.resolve()

    repeat_summary = responses_dir / f"repeat_summary_{batch_id}.json"
    verification_report = responses_dir / f"verification_report_{batch_id}.json"
    artifact_manifest = responses_dir / f"artifact_manifest_{batch_id}.txt"
    raw_files = sorted(responses_dir.glob(f"responses_{batch_id}-*.jsonl"))

    if not repeat_summary.exists():
        raise FileNotFoundError(f"repeat_summary missing: {repeat_summary}")
    if not raw_files:
        raise FileNotFoundError(f"no raw JSONL files found for batch '{batch_id}'")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "benchmark_responses").mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    files.append({"kind": "repeat_summary", **_copy_file(repeat_summary, output_dir)})

    for raw_file in raw_files:
        files.append({"kind": "raw_jsonl", **_copy_file(raw_file, output_dir)})

    if verification_report.exists():
        files.append({"kind": "verification_report", **_copy_file(verification_report, output_dir)})
    if artifact_manifest.exists():
        files.append({"kind": "artifact_manifest", **_copy_file(artifact_manifest, output_dir)})

    manifest = {
        "batch_id": batch_id,
        "artifact_name": artifact_name,
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "responses_subdir": "benchmark_responses",
        "file_count": len(files),
        "files": files,
    }

    manifest_path = output_dir / "benchmark_responses" / f"batch_manifest_{batch_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return {
        "batch_id": batch_id,
        "artifact_name": artifact_name,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "file_count": len(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage deterministic batch artifact bundle")
    parser.add_argument("--batch-id", required=True, help="Batch identifier")
    parser.add_argument("--artifact-name", required=True, help="Expected uploaded artifact name")
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=None,
        help="Responses directory (default: benchmark_responses/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output staging directory (default: artifact_bundles/<batch_id>)",
    )
    args = parser.parse_args()

    result = stage_batch_artifact(
        batch_id=args.batch_id,
        artifact_name=args.artifact_name,
        responses_dir=args.responses_dir,
        output_dir=args.output_dir,
    )

    print(f"Staged batch artifact: {result['artifact_name']}")
    print(f"  Batch: {result['batch_id']}")
    print(f"  Output dir: {result['output_dir']}")
    print(f"  Manifest: {result['manifest_path']}")
    print(f"  Files: {result['file_count']}")


if __name__ == "__main__":
    main()
