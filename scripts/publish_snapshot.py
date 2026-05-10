#!/usr/bin/env python3
"""
End-to-end publish pipeline: build → validate → sync to site → update registry.

Usage:
  python scripts/publish_snapshot.py --batch-id mini-r10-20260409
  python scripts/publish_snapshot.py --batch-id mini-r10-20260409 --snapshot-id my-custom-id
  python scripts/publish_snapshot.py --batch-id ntp3-vr1-20260411 --dry-run
  python scripts/publish_snapshot.py --batch-id mini-r10-20260409 --dry-run --allow-unverified-publication
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_snapshot import build_snapshot
from validate_snapshot import SnapshotValidator
from sync_site_data import sync, SITE_DATA_DIR

REGISTRY_PATH = PROJECT_ROOT / "registry" / "snapshots.json"
CURRENT_MANIFEST_PATH = PROJECT_ROOT / "registry" / "current-manifest.json"
RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"snapshots": []}


def save_registry(data: dict):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_registry(snapshot_id: str, batch_id: str, manifest_path: str, published_at: str):
    registry = load_registry()
    existing = [s for s in registry["snapshots"] if s["snapshot_id"] == snapshot_id]
    if existing:
        print(f"  registry: snapshot '{snapshot_id}' already exists, updating")
        existing[0]["published_at"] = published_at
        existing[0]["batch_id"] = batch_id
        existing[0]["manifest_path"] = manifest_path
        existing[0]["status"] = "published"
    else:
        registry["snapshots"].append({
            "snapshot_id": snapshot_id,
            "published_at": published_at,
            "batch_id": batch_id,
            "manifest_path": manifest_path,
            "status": "published",
        })
    save_registry(registry)
    print(f"  registry: updated {REGISTRY_PATH}")


def load_current_manifest(path: Path = CURRENT_MANIFEST_PATH) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "schema_version": "panya_current_manifest_v1",
            "current_id": None,
            "sources": [],
        }

    sources = manifest.setdefault("sources", [])
    if not isinstance(sources, list):
        raise RuntimeError(f"{path} has invalid sources field; expected list")
    manifest.setdefault("schema_version", "panya_current_manifest_v1")
    return manifest


def save_current_manifest(manifest: dict, path: Path = CURRENT_MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_current_manifest(
    *,
    snapshot_id: str,
    batch_id: str,
    snapshot_dir: Path,
    published_at: str,
    current_id: str | None = None,
    manifest_path: Path = CURRENT_MANIFEST_PATH,
) -> dict:
    """Create/update the ordered current.json source manifest.

    Source order is preserved. If the same snapshot source is already present,
    update that entry in place rather than appending a duplicate. Later sources
    win when build_current_json.py sees duplicate model_id rows.
    """
    manifest = load_current_manifest(manifest_path)
    if current_id:
        manifest["current_id"] = current_id
    elif not manifest.get("current_id"):
        manifest["current_id"] = snapshot_id

    if snapshot_dir.is_absolute():
        try:
            rel_dir = str(snapshot_dir.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_dir = str(snapshot_dir)
    else:
        rel_dir = str(snapshot_dir)
    source = {
        "snapshot_id": snapshot_id,
        "batch_id": batch_id,
        "snapshot_dir": rel_dir,
        "published_at": published_at,
    }

    sources = manifest["sources"]
    match_index = None
    for idx, existing in enumerate(sources):
        if not isinstance(existing, dict):
            raise RuntimeError(f"{manifest_path} source #{idx + 1} must be an object")
        existing_dir = existing.get("snapshot_dir") or existing.get("path") or existing.get("dir")
        if existing.get("snapshot_id") == snapshot_id or existing_dir == rel_dir:
            match_index = idx
            break

    if match_index is None:
        sources.append(source)
        print(f"  current manifest: appended {rel_dir}")
    else:
        sources[match_index] = {**sources[match_index], **source}
        print(f"  current manifest: updated {rel_dir} at position {match_index + 1}")

    manifest["merge_policy"] = "later sources replace earlier rows with the same model_id"
    save_current_manifest(manifest, manifest_path)
    print(f"  current manifest: wrote {manifest_path}")
    return manifest


SUPPORTED_VERIFICATION_PROTOCOLS = {"single_run_publication_v1", "canonical_shadow_v1"}


def verify_publishable_batch(batch_id: str) -> dict:
    """Require a passing publication verification report before publication."""
    verification_path = RESPONSES_DIR / f"verification_report_{batch_id}.json"
    if not verification_path.exists():
        raise RuntimeError(
            f"Verified publication required, but verification report is missing: {verification_path}"
        )

    with open(verification_path, encoding="utf-8") as f:
        report = json.load(f)

    if report.get("status") != "pass":
        raise RuntimeError(f"Verification report status is not pass: {report.get('status')!r}")
    protocol = report.get("protocol")
    if protocol not in SUPPORTED_VERIFICATION_PROTOCOLS:
        raise RuntimeError(f"Unsupported verification protocol: {report.get('protocol')!r}")

    models = report.get("models", [])
    has_posthoc_gate = (
        report.get("publication_mode") == "verified_posthoc_gate_v1"
        or any("publishable" in model for model in models)
    )

    if has_posthoc_gate:
        if report.get("publishable_count", 0) < 1:
            raise RuntimeError("Verification report has no publishable models")

        for model in models:
            if model.get("publishable"):
                if not model.get("canonical_run_id"):
                    raise RuntimeError(f"Publishable model missing canonical run ID: {model.get('model_id')}")
                if protocol == "canonical_shadow_v1" and not model.get("shadow_run_id"):
                    raise RuntimeError(f"Publishable model missing shadow run ID: {model.get('model_id')}")
                if protocol == "canonical_shadow_v1" and not model.get("deterministic"):
                    raise RuntimeError(f"Publishable model is not deterministic: {model.get('model_id')}")
    else:
        if not report.get("all_deterministic"):
            raise RuntimeError("Verification report indicates non-deterministic publication batch")
        if not report.get("screening_batch_id"):
            raise RuntimeError(
                "Verification report is neither verified_posthoc_gate_v1 nor legacy screening-based; "
                "missing publication_mode/publishable markers and screening_batch_id"
            )

        for model in models:
            if not model.get("deterministic"):
                raise RuntimeError(f"Model failed canonical+shadow verification: {model.get('model_id')}")
            if not model.get("canonical_run_id"):
                raise RuntimeError(f"Model missing canonical run ID: {model.get('model_id')}")
            if protocol == "canonical_shadow_v1" and not model.get("shadow_run_id"):
                raise RuntimeError(f"Model missing shadow run ID: {model.get('model_id')}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Build, validate, and publish a snapshot")
    parser.add_argument("--batch-id", required=True, help="Batch id, e.g. mini-r10-20260409")
    parser.add_argument("--snapshot-id", default=None, help="Override snapshot id")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate only, do not sync or update registry")
    parser.add_argument(
        "--update-current-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update registry/current-manifest.json and build site/data/latest/current.json from it (default: true)",
    )
    parser.add_argument("--current-id", default=None, help="Set/override registry/current-manifest.json current_id")
    parser.add_argument(
        "--allow-unverified-publication",
        "--allow-unverified",
        action="store_true",
        help="Allow building/publishing from an unverified batch (local smoke use only)",
    )
    parser.add_argument(
        "--require-verified-publication",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    batch_id = args.batch_id
    snapshot_id = args.snapshot_id or f"nt-p3-mcq-text-only-{batch_id}"

    if not args.allow_unverified_publication:
        print(f"Requiring verified publication contract for batch '{batch_id}'")
        verify_publishable_batch(batch_id)

    # Step 1: Build
    print(f"\n{'='*60}")
    print(f"Step 1: Building snapshot '{snapshot_id}' from batch '{batch_id}'")
    print(f"{'='*60}")
    out_dir = build_snapshot(batch_id, snapshot_id)

    # Step 2: Validate
    print(f"\n{'='*60}")
    print(f"Step 2: Validating snapshot")
    print(f"{'='*60}")
    validator = SnapshotValidator(out_dir)
    valid = validator.validate()
    validator.report()

    if not valid:
        print("PUBLISH ABORTED: validation failed.")
        sys.exit(1)

    if args.dry_run:
        print(f"\nDry run complete. Snapshot built and validated at {out_dir}")
        print("Skipping site sync and registry update.")
        return

    with open(out_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    if args.update_current_manifest:
        print(f"\n{'='*60}")
        print(f"Step 3: Updating current manifest")
        print(f"{'='*60}")
        update_current_manifest(
            snapshot_id=snapshot_id,
            batch_id=batch_id,
            snapshot_dir=out_dir,
            published_at=manifest["published_at"],
            current_id=args.current_id,
        )

    # Step 4: Sync to site
    print(f"\n{'='*60}")
    print(f"Step 4: Syncing to site/data/latest/")
    print(f"{'='*60}")
    if args.update_current_manifest:
        sync(out_dir, current_manifest_path=CURRENT_MANIFEST_PATH, current_id=args.current_id)
    else:
        sync(out_dir)

    # Step 5: Update registry
    print(f"\n{'='*60}")
    print(f"Step 5: Updating registry")
    print(f"{'='*60}")
    manifest_path = str(out_dir.relative_to(PROJECT_ROOT) / "manifest.json")
    update_registry(snapshot_id, batch_id, manifest_path, manifest["published_at"])

    # Summary
    print(f"\n{'='*60}")
    print(f"PUBLISHED: {snapshot_id}")
    print(f"  snapshot dir: {out_dir}")
    print(f"  site data:    {SITE_DATA_DIR}")
    print(f"  registry:     {REGISTRY_PATH}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Preview locally:  sh scripts/serve.sh")
    print("  2. Commit changes:   git add site/data registry/ dist/ && git commit")
    print("  3. Open PR to main")
    print("  4. After merge, pages-deploy.yml deploys to GitHub Pages")


if __name__ == "__main__":
    main()
