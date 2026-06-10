#!/usr/bin/env python3
"""
Sync a validated snapshot bundle into site/data/latest/ for local preview.

The dashboard's canonical data entrypoint is site/data/latest/current.json.
Legacy split JSON files are still copied for transparency/backward compatibility.

Usage:
  python scripts/sync_site_data.py --snapshot-dir ./dist/test-snapshot-verify
  python scripts/sync_site_data.py --snapshot-dir ./dist/nt-p3-mcq-text-only-mini-r10-20260409
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from build_current_json import build_current, load_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_DATA_DIR = PROJECT_ROOT / "site" / "data" / "latest"

REQUIRED_FILES = ["manifest.json", "leaderboard.json", "model_cards.json", "examples.json"]
CURRENT_BUNDLE_NAME = "current-bundle.zip"
SNAPSHOT_BUNDLE_NAME = "snapshot-bundle.zip"


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_current_bundle(
    current: dict,
    out_path: Path,
    *,
    current_manifest_path: Path | None = None,
) -> None:
    """Write a downloadable bundle that matches the merged current leaderboard."""
    bundle_root = current.get("current_id") or "current"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{bundle_root}/current.json", _json_bytes(current))
        zf.writestr(f"{bundle_root}/manifest.json", _json_bytes(current["manifest"]))
        zf.writestr(f"{bundle_root}/leaderboard.json", _json_bytes(current["leaderboard"]))
        zf.writestr(f"{bundle_root}/model_cards.json", _json_bytes(current["model_cards"]))
        zf.writestr(f"{bundle_root}/examples.json", _json_bytes(current["examples"]))
        if current_manifest_path is not None and current_manifest_path.exists():
            zf.write(current_manifest_path, f"{bundle_root}/current-manifest.json")


def sync(
    snapshot_dir: Path,
    *,
    current_snapshot_dirs: list[Path] | None = None,
    current_id: str | None = None,
    current_manifest_path: Path | None = None,
):
    if not snapshot_dir.is_dir():
        print(f"Error: {snapshot_dir} is not a directory")
        sys.exit(1)

    missing = [f for f in REQUIRED_FILES if not (snapshot_dir / f).exists()]
    if missing:
        print(f"Error: missing files in snapshot: {', '.join(missing)}")
        sys.exit(1)

    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for f in REQUIRED_FILES:
        shutil.copy2(snapshot_dir / f, SITE_DATA_DIR / f)
        print(f"  copied {f}")

    if current_manifest_path is not None:
        if current_snapshot_dirs is not None:
            print("Error: use either current_snapshot_dirs or current_manifest_path, not both")
            sys.exit(1)
        current_snapshot_dirs, manifest_current_id = load_manifest(current_manifest_path)
        current_id = current_id or manifest_current_id

    current_dirs = current_snapshot_dirs or [snapshot_dir]
    current = build_current(current_dirs, current_id=current_id)
    with open(SITE_DATA_DIR / "current.json", "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("  wrote current.json")

    for name, key in {
        "manifest.json": "manifest",
        "leaderboard.json": "leaderboard",
        "model_cards.json": "model_cards",
        "examples.json": "examples",
    }.items():
        with open(SITE_DATA_DIR / name, "w", encoding="utf-8") as f:
            json.dump(current[key], f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"  wrote current {name}")

    write_current_bundle(
        current,
        SITE_DATA_DIR / CURRENT_BUNDLE_NAME,
        current_manifest_path=current_manifest_path,
    )
    print(f"  wrote {CURRENT_BUNDLE_NAME}")

    # Copy zip bundle if available
    zip_candidates = list(snapshot_dir.parent.glob(f"{snapshot_dir.name}.zip"))
    if zip_candidates:
        shutil.copy2(zip_candidates[0], SITE_DATA_DIR / SNAPSHOT_BUNDLE_NAME)
        print(f"  copied {SNAPSHOT_BUNDLE_NAME}")
    else:
        print("  warning: no zip bundle found")

    print(f"\nSite data synced to {SITE_DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Sync snapshot data into site/data/latest/")
    parser.add_argument("--snapshot-dir", required=True, help="Path to validated snapshot directory")
    parser.add_argument("--current-manifest", type=Path, help="Optional ordered current manifest for site/data/latest/current.json")
    parser.add_argument("--current-id", default=None, help="Optional current.json identifier override")
    args = parser.parse_args()
    sync(Path(args.snapshot_dir), current_manifest_path=args.current_manifest, current_id=args.current_id)


if __name__ == "__main__":
    main()
