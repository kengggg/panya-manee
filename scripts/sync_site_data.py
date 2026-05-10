#!/usr/bin/env python3
"""
Sync a validated snapshot bundle into site/data/latest/ for local preview.

The dashboard's canonical data entrypoint is site/data/latest/current.json.
Legacy split JSON files are still copied for transparency/backward compatibility.

Usage:
  python scripts/sync_site_data.py --snapshot-dir ./dist/test-snapshot-verify
  python scripts/sync_site_data.py --snapshot-dir ./dist/nt-p3-mcq-text-only-mini-r10-20260409
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from build_current_json import build_current

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_DATA_DIR = PROJECT_ROOT / "site" / "data" / "latest"

REQUIRED_FILES = ["manifest.json", "leaderboard.json", "model_cards.json", "examples.json"]


def sync(snapshot_dir: Path):
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

    current = build_current([snapshot_dir])
    with open(SITE_DATA_DIR / "current.json", "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("  wrote current.json")

    # Copy zip bundle if available
    zip_candidates = list(snapshot_dir.parent.glob(f"{snapshot_dir.name}.zip"))
    if zip_candidates:
        shutil.copy2(zip_candidates[0], SITE_DATA_DIR / "snapshot-bundle.zip")
        print(f"  copied snapshot-bundle.zip")
    else:
        print("  warning: no zip bundle found")

    print(f"\nSite data synced to {SITE_DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Sync snapshot data into site/data/latest/")
    parser.add_argument("--snapshot-dir", required=True, help="Path to validated snapshot directory")
    args = parser.parse_args()
    sync(Path(args.snapshot_dir))


if __name__ == "__main__":
    main()
