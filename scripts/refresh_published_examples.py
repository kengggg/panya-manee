#!/usr/bin/env python3
from __future__ import annotations

"""Refresh published examples/model_cards from a bundled snapshot using current selection logic.

This is for cases where the live published snapshot exists but the original raw batch files are
not available locally. It reselects examples from bundled results.jsonl, updates model_cards.json
and examples.json, bumps manifest/registry timestamps, and rebuilds snapshot-bundle.zip.
"""

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_snapshot import (
    PROJECT_ROOT,
    compute_skill_stats,
    load_question_bank,
    pick_strengths_weaknesses,
    select_examples,
)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def refresh(site_dir: Path):
    manifest_path = site_dir / "manifest.json"
    model_cards_path = site_dir / "model_cards.json"
    examples_path = site_dir / "examples.json"
    bundle_path = site_dir / "snapshot-bundle.zip"
    registry_path = PROJECT_ROOT / "registry" / "snapshots.json"

    manifest = load_json(manifest_path)
    model_cards = load_json(model_cards_path)
    snapshot_id = manifest["snapshot_id"]

    with zipfile.ZipFile(bundle_path) as zf:
        raw_names = [
            name for name in zf.namelist()
            if name.startswith(f"{snapshot_id}/raw/") and name.endswith(".jsonl")
        ]

        rows = []
        if raw_names:
            for name in sorted(raw_names):
                payload = zf.read(name).decode("utf-8").splitlines()
                rows.extend(json.loads(line) for line in payload if line.strip())
        else:
            raw_results = zf.read(f"{snapshot_id}/results.jsonl").decode("utf-8").splitlines()
            rows = [json.loads(line) for line in raw_results if line.strip()]

    question_bank = load_question_bank()
    rows_by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_model[row["model_id"]].append(row)

    rebuilt_examples = []
    for card in model_cards["models"]:
        model_id = card["model_id"]
        model_rows = rows_by_model.get(model_id, [])
        if not model_rows:
            card["example_ids"] = {"good": [], "bad": []}
            continue

        skill_stats = compute_skill_stats(model_rows)
        strengths, weaknesses = pick_strengths_weaknesses(skill_stats)

        canonical_run_id = sorted({row["run_id"] for row in model_rows})[0]
        canonical_rows = [row for row in model_rows if row["run_id"] == canonical_run_id]

        examples = select_examples(
            canonical_rows,
            strengths,
            weaknesses,
            model_id,
            question_bank=question_bank,
        )
        rebuilt_examples.extend(examples)
        card["example_ids"] = {
            "good": [e["example_id"] for e in examples if e["is_correct"]],
            "bad": [e["example_id"] for e in examples if not e["is_correct"]],
        }

    examples = {
        "snapshot_id": snapshot_id,
        "benchmark_scope": manifest["benchmark_scope"],
        "examples": rebuilt_examples,
    }

    now = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    manifest["published_at"] = now
    registry = load_json(registry_path)
    for snap in registry.get("snapshots", []):
        if snap.get("snapshot_id") == snapshot_id:
            snap["published_at"] = now
            snap["status"] = "published"

    write_json(manifest_path, manifest)
    write_json(model_cards_path, model_cards)
    write_json(examples_path, examples)
    write_json(registry_path, registry)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        with zipfile.ZipFile(bundle_path) as zf:
            zf.extractall(tmp)
        bundle_root = tmp / snapshot_id
        shutil.copy2(manifest_path, bundle_root / "manifest.json")
        shutil.copy2(model_cards_path, bundle_root / "model_cards.json")
        shutil.copy2(examples_path, bundle_root / "examples.json")
        rebuilt_bundle = tmp / "snapshot-bundle.zip"
        with zipfile.ZipFile(rebuilt_bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(bundle_root.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(tmp))
        shutil.move(rebuilt_bundle, bundle_path)

    return {
        "snapshot_id": snapshot_id,
        "published_at": now,
        "models": len(model_cards["models"]),
        "examples": len(rebuilt_examples),
    }


def main():
    parser = argparse.ArgumentParser(description="Refresh published examples from bundled results")
    parser.add_argument("--site-dir", type=Path, default=PROJECT_ROOT / "site" / "data" / "latest")
    args = parser.parse_args()
    result = refresh(args.site_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
