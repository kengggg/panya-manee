#!/usr/bin/env python3
"""Build the dashboard's single current.json from one or more snapshot bundles.

Later inputs replace earlier rows for the same model_id. This lets a future
snapshot keep the restart baseline as its base while appending or retesting only
some models, as long as the benchmark/prompt/dataset contract is compatible.

Usage:
  python scripts/build_current_json.py --snapshot-dir dist/nt-p3-mcq-text-only-ntp3-pub-r1-20260510 --out site/data/latest/current.json
  python scripts/build_current_json.py --snapshot-dir dist/base --snapshot-dir dist/new-models --current-id ntp3-current-20260515 --out site/data/latest/current.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REQUIRED_FILES = ("manifest.json", "leaderboard.json", "model_cards.json", "examples.json")


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (snapshot_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{snapshot_dir} missing required file(s): {', '.join(missing)}")
    manifest = load_json(snapshot_dir / "manifest.json")
    leaderboard = load_json(snapshot_dir / "leaderboard.json")
    model_cards = load_json(snapshot_dir / "model_cards.json")
    examples = load_json(snapshot_dir / "examples.json")
    return {
        "dir": snapshot_dir,
        "manifest": manifest,
        "leaderboard": leaderboard,
        "model_cards": model_cards,
        "examples": examples,
    }


def rerank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -row.get("balanced_quality_score", 0),
            -row.get("overall_score_rate", 0),
            row.get("model_id", ""),
        ),
    )
    for idx, row in enumerate(ordered, 1):
        row["rank"] = idx
    return ordered


def build_current(snapshot_dirs: list[Path], current_id: str | None = None) -> dict[str, Any]:
    if not snapshot_dirs:
        raise ValueError("at least one --snapshot-dir is required")

    loaded = [load_snapshot(path) for path in snapshot_dirs]
    latest = loaded[-1]
    current_id = current_id or latest["manifest"].get("snapshot_id") or "current"

    rows_by_model: dict[str, dict[str, Any]] = {}
    cards_by_model: dict[str, dict[str, Any]] = {}
    examples_by_id: dict[str, dict[str, Any]] = {}
    source_by_model: dict[str, str] = {}

    for snap in loaded:
        sid = snap["manifest"].get("snapshot_id")
        for row in snap["leaderboard"].get("rows", []):
            model_id = row["model_id"]
            rows_by_model[model_id] = dict(row)
            source_by_model[model_id] = sid
        for card in snap["model_cards"].get("models", []):
            cards_by_model[card["model_id"]] = dict(card)
        for ex in snap["examples"].get("examples", []):
            examples_by_id[ex["example_id"]] = dict(ex)

    active_models = set(rows_by_model)
    cards = [card for model_id, card in cards_by_model.items() if model_id in active_models]
    referenced_examples = set()
    for card in cards:
        ids = card.get("example_ids", {})
        referenced_examples.update(ids.get("good", []))
        referenced_examples.update(ids.get("bad", []))

    rows = rerank(list(rows_by_model.values()))
    cards.sort(key=lambda card: next((row["rank"] for row in rows if row["model_id"] == card["model_id"]), 10**9))
    examples = [examples_by_id[eid] for eid in sorted(referenced_examples) if eid in examples_by_id]

    manifest = dict(latest["manifest"])
    manifest["snapshot_id"] = current_id
    manifest["current_json"] = True
    manifest["source_snapshots"] = [
        {
            "snapshot_id": snap["manifest"].get("snapshot_id"),
            "published_at": snap["manifest"].get("published_at"),
            "path": str(snap["dir"]),
        }
        for snap in loaded
    ]
    manifest.setdefault("snapshot_notes", {})["current_json_merge_policy"] = (
        "Built from one or more compatible snapshot bundles; later inputs replace earlier rows for duplicate model_id."
    )

    return {
        "schema_version": "panya_current_v1",
        "current_id": current_id,
        "generated_at": datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds"),
        "sources": manifest["source_snapshots"],
        "model_sources": source_by_model,
        "manifest": manifest,
        "leaderboard": {
            "snapshot_id": current_id,
            "benchmark_scope": latest["leaderboard"].get("benchmark_scope"),
            "rows": rows,
        },
        "model_cards": {
            "snapshot_id": current_id,
            "benchmark_scope": latest["model_cards"].get("benchmark_scope"),
            "models": cards,
        },
        "examples": {
            "snapshot_id": current_id,
            "benchmark_scope": latest["examples"].get("benchmark_scope"),
            "examples": examples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged dashboard current.json")
    parser.add_argument("--snapshot-dir", action="append", type=Path, required=True, help="Snapshot bundle directory; repeatable, later wins")
    parser.add_argument("--current-id", default=None, help="Current JSON/snapshot identifier")
    parser.add_argument("--out", type=Path, required=True, help="Output current.json path")
    args = parser.parse_args()

    current = build_current(args.snapshot_dir, args.current_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
