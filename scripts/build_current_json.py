#!/usr/bin/env python3
"""Build the dashboard's single current.json from one or more snapshot bundles.

Later inputs replace earlier rows for the same model_id. This lets a future
snapshot keep the restart baseline as its base while appending or retesting only
some models, as long as the benchmark/prompt/dataset contract is compatible.

Usage:
  python scripts/build_current_json.py --snapshot-dir dist/nt-p3-mcq-text-only-ntp3-pub-r1-20260510 --out site/data/latest/current.json
  python scripts/build_current_json.py --snapshot-dir dist/base --snapshot-dir dist/new-models --current-id ntp3-current-20260515 --out site/data/latest/current.json
  python scripts/build_current_json.py --manifest registry/current-manifest.json --out site/data/latest/current.json

Manifest format:
  {
    "current_id": "ntp3-current-20260510",
    "sources": [
      {"snapshot_dir": "dist/nt-p3-mcq-text-only-ntp3-pub-r1-20260510"},
      {"snapshot_dir": "dist/nt-p3-mcq-text-only-ntp3-add-r1-20260520-newmodel"}
    ]
  }

Source order matters: later sources replace earlier rows for duplicate model_id.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REQUIRED_FILES = ("manifest.json", "leaderboard.json", "model_cards.json", "examples.json")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def benchmark_scope(snapshot: dict[str, Any]) -> str | None:
    return (
        snapshot["leaderboard"].get("benchmark_scope")
        or snapshot["model_cards"].get("benchmark_scope")
        or snapshot["examples"].get("benchmark_scope")
        or snapshot["manifest"].get("benchmark_scope")
    )


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


def _best_by(rows: list[dict[str, Any]], field: str, *, reverse: bool = False) -> set[str]:
    """Return model_ids tied for the best value of a numeric leaderboard field."""
    candidates = [row for row in rows if isinstance(row.get(field), (int, float))]
    if not candidates:
        return set()
    values = [row[field] for row in candidates]
    best = min(values) if reverse else max(values)
    return {row["model_id"] for row in candidates if row[field] == best}


def assign_current_badges(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Assign badges across the merged current leaderboard.

    Individual snapshot bundles already contain badges, but a one-model append
    snapshot will naturally mark its only model as "Best Quality", "Best Thai",
    etc. Those snapshot-local badges are not meaningful once the row is merged
    into the global current leaderboard, so current.json must recompute badges
    after all source snapshots have been merged.
    """
    badge_map: dict[str, list[str]] = {row["model_id"]: [] for row in rows}

    for model_id in _best_by(rows, "balanced_quality_score"):
        badge_map[model_id].append("Best Quality")
    for model_id in _best_by(rows, "thai_score_rate"):
        badge_map[model_id].append("Best Thai")
    for model_id in _best_by(rows, "math_score_rate"):
        badge_map[model_id].append("Best Math")
    for model_id in _best_by(rows, "latency_p50_ms", reverse=True):
        badge_map[model_id].append("Fastest on Testbed")

    comfortable = [row for row in rows if row.get("ram_fit_class") == "fits_comfortably_16gb"]
    for model_id in _best_by(comfortable, "balanced_quality_score"):
        badge_map[model_id].append("Best Small Model")

    return badge_map


def build_current(snapshot_dirs: list[Path], current_id: str | None = None) -> dict[str, Any]:
    if not snapshot_dirs:
        raise ValueError("at least one --snapshot-dir is required")

    loaded = [load_snapshot(path) for path in snapshot_dirs]
    latest = loaded[-1]
    current_id = current_id or latest["manifest"].get("snapshot_id") or "current"

    scopes = {benchmark_scope(snap) for snap in loaded if benchmark_scope(snap)}
    if len(scopes) > 1:
        raise ValueError(f"current sources are not benchmark-compatible; found scopes: {sorted(scopes)}")

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
    badge_map = assign_current_badges(rows)
    for row in rows:
        row["badges"] = badge_map.get(row["model_id"], [])
    for card in cards:
        card["badges"] = badge_map.get(card["model_id"], [])
    cards.sort(key=lambda card: next((row["rank"] for row in rows if row["model_id"] == card["model_id"]), 10**9))
    examples = [examples_by_id[eid] for eid in sorted(referenced_examples) if eid in examples_by_id]

    manifest = dict(latest["manifest"])
    manifest["snapshot_id"] = current_id
    manifest["current_json"] = True
    manifest["source_snapshots"] = [
        {
            "snapshot_id": snap["manifest"].get("snapshot_id"),
            "published_at": snap["manifest"].get("published_at"),
            "path": repo_relative(snap["dir"]),
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


def load_manifest(manifest_path: Path) -> tuple[list[Path], str | None]:
    """Load an ordered current-json source manifest.

    Relative snapshot paths are resolved against the repository root.
    Each source may use `snapshot_dir`, `path`, or `dir` for readability.
    """
    manifest = load_json(manifest_path)
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{manifest_path} must contain a non-empty sources array")

    snapshot_dirs: list[Path] = []
    for idx, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            raise ValueError(f"{manifest_path} source #{idx} must be an object")
        raw = source.get("snapshot_dir") or source.get("path") or source.get("dir")
        if not raw:
            raise ValueError(f"{manifest_path} source #{idx} missing snapshot_dir/path/dir")
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        snapshot_dirs.append(path)

    current_id = manifest.get("current_id")
    return snapshot_dirs, current_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged dashboard current.json")
    parser.add_argument("--snapshot-dir", action="append", type=Path, default=[], help="Snapshot bundle directory; repeatable, later wins")
    parser.add_argument("--manifest", type=Path, help="Ordered source manifest JSON; later sources win")
    parser.add_argument("--current-id", default=None, help="Current JSON/snapshot identifier")
    parser.add_argument("--out", type=Path, required=True, help="Output current.json path")
    args = parser.parse_args()

    snapshot_dirs = args.snapshot_dir
    current_id = args.current_id
    if args.manifest:
        if snapshot_dirs:
            raise SystemExit("Use either --manifest or --snapshot-dir, not both")
        snapshot_dirs, manifest_current_id = load_manifest(args.manifest)
        current_id = current_id or manifest_current_id

    current = build_current(snapshot_dirs, current_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
