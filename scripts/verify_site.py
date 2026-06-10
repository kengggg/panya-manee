#!/usr/bin/env python3
"""
Verify that the site/ directory is correctly wired to snapshot data
and that the HTML/JS/CSS assets reference the right data paths.

Usage:
  python scripts/verify_site.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = PROJECT_ROOT / "site"
DATA_DIR = SITE_DIR / "data" / "latest"

errors = []
warnings = []


def error(msg):
    errors.append(msg)
    print(f"  FAIL: {msg}")


def ok(msg):
    print(f"  OK:   {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  WARN: {msg}")


def check_site_assets():
    print("\n[1] Site assets")
    for name in ["index.html", "model.html", "styles.css", "app.js", "model.js"]:
        path = SITE_DIR / name
        if path.exists() and path.stat().st_size > 0:
            ok(f"{name} exists ({path.stat().st_size} bytes)")
        else:
            error(f"{name} missing or empty")


def check_data_files():
    print("\n[2] Data files in site/data/latest/")
    required = ["current.json", "manifest.json", "leaderboard.json", "model_cards.json", "examples.json"]
    for name in required:
        path = DATA_DIR / name
        if path.exists() and path.stat().st_size > 0:
            ok(f"{name} exists ({path.stat().st_size} bytes)")
        else:
            error(f"{name} missing or empty")

    current_zip_path = DATA_DIR / "current-bundle.zip"
    if current_zip_path.exists():
        ok(f"current-bundle.zip exists ({current_zip_path.stat().st_size} bytes)")
    else:
        error("current-bundle.zip missing (current leaderboard download link will be broken)")

    snapshot_zip_path = DATA_DIR / "snapshot-bundle.zip"
    if snapshot_zip_path.exists():
        ok(f"snapshot-bundle.zip exists ({snapshot_zip_path.stat().st_size} bytes)")
    else:
        warn("snapshot-bundle.zip missing (latest source snapshot download link will be broken)")


def check_data_consistency():
    print("\n[3] Data consistency")
    try:
        current = json.loads((DATA_DIR / "current.json").read_text())
        split_manifest = json.loads((DATA_DIR / "manifest.json").read_text())
        split_leaderboard = json.loads((DATA_DIR / "leaderboard.json").read_text())
        split_model_cards = json.loads((DATA_DIR / "model_cards.json").read_text())
        split_examples = json.loads((DATA_DIR / "examples.json").read_text())
        manifest = current["manifest"]
        leaderboard = current["leaderboard"]
        model_cards = current["model_cards"]
        examples = current["examples"]
    except Exception as e:
        error(f"Cannot parse JSON files: {e}")
        return

    if current.get("schema_version") == "panya_current_v1":
        ok("current.json uses panya_current_v1 schema")
    else:
        error("current.json missing panya_current_v1 schema")

    for name, split_data, current_data in [
        ("manifest.json", split_manifest, manifest),
        ("leaderboard.json", split_leaderboard, leaderboard),
        ("model_cards.json", split_model_cards, model_cards),
        ("examples.json", split_examples, examples),
    ]:
        if split_data == current_data:
            ok(f"{name} mirrors current.json")
        else:
            error(f"{name} does not mirror current.json")

    # Snapshot ID consistency
    sid = manifest["snapshot_id"]
    for name, data in [("leaderboard", leaderboard), ("model_cards", model_cards), ("examples", examples)]:
        if data.get("snapshot_id") != sid:
            error(f"{name}.json snapshot_id mismatch: {data.get('snapshot_id')} != {sid}")
        else:
            ok(f"{name}.json snapshot_id matches manifest")

    # Model count consistency
    lb_models = {r["model_id"] for r in leaderboard["rows"]}
    mc_models = {m["model_id"] for m in model_cards["models"]}
    if lb_models == mc_models:
        ok(f"Model sets match: {len(lb_models)} models")
    else:
        error(f"Model set mismatch: leaderboard={lb_models}, model_cards={mc_models}")

    # Example references
    example_ids = {e["example_id"] for e in examples["examples"]}
    for card in model_cards["models"]:
        for kind in ["good", "bad"]:
            for eid in card["example_ids"].get(kind, []):
                if eid not in example_ids:
                    error(f"Model {card['model_id']}: example {eid} not found in examples.json")

    # Leaderboard required columns
    if leaderboard["rows"]:
        row = leaderboard["rows"][0]
        required_fields = [
            "rank", "model_id", "balanced_quality_score", "thai_score_rate",
            "math_score_rate", "overall_score_rate", "parseable_rate",
            "answer_only_compliance_rate", "latency_p50_ms", "latency_p95_ms",
            "questions_per_min", "correct_per_min", "item_count", "badges",
        ]
        missing = [f for f in required_fields if f not in row]
        if missing:
            error(f"Leaderboard row missing fields: {missing}")
        else:
            ok(f"All {len(required_fields)} required leaderboard columns present")

    if model_cards["models"]:
        card = model_cards["models"][0]
        metrics = card.get("metrics", {})
        if "average_output_length_chars" in metrics:
            ok("Model cards include average_output_length_chars")
        else:
            error("Model cards missing average_output_length_chars")
        if "common_failure_types" in card:
            ok("Model cards include common_failure_types")
        else:
            error("Model cards missing common_failure_types")


def check_html_wiring():
    print("\n[4] HTML/JS data path references")
    app_js = (SITE_DIR / "app.js").read_text()
    model_js = (SITE_DIR / "model.js").read_text()
    index_html = (SITE_DIR / "index.html").read_text()

    if "data/latest" in app_js:
        ok("app.js references data/latest")
    else:
        error("app.js does not reference data/latest")

    if "data/latest" in model_js:
        ok("model.js references data/latest")
    else:
        error("model.js does not reference data/latest")

    if "current.json" in app_js and "current.json" in model_js:
        ok("Dashboard JS loads current.json as the canonical data entrypoint")
    else:
        error("Dashboard JS does not load current.json")

    if "model.html?model=" in app_js or "model.html?model=" in index_html:
        ok("Leaderboard links to model.html with query param")
    else:
        error("No model.html link pattern found")

    if "current-bundle.zip" in index_html:
        ok("Download link for current leaderboard bundle present in index.html")
    else:
        error("No download link for current-bundle.zip in index.html")

    if "snapshot-bundle.zip" in index_html:
        ok("Download link for latest source snapshot bundle present in index.html")
    else:
        error("No download link for snapshot-bundle.zip in index.html")

    if "preset-groups" in index_html and "PRESET_GROUPS" in app_js:
        ok("Combinable ranking preset wiring present")
    else:
        error("Combinable ranking preset wiring missing")

    required_presets = [
        "Best Overall",
        "Best Thai",
        "Best Math",
        "Fastest",
        "Best Practical",
        "Best Small Model",
        "Thai homework",
        "Math reasoning",
        "Fast classroom use",
        "Best on 16GB Mac mini or similar",
    ]
    missing_presets = [label for label in required_presets if label not in app_js]
    if missing_presets:
        error(f"Ranking preset labels missing from app.js: {missing_presets}")
    else:
        ok("All required ranking preset labels present")

    if "questions_per_min: 1.0" in app_js and "Q/min" in app_js:
        ok("Fastest preset uses Q/min speed metric")
    else:
        error("Fastest preset does not clearly use Q/min")

    if "nerd-mode" in index_html and "metric-col" in index_html:
        ok("Full table / nerd mode metric hooks present")
    else:
        error("Full table / nerd mode metric hooks missing")

    if "model-filter" in index_html or "initModelFilter" in app_js:
        error("Legacy model filter still present; presets should sort rather than filter")
    else:
        ok("No legacy model filter remains")

    model_html = (SITE_DIR / "model.html").read_text()
    if "index.html" in model_html:
        ok("model.html has back-link to index.html")
    else:
        warn("model.html missing back-link to index.html")


def main():
    print("Verifying site/ wiring and data integrity...")

    check_site_assets()
    check_data_files()
    check_data_consistency()
    check_html_wiring()

    print(f"\n{'='*50}")
    if errors:
        print(f"FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        sys.exit(1)
    elif warnings:
        print(f"PASSED with {len(warnings)} warning(s)")
    else:
        print("ALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
