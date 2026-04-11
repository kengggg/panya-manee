#!/usr/bin/env python3
"""
Validate a snapshot bundle for cross-file consistency and metric correctness.

Usage:
  python scripts/validate_snapshot.py --dir ./dist/nt-p3-mcq-text-only-mini-r10-20260409
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = PROJECT_ROOT / "registry"
COMPATIBILITY_CONFIG = REGISTRY_DIR / "compatibility.json"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class SnapshotValidator:
    def __init__(self, snapshot_dir: Path):
        self.dir = snapshot_dir
        self.errors = []
        self.warnings = []

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def validate(self) -> bool:
        """Run all checks. Returns True if valid."""
        self._check_files_exist()
        if self.errors:
            return False

        self.manifest = load_json(self.dir / "manifest.json")
        self.leaderboard = load_json(self.dir / "leaderboard.json")
        self.model_cards = load_json(self.dir / "model_cards.json")
        self.examples = load_json(self.dir / "examples.json")

        self._check_snapshot_id_consistency()
        self._check_benchmark_scope_consistency()
        self._check_leaderboard_model_card_match()
        self._check_example_id_references()
        self._check_metric_formulas()
        self._check_metric_ranges()
        self._check_latency_ordering()
        self._check_ranking_order()
        self._check_badge_validity()
        self._check_best_small_model_badge()
        self._check_strengths_weaknesses_min_n()
        self._check_output_truncation()
        self._check_manifest_notes()
        self._check_compatibility()
        self._check_transparency_files()

        return len(self.errors) == 0

    def _check_files_exist(self):
        for name in [
            "manifest.json",
            "leaderboard.json",
            "model_cards.json",
            "examples.json",
            "repeat_summary.json",
            "results.jsonl",
        ]:
            if not (self.dir / name).exists():
                self.error(f"Missing required file: {name}")
        if not (self.dir / "raw").is_dir():
            self.error("Missing required directory: raw/")

    def _check_snapshot_id_consistency(self):
        sid = self.manifest["snapshot_id"]
        for name, data in [
            ("leaderboard.json", self.leaderboard),
            ("model_cards.json", self.model_cards),
            ("examples.json", self.examples),
        ]:
            if data.get("snapshot_id") != sid:
                self.error(f"{name} snapshot_id '{data.get('snapshot_id')}' != manifest '{sid}'")

    def _check_benchmark_scope_consistency(self):
        scope = self.manifest["benchmark_scope"]
        for name, data in [
            ("leaderboard.json", self.leaderboard),
            ("model_cards.json", self.model_cards),
            ("examples.json", self.examples),
        ]:
            if data.get("benchmark_scope") != scope:
                self.error(f"{name} benchmark_scope mismatch: '{data.get('benchmark_scope')}' != '{scope}'")

    def _check_leaderboard_model_card_match(self):
        lb_models = {row["model_id"] for row in self.leaderboard["rows"]}
        card_models = {m["model_id"] for m in self.model_cards["models"]}

        for m in lb_models - card_models:
            self.error(f"Leaderboard model '{m}' has no matching model card")
        for m in card_models - lb_models:
            self.error(f"Model card '{m}' not in leaderboard")

    def _check_example_id_references(self):
        example_ids = {e["example_id"] for e in self.examples["examples"]}
        for card in self.model_cards["models"]:
            for kind in ["good", "bad"]:
                for eid in card["example_ids"].get(kind, []):
                    if eid not in example_ids:
                        self.error(f"Model '{card['model_id']}' references missing example '{eid}'")

    def _check_metric_formulas(self):
        """Verify balanced_quality_score = (thai + math) / 2 for all models."""
        for row in self.leaderboard["rows"]:
            expected = round((row["thai_score_rate"] + row["math_score_rate"]) / 2, 4)
            actual = row["balanced_quality_score"]
            if abs(actual - expected) > 0.0001:
                self.error(
                    f"Model '{row['model_id']}': balanced_quality_score {actual} != "
                    f"(thai {row['thai_score_rate']} + math {row['math_score_rate']}) / 2 = {expected}"
                )

        # Same check in model_cards
        for card in self.model_cards["models"]:
            m = card["metrics"]
            expected = round((m["thai_score_rate"] + m["math_score_rate"]) / 2, 4)
            actual = m["balanced_quality_score"]
            if abs(actual - expected) > 0.0001:
                self.error(
                    f"Model card '{card['model_id']}': balanced_quality_score {actual} != expected {expected}"
                )

    def _check_metric_ranges(self):
        rate_fields = [
            "balanced_quality_score", "thai_score_rate", "math_score_rate",
            "overall_score_rate", "parseable_rate", "answer_only_compliance_rate",
        ]
        for row in self.leaderboard["rows"]:
            for field in rate_fields:
                v = row.get(field)
                if v is not None and not (0.0 <= v <= 1.0):
                    self.error(f"Model '{row['model_id']}': {field}={v} out of [0,1]")
            if row.get("item_count", 0) <= 0:
                self.error(f"Model '{row['model_id']}': item_count <= 0")

    def _check_latency_ordering(self):
        for row in self.leaderboard["rows"]:
            p50 = row.get("latency_p50_ms", 0)
            p95 = row.get("latency_p95_ms", 0)
            if p95 < p50:
                self.error(f"Model '{row['model_id']}': latency_p95_ms ({p95}) < latency_p50_ms ({p50})")

    def _check_ranking_order(self):
        rows = self.leaderboard["rows"]
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            curr = rows[i]
            if curr["rank"] < prev["rank"]:
                self.error(
                    f"Ranking not monotonic: model '{curr['model_id']}' rank {curr['rank']} "
                    f"< previous '{prev['model_id']}' rank {prev['rank']}"
                )
            # If rank increases, score should not increase
            if curr["rank"] > prev["rank"] and curr["balanced_quality_score"] > prev["balanced_quality_score"]:
                self.error(
                    f"Model '{curr['model_id']}' has lower rank but higher score than '{prev['model_id']}'"
                )

    def _check_badge_validity(self):
        valid_badges = {
            "Best Quality", "Best Thai", "Best Math",
            "Fastest on Testbed", "Best Small Model",
        }
        for row in self.leaderboard["rows"]:
            for b in row.get("badges", []):
                if b not in valid_badges:
                    self.error(f"Model '{row['model_id']}': unknown badge '{b}'")

    def _check_best_small_model_badge(self):
        """Best Small Model must only go to fits_comfortably_16gb models."""
        for row in self.leaderboard["rows"]:
            if "Best Small Model" in row.get("badges", []):
                if row["ram_fit_class"] != "fits_comfortably_16gb":
                    self.error(
                        f"Model '{row['model_id']}' has Best Small Model badge "
                        f"but ram_fit_class='{row['ram_fit_class']}'"
                    )

    def _check_strengths_weaknesses_min_n(self):
        for card in self.model_cards["models"]:
            for kind in ["strengths", "weaknesses"]:
                for entry in card.get(kind, []):
                    if entry["total"] < 2:
                        self.error(
                            f"Model '{card['model_id']}' {kind}: "
                            f"skill_tag '{entry['skill_tag']}' has total={entry['total']} < 2"
                        )

    def _check_output_truncation(self):
        for ex in self.examples["examples"]:
            trunc = ex.get("raw_output_truncated", "")
            full = ex.get("raw_output_full", "")
            expected_trunc = full[:200]
            if trunc != expected_trunc:
                self.error(
                    f"Example '{ex['example_id']}': truncated output doesn't match first 200 chars of full"
                )

    def _check_manifest_notes(self):
        notes = self.manifest.get("snapshot_notes", {})
        if not notes.get("ranking_excludes_image_required"):
            self.warn("Manifest missing ranking_excludes_image_required note")
        if not notes.get("ranking_excludes_human_checked"):
            self.warn("Manifest missing ranking_excludes_human_checked note")

    def _check_compatibility(self):
        """Validate compatibility block in manifest against registry contract."""
        compat_block = self.manifest.get("compatibility")
        if compat_block is None:
            self.error("Manifest missing 'compatibility' block")
            return

        # Must have required_match_fields and values
        match_fields = compat_block.get("required_match_fields")
        values = compat_block.get("values")
        if not match_fields:
            self.error("compatibility.required_match_fields is missing or empty")
            return
        if not values:
            self.error("compatibility.values is missing or empty")
            return

        # Every required_match_field must have a non-null value
        for field in match_fields:
            if field not in values or values[field] is None:
                self.error(f"compatibility.values missing required field '{field}'")

        # Cross-check against registry/compatibility.json if available
        if COMPATIBILITY_CONFIG.exists():
            registry_compat = load_json(COMPATIBILITY_CONFIG)
            registry_fields = set(registry_compat.get("required_match_fields", []))
            manifest_fields = set(match_fields)
            if manifest_fields != registry_fields:
                self.error(
                    f"compatibility.required_match_fields {sorted(manifest_fields)} "
                    f"!= registry {sorted(registry_fields)}"
                )

            # Validate values match V1 expected values
            expected = registry_compat.get("v1_expected_values", {})
            for field, expected_val in expected.items():
                actual_val = values.get(field)
                if actual_val != expected_val:
                    self.error(
                        f"compatibility.values.{field} = {actual_val!r}, "
                        f"expected {expected_val!r} per registry V1 contract"
                    )

        # Validate benchmark_scope matches manifest top-level
        manifest_scope = self.manifest.get("benchmark_scope")
        compat_scope = values.get("benchmark_scope")
        if compat_scope and manifest_scope and compat_scope != manifest_scope:
            self.error(
                f"compatibility.values.benchmark_scope '{compat_scope}' "
                f"!= manifest.benchmark_scope '{manifest_scope}'"
            )

        # If results.jsonl exists, spot-check that row-level fields match compatibility values
        results_path = self.dir / "results.jsonl"
        if results_path.exists():
            self._check_compatibility_vs_results(values)

    def _check_compatibility_vs_results(self, compat_values: dict):
        """Spot-check results.jsonl rows against compatibility values."""
        results_path = self.dir / "results.jsonl"
        with open(results_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50:  # spot-check first 50 rows
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # benchmark_scope
                row_scope = row.get("benchmark_scope")
                if row_scope and row_scope != compat_values.get("benchmark_scope"):
                    self.error(
                        f"results.jsonl row {i}: benchmark_scope '{row_scope}' "
                        f"!= compatibility '{compat_values['benchmark_scope']}'"
                    )
                    return  # one error is enough
                # benchmark_suite_id should be in the manifest suite_ids
                suite_id = row.get("benchmark_suite_id")
                manifest_suites = self.manifest.get("suite_ids", [])
                if suite_id and manifest_suites and suite_id not in manifest_suites:
                    self.error(
                        f"results.jsonl row {i}: benchmark_suite_id '{suite_id}' "
                        f"not in manifest.suite_ids"
                    )
                    return

    def _check_transparency_files(self):
        raw_files = list((self.dir / "raw").glob("*.jsonl"))
        if not raw_files:
            self.error("raw/ contains no JSONL source files")

        manifest_artifacts = self.manifest.get("artifacts", {})
        if manifest_artifacts.get("repeat_summary") != "repeat_summary.json":
            self.error("manifest.artifacts.repeat_summary missing or incorrect")
        if manifest_artifacts.get("results") != "results.jsonl":
            self.error("manifest.artifacts.results missing or incorrect")
        verification_artifact = manifest_artifacts.get("verification_report")
        if verification_artifact:
            verification_path = self.dir / verification_artifact
            if not verification_path.exists():
                self.error(f"manifest.artifacts.verification_report points to missing file: {verification_artifact}")
            else:
                report = load_json(verification_path)
                if report.get("status") != "pass":
                    self.error("verification_report.json is present but status != 'pass'")
                if not report.get("all_deterministic"):
                    self.error("verification_report.json is present but all_deterministic is false")

    def report(self):
        if self.errors:
            print(f"\n  ERRORS ({len(self.errors)}):")
            for e in self.errors:
                print(f"    ✗ {e}")
        if self.warnings:
            print(f"\n  WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"    ⚠ {w}")
        if not self.errors and not self.warnings:
            print("\n  All checks passed.")
        print()


def main():
    parser = argparse.ArgumentParser(description="Validate snapshot bundle")
    parser.add_argument("--dir", required=True, help="Path to snapshot directory")
    args = parser.parse_args()

    snapshot_dir = Path(args.dir)
    if not snapshot_dir.is_dir():
        print(f"Error: {snapshot_dir} is not a directory")
        sys.exit(1)

    print(f"Validating snapshot: {snapshot_dir}")
    v = SnapshotValidator(snapshot_dir)
    valid = v.validate()
    v.report()

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
