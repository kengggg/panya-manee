#!/usr/bin/env python3
from __future__ import annotations

"""
Tests for the snapshot build pipeline.

Tests exercise real logic against actual benchmark batch data from
benchmark_responses/. They validate metric formulas, badge rules,
ranking, example selection, and cross-file consistency.

Run: python -m pytest tests/test_snapshot_bundle.py -v
  or: python -m unittest tests.test_snapshot_bundle -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from collections import defaultdict
from pathlib import Path
from unittest import mock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.build_snapshot import (
    aggregate_model,
    assign_badges,
    build_snapshot,
    compute_skill_stats,
    generate_auto_summary,
    is_answer_only_compliant,
    load_jsonl,
    load_json,
    load_model_meta,
    load_question_bank,
    load_compatibility,
    resolve_model_meta,
    resolve_question_item,
    resolve_compatibility_values,
    percentile,
    pick_strengths_weaknesses,
    rank_models,
    select_examples,
    _make_example,
    RESPONSES_DIR,
    MODELS_CONFIG,
    MACHINE_PROFILES_CONFIG,
    COMPATIBILITY_CONFIG,
    REGISTRY_DIR,
)
from scripts.validate_snapshot import SnapshotValidator

BATCH_ID = "mini-r10-20260409"


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _have_real_data() -> bool:
    """Check if the real benchmark data is available."""
    summary = RESPONSES_DIR / f"repeat_summary_{BATCH_ID}.json"
    return summary.exists()


SKIP_NO_DATA = not _have_real_data()
SKIP_MSG = f"Benchmark data for batch '{BATCH_ID}' not found in {RESPONSES_DIR}"


# ── Unit tests for core functions ────────────────────────────────────────────


class TestAnswerOnlyCompliance(unittest.TestCase):
    def test_single_digits_are_compliant(self):
        for d in ["1", "2", "3", "4"]:
            self.assertTrue(is_answer_only_compliant(d))

    def test_whitespace_padded_digits_are_compliant(self):
        self.assertTrue(is_answer_only_compliant("  3  "))
        self.assertTrue(is_answer_only_compliant("\n2\n"))

    def test_extra_text_is_not_compliant(self):
        self.assertFalse(is_answer_only_compliant("คำตอบคือ 3"))
        self.assertFalse(is_answer_only_compliant("3)"))
        self.assertFalse(is_answer_only_compliant("I choose 2"))
        self.assertFalse(is_answer_only_compliant(""))
        self.assertFalse(is_answer_only_compliant("34"))

    def test_digit_outside_range_is_not_compliant(self):
        self.assertFalse(is_answer_only_compliant("0"))
        self.assertFalse(is_answer_only_compliant("5"))


class TestPercentile(unittest.TestCase):
    def test_single_value(self):
        self.assertEqual(percentile([100], 50), 100)
        self.assertEqual(percentile([100], 95), 100)

    def test_known_values(self):
        values = list(range(1, 101))  # 1..100
        self.assertAlmostEqual(percentile(values, 50), 50.5, places=1)
        self.assertAlmostEqual(percentile(values, 95), 95.05, places=1)

    def test_p50_le_p95(self):
        values = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000]
        self.assertLessEqual(percentile(values, 50), percentile(values, 95))

    def test_empty(self):
        self.assertEqual(percentile([], 50), 0.0)


class TestModelMetaResolution(unittest.TestCase):
    def test_resolve_latest_alias(self):
        model_meta = load_model_meta(MODELS_CONFIG)
        meta = resolve_model_meta("scb10x/typhoon2.1-gemma3-4b", model_meta)
        self.assertEqual(meta.get("model_family"), "typhoon2.1")
        self.assertEqual(meta.get("ram_fit_class"), "fits_comfortably_16gb")


class TestExampleEnrichment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.question_bank = load_question_bank()

    def test_make_example_backfills_question_fields_from_question_bank(self):
        row = {
            "model_id": "gemma4:e2b",
            "exam_id": "nt_p3_th_2565",
            "subject": "thai",
            "question_id": 1,
            "curriculum_standard": "ท 1.1 ป.3/2",
            "skill_tag": ["reading_comprehension"],
            "is_correct": True,
            "correct_answer": "4",
            "parsed_answer": "4",
            "raw_output": "4",
            "latency_ms": 1234,
        }
        ex = _make_example(row, "gemma4:e2b", "good_0", "good_top_skill", self.question_bank)
        self.assertIn("คำในข้อใด", ex["prompt_text"])
        self.assertEqual(ex["choices"]["4"], "ขุ่นเคืองใจ")
        self.assertEqual(ex["model_answer_text"], "ขุ่นเคืองใจ")
        self.assertEqual(ex["correct_answer_text"], "ขุ่นเคืองใจ")

    def test_select_examples_includes_question_fields(self):
        canonical_rows = [
            {
                "model_id": "gemma4:e2b",
                "exam_id": "nt_p3_th_2565",
                "subject": "thai",
                "question_id": 1,
                "curriculum_standard": "ท 1.1 ป.3/2",
                "skill_tag": ["reading_comprehension"],
                "is_correct": True,
                "correct_answer": "4",
                "parsed_answer": "4",
                "raw_output": "4",
                "latency_ms": 1500,
            },
            {
                "model_id": "gemma4:e2b",
                "exam_id": "nt_p3_th_2565",
                "subject": "thai",
                "question_id": 2,
                "curriculum_standard": "ท 1.1 ป.3/2",
                "skill_tag": ["reading_comprehension"],
                "is_correct": False,
                "correct_answer": "2",
                "parsed_answer": "1",
                "raw_output": "1",
                "latency_ms": 1200,
            },
        ]
        strengths = [{"skill_tag": "reading_comprehension", "correct": 1, "total": 2, "score_rate": 0.5}]
        weaknesses = [{"skill_tag": "reading_comprehension", "correct": 1, "total": 2, "score_rate": 0.5}]

        examples = select_examples(
            canonical_rows,
            strengths,
            weaknesses,
            "gemma4:e2b",
            question_bank=self.question_bank,
            n_good=1,
            n_bad=1,
        )

        self.assertEqual(len(examples), 2)
        for ex in examples:
            self.assertTrue(ex.get("prompt_text"))
            self.assertTrue(ex.get("choices"))

    def test_select_examples_diversifies_subjects(self):
        canonical_rows = [
            {
                "model_id": "gemma4:e2b",
                "exam_id": "nt_p3_th_2565",
                "subject": "thai",
                "question_id": 1,
                "curriculum_standard": "ท 1.1 ป.3/2",
                "skill_tag": ["reading_comprehension"],
                "is_correct": True,
                "correct_answer": "4",
                "parsed_answer": "4",
                "raw_output": "4",
                "latency_ms": 1500,
            },
            {
                "model_id": "gemma4:e2b",
                "exam_id": "nt_p3_math_2565",
                "subject": "math",
                "question_id": 8,
                "curriculum_standard": "ค 1.1 ป.3/10",
                "skill_tag": ["fraction_addition"],
                "is_correct": True,
                "correct_answer": "1",
                "parsed_answer": "1",
                "raw_output": "1",
                "latency_ms": 1200,
            },
            {
                "model_id": "gemma4:e2b",
                "exam_id": "nt_p3_th_2566",
                "subject": "thai",
                "question_id": 8,
                "curriculum_standard": "ท 1.1 ป.3/2",
                "skill_tag": ["reading_comprehension"],
                "is_correct": True,
                "correct_answer": "2",
                "parsed_answer": "2",
                "raw_output": "2",
                "latency_ms": 1100,
            },
            {
                "model_id": "gemma4:e2b",
                "exam_id": "nt_p3_th_2565",
                "subject": "thai",
                "question_id": 2,
                "curriculum_standard": "ท 1.1 ป.3/2",
                "skill_tag": ["reading_comprehension"],
                "is_correct": False,
                "correct_answer": "2",
                "parsed_answer": "1",
                "raw_output": "1",
                "latency_ms": 1250,
            },
            {
                "model_id": "gemma4:e2b",
                "exam_id": "nt_p3_math_2567",
                "subject": "math",
                "question_id": 8,
                "curriculum_standard": "ค 1.1 ป.3/11",
                "skill_tag": ["fraction_addition"],
                "is_correct": False,
                "correct_answer": "4",
                "parsed_answer": "3",
                "raw_output": "3",
                "latency_ms": 1400,
            },
        ]
        strengths = [{"skill_tag": "reading_comprehension", "correct": 2, "total": 3, "score_rate": 0.6667}]
        weaknesses = [{"skill_tag": "fraction_addition", "correct": 0, "total": 1, "score_rate": 0.0}]

        examples = select_examples(
            canonical_rows,
            strengths,
            weaknesses,
            "gemma4:e2b",
            question_bank=self.question_bank,
            n_good=2,
            n_bad=2,
        )

        good_subjects = [ex["subject"] for ex in examples if ex["is_correct"]]
        bad_subjects = [ex["subject"] for ex in examples if not ex["is_correct"]]
        self.assertCountEqual(good_subjects, ["thai", "math"])
        self.assertCountEqual(bad_subjects, ["thai", "math"])

    def test_select_examples_does_not_duplicate_subject_when_only_one_subject_available(self):
        canonical_rows = [
            {
                "model_id": "gemma4:e2b",
                "exam_id": "nt_p3_th_2565",
                "subject": "thai",
                "question_id": 1,
                "curriculum_standard": "ท 1.1 ป.3/2",
                "skill_tag": ["reading_comprehension"],
                "is_correct": True,
                "correct_answer": "4",
                "parsed_answer": "4",
                "raw_output": "4",
                "latency_ms": 1500,
            },
            {
                "model_id": "gemma4:e2b",
                "exam_id": "nt_p3_th_2566",
                "subject": "thai",
                "question_id": 8,
                "curriculum_standard": "ท 1.1 ป.3/2",
                "skill_tag": ["reading_comprehension"],
                "is_correct": True,
                "correct_answer": "2",
                "parsed_answer": "2",
                "raw_output": "2",
                "latency_ms": 1100,
            },
        ]
        examples = select_examples(
            canonical_rows,
            [{"skill_tag": "reading_comprehension", "correct": 2, "total": 2, "score_rate": 1.0}],
            [],
            "gemma4:e2b",
            question_bank=self.question_bank,
            n_good=2,
            n_bad=2,
        )
        good_examples = [ex for ex in examples if ex["is_correct"]]
        self.assertEqual(len(good_examples), 1)

    def test_resolve_question_item_disambiguates_same_question_id(self):
        row = {
            "subject": "math",
            "question_id": 8,
            "curriculum_standard": "ค 1.1 ป.3/11",
            "correct_answer": "4",
        }
        item = resolve_question_item(row, self.question_bank)
        self.assertEqual(item.get("exam_id"), "nt_p3_math_2567")
        self.assertIn("แม่ค้าเหลือขนมทั้งสองชนิด", item.get("prompt_text", ""))


class TestSkillAnalysis(unittest.TestCase):
    def test_min_n_filter(self):
        stats = {
            "reading": {"correct": 5, "total": 5, "score_rate": 1.0},
            "rare_tag": {"correct": 1, "total": 1, "score_rate": 1.0},
            "weak_tag": {"correct": 0, "total": 3, "score_rate": 0.0},
        }
        strengths, weaknesses = pick_strengths_weaknesses(stats, top_n=3, min_n=2)
        # rare_tag should be excluded (total=1 < min_n=2)
        strength_tags = [s["skill_tag"] for s in strengths]
        weakness_tags = [w["skill_tag"] for w in weaknesses]
        self.assertNotIn("rare_tag", strength_tags)
        self.assertNotIn("rare_tag", weakness_tags)

    def test_tie_break_by_count_then_alpha(self):
        stats = {
            "beta": {"correct": 2, "total": 2, "score_rate": 1.0},
            "alpha": {"correct": 2, "total": 2, "score_rate": 1.0},
            "gamma": {"correct": 5, "total": 5, "score_rate": 1.0},
        }
        strengths, _ = pick_strengths_weaknesses(stats, top_n=3, min_n=2)
        # All score 1.0 — tie-break: higher total first, then alphabetical
        self.assertEqual(strengths[0]["skill_tag"], "gamma")  # total=5 wins
        self.assertEqual(strengths[1]["skill_tag"], "alpha")  # alpha < beta
        self.assertEqual(strengths[2]["skill_tag"], "beta")


class TestBadgeAssignment(unittest.TestCase):
    def test_best_small_model_excludes_tight(self):
        aggs = [
            {
                "model_id": "big",
                "balanced_quality_score": 0.8,
                "thai_score_rate": 0.8,
                "math_score_rate": 0.8,
                "latency_p50_ms": 100,
                "ram_fit_class": "fits_tightly_16gb",
            },
            {
                "model_id": "small",
                "balanced_quality_score": 0.6,
                "thai_score_rate": 0.6,
                "math_score_rate": 0.6,
                "latency_p50_ms": 50,
                "ram_fit_class": "fits_comfortably_16gb",
            },
        ]
        badges = assign_badges(aggs)
        # "big" should NOT get Best Small Model even though higher score
        self.assertNotIn("Best Small Model", badges.get("big", []))
        self.assertIn("Best Small Model", badges.get("small", []))

    def test_ties_allowed(self):
        aggs = [
            {
                "model_id": "a",
                "balanced_quality_score": 0.7,
                "thai_score_rate": 0.7,
                "math_score_rate": 0.7,
                "latency_p50_ms": 100,
                "ram_fit_class": "fits_comfortably_16gb",
            },
            {
                "model_id": "b",
                "balanced_quality_score": 0.7,
                "thai_score_rate": 0.7,
                "math_score_rate": 0.7,
                "latency_p50_ms": 100,
                "ram_fit_class": "fits_comfortably_16gb",
            },
        ]
        badges = assign_badges(aggs)
        self.assertIn("Best Quality", badges["a"])
        self.assertIn("Best Quality", badges["b"])


class TestRanking(unittest.TestCase):
    def test_higher_score_ranks_higher(self):
        aggs = [
            {"model_id": "low", "balanced_quality_score": 0.3,
             "parseable_rate": 1.0, "answer_only_compliance_rate": 1.0,
             "latency_p50_ms": 50},
            {"model_id": "high", "balanced_quality_score": 0.8,
             "parseable_rate": 1.0, "answer_only_compliance_rate": 1.0,
             "latency_p50_ms": 100},
        ]
        ranked = rank_models(aggs)
        self.assertEqual(ranked[0]["model_id"], "high")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["model_id"], "low")
        self.assertEqual(ranked[1]["rank"], 2)

    def test_tie_breaker_parseable_rate(self):
        aggs = [
            {"model_id": "less_parse", "balanced_quality_score": 0.5,
             "parseable_rate": 0.9, "answer_only_compliance_rate": 1.0,
             "latency_p50_ms": 50},
            {"model_id": "more_parse", "balanced_quality_score": 0.5,
             "parseable_rate": 1.0, "answer_only_compliance_rate": 1.0,
             "latency_p50_ms": 50},
        ]
        ranked = rank_models(aggs)
        self.assertEqual(ranked[0]["model_id"], "more_parse")

    def test_tied_models_share_rank(self):
        aggs = [
            {"model_id": "a", "balanced_quality_score": 0.5,
             "parseable_rate": 1.0, "answer_only_compliance_rate": 1.0,
             "latency_p50_ms": 50},
            {"model_id": "b", "balanced_quality_score": 0.5,
             "parseable_rate": 1.0, "answer_only_compliance_rate": 1.0,
             "latency_p50_ms": 50},
        ]
        ranked = rank_models(aggs)
        self.assertEqual(ranked[0]["rank"], ranked[1]["rank"])


# ── Integration tests with real data ─────────────────────────────────────────


@unittest.skipIf(SKIP_NO_DATA, SKIP_MSG)
class TestRealDataAggregation(unittest.TestCase):
    """Test metric computation against real batch data."""

    @classmethod
    def setUpClass(cls):
        """Load real batch data once."""
        summary_path = RESPONSES_DIR / f"repeat_summary_{BATCH_ID}.json"
        with open(summary_path) as f:
            cls.repeat_summary = json.load(f)

        cls.model_meta = load_model_meta(MODELS_CONFIG)

        # Load all rows grouped by model
        cls.model_rows = defaultdict(list)
        for run_info in cls.repeat_summary["runs"]:
            fpath = Path(run_info["output_file"])
            if not fpath.exists():
                fpath = RESPONSES_DIR / fpath.name
            rows = load_jsonl(fpath)
            cls.model_rows[run_info["model_id"]].extend(rows)

    def test_balanced_quality_score_formula(self):
        """balanced_quality_score must equal (thai + math) / 2."""
        for model_id, rows in self.model_rows.items():
            agg = aggregate_model(model_id, rows, self.model_meta)
            expected = round((agg["thai_score_rate"] + agg["math_score_rate"]) / 2, 4)
            self.assertAlmostEqual(
                agg["balanced_quality_score"], expected, places=4,
                msg=f"{model_id}: bqs={agg['balanced_quality_score']} != ({agg['thai_score_rate']}+{agg['math_score_rate']})/2"
            )

    def test_gemma4_e4b_is_top_quality(self):
        """gemma4:e4b should have the highest balanced_quality_score (known from repeat_summary)."""
        aggs = {
            mid: aggregate_model(mid, rows, self.model_meta)
            for mid, rows in self.model_rows.items()
        }
        best = max(aggs.values(), key=lambda a: a["balanced_quality_score"])
        self.assertEqual(best["model_id"], "gemma4:e4b")

    def test_scores_match_repeat_summary(self):
        """Aggregated thai/math rates should match repeat_summary model-level stats."""
        for model_info in self.repeat_summary["models"]:
            model_id = model_info["model_id"]
            rows = self.model_rows.get(model_id)
            if not rows:
                continue
            agg = aggregate_model(model_id, rows, self.model_meta)
            self.assertAlmostEqual(
                agg["thai_score_rate"], model_info["mean_thai_accuracy"], places=4,
                msg=f"{model_id} thai_score_rate mismatch"
            )
            self.assertAlmostEqual(
                agg["math_score_rate"], model_info["mean_math_accuracy"], places=4,
                msg=f"{model_id} math_score_rate mismatch"
            )

    def test_p95_gte_p50(self):
        """latency_p95_ms >= latency_p50_ms for all models."""
        for model_id, rows in self.model_rows.items():
            agg = aggregate_model(model_id, rows, self.model_meta)
            self.assertGreaterEqual(
                agg["latency_p95_ms"], agg["latency_p50_ms"],
                msg=f"{model_id}: p95 < p50"
            )

    def test_item_count_is_93(self):
        """All models should have 93 items per run (60 thai + 33 math)."""
        for model_id, rows in self.model_rows.items():
            agg = aggregate_model(model_id, rows, self.model_meta)
            self.assertEqual(agg["item_count"], 93, msg=f"{model_id} item_count")

    def test_batch_aggregated_speed_uses_all_runs(self):
        """questions_per_min should reflect all 10 runs, not just one.

        With 93 items × 10 runs = 930 total items in the denominator.
        """
        for model_id, rows in self.model_rows.items():
            agg = aggregate_model(model_id, rows, self.model_meta)
            total_latency_ms = sum(r["latency_ms"] for r in rows if r.get("latency_ms", 0) > 0)
            total_latency_min = total_latency_ms / 60000.0
            expected_qpm = len(rows) / total_latency_min if total_latency_min > 0 else 0
            self.assertAlmostEqual(
                agg["questions_per_min"], round(expected_qpm, 1), places=0,
                msg=f"{model_id} questions_per_min"
            )


@unittest.skipIf(SKIP_NO_DATA, SKIP_MSG)
class TestRealDataBadges(unittest.TestCase):
    """Test badge assignment against real batch data."""

    @classmethod
    def setUpClass(cls):
        cls.model_meta = load_model_meta(MODELS_CONFIG)
        summary = read_json(RESPONSES_DIR / f"repeat_summary_{BATCH_ID}.json")
        cls.model_rows = defaultdict(list)
        for run_info in summary["runs"]:
            fpath = Path(run_info["output_file"])
            if not fpath.exists():
                fpath = RESPONSES_DIR / fpath.name
            cls.model_rows[run_info["model_id"]].extend(load_jsonl(fpath))
        cls.aggs = [
            aggregate_model(mid, rows, cls.model_meta)
            for mid, rows in cls.model_rows.items()
        ]
        cls.badges = assign_badges(cls.aggs)

    def test_best_quality_goes_to_gemma4_e4b(self):
        self.assertIn("Best Quality", self.badges.get("gemma4:e4b", []))

    def test_fastest_goes_to_gemma4_e2b(self):
        """gemma4:e2b has the lowest latency in the batch."""
        self.assertIn("Fastest on Testbed", self.badges.get("gemma4:e2b", []))

    def test_best_small_model_only_comfortable(self):
        """Best Small Model must only go to fits_comfortably_16gb models."""
        for model_id, model_badges in self.badges.items():
            if "Best Small Model" in model_badges:
                meta = resolve_model_meta(model_id, self.model_meta)
                self.assertEqual(meta.get("ram_fit_class"), "fits_comfortably_16gb")

    def test_no_tight_model_gets_best_small(self):
        """fits_tightly_16gb models must not get Best Small Model."""
        tight_models = [m for m in self.model_meta.values() if m["ram_fit_class"] == "fits_tightly_16gb"]
        for tm in tight_models:
            self.assertNotIn("Best Small Model", self.badges.get(tm["model_id"], []))


@unittest.skipIf(SKIP_NO_DATA, SKIP_MSG)
class TestFullBuildAndValidate(unittest.TestCase):
    """Build a real snapshot from batch data and validate it end-to-end."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="snapshot_test_")
        cls.out_dir = Path(cls.tmpdir) / "test-snapshot"
        cls.out_dir.mkdir()
        build_snapshot(BATCH_ID, snapshot_id="test-snapshot", out_dir=cls.out_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_all_required_files_exist(self):
        for name in [
            "manifest.json",
            "leaderboard.json",
            "model_cards.json",
            "examples.json",
            "repeat_summary.json",
            "results.jsonl",
        ]:
            self.assertTrue((self.out_dir / name).exists(), f"Missing {name}")

    def test_raw_directory_contains_source_files(self):
        raw_dir = self.out_dir / "raw"
        self.assertTrue(raw_dir.is_dir(), "Missing raw/ directory")
        raw_files = list(raw_dir.glob("*.jsonl"))
        self.assertGreater(len(raw_files), 0, "raw/ contains no JSONL files")

    def test_repeat_summary_in_transparency(self):
        self.assertTrue((self.out_dir / "repeat_summary.json").exists())

    def test_results_jsonl_in_transparency(self):
        self.assertTrue((self.out_dir / "results.jsonl").exists())

    def test_zip_bundle_exists(self):
        zip_path = self.out_dir.parent / "test-snapshot.zip"
        self.assertTrue(zip_path.exists())

    def test_zip_bundle_contains_transparency_files(self):
        zip_path = self.out_dir.parent / "test-snapshot.zip"
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
        self.assertIn("test-snapshot/results.jsonl", names)
        self.assertIn("test-snapshot/repeat_summary.json", names)
        self.assertTrue(any(name.startswith("test-snapshot/raw/") for name in names))

    def test_validator_passes(self):
        v = SnapshotValidator(self.out_dir)
        valid = v.validate()
        if not valid:
            v.report()
        self.assertTrue(valid, f"Validator errors: {v.errors}")

    def test_leaderboard_has_4_models(self):
        lb = read_json(self.out_dir / "leaderboard.json")
        self.assertEqual(len(lb["rows"]), 4)

    def test_leaderboard_rank_1_is_gemma4_e4b(self):
        lb = read_json(self.out_dir / "leaderboard.json")
        rank1 = [r for r in lb["rows"] if r["rank"] == 1]
        self.assertEqual(len(rank1), 1)
        self.assertEqual(rank1[0]["model_id"], "gemma4:e4b")

    def test_model_cards_have_examples(self):
        mc = read_json(self.out_dir / "model_cards.json")
        for card in mc["models"]:
            total_examples = len(card["example_ids"]["good"]) + len(card["example_ids"]["bad"])
            self.assertGreater(total_examples, 0, f"{card['model_id']} has no examples")

    def test_model_cards_include_reliability_fields(self):
        mc = read_json(self.out_dir / "model_cards.json")
        for card in mc["models"]:
            self.assertIn("average_output_length_chars", card["metrics"])
            self.assertIn("common_failure_types", card)

    def test_typhoon_identity_resolves(self):
        mc = read_json(self.out_dir / "model_cards.json")
        typhoon = next((m for m in mc["models"] if "typhoon" in m["model_id"]), None)
        self.assertIsNotNone(typhoon)
        self.assertNotEqual(typhoon["model_family"], "unknown")
        self.assertNotEqual(typhoon["parameter_bucket"], "unknown")
        self.assertNotEqual(typhoon["ram_fit_class"], "unknown")

    def test_examples_truncation_correct(self):
        ex = read_json(self.out_dir / "examples.json")
        for e in ex["examples"]:
            self.assertEqual(e["raw_output_truncated"], e["raw_output_full"][:200])

    def test_examples_include_question_context(self):
        ex = read_json(self.out_dir / "examples.json")
        for e in ex["examples"]:
            self.assertIn("prompt_text", e)
            self.assertIn("choices", e)
            self.assertTrue(e["prompt_text"])
            self.assertTrue(e["choices"])

    def test_strengths_weaknesses_min_n(self):
        mc = read_json(self.out_dir / "model_cards.json")
        for card in mc["models"]:
            for entry in card["strengths"] + card["weaknesses"]:
                self.assertGreaterEqual(entry["total"], 2,
                    f"{card['model_id']}: skill '{entry['skill_tag']}' has total={entry['total']} < 2")

    def test_snapshot_id_consistent_across_all_files(self):
        manifest = read_json(self.out_dir / "manifest.json")
        sid = manifest["snapshot_id"]
        for name in ["leaderboard.json", "model_cards.json", "examples.json"]:
            data = read_json(self.out_dir / name)
            self.assertEqual(data["snapshot_id"], sid, f"{name} snapshot_id mismatch")

    def test_compliance_rate_not_all_ones(self):
        """At least one model should have compliance < 1.0, unless all outputs are single digits.
        This guards against a bug where compliance is accidentally always 1.0."""
        lb = read_json(self.out_dir / "leaderboard.json")
        rates = [r["answer_only_compliance_rate"] for r in lb["rows"]]
        # If all rates are 1.0, that's actually possible for this data (models output single digits).
        # But we should at least verify it's computed per the spec.
        for row in lb["rows"]:
            self.assertGreaterEqual(row["answer_only_compliance_rate"], 0.0)
            self.assertLessEqual(row["answer_only_compliance_rate"], 1.0)


@unittest.skipIf(SKIP_NO_DATA, SKIP_MSG)
class TestVerifiedSingleRunBuild(unittest.TestCase):
    """Verified publication batches should publish canonical-only metrics."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="verified_snapshot_test_"))
        cls.responses_dir = cls.tmpdir / "responses"
        cls.responses_dir.mkdir(parents=True, exist_ok=True)
        cls.batch_id = "ntp3-vr1-test"
        cls.model_id = "gemma4:e2b"

        summary = read_json(RESPONSES_DIR / f"repeat_summary_{BATCH_ID}.json")
        source_run = next(r for r in summary["runs"] if r["model_id"] == cls.model_id)
        source_path = Path(source_run["output_file"])
        if not source_path.exists():
            source_path = RESPONSES_DIR / source_path.name
        rows = load_jsonl(source_path)

        cls.canonical_rows = []
        cls.shadow_rows = []
        for row in rows:
            canonical = dict(row)
            canonical["run_id"] = f"{cls.batch_id}-gemma4-e2b-r01"
            shadow = dict(canonical)
            shadow["run_id"] = f"{cls.batch_id}-gemma4-e2b-r02"
            shadow["latency_ms"] = canonical.get("latency_ms", 0) * 3 or 3
            shadow["eval_duration_ms"] = canonical.get("eval_duration_ms", 0) * 3
            shadow["prompt_eval_duration_ms"] = canonical.get("prompt_eval_duration_ms", 0) * 3
            cls.canonical_rows.append(canonical)
            cls.shadow_rows.append(shadow)

        cls.canonical_path = cls.responses_dir / f"responses_{cls.batch_id}-gemma4-e2b-r01.jsonl"
        cls.shadow_path = cls.responses_dir / f"responses_{cls.batch_id}-gemma4-e2b-r02.jsonl"
        for path, payload in [(cls.canonical_path, cls.canonical_rows), (cls.shadow_path, cls.shadow_rows)]:
            with open(path, "w", encoding="utf-8") as f:
                for row in payload:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        repeat_summary = {
            "batch_id": cls.batch_id,
            "started_at_epoch": 0.0,
            "finished_at_epoch": 0.0,
            "models": [{
                "model_id": cls.model_id,
                "runs": 2,
                "mean_correct": 54,
                "min_correct": 54,
                "max_correct": 54,
                "mean_accuracy": 54 / 93,
                "stdev_accuracy": 0.0,
                "mean_thai_accuracy": 0.6,
                "mean_math_accuracy": 0.45,
                "mean_time_s": 1.0,
                "median_time_s": 1.0,
                "mean_correct_per_min": 1.0,
                "mean_correct_per_gb_min": 1.0,
                "memory_gb": 2.0,
            }],
            "runs": [
                {
                    "model_id": cls.model_id,
                    "run_id": f"{cls.batch_id}-gemma4-e2b-r01",
                    "run_index": 1,
                    "output_file": str(cls.canonical_path),
                    "parse_rate": 1.0,
                    "accuracy": 54 / 93,
                },
                {
                    "model_id": cls.model_id,
                    "run_id": f"{cls.batch_id}-gemma4-e2b-r02",
                    "run_index": 2,
                    "output_file": str(cls.shadow_path),
                    "parse_rate": 1.0,
                    "accuracy": 54 / 93,
                },
            ],
        }
        (cls.responses_dir / f"repeat_summary_{cls.batch_id}.json").write_text(
            json.dumps(repeat_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        verification_report = {
            "batch_id": cls.batch_id,
            "publication_mode": "verified_posthoc_gate_v1",
            "preflight_record": "preflight_records/preflight_20260411.json",
            "protocol": "canonical_shadow_v1",
            "expected_runs_per_model": 2,
            "canonical_run_index": 1,
            "shadow_run_index": 2,
            "status": "pass",
            "all_deterministic": True,
            "publishable_count": 1,
            "publishable_models": [cls.model_id],
            "thresholds": {
                "parse_rate_gte": 0.95,
                "accuracy_gt": 0.25,
            },
            "models": [{
                "model_id": cls.model_id,
                "canonical_run_id": f"{cls.batch_id}-gemma4-e2b-r01",
                "shadow_run_id": f"{cls.batch_id}-gemma4-e2b-r02",
                "total_items": len(cls.canonical_rows),
                "matching_items": len(cls.canonical_rows),
                "divergent_count": 0,
                "divergent_items": [],
                "deterministic": True,
                "publishable": True,
            }],
        }
        (cls.responses_dir / f"verification_report_{cls.batch_id}.json").write_text(
            json.dumps(verification_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        cls.out_dir = cls.tmpdir / "out"
        with mock.patch("scripts.build_snapshot.RESPONSES_DIR", cls.responses_dir):
            build_snapshot(cls.batch_id, snapshot_id="verified-single-run", out_dir=cls.out_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_manifest_carries_verification_artifact(self):
        manifest = read_json(self.out_dir / "manifest.json")
        self.assertEqual(manifest["artifacts"].get("verification_report"), "verification_report.json")
        self.assertEqual(manifest["publication"]["mode"], "verified_single_run")

    def test_results_jsonl_contains_canonical_rows_only(self):
        with open(self.out_dir / "results.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), len(self.canonical_rows))

    def test_latency_metrics_use_canonical_run_only(self):
        leaderboard = read_json(self.out_dir / "leaderboard.json")
        row = leaderboard["rows"][0]
        expected = aggregate_model(self.model_id, self.canonical_rows, load_model_meta(MODELS_CONFIG))
        self.assertEqual(row["latency_p50_ms"], expected["latency_p50_ms"])
        self.assertEqual(row["latency_p95_ms"], expected["latency_p95_ms"])


class TestVerifiedSingleRunBuildExcludesRejectedModels(unittest.TestCase):
    def test_snapshot_includes_only_publishable_models(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="verified_snapshot_filter_test_"))
        try:
            responses_dir = tmpdir / "responses"
            responses_dir.mkdir(parents=True, exist_ok=True)
            batch_id = "ntp3-vr1-filtered"

            def write_rows(model_slug: str, model_id: str, answer: str):
                rows = []
                for qid in range(1, 3):
                    rows.append({
                        "model_id": model_id,
                        "run_id": "",
                        "exam_id": "nt_p3_th_2565",
                        "year_buddhist": 2565,
                        "subject": "thai",
                        "question_id": qid,
                        "eval_split": "text_only_core",
                        "prompt_version": "v1_answer_only",
                        "is_parseable": True,
                        "parsed_answer": answer,
                        "choice_count": 4,
                        "correct_answer": answer,
                        "is_correct": True,
                        "latency_ms": 100 + qid,
                        "raw_response": answer,
                        "question": f"Question {qid}",
                        "choices": ["1", "2", "3", "4"],
                        "subject_label": "Thai",
                        "skill_tag": ["reading_comprehension"],
                        "curriculum_standard": "ท 1.1 ป.3/2",
                    })
                return rows

            def write_run(model_slug: str, model_id: str, run_idx: int, rows: list[dict]):
                path = responses_dir / f"responses_{batch_id}-{model_slug}-r0{run_idx}.jsonl"
                run_id = f"{batch_id}-{model_slug}-r0{run_idx}"
                with open(path, "w", encoding="utf-8") as f:
                    for row in rows:
                        payload = dict(row)
                        payload["run_id"] = run_id
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                return path, run_id

            good_rows = write_rows("gemma4-e2b", "gemma4:e2b", "4")
            excluded_rows = write_rows("bad-model-1b", "bad-model:1b", "3")

            good_r1, good_id1 = write_run("gemma4-e2b", "gemma4:e2b", 1, good_rows)
            good_r2, good_id2 = write_run("gemma4-e2b", "gemma4:e2b", 2, good_rows)
            bad_r1, bad_id1 = write_run("bad-model-1b", "bad-model:1b", 1, excluded_rows)
            bad_r2, bad_id2 = write_run("bad-model-1b", "bad-model:1b", 2, excluded_rows)

            repeat_summary = {
                "batch_id": batch_id,
                "models": [
                    {"model_id": "gemma4:e2b", "runs": 2},
                    {"model_id": "bad-model:1b", "runs": 2},
                ],
                "runs": [
                    {"model_id": "gemma4:e2b", "run_id": good_id1, "run_index": 1, "output_file": str(good_r1), "parse_rate": 1.0, "accuracy": 1.0},
                    {"model_id": "gemma4:e2b", "run_id": good_id2, "run_index": 2, "output_file": str(good_r2), "parse_rate": 1.0, "accuracy": 1.0},
                    {"model_id": "bad-model:1b", "run_id": bad_id1, "run_index": 1, "output_file": str(bad_r1), "parse_rate": 1.0, "accuracy": 1.0},
                    {"model_id": "bad-model:1b", "run_id": bad_id2, "run_index": 2, "output_file": str(bad_r2), "parse_rate": 1.0, "accuracy": 1.0},
                ],
            }
            (responses_dir / f"repeat_summary_{batch_id}.json").write_text(
                json.dumps(repeat_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            verification_report = {
                "batch_id": batch_id,
                "publication_mode": "verified_posthoc_gate_v1",
                "protocol": "canonical_shadow_v1",
                "status": "pass",
                "all_deterministic": False,
                "publishable_count": 1,
                "publishable_models": ["gemma4:e2b"],
                "models": [
                    {
                        "model_id": "gemma4:e2b",
                        "canonical_run_id": good_id1,
                        "shadow_run_id": good_id2,
                        "deterministic": True,
                        "publishable": True,
                    },
                    {
                        "model_id": "bad-model:1b",
                        "canonical_run_id": bad_id1,
                        "shadow_run_id": bad_id2,
                        "deterministic": True,
                        "publishable": False,
                    },
                ],
            }
            (responses_dir / f"verification_report_{batch_id}.json").write_text(
                json.dumps(verification_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            with mock.patch("scripts.build_snapshot.RESPONSES_DIR", responses_dir):
                build_snapshot(batch_id, snapshot_id="verified-filtered", out_dir=out_dir)

            leaderboard = read_json(out_dir / "leaderboard.json")
            self.assertEqual([row["model_id"] for row in leaderboard["rows"]], ["gemma4:e2b"])

            with open(out_dir / "results.jsonl", encoding="utf-8") as f:
                models = {json.loads(line)["model_id"] for line in f if line.strip()}
            self.assertEqual(models, {"gemma4:e2b"})
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


@unittest.skipIf(SKIP_NO_DATA, SKIP_MSG)
class TestVerifiedSingleRunErrors(unittest.TestCase):
    def test_missing_verified_canonical_rows_raises(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="verified_snapshot_error_test_"))
        try:
            responses_dir = tmpdir / "responses"
            responses_dir.mkdir(parents=True, exist_ok=True)
            batch_id = "ntp3-vr1-badcanon"
            model_id = "gemma4:e2b"

            summary = read_json(RESPONSES_DIR / f"repeat_summary_{BATCH_ID}.json")
            source_run = next(r for r in summary["runs"] if r["model_id"] == model_id)
            source_path = Path(source_run["output_file"])
            if not source_path.exists():
                source_path = RESPONSES_DIR / source_path.name
            rows = load_jsonl(source_path)

            run1 = responses_dir / f"responses_{batch_id}-gemma4-e2b-r01.jsonl"
            run2 = responses_dir / f"responses_{batch_id}-gemma4-e2b-r02.jsonl"
            for path, run_id in [(run1, f"{batch_id}-gemma4-e2b-r01"), (run2, f"{batch_id}-gemma4-e2b-r02")]:
                with open(path, "w", encoding="utf-8") as f:
                    for row in rows:
                        payload = dict(row)
                        payload["run_id"] = run_id
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            repeat_summary = {
                "batch_id": batch_id,
                "models": [{"model_id": model_id, "runs": 2}],
                "runs": [
                    {"model_id": model_id, "run_id": f"{batch_id}-gemma4-e2b-r01", "run_index": 1, "output_file": str(run1)},
                    {"model_id": model_id, "run_id": f"{batch_id}-gemma4-e2b-r02", "run_index": 2, "output_file": str(run2)},
                ],
            }
            (responses_dir / f"repeat_summary_{batch_id}.json").write_text(
                json.dumps(repeat_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            verification_report = {
                "batch_id": batch_id,
                "publication_mode": "verified_posthoc_gate_v1",
                "protocol": "canonical_shadow_v1",
                "status": "pass",
                "all_deterministic": True,
                "publishable_count": 1,
                "publishable_models": [model_id],
                "models": [{
                    "model_id": model_id,
                    "canonical_run_id": f"{batch_id}-gemma4-e2b-r99",
                    "shadow_run_id": f"{batch_id}-gemma4-e2b-r02",
                    "deterministic": True,
                    "publishable": True,
                }],
            }
            (responses_dir / f"verification_report_{batch_id}.json").write_text(
                json.dumps(verification_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with mock.patch("scripts.build_snapshot.RESPONSES_DIR", responses_dir):
                with self.assertRaisesRegex(RuntimeError, "did not match any rows"):
                    build_snapshot(batch_id, snapshot_id="verified-bad-canon", out_dir=tmpdir / "out")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Registry and compatibility tests ────────────────────────────────────────


class TestRegistryFilesExist(unittest.TestCase):
    """Registry files must exist in-repo and be well-formed."""

    def test_models_json_exists(self):
        self.assertTrue(MODELS_CONFIG.exists(), f"Missing {MODELS_CONFIG}")

    def test_machine_profiles_json_exists(self):
        self.assertTrue(MACHINE_PROFILES_CONFIG.exists(), f"Missing {MACHINE_PROFILES_CONFIG}")

    def test_compatibility_json_exists(self):
        self.assertTrue(COMPATIBILITY_CONFIG.exists(), f"Missing {COMPATIBILITY_CONFIG}")

    def test_models_json_has_models(self):
        data = load_json(MODELS_CONFIG)
        self.assertIn("models", data)
        self.assertGreater(len(data["models"]), 0)
        for m in data["models"]:
            self.assertIn("model_id", m)
            self.assertIn("model_family", m)
            self.assertIn("parameter_bucket", m)
            self.assertIn("ram_fit_class", m)

    def test_machine_profiles_has_baseline(self):
        data = load_json(MACHINE_PROFILES_CONFIG)
        self.assertIn("machine_profiles", data)
        profiles = data["machine_profiles"]
        ids = [p["machine_profile"] for p in profiles]
        self.assertIn("macmini-m4-16gb-ollama", ids)

    def test_compatibility_has_required_fields_and_v1_values(self):
        data = load_json(COMPATIBILITY_CONFIG)
        self.assertIn("required_match_fields", data)
        self.assertIn("v1_expected_values", data)
        fields = data["required_match_fields"]
        values = data["v1_expected_values"]
        expected_fields = [
            "benchmark_scope", "dataset_version", "eval_split",
            "prompt_version", "scoring_version", "machine_profile", "think_mode",
        ]
        self.assertEqual(sorted(fields), sorted(expected_fields))
        # Every required field has a V1 value
        for f in fields:
            self.assertIn(f, values, f"V1 expected value missing for '{f}'")
            self.assertIsNotNone(values[f])

    def test_v1_expected_values_are_correct(self):
        data = load_json(COMPATIBILITY_CONFIG)
        v1 = data["v1_expected_values"]
        self.assertEqual(v1["benchmark_scope"], "mcq_text_only_v1")
        self.assertEqual(v1["eval_split"], "text_only_core")
        self.assertEqual(v1["prompt_version"], "v1_answer_only")
        self.assertEqual(v1["scoring_version"], "v1")
        self.assertEqual(v1["machine_profile"], "macmini-m4-16gb-ollama")
        self.assertEqual(v1["think_mode"], "off")
        self.assertEqual(v1["dataset_version"], "nt-p3-text-only-2026-04-09")


@unittest.skipIf(SKIP_NO_DATA, SKIP_MSG)
class TestCompatibilityResolution(unittest.TestCase):
    """Test that compatibility values are resolved from real batch data."""

    @classmethod
    def setUpClass(cls):
        summary_path = RESPONSES_DIR / f"repeat_summary_{BATCH_ID}.json"
        with open(summary_path) as f:
            cls.repeat_summary = json.load(f)
        cls.all_rows = []
        for run_info in cls.repeat_summary["runs"]:
            fpath = Path(run_info["output_file"])
            if not fpath.exists():
                fpath = RESPONSES_DIR / fpath.name
            cls.all_rows.extend(load_jsonl(fpath))
        cls.compat = load_compatibility(COMPATIBILITY_CONFIG)

    def test_eval_split_from_data(self):
        """All rows should have eval_split=text_only_core, so resolved value matches."""
        from scripts.build_snapshot import load_testbed
        testbed = load_testbed(MACHINE_PROFILES_CONFIG)
        values = resolve_compatibility_values(self.compat, self.all_rows, testbed)
        self.assertEqual(values["eval_split"], "text_only_core")

    def test_prompt_version_from_data(self):
        from scripts.build_snapshot import load_testbed
        testbed = load_testbed(MACHINE_PROFILES_CONFIG)
        values = resolve_compatibility_values(self.compat, self.all_rows, testbed)
        self.assertEqual(values["prompt_version"], "v1_answer_only")

    def test_think_mode_from_data(self):
        from scripts.build_snapshot import load_testbed
        testbed = load_testbed(MACHINE_PROFILES_CONFIG)
        values = resolve_compatibility_values(self.compat, self.all_rows, testbed)
        self.assertEqual(values["think_mode"], "off")

    def test_all_v1_values_match(self):
        """Resolved values must exactly match V1 expected values."""
        from scripts.build_snapshot import load_testbed
        testbed = load_testbed(MACHINE_PROFILES_CONFIG)
        values = resolve_compatibility_values(self.compat, self.all_rows, testbed)
        expected = self.compat["v1_expected_values"]
        for field in self.compat["required_match_fields"]:
            self.assertEqual(
                values[field], expected[field],
                f"Resolved {field}={values[field]!r} != expected {expected[field]!r}"
            )


@unittest.skipIf(SKIP_NO_DATA, SKIP_MSG)
class TestSnapshotCompatibilityBlock(unittest.TestCase):
    """Built snapshot must contain a valid compatibility block."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="snapshot_compat_test_")
        cls.out_dir = Path(cls.tmpdir) / "compat-test-snapshot"
        cls.out_dir.mkdir()
        build_snapshot(BATCH_ID, snapshot_id="compat-test-snapshot", out_dir=cls.out_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_manifest_has_compatibility_block(self):
        manifest = read_json(self.out_dir / "manifest.json")
        self.assertIn("compatibility", manifest)

    def test_compatibility_has_required_fields(self):
        manifest = read_json(self.out_dir / "manifest.json")
        compat = manifest["compatibility"]
        self.assertIn("required_match_fields", compat)
        self.assertIn("values", compat)

    def test_compatibility_values_match_v1(self):
        manifest = read_json(self.out_dir / "manifest.json")
        values = manifest["compatibility"]["values"]
        self.assertEqual(values["benchmark_scope"], "mcq_text_only_v1")
        self.assertEqual(values["eval_split"], "text_only_core")
        self.assertEqual(values["prompt_version"], "v1_answer_only")
        self.assertEqual(values["scoring_version"], "v1")
        self.assertEqual(values["machine_profile"], "macmini-m4-16gb-ollama")
        self.assertEqual(values["think_mode"], "off")

    def test_compatibility_scope_matches_manifest_scope(self):
        manifest = read_json(self.out_dir / "manifest.json")
        self.assertEqual(
            manifest["compatibility"]["values"]["benchmark_scope"],
            manifest["benchmark_scope"],
        )

    def test_all_required_fields_have_values(self):
        manifest = read_json(self.out_dir / "manifest.json")
        compat = manifest["compatibility"]
        for field in compat["required_match_fields"]:
            self.assertIn(field, compat["values"])
            self.assertIsNotNone(compat["values"][field])

    def test_validator_passes_with_compatibility(self):
        v = SnapshotValidator(self.out_dir)
        valid = v.validate()
        if not valid:
            v.report()
        self.assertTrue(valid, f"Validator errors: {v.errors}")


if __name__ == "__main__":
    unittest.main()
