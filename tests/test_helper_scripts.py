#!/usr/bin/env python3
"""
Tests for helper scripts: preflight, check_batch_completeness, screening_gate,
verify_batch_inputs, prepare_publication_batch.

Unit tests use synthetic data. Integration tests require real batch data
(mini-r10-20260409) and are skipped when unavailable.

Run: python -m pytest tests/test_helper_scripts.py -v
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.preflight import (
    verify_datasets,
    resolve_model_name,
    run_preflight,
)
from scripts.check_batch_completeness import (
    check_batch_completeness,
    load_jsonl_safe,
    slugify,
    REQUIRED_ROW_FIELDS,
)
from scripts.screening_gate import (
    apply_screening_gate,
    classify_tier,
    compute_mean_parse_rate,
    format_appendix_markdown,
    resolve_digest_match,
    MIN_PARSE_RATE,
    MIN_MEAN_ACCURACY,
)
from scripts.verify_batch_inputs import (
    verify_batch_inputs,
    load_compatibility,
    BATCH_ID_PATTERN,
)
from scripts.prepare_publication_batch import (
    prepare_publication_batch,
    derive_pub_batch_id,
    write_env_file,
)

RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"
BATCH_ID = "mini-r10-20260409"


# ── Preflight unit tests ───────────────────────────────────────────────────


class TestVerifyDatasets(unittest.TestCase):
    """Dataset verification should work against real dataset files."""

    def test_datasets_are_valid(self):
        counts = verify_datasets()
        self.assertIn("thai", counts)
        self.assertIn("math", counts)
        self.assertEqual(counts["thai"]["text_only_core"], 60)
        self.assertEqual(counts["math"]["text_only_core"], 33)

    def test_datasets_have_expected_splits(self):
        counts = verify_datasets()
        # Thai has text_only_core and possibly others
        self.assertGreater(sum(counts["thai"].values()), 0)
        self.assertGreater(sum(counts["math"].values()), 0)


class TestResolveModelName(unittest.TestCase):
    def test_exact_match(self):
        available = {"gemma4:e2b": {"digest": "abc"}}
        self.assertEqual(resolve_model_name("gemma4:e2b", available), "gemma4:e2b")

    def test_latest_suffix_added(self):
        available = {"gemma4:latest": {"digest": "abc"}}
        self.assertEqual(resolve_model_name("gemma4", available), "gemma4:latest")

    def test_latest_suffix_stripped(self):
        available = {"gemma4": {"digest": "abc"}}
        self.assertEqual(resolve_model_name("gemma4:latest", available), "gemma4")

    def test_not_found(self):
        available = {"gemma4:e2b": {"digest": "abc"}}
        self.assertIsNone(resolve_model_name("qwen3:8b", available))


class TestPreflightOffline(unittest.TestCase):
    """Test preflight in offline mode (no Ollama needed)."""

    def test_offline_datasets_pass(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            out = tmpdir / "preflight.json"
            record = run_preflight(
                ["gemma4:e2b"],
                offline=True,
                output_path=out,
            )
            self.assertEqual(record["status"], "pass")
            self.assertIsNone(record["ollama_healthy"])
            self.assertTrue(out.exists())

            # Verify JSON artifact is valid
            with open(out) as f:
                data = json.load(f)
            self.assertEqual(data["status"], "pass")
            self.assertIn("model_inventory", data)
            self.assertIn("dataset_counts", data)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_offline_artifact_schema(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            out = tmpdir / "preflight.json"
            record = run_preflight(
                ["model-a", "model-b"],
                offline=True,
                output_path=out,
            )
            required_keys = {
                "timestamp", "requested_models", "ollama_healthy",
                "dataset_counts", "model_inventory", "errors", "warnings", "status",
            }
            self.assertTrue(required_keys.issubset(record.keys()))
            self.assertEqual(len(record["model_inventory"]), 2)
            for entry in record["model_inventory"]:
                self.assertIn("model_id", entry)
                self.assertIn("available", entry)
                self.assertIn("digest", entry)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── check_batch_completeness unit tests ────────────────────────────────────


class TestSlugify(unittest.TestCase):
    def test_colon_to_dash(self):
        self.assertEqual(slugify("gemma4:e2b"), "gemma4-e2b")

    def test_slash_to_dash(self):
        self.assertEqual(slugify("scb10x/typhoon2.1-gemma3-4b:latest"), "scb10x-typhoon2.1-gemma3-4b-latest")


class TestLoadJsonlSafe(unittest.TestCase):
    def test_valid_jsonl(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            fpath = tmpdir / "test.jsonl"
            fpath.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
            rows, errors = load_jsonl_safe(fpath)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(errors), 0)
        finally:
            shutil.rmtree(tmpdir)

    def test_malformed_line(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            fpath = tmpdir / "bad.jsonl"
            fpath.write_text('{"a": 1}\nNOT JSON\n{"b": 2}\n', encoding="utf-8")
            rows, errors = load_jsonl_safe(fpath)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(errors), 1)
            self.assertIn("malformed JSON", errors[0])
        finally:
            shutil.rmtree(tmpdir)

    def test_empty_lines_skipped(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            fpath = tmpdir / "sparse.jsonl"
            fpath.write_text('{"a": 1}\n\n\n{"b": 2}\n', encoding="utf-8")
            rows, errors = load_jsonl_safe(fpath)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(errors), 0)
        finally:
            shutil.rmtree(tmpdir)


class TestBatchCompletenessSynthetic(unittest.TestCase):
    """Test completeness checker with synthetic data."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp())
        cls.batch_id = "synth-r2"

        # Create 2 models × 2 runs = 4 JSONL files with 3 items each
        base_row = {
            "model_id": "", "run_id": "", "question_id": 1,
            "is_correct": True, "is_parseable": True,
            "parsed_answer": "1", "correct_answer": "1",
            "raw_output": "1", "subject": "thai",
        }
        for model in ["model-a", "model-b"]:
            slug = slugify(model)
            for run_num in range(1, 3):
                run_id = f"synth-r2-{slug}-r{run_num:02d}"
                rows = []
                for q in range(1, 4):
                    row = dict(base_row, model_id=model, run_id=run_id, question_id=q)
                    rows.append(row)
                fpath = cls.tmpdir / f"responses_{run_id}_20260410_120000.jsonl"
                with open(fpath, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")

        # Create matching repeat_summary
        summary = {
            "batch_id": "synth-r2",
            "started_at_epoch": 0.0,
            "finished_at_epoch": 0.0,
            "models": [
                {"model_id": "model-a", "runs": 2, "memory_gb": 1.0,
                 "mean_accuracy": 1.0, "mean_thai_accuracy": 1.0, "mean_math_accuracy": 0.0},
                {"model_id": "model-b", "runs": 2, "memory_gb": 1.0,
                 "mean_accuracy": 1.0, "mean_thai_accuracy": 1.0, "mean_math_accuracy": 0.0},
            ],
            "runs": [
                {"model_id": "model-a", "run_id": "synth-r2-model-a-r01", "parse_rate": 1.0,
                 "accuracy": 1.0, "output_file": "x", "run_index": 1},
                {"model_id": "model-a", "run_id": "synth-r2-model-a-r02", "parse_rate": 1.0,
                 "accuracy": 1.0, "output_file": "x", "run_index": 2},
                {"model_id": "model-b", "run_id": "synth-r2-model-b-r01", "parse_rate": 1.0,
                 "accuracy": 1.0, "output_file": "x", "run_index": 1},
                {"model_id": "model-b", "run_id": "synth-r2-model-b-r02", "parse_rate": 1.0,
                 "accuracy": 1.0, "output_file": "x", "run_index": 2},
            ],
        }
        with open(cls.tmpdir / "repeat_summary_synth-r2.json", "w") as f:
            json.dump(summary, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_complete_batch_passes(self):
        result = check_batch_completeness(
            "synth-r2",
            models=["model-a", "model-b"],
            runs_per_model=2,
            responses_dir=self.tmpdir,
            expected_items=3,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["total_files"], 4)
        self.assertEqual(len(result["errors"]), 0)

    def test_missing_file_detected(self):
        result = check_batch_completeness(
            "synth-r2",
            models=["model-a", "model-b", "model-c"],
            runs_per_model=2,
            responses_dir=self.tmpdir,
            expected_items=3,
        )
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("Expected 6 JSONL files" in e for e in result["errors"]))

    def test_wrong_item_count_detected(self):
        result = check_batch_completeness(
            "synth-r2",
            models=["model-a", "model-b"],
            runs_per_model=2,
            responses_dir=self.tmpdir,
            expected_items=10,  # wrong expectation
        )
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("expected 10 rows" in e for e in result["errors"]))

    def test_infer_from_summary(self):
        """Should work without explicit models/runs_per_model."""
        result = check_batch_completeness(
            "synth-r2",
            responses_dir=self.tmpdir,
            expected_items=3,
        )
        self.assertEqual(result["status"], "pass")


class TestBatchCompletenessTruncated(unittest.TestCase):
    """Test detection of truncated/malformed JSONL files."""

    def test_malformed_jsonl_detected(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            fpath = tmpdir / "responses_bad-r1-model-a-r01_20260410.jsonl"
            fpath.write_text('{"model_id":"model-a","run_id":"bad-r1-model-a-r01"}\nBROKEN\n')
            result = check_batch_completeness(
                "bad-r1",
                models=["model-a"],
                runs_per_model=1,
                responses_dir=tmpdir,
                expected_items=1,
            )
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("malformed JSON" in e for e in result["errors"]))
        finally:
            shutil.rmtree(tmpdir)

    def test_empty_file_detected(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            fpath = tmpdir / "responses_empty-r1-model-a-r01_20260410.jsonl"
            fpath.write_text("")
            result = check_batch_completeness(
                "empty-r1",
                models=["model-a"],
                runs_per_model=1,
                responses_dir=tmpdir,
                expected_items=93,
            )
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("expected 93 rows, got 0" in e for e in result["errors"]))
        finally:
            shutil.rmtree(tmpdir)


# ── screening_gate unit tests ─────────────────────────────────────────────


class TestClassifyTier(unittest.TestCase):
    def test_core_international(self):
        self.assertEqual(classify_tier("gemma4:e2b"), "core-international")
        self.assertEqual(classify_tier("qwen3.5:9b"), "core-international")
        self.assertEqual(classify_tier("llama3.1:8b"), "core-international")
        self.assertEqual(classify_tier("phi4-mini:3.8b"), "core-international")

    def test_extended_international(self):
        self.assertEqual(classify_tier("phi3:3.8b"), "extended-international")
        self.assertEqual(classify_tier("falcon3:7b"), "extended-international")
        self.assertEqual(classify_tier("olmo2:7b"), "extended-international")

    def test_thai_specific(self):
        self.assertEqual(classify_tier("scb10x/typhoon2.1-gemma3-4b:latest"), "thai-specific")
        self.assertEqual(classify_tier("scb10x/typhoon2.5-qwen3-4b"), "thai-specific")

    def test_unknown(self):
        self.assertEqual(classify_tier("totally-new-model:1b"), "unknown")


class TestComputeMeanParseRate(unittest.TestCase):
    def test_single_model(self):
        runs = [
            {"model_id": "a", "parse_rate": 0.9},
            {"model_id": "a", "parse_rate": 1.0},
            {"model_id": "b", "parse_rate": 0.5},
        ]
        self.assertAlmostEqual(compute_mean_parse_rate(runs, "a"), 0.95)
        self.assertAlmostEqual(compute_mean_parse_rate(runs, "b"), 0.5)

    def test_missing_model(self):
        self.assertEqual(compute_mean_parse_rate([], "x"), 0.0)


class TestScreeningGateSynthetic(unittest.TestCase):
    """Test screening gate logic with synthetic repeat_summary."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp())
        # Create a summary with 3 models: one passes, one fails parse, one fails accuracy
        summary = {
            "batch_id": "screen-test",
            "started_at_epoch": 0.0,
            "finished_at_epoch": 0.0,
            "models": [
                {
                    "model_id": "good-model:7b",
                    "runs": 3,
                    "memory_gb": 5.0,
                    "mean_accuracy": 0.60,
                    "mean_thai_accuracy": 0.65,
                    "mean_math_accuracy": 0.50,
                    "mean_correct": 56,
                },
                {
                    "model_id": "bad-parse:3b",
                    "runs": 3,
                    "memory_gb": 3.0,
                    "mean_accuracy": 0.40,
                    "mean_thai_accuracy": 0.45,
                    "mean_math_accuracy": 0.30,
                    "mean_correct": 37,
                },
                {
                    "model_id": "bad-accuracy:1b",
                    "runs": 3,
                    "memory_gb": 1.0,
                    "mean_accuracy": 0.20,
                    "mean_thai_accuracy": 0.22,
                    "mean_math_accuracy": 0.15,
                    "mean_correct": 19,
                },
            ],
            "runs": [
                # good-model: 3 runs, high parse rate
                {"model_id": "good-model:7b", "run_id": "screen-test-good-model-7b-r01",
                 "parse_rate": 1.0, "accuracy": 0.60},
                {"model_id": "good-model:7b", "run_id": "screen-test-good-model-7b-r02",
                 "parse_rate": 0.98, "accuracy": 0.59},
                {"model_id": "good-model:7b", "run_id": "screen-test-good-model-7b-r03",
                 "parse_rate": 0.99, "accuracy": 0.61},
                # bad-parse: 3 runs, low parse rate
                {"model_id": "bad-parse:3b", "run_id": "screen-test-bad-parse-3b-r01",
                 "parse_rate": 0.80, "accuracy": 0.40},
                {"model_id": "bad-parse:3b", "run_id": "screen-test-bad-parse-3b-r02",
                 "parse_rate": 0.85, "accuracy": 0.38},
                {"model_id": "bad-parse:3b", "run_id": "screen-test-bad-parse-3b-r03",
                 "parse_rate": 0.82, "accuracy": 0.42},
                # bad-accuracy: 3 runs, good parse but low accuracy
                {"model_id": "bad-accuracy:1b", "run_id": "screen-test-bad-accuracy-1b-r01",
                 "parse_rate": 1.0, "accuracy": 0.20},
                {"model_id": "bad-accuracy:1b", "run_id": "screen-test-bad-accuracy-1b-r02",
                 "parse_rate": 0.99, "accuracy": 0.19},
                {"model_id": "bad-accuracy:1b", "run_id": "screen-test-bad-accuracy-1b-r03",
                 "parse_rate": 1.0, "accuracy": 0.21},
            ],
        }
        with open(cls.tmpdir / "repeat_summary_screen-test.json", "w") as f:
            json.dump(summary, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_only_good_model_promoted(self):
        result = apply_screening_gate("screen-test", responses_dir=self.tmpdir)
        self.assertEqual(result["promoted"], ["good-model:7b"])
        self.assertEqual(result["promoted_count"], 1)

    def test_bad_parse_excluded(self):
        result = apply_screening_gate("screen-test", responses_dir=self.tmpdir)
        verdicts = {v["model_id"]: v for v in result["appendix"]}
        self.assertEqual(verdicts["bad-parse:3b"]["status"], "not promoted")
        self.assertTrue(any("parse_rate" in r for r in verdicts["bad-parse:3b"]["exclusion_reasons"]))

    def test_bad_accuracy_excluded(self):
        result = apply_screening_gate("screen-test", responses_dir=self.tmpdir)
        verdicts = {v["model_id"]: v for v in result["appendix"]}
        self.assertEqual(verdicts["bad-accuracy:1b"]["status"], "not promoted")
        self.assertTrue(any("mean_accuracy" in r for r in verdicts["bad-accuracy:1b"]["exclusion_reasons"]))

    def test_csv_output(self):
        result = apply_screening_gate("screen-test", responses_dir=self.tmpdir)
        self.assertEqual(result["csv"], "good-model:7b")

    def test_appendix_has_all_models(self):
        result = apply_screening_gate("screen-test", responses_dir=self.tmpdir)
        self.assertEqual(len(result["appendix"]), 3)

    def test_appendix_fields(self):
        result = apply_screening_gate("screen-test", responses_dir=self.tmpdir)
        required = {"model_id", "tier", "runs", "mean_accuracy", "mean_parse_rate", "status"}
        for v in result["appendix"]:
            self.assertTrue(required.issubset(v.keys()), f"Missing fields: {required - set(v.keys())}")

    def test_promoted_model_has_no_exclusion_reasons(self):
        result = apply_screening_gate("screen-test", responses_dir=self.tmpdir)
        verdicts = {v["model_id"]: v for v in result["appendix"]}
        self.assertIsNone(verdicts["good-model:7b"]["exclusion_reasons"])

    def test_deterministic_csv_sorted(self):
        """CSV output should be sorted alphabetically for determinism."""
        result = apply_screening_gate("screen-test", responses_dir=self.tmpdir)
        models = result["csv"].split(",") if result["csv"] else []
        self.assertEqual(models, sorted(models))


class TestScreeningGateEdgeCases(unittest.TestCase):
    """Edge cases for promotion thresholds."""

    def _make_summary(self, mean_accuracy: float, parse_rate: float, runs: int = 3):
        tmpdir = Path(tempfile.mkdtemp())
        summary = {
            "batch_id": "edge-test",
            "started_at_epoch": 0.0,
            "finished_at_epoch": 0.0,
            "models": [{
                "model_id": "edge:1b",
                "runs": runs,
                "memory_gb": 1.0,
                "mean_accuracy": mean_accuracy,
                "mean_thai_accuracy": mean_accuracy,
                "mean_math_accuracy": mean_accuracy,
            }],
            "runs": [{
                "model_id": "edge:1b",
                "run_id": f"edge-test-edge-1b-r{i:02d}",
                "parse_rate": parse_rate,
                "accuracy": mean_accuracy,
            } for i in range(1, runs + 1)],
        }
        with open(tmpdir / "repeat_summary_edge-test.json", "w") as f:
            json.dump(summary, f)
        return tmpdir

    def test_exactly_25_percent_accuracy_fails(self):
        """mean accuracy must be > 25%, not >=."""
        tmpdir = self._make_summary(0.25, 1.0)
        try:
            result = apply_screening_gate("edge-test", responses_dir=tmpdir)
            self.assertEqual(result["promoted"], [])
        finally:
            shutil.rmtree(tmpdir)

    def test_just_above_25_percent_passes(self):
        tmpdir = self._make_summary(0.2501, 1.0)
        try:
            result = apply_screening_gate("edge-test", responses_dir=tmpdir)
            self.assertEqual(result["promoted"], ["edge:1b"])
        finally:
            shutil.rmtree(tmpdir)

    def test_exactly_95_percent_parse_passes(self):
        """parse_rate >= 95% (inclusive)."""
        tmpdir = self._make_summary(0.5, 0.95)
        try:
            result = apply_screening_gate("edge-test", responses_dir=tmpdir)
            self.assertEqual(result["promoted"], ["edge:1b"])
        finally:
            shutil.rmtree(tmpdir)

    def test_just_below_95_percent_parse_fails(self):
        tmpdir = self._make_summary(0.5, 0.9499)
        try:
            result = apply_screening_gate("edge-test", responses_dir=tmpdir)
            self.assertEqual(result["promoted"], [])
        finally:
            shutil.rmtree(tmpdir)

    def test_incomplete_runs_fails(self):
        tmpdir = self._make_summary(0.5, 1.0, runs=2)
        try:
            result = apply_screening_gate("edge-test", responses_dir=tmpdir, expected_runs=3)
            self.assertEqual(result["promoted"], [])
            verdicts = {v["model_id"]: v for v in result["appendix"]}
            self.assertTrue(any("incomplete" in r for r in verdicts["edge:1b"]["exclusion_reasons"]))
        finally:
            shutil.rmtree(tmpdir)


class TestScreeningGateDigestCheck(unittest.TestCase):
    """Test digest verification with preflight records."""

    def test_missing_from_preflight_fails(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            summary = {
                "batch_id": "digest-test",
                "started_at_epoch": 0.0,
                "finished_at_epoch": 0.0,
                "models": [{
                    "model_id": "model-x:1b",
                    "runs": 3,
                    "memory_gb": 1.0,
                    "mean_accuracy": 0.5,
                    "mean_thai_accuracy": 0.5,
                    "mean_math_accuracy": 0.5,
                }],
                "runs": [
                    {"model_id": "model-x:1b", "run_id": f"digest-test-r{i:02d}",
                     "parse_rate": 1.0, "accuracy": 0.5}
                    for i in range(1, 4)
                ],
            }
            with open(tmpdir / "repeat_summary_digest-test.json", "w") as f:
                json.dump(summary, f)

            # Preflight record that does NOT include model-x:1b
            preflight = {
                "model_inventory": [
                    {"model_id": "other-model:1b", "digest": "sha256:abc123"},
                ]
            }
            preflight_path = tmpdir / "preflight.json"
            with open(preflight_path, "w") as f:
                json.dump(preflight, f)

            result = apply_screening_gate(
                "digest-test", responses_dir=tmpdir, preflight_path=preflight_path
            )
            self.assertEqual(result["promoted"], [])
            verdicts = {v["model_id"]: v for v in result["appendix"]}
            self.assertTrue(any("not in preflight" in r for r in verdicts["model-x:1b"]["exclusion_reasons"]))
        finally:
            shutil.rmtree(tmpdir)

    def test_present_in_preflight_passes(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            summary = {
                "batch_id": "digest-ok",
                "started_at_epoch": 0.0,
                "finished_at_epoch": 0.0,
                "models": [{
                    "model_id": "model-x:1b",
                    "runs": 3,
                    "memory_gb": 1.0,
                    "mean_accuracy": 0.5,
                    "mean_thai_accuracy": 0.5,
                    "mean_math_accuracy": 0.5,
                }],
                "runs": [
                    {"model_id": "model-x:1b", "run_id": f"digest-ok-r{i:02d}",
                     "parse_rate": 1.0, "accuracy": 0.5}
                    for i in range(1, 4)
                ],
            }
            with open(tmpdir / "repeat_summary_digest-ok.json", "w") as f:
                json.dump(summary, f)

            preflight = {
                "model_inventory": [
                    {"model_id": "model-x:1b", "digest": "sha256:abc123"},
                ]
            }
            preflight_path = tmpdir / "preflight.json"
            with open(preflight_path, "w") as f:
                json.dump(preflight, f)

            with mock.patch(
                "scripts.screening_gate.list_current_ollama_digests",
                return_value={"model-x:1b": "sha256:abc123"},
            ):
                result = apply_screening_gate(
                    "digest-ok", responses_dir=tmpdir, preflight_path=preflight_path
                )
            self.assertEqual(result["promoted"], ["model-x:1b"])
        finally:
            shutil.rmtree(tmpdir)

    def test_digest_mismatch_fails(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            summary = {
                "batch_id": "digest-mismatch",
                "started_at_epoch": 0.0,
                "finished_at_epoch": 0.0,
                "models": [{
                    "model_id": "model-x:1b",
                    "runs": 3,
                    "memory_gb": 1.0,
                    "mean_accuracy": 0.5,
                    "mean_thai_accuracy": 0.5,
                    "mean_math_accuracy": 0.5,
                }],
                "runs": [
                    {"model_id": "model-x:1b", "run_id": f"digest-mismatch-r{i:02d}",
                     "parse_rate": 1.0, "accuracy": 0.5}
                    for i in range(1, 4)
                ],
            }
            with open(tmpdir / "repeat_summary_digest-mismatch.json", "w") as f:
                json.dump(summary, f)

            preflight = {
                "model_inventory": [
                    {"model_id": "model-x:1b", "digest": "sha256:expected"},
                ]
            }
            preflight_path = tmpdir / "preflight.json"
            with open(preflight_path, "w") as f:
                json.dump(preflight, f)

            with mock.patch(
                "scripts.screening_gate.list_current_ollama_digests",
                return_value={"model-x:1b": "sha256:actual"},
            ):
                result = apply_screening_gate(
                    "digest-mismatch", responses_dir=tmpdir, preflight_path=preflight_path
                )
            self.assertEqual(result["promoted"], [])
            verdicts = {v["model_id"]: v for v in result["appendix"]}
            self.assertTrue(any("digest" in r for r in verdicts["model-x:1b"]["exclusion_reasons"]))
            self.assertFalse(verdicts["model-x:1b"]["digest_checked"])
        finally:
            shutil.rmtree(tmpdir)


class TestResolveDigestMatch(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(resolve_digest_match("gemma4:e2b", {"gemma4:e2b": "sha256:abc"}, "sha256:abc"))

    def test_latest_suffix_match(self):
        self.assertTrue(resolve_digest_match("gemma4:e2b", {"gemma4:e2b:latest": "sha256:abc"}, "sha256:abc"))

    def test_unreachable_inventory_returns_none(self):
        self.assertIsNone(resolve_digest_match("gemma4:e2b", {}, "sha256:abc"))


class TestFormatAppendixMarkdown(unittest.TestCase):
    def test_markdown_format(self):
        appendix = [
            {
                "model_id": "good:1b",
                "tier": "core-international",
                "runs": 3,
                "mean_accuracy": 0.6,
                "mean_parse_rate": 1.0,
                "status": "promoted",
                "exclusion_reasons": None,
            },
            {
                "model_id": "bad:1b",
                "tier": "extended-international",
                "runs": 3,
                "mean_accuracy": 0.1,
                "mean_parse_rate": 0.5,
                "status": "not promoted",
                "exclusion_reasons": ["low parse rate", "low accuracy"],
            },
        ]
        md = format_appendix_markdown(appendix)
        self.assertIn("| Model |", md)
        self.assertIn("good:1b", md)
        self.assertIn("bad:1b", md)
        self.assertIn("promoted", md)
        self.assertIn("not promoted", md)
        lines = md.strip().split("\n")
        self.assertEqual(len(lines), 4)  # header + separator + 2 data rows


# ── verify_batch_inputs unit tests ────────────────────────────────────────


class TestBatchIdPattern(unittest.TestCase):
    def test_valid_batch_ids(self):
        for bid in ["ntp3-screen-r3-20260411", "mini-r10-20260409", "test_batch.v1"]:
            self.assertTrue(BATCH_ID_PATTERN.match(bid), f"Should be valid: {bid}")

    def test_invalid_batch_ids(self):
        for bid in ["has spaces", "has\ttab", "has/slash"]:
            self.assertIsNone(BATCH_ID_PATTERN.match(bid), f"Should be invalid: {bid!r}")
        # Empty string should not match
        self.assertIsNone(BATCH_ID_PATTERN.match(""), "Empty string should be invalid")


class TestVerifyBatchInputs(unittest.TestCase):
    def test_valid_inputs_pass(self):
        result = verify_batch_inputs(
            "ntp3-screen-r3-20260411",
            models=["gemma4:e2b", "gemma4:e4b"],
            runs_per_model=3,
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["checks"]["compatibility_loaded"])
        self.assertTrue(result["checks"]["dataset_integrity"])

    def test_empty_models_fails(self):
        result = verify_batch_inputs(
            "test-batch",
            models=[],
            runs_per_model=3,
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["checks"]["models_non_empty"])

    def test_invalid_batch_id_fails(self):
        result = verify_batch_inputs(
            "bad batch id",
            models=["gemma4:e2b"],
            runs_per_model=1,
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["checks"]["batch_id_format"])

    def test_think_budget_with_off_contract_fails(self):
        result = verify_batch_inputs(
            "test-batch",
            models=["gemma4:e2b"],
            runs_per_model=1,
            think_budget=4096,
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["checks"]["think_mode_consistent"])

    def test_wrong_subjects_fails(self):
        result = verify_batch_inputs(
            "test-batch",
            models=["gemma4:e2b"],
            runs_per_model=1,
            subjects="thai",
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["checks"]["subjects_match"])

    def test_zero_runs_fails(self):
        result = verify_batch_inputs(
            "test-batch",
            models=["gemma4:e2b"],
            runs_per_model=0,
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["checks"]["runs_per_model_valid"])

    def test_contract_loaded(self):
        compat = load_compatibility()
        self.assertIn("v1_expected_values", compat)
        self.assertEqual(compat["v1_expected_values"]["think_mode"], "off")


# ── prepare_publication_batch unit tests ──────────────────────────────────


class TestDerivePubBatchId(unittest.TestCase):
    def test_standard_screening_id(self):
        self.assertEqual(
            derive_pub_batch_id("ntp3-screen-r3-20260411", 10),
            "ntp3-pub-r10-20260411",
        )

    def test_different_run_count(self):
        self.assertEqual(
            derive_pub_batch_id("ntp3-screen-r3-20260411", 5),
            "ntp3-pub-r5-20260411",
        )

    def test_nonstandard_id_fallback(self):
        result = derive_pub_batch_id("custom-screen-batch", 10)
        self.assertIn("ntp3-pub-r10", result)
        self.assertIn("custom-screen-batch", result)


class TestPreparePublicationBatch(unittest.TestCase):
    """Test publication batch preparation with synthetic screening data."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp())
        # Create a screening decision JSON
        cls.screening_decision = {
            "batch_id": "ntp3-screen-r3-20260411",
            "expected_runs_per_model": 3,
            "total_models": 3,
            "promoted_count": 2,
            "promoted": ["gemma4:e2b", "qwen3:8b"],
            "csv": "gemma4:e2b,qwen3:8b",
            "appendix": [],
        }
        cls.screening_path = cls.tmpdir / "screening_decision.json"
        with open(cls.screening_path, "w") as f:
            json.dump(cls.screening_decision, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_from_screening_json(self):
        result = prepare_publication_batch(
            screening_json_path=self.screening_path,
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["promoted_models"], ["gemma4:e2b", "qwen3:8b"])
        self.assertEqual(result["promoted_count"], 2)
        self.assertEqual(result["pub_runs"], 10)
        self.assertEqual(result["pub_batch_id"], "ntp3-pub-r10-20260411")

    def test_total_expected_runs(self):
        result = prepare_publication_batch(
            screening_json_path=self.screening_path,
        )
        self.assertEqual(result["total_expected_runs"], 20)

    def test_custom_pub_batch_id(self):
        result = prepare_publication_batch(
            screening_json_path=self.screening_path,
            pub_batch_id="custom-pub-20260412",
        )
        self.assertEqual(result["pub_batch_id"], "custom-pub-20260412")

    def test_custom_pub_runs(self):
        result = prepare_publication_batch(
            screening_json_path=self.screening_path,
            pub_runs=5,
        )
        self.assertEqual(result["pub_runs"], 5)
        self.assertEqual(result["total_expected_runs"], 10)
        self.assertEqual(result["pub_batch_id"], "ntp3-pub-r5-20260411")

    def test_csv_output(self):
        result = prepare_publication_batch(
            screening_json_path=self.screening_path,
        )
        self.assertEqual(result["promoted_csv"], "gemma4:e2b,qwen3:8b")

    def test_no_models_promoted(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            empty_decision = {
                "batch_id": "ntp3-screen-r3-20260411",
                "promoted_count": 0,
                "promoted": [],
                "csv": "",
                "appendix": [],
            }
            path = tmpdir / "empty.json"
            with open(path, "w") as f:
                json.dump(empty_decision, f)
            result = prepare_publication_batch(screening_json_path=path)
            self.assertEqual(result["status"], "no_models_promoted")
            self.assertEqual(result["promoted_count"], 0)
        finally:
            shutil.rmtree(tmpdir)

    def test_contract_included(self):
        result = prepare_publication_batch(
            screening_json_path=self.screening_path,
        )
        self.assertIn("contract", result)
        # Contract should have V1 fields if compatibility.json exists
        if result["contract"]:
            self.assertIn("think_mode", result["contract"])


class TestWriteEnvFile(unittest.TestCase):
    def test_env_file_content(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            env_path = tmpdir / "pub.env"
            result = {
                "pub_batch_id": "ntp3-pub-r10-20260411",
                "promoted_csv": "gemma4:e2b,qwen3:8b",
                "pub_runs": 10,
                "screening_batch_id": "ntp3-screen-r3-20260411",
            }
            write_env_file(result, env_path)
            content = env_path.read_text()
            self.assertIn('PUB_BATCH_ID="ntp3-pub-r10-20260411"', content)
            self.assertIn('PROMOTED_MODELS="gemma4:e2b,qwen3:8b"', content)
            self.assertIn('PUB_RUNS="10"', content)
            self.assertIn('SCREENING_BATCH_ID="ntp3-screen-r3-20260411"', content)
        finally:
            shutil.rmtree(tmpdir)


class TestPreparePublicationFromScreeningGate(unittest.TestCase):
    """End-to-end: screening gate -> publication preparation."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp())
        # Create a repeat_summary that the screening gate will read
        summary = {
            "batch_id": "ntp3-screen-r3-20260411",
            "started_at_epoch": 0.0,
            "finished_at_epoch": 0.0,
            "models": [
                {"model_id": "gemma4:e2b", "runs": 3, "memory_gb": 2.0,
                 "mean_accuracy": 0.55, "mean_thai_accuracy": 0.60, "mean_math_accuracy": 0.45},
                {"model_id": "bad-model:1b", "runs": 3, "memory_gb": 1.0,
                 "mean_accuracy": 0.15, "mean_thai_accuracy": 0.10, "mean_math_accuracy": 0.20},
            ],
            "runs": [
                {"model_id": "gemma4:e2b", "run_id": "r01", "parse_rate": 1.0, "accuracy": 0.55},
                {"model_id": "gemma4:e2b", "run_id": "r02", "parse_rate": 0.98, "accuracy": 0.54},
                {"model_id": "gemma4:e2b", "run_id": "r03", "parse_rate": 0.99, "accuracy": 0.56},
                {"model_id": "bad-model:1b", "run_id": "r01", "parse_rate": 0.70, "accuracy": 0.15},
                {"model_id": "bad-model:1b", "run_id": "r02", "parse_rate": 0.65, "accuracy": 0.14},
                {"model_id": "bad-model:1b", "run_id": "r03", "parse_rate": 0.72, "accuracy": 0.16},
            ],
        }
        with open(cls.tmpdir / "repeat_summary_ntp3-screen-r3-20260411.json", "w") as f:
            json.dump(summary, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_end_to_end(self):
        result = prepare_publication_batch(
            screening_batch_id="ntp3-screen-r3-20260411",
            responses_dir=self.tmpdir,
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["promoted_models"], ["gemma4:e2b"])
        self.assertEqual(result["promoted_count"], 1)
        self.assertEqual(result["pub_batch_id"], "ntp3-pub-r10-20260411")
        self.assertEqual(result["total_expected_runs"], 10)


# ── Integration tests with real data ─────────────────────────────────────


def _have_real_data() -> bool:
    return (RESPONSES_DIR / f"repeat_summary_{BATCH_ID}.json").exists()


SKIP_NO_DATA = not _have_real_data()
SKIP_MSG = f"Benchmark data for batch '{BATCH_ID}' not found"


@unittest.skipIf(SKIP_NO_DATA, SKIP_MSG)
class TestBatchCompletenessRealData(unittest.TestCase):
    """Verify real mini-r10-20260409 batch passes completeness check."""

    def test_real_batch_passes(self):
        result = check_batch_completeness(BATCH_ID, responses_dir=RESPONSES_DIR)
        self.assertEqual(result["status"], "pass", f"Errors: {result['errors']}")

    def test_real_batch_has_40_files(self):
        """4 models × 10 runs = 40 files."""
        result = check_batch_completeness(BATCH_ID, responses_dir=RESPONSES_DIR)
        self.assertEqual(result["total_files"], 40)

    def test_all_files_have_93_items(self):
        result = check_batch_completeness(BATCH_ID, responses_dir=RESPONSES_DIR)
        for detail in result["file_details"]:
            self.assertEqual(detail["rows"], 93, f"{detail['file']} has {detail['rows']} rows")


@unittest.skipIf(SKIP_NO_DATA, SKIP_MSG)
class TestScreeningGateRealData(unittest.TestCase):
    """Run screening gate on real data to verify all 4 models pass."""

    def test_all_models_promoted(self):
        """All 4 models in mini-r10-20260409 should pass screening gates."""
        result = apply_screening_gate(BATCH_ID, responses_dir=RESPONSES_DIR)
        self.assertEqual(result["promoted_count"], 4)

    def test_csv_has_4_models(self):
        result = apply_screening_gate(BATCH_ID, responses_dir=RESPONSES_DIR)
        models = result["csv"].split(",")
        self.assertEqual(len(models), 4)

    def test_all_parse_rates_above_threshold(self):
        result = apply_screening_gate(BATCH_ID, responses_dir=RESPONSES_DIR)
        for v in result["appendix"]:
            self.assertGreaterEqual(
                v["mean_parse_rate"], MIN_PARSE_RATE,
                f"{v['model_id']} parse_rate {v['mean_parse_rate']}"
            )

    def test_all_accuracies_above_threshold(self):
        result = apply_screening_gate(BATCH_ID, responses_dir=RESPONSES_DIR)
        for v in result["appendix"]:
            self.assertGreater(
                v["mean_accuracy"], MIN_MEAN_ACCURACY,
                f"{v['model_id']} accuracy {v['mean_accuracy']}"
            )


if __name__ == "__main__":
    unittest.main()
