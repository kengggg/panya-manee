#!/usr/bin/env python3
"""
Tests for scripts/summarize_batch.py — repeat_summary generation.

Validates that generated summaries match the expected schema and that
metrics are consistent with the raw JSONL data they're derived from.

Run: python -m pytest tests/test_summarize_batch.py -v
"""

import json
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.summarize_batch import (
    generate_repeat_summary,
    summarize_run,
    summarize_model,
    extract_run_index,
    load_jsonl,
)

RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"
BATCH_ID = "mini-r10-20260409"


class TestExtractRunIndex(unittest.TestCase):
    def test_standard_format(self):
        self.assertEqual(extract_run_index("mini-r10-20260409-gemma4-e4b-r03", "mini-r10-20260409"), 3)

    def test_double_digit(self):
        self.assertEqual(extract_run_index("mini-r10-20260409-qwen3-0.6b-r10", "mini-r10-20260409"), 10)

    def test_no_match(self):
        self.assertEqual(extract_run_index("unrelated-id", "mini-r10-20260409"), 0)


@unittest.skipUnless(
    (RESPONSES_DIR / f"repeat_summary_{BATCH_ID}.json").exists(),
    "Requires existing repeat_summary for comparison"
)
class TestGeneratedMatchesExisting(unittest.TestCase):
    """Compare generated summary against the known-good existing summary."""

    @classmethod
    def setUpClass(cls):
        # Load the existing hand-crafted summary
        with open(RESPONSES_DIR / f"repeat_summary_{BATCH_ID}.json") as f:
            cls.existing = json.load(f)

        # Generate a fresh summary from the raw JSONL files into a temp dir
        cls.tmpdir = Path(tempfile.mkdtemp())
        # Copy JSONL files to temp dir so generation doesn't overwrite existing
        for fpath in RESPONSES_DIR.glob(f"responses_{BATCH_ID}-*.jsonl"):
            shutil.copy2(fpath, cls.tmpdir / fpath.name)
        cls.generated = generate_repeat_summary(BATCH_ID, cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_batch_id_matches(self):
        self.assertEqual(self.generated["batch_id"], BATCH_ID)

    def test_same_model_ids(self):
        existing_ids = sorted(m["model_id"] for m in self.existing["models"])
        generated_ids = sorted(m["model_id"] for m in self.generated["models"])
        self.assertEqual(existing_ids, generated_ids)

    def test_same_run_count(self):
        self.assertEqual(len(self.generated["runs"]), len(self.existing["runs"]))

    def test_run_ids_match(self):
        existing_ids = sorted(r["run_id"] for r in self.existing["runs"])
        generated_ids = sorted(r["run_id"] for r in self.generated["runs"])
        self.assertEqual(existing_ids, generated_ids)

    def test_per_run_accuracy_matches(self):
        existing_by_id = {r["run_id"]: r for r in self.existing["runs"]}
        for run in self.generated["runs"]:
            ex = existing_by_id.get(run["run_id"])
            if ex is None:
                continue
            self.assertAlmostEqual(
                run["accuracy"], ex["accuracy"], places=6,
                msg=f"{run['run_id']} accuracy mismatch"
            )
            self.assertEqual(run["correct"], ex["correct"], msg=f"{run['run_id']} correct mismatch")
            self.assertEqual(run["questions"], ex["questions"], msg=f"{run['run_id']} questions mismatch")

    def test_per_run_subject_breakdown_matches(self):
        existing_by_id = {r["run_id"]: r for r in self.existing["runs"]}
        for run in self.generated["runs"]:
            ex = existing_by_id.get(run["run_id"])
            if ex is None:
                continue
            self.assertEqual(run["thai_correct"], ex["thai_correct"], msg=f"{run['run_id']} thai_correct")
            self.assertEqual(run["math_correct"], ex["math_correct"], msg=f"{run['run_id']} math_correct")
            self.assertEqual(run["thai_total"], ex["thai_total"], msg=f"{run['run_id']} thai_total")
            self.assertEqual(run["math_total"], ex["math_total"], msg=f"{run['run_id']} math_total")

    def test_model_mean_accuracy_matches(self):
        existing_by_id = {m["model_id"]: m for m in self.existing["models"]}
        for model in self.generated["models"]:
            ex = existing_by_id.get(model["model_id"])
            if ex is None:
                continue
            self.assertAlmostEqual(
                model["mean_accuracy"], ex["mean_accuracy"], places=6,
                msg=f"{model['model_id']} mean_accuracy mismatch"
            )
            self.assertAlmostEqual(
                model["mean_thai_accuracy"], ex["mean_thai_accuracy"], places=6,
                msg=f"{model['model_id']} mean_thai_accuracy mismatch"
            )
            self.assertAlmostEqual(
                model["mean_math_accuracy"], ex["mean_math_accuracy"], places=6,
                msg=f"{model['model_id']} mean_math_accuracy mismatch"
            )

    def test_model_memory_gb_matches(self):
        existing_by_id = {m["model_id"]: m for m in self.existing["models"]}
        for model in self.generated["models"]:
            ex = existing_by_id.get(model["model_id"])
            if ex is None:
                continue
            self.assertAlmostEqual(
                model["memory_gb"], ex["memory_gb"], places=1,
                msg=f"{model['model_id']} memory_gb mismatch"
            )


class TestSummarySchema(unittest.TestCase):
    """Validate the output schema matches what build_snapshot.py expects."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp())
        # Create minimal synthetic JSONL
        rows = [
            {
                "model_id": "test-model:1b",
                "run_id": "test-batch-test-model-1b-r01",
                "subject": "thai",
                "question_id": "q1",
                "is_correct": True,
                "is_parseable": True,
                "latency_ms": 1000,
                "raw_output": "1",
                "parsed_answer": 1,
                "correct_answer": 1,
            },
            {
                "model_id": "test-model:1b",
                "run_id": "test-batch-test-model-1b-r01",
                "subject": "math",
                "question_id": "q2",
                "is_correct": False,
                "is_parseable": True,
                "latency_ms": 2000,
                "raw_output": "2",
                "parsed_answer": 2,
                "correct_answer": 3,
            },
        ]
        fpath = cls.tmpdir / "responses_test-batch-test-model-1b-r01_20260409_120000.jsonl"
        with open(fpath, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        cls.summary = generate_repeat_summary("test-batch", cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_top_level_keys(self):
        required = {"batch_id", "started_at_epoch", "finished_at_epoch", "models", "runs"}
        self.assertTrue(required.issubset(set(self.summary.keys())))

    def test_run_has_required_fields(self):
        """build_snapshot.py reads model_id, output_file, run_index, run_id from each run."""
        required = {"model_id", "output_file", "run_index", "run_id"}
        for run in self.summary["runs"]:
            self.assertTrue(required.issubset(set(run.keys())), f"Missing fields in run: {required - set(run.keys())}")

    def test_model_has_required_fields(self):
        required = {"model_id", "memory_gb", "runs", "mean_accuracy", "mean_thai_accuracy", "mean_math_accuracy"}
        for model in self.summary["models"]:
            self.assertTrue(required.issubset(set(model.keys())), f"Missing fields in model: {required - set(model.keys())}")

    def test_accuracy_is_correct(self):
        run = self.summary["runs"][0]
        # 1 correct out of 2 questions
        self.assertAlmostEqual(run["accuracy"], 0.5)
        self.assertEqual(run["correct"], 1)
        self.assertEqual(run["questions"], 2)

    def test_subject_breakdown(self):
        run = self.summary["runs"][0]
        self.assertEqual(run["thai_correct"], 1)
        self.assertEqual(run["thai_total"], 1)
        self.assertEqual(run["math_correct"], 0)
        self.assertEqual(run["math_total"], 1)

    def test_output_file_written(self):
        out_path = self.tmpdir / "repeat_summary_test-batch.json"
        self.assertTrue(out_path.exists())
        with open(out_path) as f:
            disk_data = json.load(f)
        self.assertEqual(disk_data["batch_id"], "test-batch")


if __name__ == "__main__":
    unittest.main()
