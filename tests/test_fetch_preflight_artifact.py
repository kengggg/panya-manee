#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ci.fetch_preflight_artifact import (  # noqa: E402
    PreflightFetchError,
    choose_latest_artifact,
    materialize_preflight_record,
)


class TestChooseLatestArtifact(unittest.TestCase):
    def test_picks_latest_non_expired_exact_name_match(self):
        artifacts = [
            {
                "id": 1,
                "name": "preflight-20260412",
                "expired": False,
                "updated_at": "2026-04-12T03:50:00Z",
                "created_at": "2026-04-12T03:49:00Z",
            },
            {
                "id": 2,
                "name": "preflight-20260412",
                "expired": True,
                "updated_at": "2026-04-12T04:00:00Z",
                "created_at": "2026-04-12T03:59:00Z",
            },
            {
                "id": 3,
                "name": "preflight-20260412",
                "expired": False,
                "updated_at": "2026-04-12T04:10:00Z",
                "created_at": "2026-04-12T04:09:00Z",
            },
            {
                "id": 4,
                "name": "preflight-20260411",
                "expired": False,
                "updated_at": "2026-04-11T04:10:00Z",
                "created_at": "2026-04-11T04:09:00Z",
            },
        ]
        chosen = choose_latest_artifact(artifacts, "preflight-20260412")
        self.assertEqual(chosen["id"], 3)

    def test_raises_when_missing(self):
        with self.assertRaises(PreflightFetchError):
            choose_latest_artifact([], "preflight-20260412")


class TestMaterializePreflightRecord(unittest.TestCase):
    def test_copies_nested_preflight_json_to_requested_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "preflight-20260412" / "preflight_20260412.json"
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_text('{"ok": true}\n', encoding="utf-8")

            output = root / "out" / "preflight_20260412.json"
            materialize_preflight_record("20260412", root, output)

            self.assertTrue(output.exists())
            self.assertEqual(output.read_text(encoding="utf-8"), '{"ok": true}\n')

    def test_raises_when_json_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(PreflightFetchError):
                materialize_preflight_record("20260412", root, root / "out.json")


if __name__ == "__main__":
    unittest.main()
