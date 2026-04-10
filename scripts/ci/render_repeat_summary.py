#!/usr/bin/env python3
import json
import os
from pathlib import Path

batch_id = os.environ["BATCH_ID"]
path = Path(f"benchmark_responses/repeat_summary_{batch_id}.json")
d = json.loads(path.read_text(encoding="utf-8"))

print()
print(f'- Models: **{len(d.get("models", []))}**')
print(f'- Runs: **{len(d.get("runs", []))}**')
