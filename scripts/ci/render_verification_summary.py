#!/usr/bin/env python3
import json
import os
from pathlib import Path

path = Path(os.environ["REPORT"])
report = json.loads(path.read_text(encoding="utf-8"))
print(f"- Status: **{report.get('status', 'unknown')}**")
print(f"- Publishable models: **{report.get('publishable_count', 0)}**")
print(f"- All deterministic: **{report.get('all_deterministic', False)}**")
for model in report.get("models", []):
    marker = "publish" if model.get("publishable") else "exclude"
    print(f"  - `{model['model_id']}`: {marker}")
