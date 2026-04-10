#!/usr/bin/env python3
import json
import os
from pathlib import Path

path = Path(os.environ["SUMMARY"])
d = json.loads(path.read_text(encoding="utf-8"))
print(f"  {len(d.get('models', []))} models, {len(d.get('runs', []))} runs")
