#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from screening_gate import format_appendix_markdown  # noqa: E402

path = Path(os.environ["APPENDIX"])
appendix = json.loads(path.read_text(encoding="utf-8"))
print(format_appendix_markdown(appendix))
