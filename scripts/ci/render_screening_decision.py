#!/usr/bin/env python3
import json
import os
from pathlib import Path

path = Path(os.environ["DECISION"])
d = json.loads(path.read_text(encoding="utf-8"))
print(f"**{d['promoted_count']}/{d['total_models']}** models promoted")
print()
print('Promoted: `' + d['csv'] + '`')
