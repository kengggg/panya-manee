#!/usr/bin/env python3
import json
import os
from pathlib import Path

path = Path(os.environ["PUB"])
d = json.loads(path.read_text(encoding="utf-8"))
if d["status"] == "ready":
    print('Dispatch publication batch:')
    print()
    print('```bash')
    print('gh workflow run benchmark-run.yml \\\\')
    print(f'  -f batch_id="{d["pub_batch_id"]}" \\\\')
    print(f'  -f models="{d["promoted_csv"]}" \\\\')
    print(f'  -f runs_per_model="{d["pub_runs"]}"')
    print('```')
else:
    print('No models promoted — cannot proceed to publication.')
