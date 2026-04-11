#!/usr/bin/env python3
import json
import os
from pathlib import Path

path = Path(os.environ["PUB"])
d = json.loads(path.read_text(encoding="utf-8"))
if d["status"] == "ready":
    print('Dispatch verified publication batch:')
    print()
    print('```bash')
    print('gh workflow run benchmark-verified.yml \\\\')
    print(f'  -f screening_batch_id="{d["screening_batch_id"]}"')
    print('```')
else:
    print('No models promoted — cannot proceed to publication.')
