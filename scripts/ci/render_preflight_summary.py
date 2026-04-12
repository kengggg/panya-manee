#!/usr/bin/env python3
import json
import os
from pathlib import Path

path = Path(os.environ["PREFLIGHT_OUTPUT"])
d = json.loads(path.read_text(encoding="utf-8"))
date = os.environ.get("PREFLIGHT_DATE", "")

print()
print(f'- Status: **{d["status"]}**')
print(f'- Output: `{path}`')
print(f'- Models checked: **{len(d.get("model_inventory", []))}**')
print(f'- Errors: **{len(d.get("errors", []))}**')
print(f'- Warnings: **{len(d.get("warnings", []))}**')

if d.get("errors"):
    print('- Error list:')
    for item in d["errors"]:
        print(f'  - {item}')

if d.get("warnings"):
    print('- Warning list:')
    for item in d["warnings"]:
        print(f'  - {item}')

if d.get("status") == "pass":
    print()
    print('### Next Step')
    print('Dispatch verified publication batch:')
    print()
    print('```bash')
    print('gh workflow run benchmark-verified.yml \\\\')
    print(f'  -f batch_id="ntp3-vr1-{date}" \\\\')
    print(f'  -f models="{",".join(d["requested_models"])}" \\\\')
    print(f'  -f preflight_date="{date}" \\\\')
    print('  -f dry_run="false"')
    print('```')
