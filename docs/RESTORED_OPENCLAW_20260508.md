# Restored OpenClaw workspace note — 2026-05-08

This repository was restored from the previous OpenClaw workspace backup into:

`/Users/openclaw/.openclaw/workspace/projects/panya-manee/`

Restoration choices:

- Preserved the Git repository and history from the backup.
- Excluded local cache/runtime clutter: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.DS_Store`.
- Kept benchmark artifacts, `dist/`, `site/data/latest/`, registry files, workflows, tests, and NT test datasets because they are useful for reproducibility.
- Restored the earlier dashboard planning docs under `docs/dashboard/` as project reference material.

Local verification on 2026-05-08:

```bash
uv run pytest -q
# 180 passed

uv run python scripts/verify_site.py
# ALL CHECKS PASSED

uv run python main.py run --model qwen3:0.6b --subjects thai --run-id restore-smoke-20260508 --dry-run
# dry-run completed; no Ollama/API calls
```

Small improvement made during restoration:

- Added `pytest>=8` as a dev dependency so the documented test suite works with `uv run pytest` after a fresh checkout/restore.
