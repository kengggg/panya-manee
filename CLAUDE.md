# CLAUDE.md

## Project overview

panya-manee is a local LLM benchmark for Thailand's NT (National Test) Grade 3 exams. It runs models via Ollama and evaluates Thai language comprehension and math reasoning.

## Architecture

- `main.py` — CLI entry point (argparse)
- `config.py` — shared config, dataset paths, prompt template, data validation (`validate_items`)
- `benchmark_runner.py` — runs one model against the test set, live progress via `rich`, saves JSONL
- `summarize_results.py` — aggregates JSONL results across models, rich tables
- `nt-tests/` — test data (JSON) and original exam PDFs

## Commands

```bash
uv sync                                              # install deps
uv run python main.py run --model <name> --dry-run   # test without API calls
uv run python main.py run --model <name>             # real benchmark run
uv run python main.py summarize                      # compare all models
```

## Key conventions

- All console output uses `rich` (Console, Panel, Table, Live)
- Data validation happens at load time via `validate_items()` in `config.py` — fail fast, never silently fix data
- Test data must not contain UTF-8 BOM characters — clean at source
- `--think N` enables thinking mode with N-token budget; default is thinking disabled
- JSONL output goes to `benchmark_responses/` with one file per run
- Eval splits: `text_only_core` (runnable), `vision_extended` (needs images), `written_manual`/`written_auto` (needs scorers)

## CI/CD pipeline (workflow_dispatch, ordered)

1. **benchmark-preflight.yml** — manually dispatched on the self-hosted runner; verifies Ollama, datasets, and model availability, and writes `preflight_records/preflight_YYYYMMDD.json`
2. **benchmark-screen.yml** — manually dispatched after preflight; runs the 3x screening batch, checks completeness, applies the screening gate, and prepares publication metadata
3. **benchmark-run.yml** — manually dispatched for publication batches; runs the real benchmark batch and produces `repeat_summary` artifacts
4. **snapshot-pr.yml** — manually dispatched after benchmark-run completes; consumes prior batch data, builds/validates the dashboard, syncs `site/data/latest`, updates `registry/snapshots.json`, and opens a PR
5. **pages-deploy.yml** — triggered automatically when the snapshot PR merges to `main`; deploys the updated site to GitHub Pages

Steps 1 through 4 are `workflow_dispatch` only and must be run in order. Step 5 is automatic.

## Ops policy

- Samantha stays in the control plane: launch workflows, monitor turning points, and report status.
- Long-running benchmark work belongs in GitHub Actions or a detached worker session, not the live chat lane.
- Local foreground execution is for short bounded checks only, not preflight, screening, or publication runs.
- GitHub job summaries and uploaded artifacts are the preferred inspection surface; do not rely solely on the self-hosted runner filesystem.

## Dependencies

- Python 3.12+
- `rich` (declared in pyproject.toml)
- Ollama running locally
