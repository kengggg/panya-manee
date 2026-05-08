# Workflow and Orchestration Spec

## 1. Principle

Benchmark execution, publication, and deployment should be separated.

- benchmark execution = on the fixed Mac mini baseline
- snapshot publication = controlled PR workflow
- dashboard deployment = GitHub Pages workflow

Samantha should orchestrate these with deterministic scripts and `gh`, not manual stitching.

---

## 2. GitHub Actions Workflow Set

## A. `benchmark-run.yml`

### Purpose
Run benchmark batches on the Mac mini baseline using a self-hosted runner.

### Trigger
`workflow_dispatch` only

### Runner
```yaml
runs-on: [self-hosted, macOS, ARM64, macmini-m4-16gb, panya-manee-baseline]
```

### Inputs
- `batch_id`
- `models_json` or `models_manifest_path`
- `runs_per_model`
- `subjects`
- `think_mode`
- `machine_profile`

### Concurrency
Use workflow concurrency so only one benchmark job runs at a time.

Example:
```yaml
concurrency:
  group: panya-manee-benchmark-runner
  cancel-in-progress: false
```

### High-level job steps
1. checkout trusted ref from `main`
2. setup Python / uv
3. cleanup runner workspace
4. run `scripts/run_batch.py`
5. run `scripts/aggregate_model_batch.py` for each model
6. run `scripts/aggregate_batch_candidate.py`
7. upload artifacts

### Notes
- do not use matrix execution for benchmark jobs on this one machine
- serial execution is the point
- workflow should not auto-publish

---

## B. `snapshot-pr.yml`

### Purpose
Create a PR that updates the public snapshot and dashboard bundle.

### Trigger
`workflow_dispatch`

### Runner
GitHub-hosted:
```yaml
runs-on: ubuntu-latest
```

### Inputs
- `snapshot_version`
- `selection_file`

### High-level job steps
1. checkout repo
2. setup Python and site build dependencies
3. run `scripts/validate_snapshot.py`
4. run `scripts/publish_snapshot.py`
5. run `scripts/build_dashboard_bundle.py`
6. update `registry/snapshots.json`
7. create branch
8. commit files
9. open PR using `gh pr create`

### PR contents
Should include only:
- new published snapshot files
- updated dashboard bundle files
- registry updates if needed

---

## C. `pages-deploy.yml`

### Purpose
Deploy static dashboard to GitHub Pages after snapshot PR merge.

### Trigger
On push to `main` when relevant files change.

### Runner
GitHub-hosted.

### Required GitHub Pages actions
- `actions/configure-pages`
- `actions/upload-pages-artifact`
- `actions/deploy-pages`

### High-level job steps
1. checkout repo
2. install site dependencies
3. build site from `site/`
4. upload Pages artifact
5. deploy

### Notes
This matches GitHub’s current custom workflow model for Pages deployment.

---

## 3. Suggested Workflow Permissions

### `benchmark-run.yml`
Minimal permissions.

Example:
```yaml
permissions:
  contents: read
```

### `snapshot-pr.yml`
Needs PR and contents write.

Example:
```yaml
permissions:
  contents: write
  pull-requests: write
```

### `pages-deploy.yml`
Needs Pages deployment permissions.

Example:
```yaml
permissions:
  contents: read
  pages: write
  id-token: write
```

---

## 4. Self-hosted Runner Safety Model

Because the repo is public, benchmark workflow safety matters a lot.

### Safe pattern
- self-hosted runner used only for `workflow_dispatch`
- benchmark workflow checks out trusted branch only
- no `pull_request` jobs on self-hosted runner
- no fork code on self-hosted runner
- no arbitrary branch execution unless explicitly trusted

### Nice-to-have extra safeguards
- dedicated runner user
- explicit workspace cleanup before each run
- benchmark lock file for mutual exclusion
- keep secrets minimal

---

## 5. Samantha Orchestration Flow with `gh`

## A. Start a benchmark batch

Samantha should:
1. prepare batch id and model selection
2. trigger benchmark workflow
3. monitor run
4. read artifacts/results

Representative commands:
```bash
gh workflow run benchmark-run.yml \
  --repo kengggg/panya-manee \
  -f batch_id=mini-r10-20260415 \
  -f runs_per_model=10 \
  -f subjects=thai,math \
  -f think_mode=off \
  -f machine_profile=macmini-m4-16gb-ollama
```

Then:
```bash
gh run list --repo kengggg/panya-manee --workflow benchmark-run.yml --limit 5
gh run watch <run-id> --repo kengggg/panya-manee
```

## B. Publish a snapshot

After stable compatible artifacts exist, Samantha should:
1. prepare snapshot selection file
2. trigger snapshot PR workflow
3. review generated PR
4. report summary

Representative commands:
```bash
gh workflow run snapshot-pr.yml \
  --repo kengggg/panya-manee \
  -f snapshot_version=2026-04-15-launch \
  -f selection_file=benchmark_artifacts/selections/2026-04-15-launch.json
```

Then:
```bash
gh run list --repo kengggg/panya-manee --workflow snapshot-pr.yml --limit 5
gh run watch <run-id> --repo kengggg/panya-manee
gh pr list --repo kengggg/panya-manee --state open
```

## C. Review Pages deploy

After PR merge:
```bash
gh run list --repo kengggg/panya-manee --workflow pages-deploy.yml --limit 5
gh run watch <run-id> --repo kengggg/panya-manee
```

---

## 6. Snapshot Selection File

Samantha should not improvise model selection in chat. Use an explicit selection file.

Suggested path:
- `benchmark_artifacts/selections/<snapshot_version>.json`

Suggested shape:
```json
{
  "snapshot_version": "2026-04-15-launch",
  "selection_policy": "latest_stable_compatible",
  "machine_profile": "macmini-m4-16gb-ollama",
  "required_match": {
    "benchmark_scope": "mcq_text_only_v1",
    "dataset_version": "nt-p3-text-only-2026-04-09",
    "eval_split": "text_only_core",
    "prompt_version": "v1_answer_only",
    "scoring_version": "v1",
    "think_mode": "off"
  },
  "models": [
    {
      "model_id": "gemma4:e4b",
      "source_model_batch": "benchmark_artifacts/model_batches/mini-r10-20260409/gemma4-e4b.json"
    },
    {
      "model_id": "qwen3.5:9b",
      "source_model_batch": "benchmark_artifacts/model_batches/mini-r10-20260409/qwen35-9b.json"
    }
  ]
}
```

---

## 7. Dashboard Data Bundle

The site should consume only published snapshot data.

Recommended generated files under `site/public/data/`:
- `latest.json`
- `leaderboard.json`
- `models.json`
- `snapshots.json`
- `downloads.json`

### `latest.json`
- points to current published snapshot
- gives the site one stable entry point

Example:
```json
{
  "snapshot_version": "2026-04-15-launch",
  "leaderboard_path": "/data/leaderboard.json",
  "models_path": "/data/models.json"
}
```

---

## 8. PR Review Model

Because publication is PR-only, every snapshot update becomes reviewable.

Recommended PR body sections:
- snapshot version
- selected model aggregates
- compatibility gate summary
- stable threshold summary
- notable changes from previous snapshot
- links to raw bundle

---

## 9. Suggested Implementation Order

### Phase 1
Add deterministic scripts only.

### Phase 2
Add artifact and selection schema.

### Phase 3
Add benchmark-run workflow on self-hosted Mac mini.

### Phase 4
Add snapshot PR workflow.

### Phase 5
Add static dashboard in `site/` and Pages deploy.

This keeps risk low and lets you verify each layer.

---

## 10. Coding Execution Rule

When it is time to implement code:
- use **Claude**
- explicitly ask Claude to **ultrathink**
- keep changes script-driven and reviewable
- prefer small PRs, one layer at a time

Suggested implementation slices:
1. model-batch aggregation
2. snapshot validation + publication
3. dashboard bundle generation
4. GitHub workflow wiring
5. frontend site

---

## 11. Recommendation Summary

Best working model:
- one Mac mini baseline machine
- one self-hosted runner on that machine
- benchmark workflow manually triggered only
- no untrusted PRs on self-hosted runner
- snapshots published by PR only
- static site in `site/`
- GitHub Pages deploy from merged `main`
- Samantha orchestrates via `gh`
