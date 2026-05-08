# Panya-Manee Repo Integration + GitHub Actions Plan

## Goal

Turn `panya-manee` from a local benchmark script repo into a deterministic benchmark + publication pipeline where:

- benchmark execution stays tied to the **single Mac mini baseline machine**
- benchmark artifacts are produced by **deterministic scripts inside the repo**
- public publication happens only through explicit **snapshot releases**
- the dashboard is a **static site hosted on GitHub Pages** from the same repo
- Samantha can orchestrate runs and publishing with `gh`
- implementation work should use **Claude with an ultrathink prompt** when coding begins

---

## Core Recommendation

Use a hybrid architecture:

- **Self-hosted GitHub Actions runner on the Mac mini** for benchmark execution only
- **GitHub-hosted Actions** for validation, PR creation, static site build, and GitHub Pages deploy
- **PR-only publication** for public dashboard data

This keeps the benchmark faithful to the fixed hardware while still making the system programmatic and auditable.

---

## Important Security Note

Because `kengggg/panya-manee` is a **public repo**, a self-hosted runner must be handled carefully.

### Recommendation

Self-hosted runner is acceptable **only if**:

- it is used for **trusted manual workflows only**
- it is **never** used for `pull_request` from forks
- it is **never** used for arbitrary untrusted branch code
- benchmark workflows are triggered via `workflow_dispatch` on trusted refs only
- runner job concurrency is limited to one benchmark job at a time

### Hard rule

Do **not** run untrusted PR code on the Mac mini self-hosted runner.

---

## Suggested Runner Setup

### Runner scope

- Repository-level self-hosted runner dedicated to `kengggg/panya-manee`

### Suggested labels

- `self-hosted`
- `macOS`
- `ARM64`
- `macmini-m4-16gb`
- `panya-manee-baseline`

### Suggested operational constraints

- single runner only
- dedicated OS user if practical
- deterministic workspace path
- cleanup step before/after benchmark job
- no parallel benchmark jobs
- explicit concurrency group in workflows

---

## Repo Structure Plan

```text
panya-manee/
  benchmark_runner.py
  summarize_results.py
  main.py
  nt-tests/

  scripts/
    run_batch.py
    aggregate_model_batch.py
    aggregate_snapshot_candidate.py
    publish_snapshot.py
    build_dashboard_bundle.py
    validate_snapshot.py
    cleanup_runner_workspace.sh

  benchmark_responses/
    # raw per-run JSONL outputs (Layer 1)

  benchmark_artifacts/
    model_batches/
      <batch_id>/
        <model_slug>.json
    snapshot_candidates/
      <batch_id>.json
    snapshots/
      <snapshot_version>/
        leaderboard.json
        models.json
        manifest.json
        raw-bundle.zip

  registry/
    models.json
    machine_profiles.json
    benchmark_compatibility.json
    snapshots.json

  site/
    # static dashboard app

  .github/workflows/
    benchmark-run.yml
    snapshot-pr.yml
    pages-deploy.yml
```

---

## Artifact Model

### Layer 1 — Individual run artifact

Source of truth for each execution.

Suggested fields:
- `run_id`
- `batch_id`
- `model_id`
- `model_alias`
- `machine_profile`
- `dataset_version`
- `eval_split`
- `prompt_version`
- `scoring_version`
- `think_mode`
- `started_at`
- `finished_at`
- `output_file`
- per-question rows remain in JSONL

### Layer 2 — Model aggregate within batch

One canonical published unit per model per batch.

Suggested fields:
- `batch_id`
- `model_id`
- `runs_total`
- `status` = `testing` or `stable`
- `stable_threshold` = 10
- `mean_accuracy`
- `stdev_accuracy`
- `mean_thai_accuracy`
- `mean_math_accuracy`
- `mean_correct`
- `min_correct`
- `max_correct`
- `mean_time_s`
- `median_time_s`
- `mean_questions_per_min`
- `mean_correct_per_min`
- `mean_correct_per_gb_min`
- `memory_gb`
- `run_files[]`
- compatibility metadata block

### Layer 3 — Snapshot aggregate across models

Public homepage source.

Suggested fields:
- `snapshot_version`
- `published_at`
- `selected_model_results[]`
- `leaderboard_sort`
- `benchmark_scope`
- `dataset_version`
- `eval_split`
- `prompt_version`
- `scoring_version`
- `machine_profile`
- `think_mode`
- downloadable bundle links

---

## Compatibility Gates

A model aggregate may enter the same public snapshot only if these match exactly:

- `benchmark_scope`
- `dataset_version`
- `eval_split`
- `prompt_version`
- `scoring_version`
- `machine_profile`
- `think_mode`

Keep these as hard validation checks in `validate_snapshot.py`.

Soft metadata for audit only:
- Ollama version
- macOS version
- repo git SHA
- model digest
- notes

---

## Status Rules

- `< 10 runs` = `testing`
- `>= 10 runs` = `stable`
- Public homepage should include **published snapshot entries only**
- New model results should wait until included in a published snapshot

---

## Workflow Plan

### 1. `benchmark-run.yml`

**Purpose:** Run one or more models on the Mac mini baseline.

**Runner:** self-hosted Mac mini only

**Trigger:** `workflow_dispatch`

**Inputs:**
- `batch_id`
- `models_manifest_path` or JSON string
- `runs_per_model`
- `publish_raw_artifacts` (optional)

**Behavior:**
- checkout trusted ref
- cleanup workspace
- run models serially
- write raw Layer 1 outputs
- run `aggregate_model_batch.py` for each model
- run `aggregate_snapshot_candidate.py` for the batch
- upload generated artifacts

**Important:**
- single-job serial execution is preferred over matrix because the testbed is one machine
- add workflow `concurrency` so only one benchmark job can run at once

### 2. `snapshot-pr.yml`

**Purpose:** Create a PR that adds or updates published snapshot artifacts.

**Runner:** GitHub-hosted (`ubuntu-latest`)

**Trigger:** `workflow_dispatch`

**Inputs:**
- `snapshot_version`
- selected `batch_id` / selected model aggregate refs

**Behavior:**
- download or fetch stable model aggregates
- run `validate_snapshot.py`
- run `publish_snapshot.py`
- run `build_dashboard_bundle.py`
- create branch
- commit snapshot + dashboard data changes
- open PR

### 3. `pages-deploy.yml`

**Purpose:** Build and deploy static site to GitHub Pages.

**Runner:** GitHub-hosted (`ubuntu-latest`)

**Trigger:**
- on merge to `main` affecting `site/` or published snapshot data

**Behavior:**
- install frontend deps
- build static site
- use GitHub Pages Actions:
  - `actions/configure-pages`
  - `actions/upload-pages-artifact`
  - `actions/deploy-pages`

---

## Why a Self-hosted Runner Is Worth It

### Pros

- preserves benchmark integrity on the exact Mac mini baseline
- workflow becomes reproducible and auditable
- Samantha can drive it with `gh workflow run`
- easier long-term than ad hoc local shell sessions

### Cons

- security risk if misconfigured on a public repo
- runner machine maintenance is now part of the system
- machine availability controls workflow availability

### Recommendation

Use the self-hosted runner, but only for the benchmark-run workflow and only for trusted manual dispatches.

---

## Samantha Orchestration Plan

Samantha should orchestrate at the workflow level, not manually recalculate publication data in chat.

### Typical flow

1. Prepare a batch manifest
2. Trigger benchmark workflow with `gh workflow run`
3. Watch workflow with `gh run watch`
4. Review artifacts/status
5. Trigger snapshot PR workflow
6. Review PR / CI
7. Merge PR
8. Pages deploy runs automatically

### Samantha responsibilities

- choose compatible inputs
- enforce stable threshold rules
- enforce snapshot gating rules
- explain results to user
- use deterministic scripts as the only production path

---

## Coding Policy for Implementation

When implementation starts:

- use **Claude** for coding work
- explicitly ask Claude to **ultrathink**
- keep deterministic logic in scripts, not buried in chat
- prefer small reviewable PRs over one huge rewrite

---

## Recommended Implementation Phases

### Phase 1 — Internal deterministic artifact layer

Add scripts and artifact schemas inside repo:
- `run_batch.py`
- `aggregate_model_batch.py`
- `aggregate_snapshot_candidate.py`
- `validate_snapshot.py`

Goal:
- make current local process reproducible without changing public publishing yet

### Phase 2 — Snapshot publication layer

Add:
- `publish_snapshot.py`
- `build_dashboard_bundle.py`
- `registry/`

Goal:
- produce canonical publishable JSON bundle for the dashboard

### Phase 3 — GitHub Actions integration

Add:
- self-hosted benchmark workflow
- snapshot PR workflow
- Pages deploy workflow

Goal:
- full programmatic orchestration

### Phase 4 — Static dashboard in `site/`

Build a static site consuming published snapshot JSON.

Goal:
- homepage leaderboard
- model detail pages
- downloadable raw JSON bundles

---

## Final Recommendation

Best architecture for V1:

- **One repo**: `panya-manee`
- **One baseline benchmark machine**: Mac mini M4 16 GB
- **One self-hosted runner on that machine** for trusted benchmark execution only
- **PR-only snapshot publication**
- **Static dashboard in `site/`**
- **GitHub Pages** as the public host
- **Samantha orchestrates with `gh` + deterministic scripts**
