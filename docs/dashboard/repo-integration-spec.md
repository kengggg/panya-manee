# Panya-Manee Repo Integration Spec

## 1. Goal

Move the benchmark pipeline into the `panya-manee` repo in a deterministic, production-shaped way.

This spec assumes:
- one fixed benchmark machine: **Mac mini M4 / 16GB / Ollama**
- benchmark execution must stay tied to that machine
- public publishing is controlled by **snapshot release**, not by every raw run
- public dashboard is hosted from the repo via **GitHub Pages**
- publication happens through **PR-only** updates

---

## 2. Recommended Repo Layout

```text
panya-manee/
  README.md
  main.py
  benchmark_runner.py
  summarize_results.py
  config.py
  nt-tests/

  scripts/
    run_batch.py
    aggregate_model_batch.py
    aggregate_batch_candidate.py
    validate_snapshot.py
    publish_snapshot.py
    build_dashboard_bundle.py
    runner_cleanup.sh

  benchmark_responses/
    # raw per-run jsonl files, one file per actual run

  benchmark_artifacts/
    model_batches/
      <batch_id>/
        <model_slug>.json
    batch_candidates/
      <batch_id>.json
    snapshots/
      <snapshot_version>/
        manifest.json
        leaderboard.json
        models.json
        raw-bundle.zip

  registry/
    models.json
    machine_profiles.json
    compatibility.json
    snapshots.json

  site/
    package.json
    src/
    public/

  .github/
    workflows/
      benchmark-run.yml
      snapshot-pr.yml
      pages-deploy.yml
```

---

## 3. Artifact Layers

### Layer 1, raw run artifact

Unit: one actual benchmark execution.

Storage:
- `benchmark_responses/responses_<run_id>_<timestamp>.jsonl`

Characteristics:
- immutable
- produced directly by benchmark execution
- includes per-question outputs

### Layer 2, model aggregate in a batch

Unit: one model aggregated across all runs in one batch.

Storage:
- `benchmark_artifacts/model_batches/<batch_id>/<model_slug>.json`

Characteristics:
- deterministic summary of raw Layer 1 files
- canonical model result for that model within that batch
- can be `testing` or `stable`

### Layer 3, batch candidate aggregate

Unit: all eligible model aggregates found for one execution batch.

Storage:
- `benchmark_artifacts/batch_candidates/<batch_id>.json`

Characteristics:
- internal candidate artifact
- not automatically public
- used to help compose a future snapshot

### Layer 4, published snapshot

Unit: one public publication bundle.

Storage:
- `benchmark_artifacts/snapshots/<snapshot_version>/...`

Characteristics:
- contains one chosen published result per model
- public homepage reads this only
- created only through explicit publish step

---

## 4. Status Model

### Raw run status
- always raw
- never public by itself

### Model batch status
- `testing` if runs < 10
- `stable` if runs >= 10

### Snapshot status
- `draft` before PR merge
- `published` after merge to `main`

Public rules:
- homepage includes **published snapshot entries only**
- models with fewer than 10 runs do not enter public homepage
- new model results wait for a snapshot release

---

## 5. Compatibility Gates

The following must match for model aggregates to appear in the same public snapshot:

- `benchmark_scope`
- `dataset_version`
- `eval_split`
- `prompt_version`
- `scoring_version`
- `machine_profile`
- `think_mode`

Suggested source of truth:
- `registry/compatibility.json`

Example:
```json
{
  "required_match_fields": [
    "benchmark_scope",
    "dataset_version",
    "eval_split",
    "prompt_version",
    "scoring_version",
    "machine_profile",
    "think_mode"
  ]
}
```

Soft metadata for audit, not blocking:
- `ollama_version`
- `repo_git_sha`
- `model_digest`
- `os_version`
- freeform notes

---

## 6. Per-model Selection Rule for Snapshot

A model may have multiple stable historical model-batch aggregates.

Recommended snapshot selection rule:
- choose the **latest stable compatible model aggregate** for each model

This should be explicit in snapshot manifest, not inferred later.

---

## 7. Registry Files

### `registry/models.json`

Purpose:
- canonical list of tracked models
- human labels and metadata for dashboard

Suggested shape:
```json
{
  "models": [
    {
      "model_id": "gemma4:e4b",
      "slug": "gemma4-e4b",
      "display_name": "Gemma 4 e4b",
      "provider": "ollama",
      "family": "gemma4",
      "size_label": "e4b",
      "active": true
    }
  ]
}
```

### `registry/machine_profiles.json`

Purpose:
- define approved benchmark testbeds

Suggested shape:
```json
{
  "machine_profiles": [
    {
      "machine_profile": "macmini-m4-16gb-ollama",
      "label": "Apple Mac mini M4 / 16GB / Ollama",
      "cpu": "Apple M4",
      "memory_gb": 16,
      "runner_label": "panya-manee-baseline"
    }
  ]
}
```

### `registry/snapshots.json`

Purpose:
- append-only index of published snapshots

Suggested shape:
```json
{
  "snapshots": [
    {
      "snapshot_version": "2026-04-15-launch",
      "published_at": "2026-04-15T12:00:00Z",
      "manifest_path": "benchmark_artifacts/snapshots/2026-04-15-launch/manifest.json"
    }
  ]
}
```

---

## 8. Script Responsibilities

### `scripts/run_batch.py`

Purpose:
- run one or more models serially for N runs each on the Mac mini

Inputs:
- `--batch-id`
- `--models-file` or `--model`
- `--runs-per-model`
- `--subjects`
- `--machine-profile`
- `--think-mode`

Outputs:
- raw JSONL files in `benchmark_responses/`
- optional local batch manifest

Hard rules:
- serial only
- deterministic run naming
- explicit machine metadata capture

### `scripts/aggregate_model_batch.py`

Purpose:
- gather all raw runs for one model in one batch
- compute Layer 2 canonical aggregate

Inputs:
- `--batch-id`
- `--model-id`

Outputs:
- `benchmark_artifacts/model_batches/<batch_id>/<model_slug>.json`

### `scripts/aggregate_batch_candidate.py`

Purpose:
- combine all model aggregates present in a batch into one candidate file

Inputs:
- `--batch-id`

Outputs:
- `benchmark_artifacts/batch_candidates/<batch_id>.json`

### `scripts/validate_snapshot.py`

Purpose:
- validate snapshot input selection against hard gates and stable threshold

Inputs:
- `--selection-file`

Outputs:
- exit code only or validation report JSON

### `scripts/publish_snapshot.py`

Purpose:
- build the public snapshot bundle from selected stable model aggregates

Inputs:
- `--snapshot-version`
- `--selection-file`

Outputs:
- `manifest.json`
- `leaderboard.json`
- `models.json`
- `raw-bundle.zip`

### `scripts/build_dashboard_bundle.py`

Purpose:
- convert published snapshot into static dashboard JSON contract

Inputs:
- `--snapshot-version`

Outputs:
- files under `site/public/data/` or equivalent

---

## 9. File Naming Recommendations

### Batch id
Execution identity.

Recommended pattern:
- `mini-r10-20260409`
- `mini-r10-20260415-newmodels`

### Snapshot version
Publication identity.

Recommended pattern:
- `2026-04-15-launch`
- `2026-04-22-update-1`

Recommendation:
- human-readable snapshot names are better than opaque counters

---

## 10. Data Flow

### Initial launch snapshot

1. choose launch model set
2. run all models on baseline Mac mini
3. aggregate each model in batch
4. aggregate batch candidate
5. validate stable threshold and compatibility
6. publish snapshot
7. build site bundle
8. open PR
9. merge
10. deploy Pages

### Later new model

1. create new batch id
2. run only new model on baseline Mac mini
3. aggregate model batch
4. leave as internal if needed
5. when ready, compose future snapshot using latest stable compatible result per model
6. publish snapshot via PR

### Retest of old model

1. create new batch id
2. run retest
3. aggregate model batch
4. future snapshot decides whether this new retest replaces old published result

---

## 11. PR-only Publication Policy

Anything public-facing should land only by PR:
- published snapshot files
- dashboard bundle files
- `site/` changes
- registry updates

This keeps homepage publication controlled and reviewable.

---

## 12. Strong Recommendation

For V1, do not overcomplicate with databases or live APIs.

Use:
- deterministic Python scripts
- JSON artifacts in repo
- static frontend in `site/`
- GitHub Pages deploy

That is enough for a clean, transparent first version.
