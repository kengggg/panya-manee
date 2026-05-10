# panya-manee

Local LLM arena that benchmarks small models on Thailand's NT (National Test) Grade 3 exam questions via Ollama. Tests Thai language comprehension and math reasoning.

License: Apache-2.0

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) running locally
- At least one model pulled (e.g. `ollama pull qwen3:0.6b`)

## Setup

```bash
uv sync            # or: pip install -e .
```

## Usage

### Run a benchmark

```bash
uv run python main.py run --model qwen3:0.6b
uv run python main.py run --model gemma3:1b --subjects thai --run-id run002
uv run python main.py run --model llama3.2:1b --dry-run  # no API calls
```

### Thinking models

For models with thinking/reasoning (e.g. qwen3.5), use `--think` to set a token budget:

```bash
uv run python main.py run --model qwen3.5:9b --think 4096   # enable thinking, 4096 token budget
uv run python main.py run --model qwen3.5:9b --no-think     # disable thinking (default)
```

If the model hits the token limit mid-think (empty content), the runner will:
- Attempt to parse an answer from the thinking output (`thinking_fallback`)
- Show `⚠ BUDGET` warnings per-item and in the summary

### Summarize results

```bash
uv run python main.py summarize                  # all runs
uv run python main.py summarize --run-id run001  # specific run
```

## Dashboard

A static public dashboard shows benchmark results at a glance. No build step required — plain HTML/CSS/JS consuming pre-built JSON.

### Local preview

```bash
# If you have a verified snapshot in dist/, publish it to site/data/latest/:
python scripts/publish_snapshot.py --batch-id ntp3-pub-r1-20260510

# Or just preview what's already synced:
sh scripts/serve.sh
# Open http://localhost:8000
```

### Publish flow

```bash
# 1. Build, validate, sync to site, update registry — one command:
python scripts/publish_snapshot.py --batch-id ntp3-pub-r1-20260510

# 2. Dry-run (build + validate only, no side effects):
python scripts/publish_snapshot.py --batch-id ntp3-pub-r1-20260510 --dry-run

# 3. Commit the updated site data and registry:
git add site/data/ registry/
git commit -m "Publish snapshot nt-p3-mcq-text-only-ntp3-pub-r1-20260510"

# 4. Open a PR to main. After merge, GitHub Pages deploys automatically.
```

### Benchmark publication pipeline

```mermaid
flowchart TD
    A[Select candidate models] --> B[Benchmark Preflight<br/>benchmark-preflight.yml]
    B --> B1[Check Ollama reachable]
    B --> B2[Verify dataset plus inputs]
    B --> B3[Record model digests]
    B --> B4[Upload preflight artifact<br/>preflight-YYYYMMDD]

    B --> C{Optional smoke?}
    C -->|Yes| D[Smoke batch<br/>operational confidence only]
    C -->|No| E[Verified batch]
    D --> E

    E[Benchmark Verified<br/>benchmark-verified.yml] --> E1[Run full benchmark<br/>1 run per model]
    E1 --> E2[Check batch completeness]
    E2 --> E3[Fetch matching preflight artifact]
    E3 --> E4[Verification gate<br/>single canonical run]

    E4 --> G2{Parse rate >= 95%?}
    G2 -->|No| X1[Exclude model]
    G2 -->|Yes| G3{Accuracy > 25%?}
    G3 -->|No| X1
    G3 -->|Yes| G4{Digest matches preflight<br/>when requested?}
    G4 -->|No| X1
    G4 -->|Yes| P[Publishable survivor]

    X1 --> R[verification_report.json]
    P --> R

    R --> S[Stage verified artifact bundle]
    S --> T[Upload verified artifact]

    T --> U[Snapshot PR<br/>snapshot-pr.yml]
    U --> U1[Fetch verified artifact bundle]
    U1 --> U2[Build snapshot]
    U2 --> U3[Validate snapshot]
    U3 --> U4[Sync site/data/latest]
    U4 --> U5[Update registry/snapshots.json]
    U5 --> U6[Open PR]

    U6 --> V[Merge PR]
    V --> W[pages-deploy.yml]
    W --> Z[Published dashboard]

    N[Deprecated old flow:<br/>benchmark-screen.yml] -. stop sign only .-> E
```

Current production path:

- `benchmark-preflight.yml`
- optional smoke batch
- `benchmark-verified.yml` on all candidates, 1x each
- post-hoc survivor filter from `verification_report_{batch_id}.json`
- `snapshot-pr.yml`
- `pages-deploy.yml`

The dashboard consumes one canonical `site/data/latest/current.json`. That file
is generated from one or more compatible snapshot bundles; when a later input
contains the same `model_id`, it replaces the earlier row for the next current
view. This keeps the 2026-05-10 restart baseline comparable while allowing
future snapshots to append only new or retested model batches.

### New model lifecycle

Use this path when adding a model after the `ntp3-pub-r1-20260510` restart
baseline. Benchmarks are manual because they must run on the fixed Mac mini
self-hosted runner (`panya-manee-baseline`).

#### 1. Register the model if needed

1. Pull and smoke-test it on the Mac mini baseline:

   ```bash
   ollama pull granite4.1:8b
   ollama run granite4.1:8b 'ตอบเป็นเลข 1-4 เท่านั้น: 2+2=? 1)3 2)4 3)5 4)6'
   ```

2. Add the Ollama model ID to `registry/active_roster.json`.
3. Add metadata to `registry/models.json` if the model is not already there.
4. Verify inputs locally:

   ```bash
   uv run python scripts/verify_batch_inputs.py \
     --batch-id ntp3-add-r1-YYYYMMDD-modelslug \
     --models 'granite4.1:8b' \
     --runs-per-model 1 \
     --subjects thai,math
   ```

Commit and push those registry/doc changes before running Actions if the model
should become part of the future active roster. Historical snapshots are not
edited; roster removals affect future snapshots only.

#### 2. Run the benchmark

Run only the new model:

```bash
gh workflow run benchmark-verified.yml \
  --repo kengggg/panya-manee \
  -f batch_id='ntp3-add-r1-YYYYMMDD-modelslug' \
  -f models='granite4.1:8b' \
  -f dry_run=false
```

Or rerun the whole active roster:

```bash
MODELS=$(jq -r '.models | join(",")' registry/active_roster.json)
gh workflow run benchmark-verified.yml \
  --repo kengggg/panya-manee \
  -f batch_id='ntp3-pub-r1-YYYYMMDD' \
  -f models="$MODELS" \
  -f dry_run=false
```

GitHub UI equivalent:

1. GitHub repo → **Actions** → **Benchmark Verified** → **Run workflow**
2. Branch: `main`
3. `batch_id`: e.g. `ntp3-add-r1-YYYYMMDD-modelslug`
4. `models`: comma-separated Ollama model IDs
5. `dry_run`: `false`

Watch and download the artifact:

```bash
gh run list --repo kengggg/panya-manee --workflow 'Benchmark Verified' --limit 3
gh run watch <run-id> --repo kengggg/panya-manee --exit-status
gh run download <run-id> --repo kengggg/panya-manee \
  --name benchmark-verified-ntp3-add-r1-YYYYMMDD-modelslug
```

The artifact contains `repeat_summary_<batch_id>.json`, raw per-model JSONL
responses, and `verification_report_<batch_id>.json`.

#### 3. Publish the result

Publish via **Snapshot PR** after `Benchmark Verified` succeeds:

```bash
gh workflow run snapshot-pr.yml \
  --repo kengggg/panya-manee \
  -f batch_id='ntp3-add-r1-YYYYMMDD-modelslug' \
  -f source_run_id='<benchmark-verified-run-id>' \
  -f update_current_manifest=true \
  -f dry_run=false
```

GitHub UI equivalent:

1. GitHub repo → **Actions** → **Snapshot PR** → **Run workflow**
2. Branch: `main`
3. `batch_id`: the batch that just passed verification
4. `source_run_id`: the numeric `Benchmark Verified` run ID
5. leave `snapshot_id` blank unless intentionally overriding it
6. `update_current_manifest`: `true` for the normal current-dashboard path
7. `current_id`: optional; set only when intentionally renaming the current view
8. `dry_run`: `false`
9. Merge the PR it opens. `pages-deploy.yml` deploys automatically after merge.

For append-only publication, keep `ntp3-pub-r1-20260510` as the first/current
baseline source and add the new snapshot after it. The ordered current manifest
is just the source list used to build `site/data/latest/current.json`; later
sources replace earlier rows for duplicate `model_id` values:

```json
{
  "current_id": "ntp3-current-YYYYMMDD",
  "sources": [
    {"snapshot_dir": "dist/nt-p3-mcq-text-only-ntp3-pub-r1-20260510"},
    {"snapshot_dir": "dist/nt-p3-mcq-text-only-ntp3-add-r1-YYYYMMDD-modelslug"}
  ]
}
```

`snapshot-pr.yml` updates this manifest automatically by default. If the new
snapshot source is already present, it updates that source in place rather than
creating a duplicate, preserving the original source order. The workflow also
commits `dist/<snapshot_id>/` and `dist/<snapshot_id>.zip` because future
manifest rebuilds need those source bundles to exist in the repo.

Build from that manifest when preparing or checking `current.json` locally:

```bash
uv run python scripts/build_current_json.py \
  --manifest registry/current-manifest.json \
  --out site/data/latest/current.json
uv run python scripts/verify_site.py
```

#### 4. View results

- Public dashboard: <https://kengggg.github.io/panya-manee/>
- GitHub Pages deployment status: repo → **Actions** → **Deploy Dashboard to GitHub Pages**
- Raw benchmark artifact: repo → **Actions** → the `Benchmark Verified` run → **Artifacts**
- Snapshot publication PR: repo → **Pull requests**, title starts with `Publish snapshot:`

`benchmark-screen.yml` is deprecated and intentionally not part of the live publication flow.

### Individual scripts

| Script | Purpose |
|---|---|
| `scripts/build_snapshot.py` | Build snapshot bundle from batch data |
| `scripts/build_current_json.py` | Merge one or more snapshot bundles into the single dashboard `current.json` |
| `scripts/validate_snapshot.py` | Validate snapshot cross-file consistency |
| `scripts/sync_site_data.py` | Copy snapshot files to `site/data/latest/` and write `current.json` |
| `scripts/verify_site.py` | Verify site HTML/JS wiring to data |
| `scripts/publish_snapshot.py` | End-to-end: build → validate → sync → registry |

### GitHub Pages deployment

The `.github/workflows/pages-deploy.yml` workflow deploys `site/` to GitHub Pages on push to `main` when `site/**` or `registry/snapshots.json` changes. Prerequisites:

1. In repo Settings → Pages → Source, select **GitHub Actions**
2. Ensure `site/data/latest/` contains snapshot data (committed via publish flow above)


## Development checks

```bash
uv sync --dev
uv run pytest -q
uv run python scripts/verify_site.py
```

For a no-network/no-Ollama smoke run of the benchmark path:

```bash
uv run python main.py run --model qwen3:0.6b --subjects thai --run-id restore-smoke --dry-run
```

Restored OpenClaw workspace notes live in `docs/RESTORED_OPENCLAW_20260508.md`. Dashboard planning/reference docs from the earlier workspace are under `docs/dashboard/`.

## License

This project is licensed under the Apache License, Version 2.0. See `LICENSE`.

## How it works

1. Loads MCQ items from `nt-tests/` JSON files
2. Validates data integrity (no BOM, required keys present)
3. Prompts the model in Thai to answer with just a digit (1-4)
4. Parses model output and compares to ground truth
5. Saves detailed JSONL results to `benchmark_responses/`
6. Displays live progress and rich summary by subject and skill tag

## Test Data Landscape (as of 2026-04-09)

**Source**: NT (National Test) Grade 3, Thailand — Years 2565, 2566, 2567 (2022–2024)

180 total items: 90 Thai + 90 Math, evenly split across 3 years (30 per year per subject).

### Eval splits

| Split | Thai | Math | Status |
|---|---|---|---|
| `text_only_core` | 60 | 33 | Runnable now |
| `vision_extended` | 18 | 45 | Needs image extraction from PDFs |
| `written_manual` | 12 | 3 | Needs human/rubric scoring |
| `written_auto` | — | 9 | Needs exact-match scorer |

### Answer types

| Type | Thai | Math |
|---|---|---|
| `mcq_single` | 78 | 78 |
| `short_written` | 9 | — |
| `free_written` | 3 | — |
| `numeric_short` | — | 9 |
| `worked_solution` | — | 3 |

### Runnable now — skill tags

#### Thai (60 items, all MCQ)

| skill_tag | count |
|---|---|
| reading_comprehension | 19 |
| reading_literature | 11 |
| parts_of_speech | 7 |
| moral_application | 6 |
| moral_extraction | 6 |
| sentence_type | 6 |
| reasoning_question | 5 |
| word_meaning | 4 |
| story_prediction | 3 |
| applied_reading | 3 |
| judgment_from_text | 3 |
| verb_identification | 3 |
| standard_thai | 3 |

Plus 27 more tags with 1–2 items each (spelling, classification, pronoun identification, etc.)

#### Math (33 items, all MCQ)

| skill_tag | count |
|---|---|
| word_problem | 7 |
| fraction_addition | 7 |
| fraction_subtraction | 7 |
| data_interpretation | 6 |
| comparison | 6 |
| length | 5 |
| unit_conversion | 5 |
| two_step_problem | 4 |
| table_reading | 4 |
| number_pattern | 3 |
| pictograph | 3 |
| addition_subtraction_with_units | 2 |
| number_ordering | 2 |
| division | 2 |
| multiplication | 2 |
| time | 2 |
| duration | 2 |
| money | 1 |
| capacity | 1 |

### Not runnable yet — skill tags

#### Thai (30 items: 18 vision + 12 written)

| skill_tag | mcq_single | free_written | short_written | total |
|---|---|---|---|---|
| visual_reading | 8 | | | 8 |
| creative_writing | | 3 | 3 | 6 |
| information_identification | 4 | | | 4 |
| image_description | | | 3 | 3 |
| sentence_writing | | | 3 | 3 |
| slogan_writing | | | 3 | 3 |
| short_reasoning | | | 3 | 3 |
| literature_response | | | 3 | 3 |
| imaginative_story | | 3 | | 3 |
| map_reading | 3 | | | 3 |
| visual_inference | 3 | | | 3 |

Plus 14 more tags with 1–2 items each.

#### Math (57 items: 45 vision + 9 written_auto + 3 written_manual)

| skill_tag | mcq_single | numeric_short | worked_solution | total |
|---|---|---|---|---|
| word_problem | 7 | 6 | 3 | 16 |
| multiplication | 5 | 3 | 2 | 10 |
| addition_subtraction_with_units | 7 | 1 | 1 | 9 |
| weight | 6 | 3 | | 9 |
| money | 3 | 4 | 1 | 8 |
| time | 7 | | | 7 |
| duration | 7 | | | 7 |
| unit_conversion | 4 | 1 | | 5 |
| capacity | 5 | | | 5 |
| addition | 2 | 1 | 2 | 5 |
| fraction_comparison | 4 | | | 4 |
| length | 4 | | | 4 |

Plus 17 more tags with 1–3 items each.
