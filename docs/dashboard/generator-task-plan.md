# Panya Manee Dashboard V1 Generator Task Plan

## Goal

Turn benchmark batch outputs into a published static snapshot bundle for the dashboard.

Target outputs:
- `manifest.json`
- `leaderboard.json`
- `model_cards.json`
- `examples.json`
- optional transparency artifacts

---

## 1. Source inputs

The generator should read from:
- benchmark row files from published batch runs
- repeat batch summary, if available
- model metadata mapping
- testbed metadata

Recommended input sources:
- `panya-manee/benchmark_responses/*.jsonl`
- `panya-manee/benchmark_responses/repeat_summary_*.json`
- `panya-manee-dashboard/config/models.json`
- `panya-manee-dashboard/config/testbed.json`

---

## 2. Generator stages

### Stage 1. Snapshot selection

Input:
- batch id or snapshot id

Tasks:
- identify included models
- identify included row files
- verify all included runs share the same benchmark scope
- verify all included runs belong to the same published batch
- assign `snapshot_id`

Output:
- normalized in-memory snapshot manifest

### Stage 2. Row normalization

Tasks:
- load each JSONL row
- validate required row fields
- derive missing metadata where possible:
  - `model_family`
  - `parameter_bucket`
  - `ram_fit_class`
  - `answer_only_compliant`
  - `output_length_chars`
- normalize skill tags to arrays
- attach `snapshot_id`, `host_label`, `benchmark_scope`, `benchmark_suite_id`

Output:
- normalized row list

### Stage 3. Aggregate model metrics

**Batch-aggregation rule:** Speed metrics and latency percentiles use data from **all runs in the published batch**, not a single run.

Tasks per model:
- compute Thai score rate
- compute Math score rate
- compute overall score rate
- compute balanced quality score
- compute parseable rate
- compute answer-only compliance rate
- compute p50 latency — across all per-question latencies from all batch runs
- compute p95 latency — across all per-question latencies from all batch runs
- compute questions per minute — batch-aggregated total items / total time
- compute correct per minute — batch-aggregated total correct / total time
- optionally compute tokens per second
- compute strongest and weakest skill tags (minimum n >= 2 items per tag; tie-break: higher item count, then alphabetical)

Output:
- model aggregate objects

### Stage 4. Badge assignment

Tasks:
- assign `Best Quality`
- assign `Best Thai`
- assign `Best Math`
- assign `Fastest on Testbed`
- assign `Best Small Model`
- allow ties

Output:
- leaderboard rows with badges
- model cards with badges

### Stage 5. Deterministic example selection

**Source:** Use the **canonical run** (the run with the lowest `run_index` in the published batch) for example selection, not aggregated data. The canonical run is used only for examples and raw outputs, not for ranking metrics.

Tasks per model:
- choose 2 good examples
- choose 2 bad examples
- use deterministic rules only
- generate `raw_output_truncated` = **first 200 characters** of raw output

Recommended rule order:
1. prioritize strongest skill tags for good examples
2. prioritize weakest skill tags for bad examples
3. prefer higher latency within tied candidates
4. break remaining ties with stable sort:
   - subject
   - question_id
   - raw_output

Output:
- example objects
- example ids per model

### Stage 6. Auto-summary generation

Tasks per model:
- map metric thresholds to fixed phrases
- generate deterministic summary text
- avoid freeform LLM generation in V1

Output:
- `auto_summary` field in model cards

### Stage 7. Artifact emission

Tasks:
- write `manifest.json`
- write `leaderboard.json`
- write `model_cards.json`
- write `examples.json`
- write `repeat_summary.json` into the transparency bundle
- copy raw row-level source files into the transparency bundle

Output:
- publishable snapshot directory

---

## 3. Recommended script layout

Suggested generator scripts:
- `scripts/build_snapshot.py`
- `scripts/validate_snapshot.py`
- `scripts/publish_snapshot.py`

### `build_snapshot.py`
Responsibilities:
- load inputs
- normalize rows
- compute aggregates
- assign badges
- select examples
- emit artifacts

### `validate_snapshot.py`
Responsibilities:
- schema checks
- cross-file consistency checks
- metric sanity checks
- duplicate id checks

### `publish_snapshot.py`
Responsibilities:
- copy validated snapshot bundle to publish directory
- optionally create downloadable zip bundle

---

## 4. Required config files

### `config/models.json`
Should define per-model metadata:
- `model_id`
- `model_family`
- `parameter_bucket`
- `ram_fit_class`
- optional human label

### `config/testbed.json`
Should define:
- `host_label`
- `chip`
- `ram_gb`
- `backend`

### `config/example_rules.json` (optional)
Should define:
- number of good examples
- number of bad examples
- tie-break ordering
- truncation rules

---

## 5. Recommended build command

Example:

```bash
python scripts/build_snapshot.py \
  --batch-id mini-r10-20260409 \
  --snapshot-id nt-p3-mcq-text-only-2026-04-09 \
  --out ./dist/nt-p3-mcq-text-only-2026-04-09
```

Then:

```bash
python scripts/validate_snapshot.py --dir ./dist/nt-p3-mcq-text-only-2026-04-09
python scripts/publish_snapshot.py --dir ./dist/nt-p3-mcq-text-only-2026-04-09
```

---

## 6. Validation checklist for generator output

The generator is correct if:
- every artifact has the same `snapshot_id`
- every model in leaderboard has a matching model card
- every example id referenced by model cards exists
- balanced quality score is computed correctly
- tie-breakers are applied correctly
- badges are reproducible
- example selection is reproducible
- output truncation is deterministic
- raw bundle matches the published snapshot

---

## 7. Suggested implementation order

1. add missing row fields in benchmark script
2. create `models.json` and `testbed.json`
3. implement row normalization helpers
4. implement metric aggregation
5. implement badge logic
6. implement deterministic example selection
7. emit snapshot artifacts
8. validate bundle
9. zip raw bundle for transparency
