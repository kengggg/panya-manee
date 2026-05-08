# Panya Manee Dashboard V1 JSON Schema Spec

## Purpose

This document defines the static JSON bundle for the public **NT P3 MCQ Text-Only** dashboard.

The bundle is snapshot-based.
A snapshot is the atomic published unit for the dashboard.

---

## 1. Global rules

### 1.1 Encoding
- UTF-8

### 1.2 Numeric conventions
- rates are decimal fractions from `0.0` to `1.0`
- latency is in milliseconds unless otherwise stated
- throughput metrics are explicit in their unit name

### 1.3 Snapshot rule
Every artifact in the same published bundle must share the same:
- `snapshot_id`
- `benchmark_scope`
- `testbed.host_label`

### 1.4 V1 benchmark scope
- `mcq_text_only_v1`

### 1.5 Active V1 suite ids
- `thai_mcq_text_only_all`
- `math_mcq_text_only_all`
- `overall_mcq_text_only_all`

---

## 2. Artifact list

Required artifacts:
- `manifest.json`
- `leaderboard.json`
- `model_cards.json`
- `examples.json`

Optional transparency artifacts:
- `results.jsonl`
- `repeat_summary.json`

---

## 3. Enum definitions

### 3.1 `benchmark_scope`
Allowed values for V1:
- `mcq_text_only_v1`

### 3.2 `parameter_bucket`
Allowed values:
- `<1B`
- `1B–4B`
- `4B–8B`
- `8B+`
- `unknown`

### 3.3 `ram_fit_class`
Allowed values:
- `fits_comfortably_16gb`
- `fits_tightly_16gb`
- `does_not_fit_16gb`
- `unknown`

### 3.4 `run_status`
Allowed values:
- `success`
- `partial`
- `failed`

### 3.5 `subject`
Allowed values for V1:
- `thai`
- `math`

### 3.6 badge values
Allowed values:
- `Best Quality`
- `Best Thai`
- `Best Math`
- `Fastest on Testbed`
- `Best Small Model`

---

## 4. Derived metric formulas

### 4.1 Balanced quality score
```text
balanced_quality_score = (thai_score_rate + math_score_rate) / 2
```

### 4.2 Overall score rate
```text
overall_score_rate = total_correct / total_items
```

### 4.3 Parseable rate
```text
parseable_rate = parseable_items / total_items
```

### 4.4 Answer-only compliance rate
```text
answer_only_compliance_rate = compliant_items / total_items
```

### 4.5 Questions per minute
```text
questions_per_min = total_items_across_batch / total_latency_minutes_across_batch
```

### 4.6 Correct per minute
```text
correct_per_min = total_correct_across_batch / total_latency_minutes_across_batch
```

### 4.7 Tokens per second
Optional technical metric.
```text
throughput_toks_per_sec = total_eval_tokens / total_eval_seconds
```

---

## 5. `manifest.json`

### Purpose
Describes the published snapshot and the artifact bundle.

### Required fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `snapshot_id` | string | yes | unique published snapshot id |
| `published_at` | string | yes | ISO-8601 timestamp |
| `benchmark_name` | string | yes | human-readable benchmark name |
| `benchmark_label` | string | yes | public label, e.g. `NT P3 MCQ Text-Only` |
| `benchmark_scope` | string | yes | must be `mcq_text_only_v1` for V1 |
| `suite_ids` | array[string] | yes | active suite ids included in snapshot |
| `testbed` | object | yes | fixed benchmark machine info |
| `snapshot_notes` | object | yes | UI / transparency notes |
| `artifacts` | object | yes | file names for bundle members |

### `testbed` fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `host_label` | string | yes | public machine label |
| `chip` | string | yes | e.g. `M4` |
| `ram_gb` | number | yes | e.g. `16` |
| `backend` | string | yes | e.g. `ollama` |
| `context_window` | number or null | yes | nullable |
| `quantization` | string or null | yes | nullable if mixed / unknown |

### Example
```json
{
  "snapshot_id": "nt-p3-mcq-text-only-2026-04-09",
  "published_at": "2026-04-09T14:40:00+07:00",
  "benchmark_name": "NT P3 Local LLM Benchmark",
  "benchmark_label": "NT P3 MCQ Text-Only",
  "benchmark_scope": "mcq_text_only_v1",
  "suite_ids": [
    "thai_mcq_text_only_all",
    "math_mcq_text_only_all",
    "overall_mcq_text_only_all"
  ],
  "testbed": {
    "host_label": "Apple Mac mini M4 / 16GB / Ollama",
    "chip": "M4",
    "ram_gb": 16,
    "backend": "ollama",
    "context_window": null,
    "quantization": null
  },
  "snapshot_notes": {
    "public_outputs_truncated": true,
    "badges_enabled": true,
    "auto_model_summary": true,
    "ranking_excludes_image_required": true,
    "ranking_excludes_human_checked": true,
    "raw_bundle_downloadable": true
  },
  "artifacts": {
    "leaderboard": "leaderboard.json",
    "model_cards": "model_cards.json",
    "examples": "examples.json"
  }
}
```

---

## 6. `leaderboard.json`

### Purpose
One row per model for the public leaderboard.

### Top-level fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `snapshot_id` | string | yes | must match manifest |
| `benchmark_scope` | string | yes | must match manifest |
| `rows` | array[object] | yes | leaderboard rows |

### Row fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `rank` | integer | yes | after tie-breakers |
| `model_id` | string | yes | display identifier |
| `model_family` | string | yes | e.g. `gemma4` |
| `parameter_bucket` | string | yes | enum |
| `ram_fit_class` | string | yes | enum |
| `balanced_quality_score` | number | yes | main rank metric |
| `thai_score_rate` | number | yes | 0-1 |
| `math_score_rate` | number | yes | 0-1 |
| `overall_score_rate` | number | yes | item-weighted overall |
| `parseable_rate` | number | yes | 0-1 |
| `answer_only_compliance_rate` | number | yes | 0-1 |
| `latency_p50_ms` | number | yes | median latency across all batch runs |
| `latency_p95_ms` | number | yes | p95 latency across all batch runs |
| `questions_per_min` | number | yes | batch-aggregated speed metric |
| `correct_per_min` | number | yes | batch-aggregated efficiency metric |
| `item_count` | integer | yes | total included items |
| `badges` | array[string] | yes | 0+ badge labels |

### Notes
- ties are allowed
- leaderboard should still output a stable rank value
- if tied, tied rows may share the same rank
- `latency_p50_ms` and `latency_p95_ms` are computed across all individual question latencies from **all runs in the batch**
- `questions_per_min` and `correct_per_min` are **batch-aggregated** across the full published batch, not from a single canonical run

---

## 7. `model_cards.json`

### Purpose
One detail object per model.

### Top-level fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `snapshot_id` | string | yes | must match manifest |
| `benchmark_scope` | string | yes | must match manifest |
| `models` | array[object] | yes | one detail object per model |

### Model object fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `model_id` | string | yes | same as leaderboard |
| `model_family` | string | yes | same as leaderboard |
| `parameter_bucket` | string | yes | enum |
| `ram_fit_class` | string | yes | enum |
| `testbed` | object | yes | minimal machine info |
| `metrics` | object | yes | detail metrics |
| `strengths` | array[object] | yes | strongest 3 skill tags |
| `weaknesses` | array[object] | yes | weakest 3 skill tags |
| `badges` | array[string] | yes | badge labels |
| `auto_summary` | string | yes | deterministic generated text |
| `example_ids` | object | yes | references into `examples.json` |

### `metrics` fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `balanced_quality_score` | number | yes | 0-1 |
| `thai_score_rate` | number | yes | 0-1 |
| `math_score_rate` | number | yes | 0-1 |
| `overall_score_rate` | number | yes | 0-1 |
| `parseable_rate` | number | yes | 0-1 |
| `answer_only_compliance_rate` | number | yes | 0-1 |
| `latency_p50_ms` | number | yes | across all batch runs |
| `latency_p95_ms` | number | yes | across all batch runs |
| `questions_per_min` | number | yes | batch-aggregated speed metric |
| `correct_per_min` | number | yes | batch-aggregated efficiency metric |
| `throughput_toks_per_sec` | number | no | optional technical metric |
| `item_count` | integer | yes | total items |
| `total_correct` | integer | yes | absolute count |

### `strengths` / `weaknesses` item fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `skill_tag` | string | yes | one skill tag |
| `score_rate` | number | yes | 0-1 |
| `correct` | integer | yes | absolute count |
| `total` | integer | yes | denominator, must be >= 2 |

**Rules:**
- Only include skill tags with **n >= 2** items (minimum 2 questions for that tag).
- Tie-break for ranking: higher item count first, then alphabetical by tag name.
- If fewer than 3 tags qualify, show only the qualifying tags.

### `example_ids` fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `good` | array[string] | yes | recommended size 2 |
| `bad` | array[string] | yes | recommended size 2 |

---

## 8. `examples.json`

### Purpose
Stores heavier example payloads outside the leaderboard and model card payloads.

### Top-level fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `snapshot_id` | string | yes | must match manifest |
| `benchmark_scope` | string | yes | must match manifest |
| `examples` | array[object] | yes | referenced by `example_ids` |

### Example fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `example_id` | string | yes | unique within snapshot |
| `model_id` | string | yes | source model |
| `subject` | string | yes | `thai` or `math` |
| `question_id` | integer | yes | benchmark item id |
| `skill_tag` | array[string] | yes | 1+ tags |
| `curriculum_standard` | string or null | yes | nullable if missing |
| `is_correct` | boolean | yes | correctness |
| `correct_answer` | string | yes | expected answer token |
| `parsed_answer` | string or null | yes | nullable if unparsable |
| `raw_output_truncated` | string | yes | first 200 characters of raw output |
| `raw_output_full` | string | yes | expanded payload |
| `latency_ms` | number | yes | per-question latency |
| `selection_reason` | string | yes | deterministic selection label |

### Recommended `selection_reason` values
- `good_top_skill`
- `good_high_latency`
- `bad_weak_skill`
- `bad_high_latency`

---

## 9. Optional raw row schema

This is not necessarily rendered directly in the UI, but it should be consistent enough for transparent reuse.

### Required row fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `snapshot_id` | string | yes | published snapshot id |
| `run_id` | string | yes | source run id |
| `benchmark_scope` | string | yes | scope id |
| `benchmark_suite_id` | string | yes | suite id |
| `run_status` | string | yes | enum |
| `host_label` | string | yes | testbed label |
| `model_id` | string | yes | model identifier |
| `model_family` | string | yes | derived family |
| `parameter_bucket` | string | yes | enum |
| `ram_fit_class` | string | yes | enum |
| `subject` | string | yes | enum |
| `question_id` | integer | yes | benchmark item id |
| `skill_tag` | array[string] | yes | 0+ tags |
| `curriculum_standard` | string or null | yes | nullable |
| `raw_output` | string | yes | raw model text |
| `parsed_answer` | string or null | yes | parsed token |
| `correct_answer` | string | yes | expected answer |
| `is_parseable` | boolean | yes | parse success |
| `answer_only_compliant` | boolean | yes | strict exact-format compliance |
| `is_correct` | boolean | yes | correctness |
| `latency_ms` | number | yes | per-question latency |
| `prompt_tokens` | integer | yes | input token count |
| `eval_tokens` | integer | yes | output token count |
| `output_length_chars` | integer | yes | length of raw output |
| `error_type` | string or null | yes | nullable |

---

## 10. Validation rules

### Bundle-level validation
- every artifact in the bundle must share the same `snapshot_id`
- every artifact in the bundle must share the same `benchmark_scope`
- every leaderboard row must have a matching model card
- every `example_id` in model cards must exist in `examples.json`

### Metric validation
- `balanced_quality_score` must equal average of Thai and Math score rates
- `overall_score_rate` must be in `[0,1]`
- `parseable_rate` must be in `[0,1]`
- `answer_only_compliance_rate` must be in `[0,1]`
- `latency_p95_ms >= latency_p50_ms`
- `item_count > 0`

### Public transparency validation
- raw bundle must include the same `snapshot_id` as the public UI bundle
- public caveat about excluded item types must be present in manifest notes or UI copy source

### Test integrity
- tests must validate actual behavior honestly, not be engineered to pass trivially
- do not use tautological assertions, trivial happy-path-only tests, or mock away the logic under test

---

## 11. Current V1 implementation recommendation

For V1, the easiest implementation path is:
1. extend benchmark rows
2. write one snapshot aggregation script
3. emit the 4 required public JSON artifacts
4. include raw source rows and repeat summary in a downloadable transparency bundle
5. build the static UI against these stable artifact shapes
