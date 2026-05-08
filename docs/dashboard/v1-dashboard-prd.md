# Panya Manee Dashboard V1 PRD

## 1. Product Definition

### 1.1 Product name

**Small Model Thai Performance Dashboard**

A public benchmark dashboard showing how local LLMs perform on Thailand NT Grade 3 benchmark tasks on a fixed local testbed.

### 1.2 V1 benchmark label

**NT P3 MCQ Text-Only**

### 1.3 Product goal

Help users compare small local models on Thai language and math performance under a fixed and transparent benchmark setup.

This is a **public static benchmark UI**, not an arena, not a live inference playground, and not a human-voted leaderboard.

---

## 2. V1 Scope

V1 includes only:
- latest **published benchmark snapshot** per model set
- static data only
- objective scoring only
- MCQ text-only items only
- Thai + Math
- one fixed testbed: **Apple Mac mini M4 / 16GB / Ollama**
- model results from the same benchmark publication batch
- per-model quality, reliability, speed, and examples

V1 excludes:
- live inference
- human voting
- image-required items in ranking
- human-checked items in ranking
- written-response ranking
- multi-snapshot trend UI
- user-uploaded tests
- quality-speed blended rank score

**Public caveat that must be visible:**

> V1 includes only MCQ text-only items. Image-required and human-checked tasks are excluded from this ranking.

---

## 3. Core Product Decisions

### 3.1 Unit of publication

The public dashboard should rank models using a **published benchmark snapshot**, not an arbitrary latest single run.

A snapshot is a coherent release bundle where:
- all models were run on the same benchmark scope
- all models used the same prompt rules
- all models used the same testbed
- all models used the same script/version contract

Reason:
- this avoids mixing runs from different dates, scripts, or benchmark definitions
- this matches the current benchmark workflow better than “latest successful run per model”

### 3.2 Main ranking metric

Use:

`balanced_quality_score = (thai_score_rate + math_score_rate) / 2`

Reason:
- Thai and Math should have equal product importance
- raw item-weighted ranking would overweight Thai because Thai has more items than Math in the current runnable set

### 3.3 Supporting metrics

Show these separately. Do not blend them into the main rank:
- item-weighted overall score rate
- parseable rate
- answer-only compliance rate
- p50 latency
- p95 latency
- questions per minute
- correct per minute
- optional technical metric: tokens per second

**Speed metric aggregation rule:**
- `questions_per_min` and `correct_per_min` are **batch-aggregated** across the full published batch (all runs), not taken from a single canonical run.
- `latency_p50_ms` and `latency_p95_ms` are computed across **all individual question latencies from all runs in the batch**.

### 3.4 Tie-breakers

Tie-breakers for the leaderboard:
1. higher parseable rate
2. higher answer-only compliance rate
3. lower p50 latency

---

## 4. User-Facing Experience

### 4.1 Arena / Leaderboard page

#### A. Header

Show:
- benchmark name
- benchmark label: **NT P3 MCQ Text-Only**
- benchmark scope id
- published snapshot timestamp
- model count
- item count
- testbed label
- visible caveat that image-required and human-checked tasks are excluded in V1

#### B. Main leaderboard

Columns:
- Rank
- Model
- Balanced Quality Score
- Thai
- Math
- Overall
- Parseable
- Compliance
- p50 latency
- p95 latency
- Questions/min
- Correct/min
- N items
- Badges

#### C. Filters

V1 minimal filters:
- model

**Subject filter** (`all / thai / math`) is **deferred from the initial V1 launch**. It can be added later without schema changes.

Skill-tag filtering is **not required on the main leaderboard in V1**.
Skill tags should appear in model detail views instead.

#### D. Model detail entry point

Users can click a model row to open a separate model detail page.

#### E. Failure examples section

For each model, show selected examples with:
- truncated raw output
- expand to full output
- parsed answer
- correct answer
- question id
- subject
- skill tag
- latency

---

## 5. Model Performance Card / Detail View

Each model detail view should contain the following sections.

### A. Identity
- model name
- model family
- parameter bucket
- RAM-fit class
- backend
- testbed label

### B. Quality
- balanced quality score
- Thai score rate
- Math score rate
- item-weighted overall score rate
- total correct / total items

### C. Reliability
- parseable rate
- answer-only compliance rate
- average output length
- common failure types

### D. Speed
- p50 latency
- p95 latency
- questions per minute
- correct per minute
- optional technical metric: tokens per second

### E. Strengths / Weaknesses
- strongest 3 skill tags
- weakest 3 skill tags

**Skill tag rules:**
- Only include skill tags with **n >= 2** items (minimum 2 questions answered for that tag).
- Tie-break: higher item count first, then alphabetical by tag name.
- If fewer than 3 tags qualify, show only the qualifying tags.

### F. Examples
- 2 good examples (from canonical run)
- 2 bad examples (from canonical run)
- truncated by default with expand
- `raw_output_truncated` = **first 200 characters** of raw output

### G. Auto summary

One deterministic generated paragraph, for example:

> Strong Thai reading and stable formatting, but weaker on fraction reasoning and number patterns. On the current Mac mini testbed, this model is mid-pack on speed.

---

## 6. Badge System

Show both metrics and badges.

V1 badges:
- Best Quality
- Best Thai
- Best Math
- Fastest on Testbed
- Best Small Model

### Badge rules

**Best Quality**
- highest `balanced_quality_score`

**Best Thai**
- highest `thai_score_rate`

**Best Math**
- highest `math_score_rate`

**Fastest on Testbed**
- lowest `latency_p50_ms`

**Best Small Model**
- highest `balanced_quality_score` among models with `ram_fit_class = fits_comfortably_16gb` only
- models classified as `fits_tightly_16gb` are **not eligible** for this badge

### Not required in V1

Do not require a **Best Compliance** badge in V1 unless compliance differences become meaningful in the actual data.

---

## 7. Model Classification

Expose two descriptors.

### 7.1 Parameter bucket

Suggested buckets:
- `<1B`
- `1B–4B`
- `4B–8B`
- `8B+`

### 7.2 RAM-fit class on testbed

Allowed values:
- `fits_comfortably_16gb`
- `fits_tightly_16gb`
- `unknown`

RAM-fit class is descriptive and should not depend on guesses beyond measured or manually assigned knowledge.

---

## 8. Benchmark Suite Registry

Even though V1 publishes one public benchmark family, the contract should be future-ready.

### V1 active suite ids
- `thai_mcq_text_only_all`
- `math_mcq_text_only_all`
- `overall_mcq_text_only_all`

### Reserved future suite ids
- `thai_mcq_vision_all`
- `math_mcq_vision_all`
- `overall_mcq_all`
- `thai_written_all`
- `math_written_all`

---

## 9. Static Data Delivery

Do not build an API for V1.
Use a static JSON bundle.

### Recommended artifacts
- `manifest.json`
- `leaderboard.json`
- `model_cards.json`
- `examples.json`
- downloadable transparency bundle including:
  - `results.jsonl` or equivalent raw row-level source rows
  - `repeat_summary.json`
  - underlying per-run source files for the published batch when available

---

## 10. Data Contract Principles

### 10.1 Snapshot-first contract

Every public JSON artifact should be tied to a published snapshot.

Recommended top-level identifiers:
- `snapshot_id`
- `published_at`
- `benchmark_scope`
- `suite_ids`
- `testbed`

### 10.2 Latest public data behavior

V1 UI should load one latest published snapshot.
The raw bundle can still include underlying per-run files for transparency.

---

## 11. JSON Contract

### 11.1 manifest.json

Purpose:
- defines the published snapshot
- defines the benchmark scope
- defines the testbed
- describes which artifacts belong to the snapshot

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
    "ranking_excludes_human_checked": true
  },
  "artifacts": {
    "leaderboard": "leaderboard.json",
    "model_cards": "model_cards.json",
    "examples": "examples.json"
  }
}
```

### 11.2 leaderboard.json

Purpose:
- one row per model for the main public leaderboard

```json
{
  "snapshot_id": "nt-p3-mcq-text-only-2026-04-09",
  "benchmark_scope": "mcq_text_only_v1",
  "rows": [
    {
      "rank": 1,
      "model_id": "gemma4:e4b",
      "model_family": "gemma4",
      "parameter_bucket": "4B–8B",
      "ram_fit_class": "fits_tightly_16gb",
      "balanced_quality_score": 0.5455,
      "thai_score_rate": 0.6667,
      "math_score_rate": 0.4242,
      "overall_score_rate": 0.5806,
      "parseable_rate": 1.0,
      "answer_only_compliance_rate": 1.0,
      "latency_p50_ms": 51847,
      "latency_p95_ms": 57500,
      "questions_per_min": 106.9,
      "correct_per_min": 62.0,
      "item_count": 93,
      "badges": ["Best Quality"]
    }
  ]
}
```

### 11.3 model_cards.json

Purpose:
- one object per model for detail views

```json
{
  "snapshot_id": "nt-p3-mcq-text-only-2026-04-09",
  "benchmark_scope": "mcq_text_only_v1",
  "models": [
    {
      "model_id": "gemma4:e4b",
      "model_family": "gemma4",
      "parameter_bucket": "4B–8B",
      "ram_fit_class": "fits_tightly_16gb",
      "testbed": {
        "host_label": "Apple Mac mini M4 / 16GB / Ollama"
      },
      "metrics": {
        "balanced_quality_score": 0.5455,
        "thai_score_rate": 0.6667,
        "math_score_rate": 0.4242,
        "overall_score_rate": 0.5806,
        "parseable_rate": 1.0,
        "answer_only_compliance_rate": 1.0,
        "latency_p50_ms": 51847,
        "latency_p95_ms": 57500,
        "questions_per_min": 106.9,
        "correct_per_min": 62.0,
        "item_count": 93
      },
      "strengths": [
        {"skill_tag": "reading_literature", "score_rate": 0.9091, "correct": 10, "total": 11},
        {"skill_tag": "moral_application", "score_rate": 1.0, "correct": 3, "total": 3},
        {"skill_tag": "judgment_from_text", "score_rate": 1.0, "correct": 2, "total": 2}
      ],
      "weaknesses": [
        {"skill_tag": "verb_identification", "score_rate": 0.0, "correct": 0, "total": 2},
        {"skill_tag": "number_pattern", "score_rate": 0.0, "correct": 0, "total": 2},
        {"skill_tag": "story_prediction", "score_rate": 0.3333, "correct": 1, "total": 3}
      ],
      "badges": ["Best Quality"],
      "auto_summary": "Strong on Thai reading and the best overall on the current benchmark snapshot. Weaker on some grammar and math pattern tasks. On the current Mac mini testbed, it is mid-pack on speed.",
      "example_ids": {
        "good": ["ex_001", "ex_002"],
        "bad": ["ex_101", "ex_102"]
      }
    }
  ]
}
```

### 11.4 examples.json

Purpose:
- separate heavy example content from the main card payload

```json
{
  "snapshot_id": "nt-p3-mcq-text-only-2026-04-09",
  "benchmark_scope": "mcq_text_only_v1",
  "examples": [
    {
      "example_id": "ex_101",
      "model_id": "gemma4:e4b",
      "subject": "math",
      "question_id": 8,
      "skill_tag": ["fraction_addition"],
      "curriculum_standard": "ค 1.1 ป.3/11",
      "is_correct": false,
      "correct_answer": "3",
      "parsed_answer": "1",
      "raw_output_truncated": "1",
      "raw_output_full": "1",
      "latency_ms": 496
    }
  ]
}
```

---

## 12. Required Row-Level Fields for the Benchmark Script

### Must add
- `snapshot_id`
- `benchmark_scope`
- `benchmark_suite_id`
- `answer_only_compliant`
- `questions_per_min` or enough data to derive it at aggregation time
- `model_family`
- `parameter_bucket`
- `ram_fit_class`
- `host_label`
- `run_status`

### Should add
- `output_length_chars`
- `prompt_tokens`
- `eval_tokens`
- `error_type`

Notes:
- `questions_per_min` can also be derived at aggregate level rather than stored per row
- if both are available, raw rows should still preserve the inputs needed to recompute aggregate metrics

---

## 13. Compliance Rules

### 13.1 Parseable rate

A row is parseable if the system can extract a valid answer in the allowed answer set.

For V1 MCQ text-only:
- valid answers are `1`, `2`, `3`, `4`

### 13.2 Answer-only compliance rate

A row is answer-only compliant if the raw output is exactly one allowed answer token and contains no extra text.

For V1 MCQ text-only:
- compliant examples: `"1"`, `"2"`, `"3"`, `"4"`
- non-compliant but parseable examples: `"คำตอบคือ 3"`, `"3)"`, `"I choose 2"`

This metric matters because the benchmark is explicitly testing instruction following as part of the output contract.

---

## 14. Auto-Summary Rules

The summary should be deterministic, brief, and boring.

Inputs:
- Thai score rate
- Math score rate
- parseable rate
- compliance rate
- strongest 3 skills
- weakest 3 skills
- speed percentile within the snapshot

Template:

> {model_id} performs {strength_phrase}. It is {reliability_phrase} on answer-only formatting. Its strongest areas are {top_skills}. Its weakest areas are {weak_skills}. On the current Mac mini testbed, it is {speed_phrase}.

Example phrase sets:
- `strength_phrase`: `best on Thai`, `balanced across subjects`, `weaker on math than Thai`
- `reliability_phrase`: `very reliable`, `mostly reliable`, `format-unstable`
- `speed_phrase`: `among the fastest`, `mid-pack`, `slower than peers`

---

## 15. Acceptance Criteria for V1

**Test integrity principle:** Tests must validate actual behavior honestly. Do not engineer tests to pass trivially (e.g., testing only happy paths, using tautological assertions, or mocking away the logic under test).

V1 is done if:
- main leaderboard ranks by `balanced_quality_score`
- public UI clearly says MCQ text-only
- one published snapshot is displayed consistently across all models
- Thai and Math subscores are visible
- parseable rate and compliance rate are visible
- latency, questions-per-minute, and correct-per-minute are visible but not blended into rank
- badges are shown
- per-model detail view exists
- outputs are truncated with expand
- strengths and weaknesses by skill tag are shown
- deterministic auto-summary appears on the model detail view

---

## 16. Recommended Implementation Order

Do not build the UI first.

Lock these first:
1. `benchmark_scope = mcq_text_only_v1`
2. snapshot-first static JSON bundle contract
3. row-level field additions in the benchmark script
4. aggregation logic for leaderboard and model cards
5. only then implement the static dashboard UI

---

## 17. Confirmed Product Decisions

The following decisions are now locked for V1:

1. **Publication unit**
   - Public results should use a **batch run result**, not an arbitrary single run.
   - Speed metrics (`questions_per_min`, `correct_per_min`) and latency percentiles (`latency_p50_ms`, `latency_p95_ms`) are **batch-aggregated across all runs in the published batch**.
   - The **canonical run** is the run with the lowest `run_index` in the published batch.
   - The canonical run is used only for selecting examples and raw outputs, not for ranking metrics.

2. **Public speed metrics**
   - Show **both**:
     - `questions_per_min`
     - `correct_per_min`
   - These are batch-aggregated, not from a single canonical run.

3. **Model detail UI**
   - Use a **separate page** per model.

4. **Badge winners**
   - **Ties are allowed**.

5. **Best Small Model definition**
   - Eligible only if `ram_fit_class = fits_comfortably_16gb`.
   - Models classified as `fits_tightly_16gb` are not eligible.
   - For V1, this uses the RAM-fit class on the benchmark machine rather than a pure parameter-count cutoff.

6. **Example selection**
   - Examples should be **deterministically selected by rules**, not manually curated.
   - This keeps the dashboard reproducible and benchmark-like.

7. **Transparency**
   - The public dashboard should also expose **downloadable raw JSON bundles** for the published snapshot.
   - The UI still centers the latest published snapshot, but raw artifacts should be available for inspection and reuse.
   - The downloadable transparency bundle should include `repeat_summary.json` and raw row-level source rows for the published snapshot.

---

## 18. Current Recommendation Based on Existing Results

Given current benchmark behavior on the Mac mini:
- quality winner: `gemma4:e4b`
- speed-efficiency winner: `gemma4:e2b`
- lightweight efficiency winner: `scb10x/typhoon2.1-gemma3-4b:latest`

This supports the product direction of showing:
- a main quality rank
- separate speed metrics
- separate small-model / lightweight recognition
