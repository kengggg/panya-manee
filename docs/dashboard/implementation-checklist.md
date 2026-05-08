# Panya Manee Dashboard V1 Implementation Checklist

## Goal

Ship a static public dashboard for the **NT P3 MCQ Text-Only** benchmark using a published batch snapshot and downloadable raw JSON bundles.

---

## Phase 0. Lock the contract

- [ ] Freeze `benchmark_scope = mcq_text_only_v1`
- [ ] Freeze active suite ids:
  - [ ] `thai_mcq_text_only_all`
  - [ ] `math_mcq_text_only_all`
  - [ ] `overall_mcq_text_only_all`
- [ ] Freeze testbed label:
  - [ ] `Apple Mac mini M4 / 16GB / Ollama`
- [ ] Freeze ranking formula:
  - [ ] `balanced_quality_score = (thai_score_rate + math_score_rate) / 2`
- [ ] Freeze badge rules
- [ ] Freeze deterministic example-selection rules

---

## Phase 1. Update benchmark row output

Add required row-level fields to the benchmark output.

### Must add
- [ ] `snapshot_id`
- [ ] `benchmark_scope`
- [ ] `benchmark_suite_id`
- [ ] `answer_only_compliant`
- [ ] `model_family`
- [ ] `parameter_bucket`
- [ ] `ram_fit_class`
- [ ] `host_label`
- [ ] `run_status`

### Should add
- [ ] `output_length_chars`
- [ ] `error_type`

### Keep / verify existing fields
- [ ] `model_id`
- [ ] `run_id`
- [ ] `subject`
- [ ] `question_id`
- [ ] `skill_tag`
- [ ] `curriculum_standard`
- [ ] `raw_output`
- [ ] `parsed_answer`
- [ ] `correct_answer`
- [ ] `is_parseable`
- [ ] `is_correct`
- [ ] `latency_ms`
- [ ] `prompt_tokens`
- [ ] `eval_tokens`

---

## Phase 2. Define deterministic example-selection rules

Pick one ruleset and keep it fixed.

### Recommended V1 rule set

For each model:
- [ ] select 2 good examples from strongest skill tags
- [ ] select 2 bad examples from weakest skill tags
- [ ] break ties by highest latency first
- [ ] break remaining ties by stable sort order:
  - [ ] subject
  - [ ] question_id
  - [ ] model_id

### Source for examples
- [ ] use canonical run (lowest `run_index` in the published batch) for example selection, not aggregated data

### Output behavior
- [ ] keep `raw_output_full`
- [ ] generate `raw_output_truncated` = **first 200 characters** of raw output
- [ ] expose both in `examples.json`

---

## Phase 3. Build snapshot aggregation script

Create an aggregation step that takes batch-run outputs and produces the public snapshot bundle.

### Inputs
- [ ] one published batch summary
- [ ] row-level run files for all included models
- [ ] manual model metadata map if needed

### Outputs
- [ ] `manifest.json`
- [ ] `leaderboard.json`
- [ ] `model_cards.json`
- [ ] `examples.json`
- [ ] downloadable transparency bundle including `repeat_summary.json` and raw row-level source rows

### Aggregate metrics to compute

**Batch-aggregation rule:** Speed metrics and latency percentiles are computed across **all runs in the published batch**, not from a single run.

- [ ] `balanced_quality_score`
- [ ] `thai_score_rate`
- [ ] `math_score_rate`
- [ ] `overall_score_rate`
- [ ] `parseable_rate`
- [ ] `answer_only_compliance_rate`
- [ ] `latency_p50_ms` — across all per-question latencies from all batch runs
- [ ] `latency_p95_ms` — across all per-question latencies from all batch runs
- [ ] `questions_per_min` — batch-aggregated across all runs
- [ ] `correct_per_min` — batch-aggregated across all runs
- [ ] optional `throughput_toks_per_sec`

---

## Phase 4. Badge assignment

- [ ] assign `Best Quality`
- [ ] assign `Best Thai`
- [ ] assign `Best Math`
- [ ] assign `Fastest on Testbed`
- [ ] assign `Best Small Model`
- [ ] allow ties
- [ ] ensure badge logic uses only snapshot-local data

### Best Small Model rule
- [ ] eligible only if `ram_fit_class = fits_comfortably_16gb`
- [ ] `fits_tightly_16gb` models are **not eligible**
- [ ] use RAM-fit class, not parameter count, as source of truth

---

## Phase 5. Auto-summary generation

Implement deterministic summary generation.

### Inputs
- [ ] Thai score rate
- [ ] Math score rate
- [ ] parseable rate
- [ ] compliance rate
- [ ] strongest 3 skill tags (minimum n >= 2 per tag; tie-break: higher item count, then alphabetical)
- [ ] weakest 3 skill tags (minimum n >= 2 per tag; tie-break: higher item count, then alphabetical)
- [ ] speed rank / percentile within snapshot

### Output rules
- [ ] no freeform LLM generation required for V1
- [ ] use phrase templates only
- [ ] deterministic wording from metric thresholds

---

## Phase 6. Static UI build

### Leaderboard page
- [ ] header with benchmark metadata
- [ ] visible caveat about excluded item types
- [ ] leaderboard table
- [ ] minimal filter: model (subject filter deferred from initial V1 launch)
- [ ] badge rendering
- [ ] raw JSON download links

### Model detail page
- [ ] identity section
- [ ] quality section
- [ ] reliability section
- [ ] speed section
- [ ] strengths / weaknesses section
- [ ] examples section
- [ ] deterministic auto summary

---

## Phase 7. Transparency bundle

- [ ] provide downloadable snapshot JSON bundle
- [ ] include `manifest.json`
- [ ] include public artifacts
- [ ] include `repeat_summary.json`
- [ ] include raw row-level source files for the published snapshot
- [ ] expose snapshot id clearly in UI and bundle

---

## Phase 8. Validation checklist

Before publishing a snapshot:
- [ ] all models come from the same published batch
- [ ] all rows use the same benchmark scope
- [ ] item count is correct
- [ ] ranking matches `balanced_quality_score`
- [ ] tie-breakers work correctly
- [ ] badge logic matches spec (Best Small Model = `fits_comfortably_16gb` only)
- [ ] `answer_only_compliance_rate` is computed correctly
- [ ] `p50` and `p95` latency computed across all batch runs
- [ ] `questions_per_min` and `correct_per_min` are batch-aggregated
- [ ] examples are reproducible from rules using canonical run
- [ ] `raw_output_truncated` is first 200 characters
- [ ] strength/weakness tags have minimum n >= 2
- [ ] raw JSON bundle downloads correctly
- [ ] caveat text is visible in UI
- [ ] tests validate actual behavior honestly, not engineered to pass trivially

---

## Phase 9. Nice-to-have after V1

- [ ] historical snapshot browser
- [ ] trend charts across snapshots
- [ ] image-required benchmark suites
- [ ] written-response suites
- [ ] richer skill-tag exploration
- [ ] comparison page across 2-3 models
