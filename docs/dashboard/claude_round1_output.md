---

# Panya Manee Dashboard V1 — Plan Review

---

## 1. VERDICT

**Yes, with important fixes.**

The plan is directionally correct and surprisingly well-scoped for a V1. The PRD, JSON schema spec, and repo integration spec are coherent with each other. But there are real gaps in the data contract, under-specified aggregation rules, and the repo integration spec introduces scope that will sink V1 if you try to build it all. The fixes are tractable — this is not a redesign, it's a tightening pass.

---

## 2. WHAT IS STRONG

- **Snapshot-first publication model is the right call.** The PRD correctly rejects "latest run per model" in favor of a coherent batch snapshot. This avoids the most common leaderboard corruption problem.
- **Balanced quality score is a defensible ranking metric.** Equal-weighting Thai and Math avoids the trap of letting item count drive rank. The formula is simple, auditable, and documented.
- **Static JSON bundle with no API is the right V1 architecture.** No backend, no database, no live inference. This is the fastest path to a shippable product.
- **Deterministic example selection is explicitly required.** The PRD bans LLM-generated summaries and mandates rule-based selection. This is correct for a benchmark product.
- **The JSON schema spec is unusually complete for a V1.** Field-level types, enums, validation rules, and cross-file consistency checks are all defined. This is the strongest single document.
- **The 10-run repeat data shows zero variance in accuracy.** This means the benchmark is deterministic under current setup (temp=0), which simplifies the entire aggregation story — you don't need statistical summaries for V1.

---

## 3. WHAT IS BROKEN OR UNDER-SPECIFIED

### 3.1 `correct_per_min` is in the JSON spec but not in the PRD acceptance criteria

The PRD Section 17 says `correct_per_min` is a confirmed public metric. The JSON schema spec includes it in leaderboard rows. But the PRD Section 4.1 (leaderboard columns) lists `Questions/min` and does **not** list `Correct/min`. The implementation checklist includes it in Phase 3 aggregates but not in Phase 6 UI columns. This will cause confusion during implementation — is it shown or not?

**Fix:** Decide now. Either add it to the leaderboard column list in the PRD, or mark it model-card-only.

### 3.2 Strength/weakness skill tag selection rules are not defined

The PRD says "strongest 3 skill tags" and "weakest 3 skill tags." The generator task plan says "compute strongest and weakest skill tags." But **nowhere** is the actual selection rule defined:
- What happens if a skill tag has only 1 item? Is `1/1 = 100%` the "strongest"?
- What is the minimum item count for a skill tag to be eligible?
- How are ties broken among skill tags with the same score rate?
- What if a model has fewer than 3 distinct skill tags with nonzero items?

This is not optional. Skill tags with tiny denominators will produce misleading 0% or 100% rates that dominate the strength/weakness lists.

**Fix:** Define a minimum item threshold (e.g., `n >= 2`) and a tie-break rule (e.g., higher item count wins, then alphabetical).

### 3.3 `questions_per_min` computation is ambiguous for batch runs

The formula says `total_items / total_latency_minutes`. But with 10 runs of 93 items each:
- Is it `93 / median_run_time_minutes`?
- Is it `930 / total_batch_time_minutes`?
- Is it the mean of per-run `questions_per_min`?

The repeat summary JSON has `mean_correct_per_min` but no `questions_per_min` field at all. The raw row-level data has per-question `latency_ms`, so you could sum those per run, but the spec doesn't say whether to use wall-clock time or summed per-question latency.

**Fix:** Pin the definition: `questions_per_min = item_count / (sum_of_per_question_latency_ms / 60000)` for a single canonical run. State which run is used when you have 10 runs (the median run, or the mean across runs).

### 3.4 No definition of which run's data populates the snapshot when runs are identical

The 10-run batch shows identical accuracy across all runs. But raw outputs, latencies, and per-question details will differ slightly between runs. The plan never says:
- Which of the 10 runs is the "canonical" run whose row-level data populates examples, latency percentiles, and raw outputs?
- Or are latency metrics aggregated across all 10 runs?

**Fix:** Define explicitly. Recommended: use run 1 (or the median-latency run) as the canonical row source. Aggregate latency stats across all 10 runs.

### 3.5 `ram_fit_class` assignment is manual but no mapping exists

The PRD defines `fits_comfortably_16gb`, `fits_tightly_16gb`, etc. The benchmark context doc has observed memory footprints. But no document defines the threshold between "comfortably" and "tightly." The `config/models.json` doesn't exist yet, and there's no rule like "comfortably = loaded size < 8GB."

**Fix:** Define the threshold in `models.json` comments or in the spec. Based on the observed data: `<= 8 GB loaded = fits_comfortably_16gb`, `> 8 GB loaded = fits_tightly_16gb`.

### 3.6 `Best Small Model` badge is paradoxical with current model set

The PRD says Best Small Model goes to the highest `balanced_quality_score` among models whose `ram_fit_class` indicates they fit the 16GB testbed. But **all four models** fit the 16GB testbed — that's the entire point of this benchmark. So `Best Small Model` = `Best Quality` in every case.

**Fix:** Either redefine Best Small Model as `fits_comfortably_16gb` only (which excludes gemma4:e4b), or defer this badge until you have models that don't fit.

### 3.7 The repo integration spec is a different project from V1

The repo integration spec defines 4 artifact layers, 6 scripts, 3 GitHub Actions workflows, a self-hosted runner security model, registry files, compatibility gates, and a selection file schema. This is a full production pipeline spec. It is **not V1 of the dashboard** — it is V2+ of the infrastructure.

The PRD says: lock the JSON contract, build one aggregation script, emit 4 artifacts, build a static UI. The repo integration spec says: build `run_batch.py`, `aggregate_model_batch.py`, `aggregate_batch_candidate.py`, `validate_snapshot.py`, `publish_snapshot.py`, `build_dashboard_bundle.py`, plus 3 GitHub Actions workflows, plus a registry system.

If you try to build both simultaneously, V1 will not ship.

### 3.8 Auto-summary template is defined but phrase thresholds are not

The PRD defines a template with placeholders like `{strength_phrase}`, `{speed_phrase}`. It gives example phrases. But it never defines the numeric thresholds that map to each phrase. At what `thai_score_rate` does a model qualify for "best on Thai" vs "balanced across subjects"? At what percentile is a model "among the fastest"?

**Fix:** Define the thresholds as a simple lookup table in the spec. Example: `speed_phrase = "among the fastest" if speed_rank <= 1, "mid-pack" if speed_rank <= ceil(n/2), "slower than peers" otherwise`.

### 3.9 Output truncation length is never specified

The PRD and schema spec say `raw_output_truncated` is required. But no document says how many characters to truncate to. 50? 100? 200? This affects both the generator and the UI.

**Fix:** Pick a number. 200 chars is reasonable for MCQ outputs that are mostly single-digit answers.

---

## 4. WHAT SHOULD BE CUT OR DEFERRED FROM V1

1. **The entire repo integration spec (Layers 1-4, registry files, compatibility gates, 3 GitHub Actions workflows, self-hosted runner setup).** V1 should ship with one Python script that reads existing benchmark data and emits the 4 JSON artifacts. The pipeline automation is V2.

2. **`correct_per_gb_min` / size-aware efficiency metric.** The benchmark context doc discusses it, but the PRD doesn't include it in the leaderboard or model card spec. Don't add it now.

3. **`Best Small Model` badge.** As noted above, it's redundant with the current model set. Defer until you add a model that doesn't fit the testbed.

4. **`results.jsonl` raw row transparency artifact.** The PRD marks it optional. Don't block V1 on producing a normalized raw row bundle. Ship the 4 required artifacts first. Add the raw bundle as a fast follow.

5. **`config/example_rules.json` as a separate config file.** Hardcode the example selection rules in the generator script for V1. A config file adds indirection with no benefit for 4 models.

6. **Subject filter on the leaderboard.** The PRD says "minimal filters: subject, model." For 4 models and 93 items, a filter UI adds implementation time with near-zero user value. Defer to V1.1.

---

## 5. V1 TEST PLAN

### A. Data Contract Tests

**Goal:** Every JSON artifact conforms to the schema spec and is internally consistent.

Checks:
- `manifest.json`, `leaderboard.json`, `model_cards.json`, `examples.json` all parse as valid JSON
- All four files share the same `snapshot_id` and `benchmark_scope`
- Every enum field (`parameter_bucket`, `ram_fit_class`, `subject`, `run_status`, badge values) contains only allowed values
- Every rate field is in `[0.0, 1.0]`
- `latency_p95_ms >= latency_p50_ms` for every model
- `item_count > 0` for every model
- Every `model_id` in `leaderboard.json` has a matching entry in `model_cards.json`
- Every `example_id` referenced in `model_cards.json` exists in `examples.json`
- Every example in `examples.json` references a `model_id` that exists in the leaderboard
- `manifest.artifacts` keys point to files that exist in the bundle directory

**Pass condition:** Zero schema violations. Zero dangling references.

### B. Aggregation Logic Tests

**Goal:** Computed metrics are mathematically correct against known source data.

Checks:
- For each model, recompute `balanced_quality_score` from `thai_score_rate` and `math_score_rate`. Must match to 4 decimal places.
- For each model, recompute `overall_score_rate` from `total_correct / item_count`. Must match.
- Verify `thai_score_rate + math_score_rate` / 2 independently from raw rows for at least one model (gemma4:e4b: `(0.6667 + 0.4242) / 2 = 0.5455`).
- Verify `parseable_rate` = parseable_items / total_items from raw rows.
- Verify `answer_only_compliance_rate` from raw rows.
- Verify `latency_p50_ms` and `latency_p95_ms` match the actual 50th and 95th percentile of per-question latencies from source rows.
- Verify `questions_per_min` matches the defined formula.

**Pass condition:** All recomputed values match within floating-point tolerance (1e-4).

### C. Badge / Ranking Tests

**Goal:** Badges and ranks are deterministically correct.

Checks:
- The model with the highest `balanced_quality_score` has `Best Quality` badge.
- The model with the highest `thai_score_rate` has `Best Thai` badge.
- The model with the highest `math_score_rate` has `Best Math` badge.
- The model with the lowest `latency_p50_ms` has `Fastest on Testbed` badge.
- No model has a badge it shouldn't have.
- Rank 1 has the highest `balanced_quality_score`.
- Ranks are monotonically non-decreasing as `balanced_quality_score` decreases.
- Tie-breakers are applied in order: parseable_rate, compliance_rate, p50 latency.
- If two models tie on all tie-breakers, they share the same rank.
- Re-running the generator produces identical ranks and badges.

**Pass condition:** Badge assignment matches manual verification. Ranks are stable and reproducible.

### D. Deterministic Example-Selection Tests

**Goal:** Example selection is reproducible and rule-compliant.

Checks:
- Each model has exactly 2 good and 2 bad examples.
- Good examples come from the model's strongest skill tags.
- Bad examples come from the model's weakest skill tags.
- Good examples have `is_correct = true`.
- Bad examples have `is_correct = false`.
- Each example has a valid `selection_reason` from the allowed enum.
- Running the generator twice produces identical example selections (same `example_id`s in same order).
- `raw_output_truncated` is a prefix of `raw_output_full` (or equal if output is short).

**Pass condition:** 100% reproducibility. Zero examples violate the selection rules.

### E. UI Rendering / UX Acceptance Tests

**Goal:** The static dashboard renders correctly and meets PRD requirements.

Checks:
- Leaderboard page loads and shows all 4 models.
- Leaderboard is sorted by `balanced_quality_score` descending.
- All required columns are visible: Rank, Model, Balanced Quality Score, Thai, Math, Overall, Parseable, Compliance, p50 latency, Questions/min, N items, Badges.
- Badge icons/labels render next to the correct models.
- Benchmark label "NT P3 MCQ Text-Only" is visible in the header.
- Public caveat text about excluded item types is visible.
- Snapshot timestamp is visible.
- Testbed label is visible.
- Clicking a model row navigates to a model detail page.
- Model detail page shows: identity, quality, reliability, speed, strengths/weaknesses, examples, auto-summary.
- Example outputs are truncated by default.
- Expanding an example shows full output.
- Raw JSON download links work and download valid JSON.

**Pass condition:** All items above pass on a single browser (Chrome). No broken links, no missing data, no layout crashes.

### F. Publication Workflow Tests

**Goal:** The generator produces a valid, publishable bundle from real data.

Checks:
- `python scripts/build_snapshot.py` exits 0 and produces all 4 required files.
- `python scripts/validate_snapshot.py` exits 0 on the generated bundle.
- Generated files are deterministic: running `build_snapshot.py` twice produces byte-identical output (or identical after JSON normalization).
- Bundle can be served by a local static file server and the dashboard loads correctly from it.
- Bundle directory structure matches the spec.

**Pass condition:** End-to-end from raw data to served dashboard with zero manual intervention.

---

## 6. IMPLEMENTATION PLAN

### Phase 0: Lock decisions (half day)

Before writing code:
1. Resolve the 9 issues listed in Section 3 above. Specifically:
   - Pin `correct_per_min` to leaderboard or model-card-only
   - Define skill tag minimum item threshold and tie-break rule
   - Define `questions_per_min` computation from batch runs
   - Define which run is the canonical row source
   - Define `ram_fit_class` thresholds
   - Decide on `Best Small Model` badge (defer or redefine)
   - Define auto-summary phrase thresholds
   - Define truncation length
2. Write `config/models.json` with the 4 current models and their metadata (family, parameter_bucket, ram_fit_class).
3. Write `config/testbed.json`.

**Deliverable:** Updated spec decisions + 2 config files.

### Phase 1: Snapshot generator script (1-2 days)

Build `scripts/build_snapshot.py` as a single script that:
1. Reads raw JSONL benchmark response files from `benchmark_responses/`
2. Reads `config/models.json` and `config/testbed.json`
3. Normalizes rows (derive `answer_only_compliant`, `output_length_chars`, attach metadata)
4. Computes per-model aggregate metrics
5. Assigns badges
6. Selects deterministic examples
7. Generates auto-summaries from phrase templates
8. Emits `manifest.json`, `leaderboard.json`, `model_cards.json`, `examples.json` to an output directory

**Deliverable:** One script that produces the complete bundle. No registry, no layers, no workflows.

### Phase 2: Snapshot validator script (half day)

Build `scripts/validate_snapshot.py` that:
1. Loads all 4 JSON artifacts
2. Runs all data contract tests from Section 5A
3. Runs all aggregation logic cross-checks from Section 5B
4. Exits 0 or prints validation errors

**Deliverable:** One validation script.

### Phase 3: Generate and validate the real V1 snapshot (half day)

1. Run `build_snapshot.py` against the `mini-r10-20260409` batch data
2. Run `validate_snapshot.py` against the output
3. Manually verify badge assignments and rankings against the benchmark context doc
4. Fix any bugs found

**Deliverable:** A validated `dist/nt-p3-mcq-text-only-2026-04-09/` directory with all 4 artifacts.

### Phase 4: Static dashboard UI (2-3 days)

Build a minimal static site in `site/`:
1. Pick a lightweight framework (plain HTML + vanilla JS, or a single-page React/Preact app — the simpler the better for V1)
2. Leaderboard page: load `leaderboard.json`, render table, show header/metadata from `manifest.json`, show badges, show caveat
3. Model detail page: load `model_cards.json` and `examples.json`, render all sections from PRD Section 5
4. Raw JSON download links
5. Test locally with a static file server

**Deliverable:** Working static site that reads from the generated JSON bundle.

### Phase 5: Deploy to GitHub Pages (half day)

1. Add the generated JSON bundle to `site/public/data/`
2. Configure GitHub Pages deployment (can be as simple as pushing `site/` output to `gh-pages` branch, or using the Actions workflow from the orchestration spec)
3. Verify the live site loads and renders correctly

**Deliverable:** Live public dashboard at the GitHub Pages URL.

### Phase 6: Publication workflow automation (V1.1, after launch)

Only after the dashboard is live:
1. Add `scripts/publish_snapshot.py` for zip bundling
2. Add the `snapshot-pr.yml` GitHub Actions workflow
3. Add the `benchmark-run.yml` self-hosted runner workflow
4. Set up registry files

This is explicitly post-V1.

---

## 7. STOP CONDITIONS

V1 is ready when **all** of the following are true:

1. `build_snapshot.py` produces all 4 required JSON artifacts from the `mini-r10-20260409` batch data without errors.
2. `validate_snapshot.py` passes with zero validation errors on the generated bundle.
3. Running `build_snapshot.py` twice produces identical output (determinism check).
4. The leaderboard ranks match manual verification: gemma4:e4b #1, qwen3.5:9b #2, gemma4:e2b #3, typhoon #4.
5. Badge assignments match manual verification.
6. All 4 models have exactly 2 good + 2 bad examples each, selected by rules, reproducible.
7. The static dashboard loads in a browser from the generated bundle.
8. All required leaderboard columns are visible.
9. Model detail pages render all PRD Section 5 sections.
10. The public caveat text ("MCQ text-only, image-required and human-checked excluded") is visible.
11. Raw JSON download links work.
12. The site is deployed and accessible at a public URL.

---

## 8. OPEN QUESTIONS THAT MUST BE DECIDED NOW

1. **Which run is the canonical row source for examples and per-question latency?**
   All 10 runs produce identical accuracy. Pick one. Recommendation: use run index 0 (first run) as canonical.

2. **What is the minimum skill-tag item count for strength/weakness eligibility?**
   Recommendation: `n >= 2`. This prevents single-item tags from dominating.

3. **Is `correct_per_min` a leaderboard column or model-card-only?**
   The PRD Section 17 says show both `questions_per_min` and `correct_per_min`. The leaderboard column list doesn't include it. Decide.

4. **What is the `ram_fit_class` threshold?**
   Recommendation: loaded model size `<= 8 GB` = `fits_comfortably_16gb`, `> 8 GB` = `fits_tightly_16gb`.

5. **Should `Best Small Model` badge ship in V1?**
   With current models, it's identical to `Best Quality`. Recommendation: defer, or redefine as `fits_comfortably_16gb` models only.

6. **What is the truncation length for `raw_output_truncated`?**
   Recommendation: 200 characters. Most MCQ outputs are single digits, so this is generous.

7. **What frontend stack for `site/`?**
   This affects implementation time. Recommendation for V1: static HTML + vanilla JS, or a minimal Astro/Vite setup. Avoid heavy frameworks.
