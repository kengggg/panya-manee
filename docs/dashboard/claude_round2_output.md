# Panya Manee Dashboard V1 — Convergence Review (Round 2)

---

## 1. FINAL CONSENSUS VERDICT

The Round 1 review and Samantha's counterpoints are almost fully aligned. The three real corrections are: (a) speed metrics (`questions_per_min`, `correct_per_min`) must be aggregated across the full published batch, not pulled from one canonical run — the canonical run is only for row-level examples and raw outputs; (b) downloadable raw JSON bundles are non-negotiable for V1, though the implementation can be minimal (zip of the 4 artifacts plus a flat `results.jsonl`); (c) Best Small Model ships in V1 with a tightened definition of `fits_comfortably_16gb` only. Everything else from Round 1 stands. The plan is buildable.

---

## 2. FINAL V1 GROUND TRUTH

These are the locked decisions to build against.

### Publication model
- Unit of publication: **batch run result** (a snapshot where all models ran the same scope, prompts, testbed, and script version).
- The dashboard displays exactly **one latest published snapshot**.
- A snapshot has a `snapshot_id` and `published_at` timestamp.

### Ranking
- Main rank metric: `balanced_quality_score = (thai_score_rate + math_score_rate) / 2`
- Tie-breakers in order: (1) higher `parseable_rate`, (2) higher `answer_only_compliance_rate`, (3) lower `latency_p50_ms`.
- Tied models after all tie-breakers share the same rank.

### Public speed metrics (both required on leaderboard)
- `questions_per_min = total_items_across_batch / total_latency_minutes_across_batch`
  - "across batch" means: sum all per-question latencies from all runs in the batch for that model. `total_latency_minutes = sum(all_question_latency_ms) / 60000`. `total_items_across_batch = item_count × num_runs`.
- `correct_per_min = total_correct_across_batch / total_latency_minutes_across_batch`
  - Same denominator as above. Numerator is total correct answers across all runs.
- These are **batch-aggregated**, not single-run metrics.

### Canonical run
- One run from the batch is designated canonical (recommendation: **run index 0**, or median-latency run).
- The canonical run is used **only** for:
  - Row-level examples (good/bad examples in `examples.json`)
  - Raw output text in examples
  - Per-question latency values displayed in examples
- The canonical run is **not** used for leaderboard rank metrics or speed metrics.

### Latency metrics
- `latency_p50_ms`: 50th percentile of per-question latencies **across all runs in the batch**.
- `latency_p95_ms`: 95th percentile of per-question latencies **across all runs in the batch**.

### Badges (V1)
| Badge | Rule |
|---|---|
| Best Quality | Highest `balanced_quality_score`. Ties allowed. |
| Best Thai | Highest `thai_score_rate`. Ties allowed. |
| Best Math | Highest `math_score_rate`. Ties allowed. |
| Fastest on Testbed | Lowest `latency_p50_ms`. Ties allowed. |
| Best Small Model | Highest `balanced_quality_score` among models with `ram_fit_class = fits_comfortably_16gb`. Ties allowed. |

### Best Small Model definition
- Eligible models: `ram_fit_class == "fits_comfortably_16gb"` only.
- Threshold: loaded model size ≤ 8 GB = `fits_comfortably_16gb`, > 8 GB = `fits_tightly_16gb`.
- This makes the badge non-redundant with Best Quality on the current model set (gemma4:e4b is `fits_tightly_16gb`).

### Skill tag rules
- Minimum item count for strength/weakness eligibility: `n >= 2`.
- Tie-break for skill tags with equal score rate: higher item count wins, then alphabetical.
- Show top 3 strongest and top 3 weakest per model. If fewer than 3 eligible tags exist, show what's available.

### Example selection
- Deterministic, rule-based. No manual curation.
- Per model: 2 good examples (correct, from strongest skill tags) + 2 bad examples (incorrect, from weakest skill tags).
- Tie-break within candidates: higher latency first, then by subject alphabetically, then by question_id ascending.
- Source data: canonical run only.

### Auto-summary
- Deterministic phrase template. No LLM generation.
- Phrase thresholds:
  - `strength_phrase`: "best on Thai" if rank 1 on thai_score_rate, "best on Math" if rank 1 on math_score_rate, "balanced across subjects" if |thai - math| < 0.1, else "stronger on {higher subject} than {lower subject}"
  - `reliability_phrase`: "very reliable" if compliance ≥ 0.95, "mostly reliable" if compliance ≥ 0.8, "format-unstable" otherwise
  - `speed_phrase`: "among the fastest" if speed_rank ≤ 1, "mid-pack" if speed_rank ≤ ceil(n/2), "slower than peers" otherwise

### Output truncation
- `raw_output_truncated`: first 200 characters of `raw_output_full`.

### Transparency (non-negotiable in V1)
- The published snapshot includes a **downloadable raw JSON bundle**.
- Minimal V1 implementation: a single downloadable zip containing:
  - The 4 published artifacts (`manifest.json`, `leaderboard.json`, `model_cards.json`, `examples.json`)
  - A flat `results.jsonl` with normalized per-question rows from the canonical run (schema per JSON schema spec Section 9)
- The UI provides a download link. No viewer/browser for raw data in V1.

### Subject filter
- **Cut from V1 initial release.** The leaderboard shows all subjects. Subject breakdown is visible per-model in the detail view.

### Leaderboard columns (final)
| Column | Source |
|---|---|
| Rank | computed |
| Model | `model_id` |
| Balanced Quality Score | `balanced_quality_score` |
| Thai | `thai_score_rate` |
| Math | `math_score_rate` |
| Overall | `overall_score_rate` |
| Parseable | `parseable_rate` |
| Compliance | `answer_only_compliance_rate` |
| p50 Latency | `latency_p50_ms` |
| Questions/min | `questions_per_min` |
| Correct/min | `correct_per_min` |
| N Items | `item_count` |
| Badges | `badges` |

### Model detail page
- Separate page per model (not a modal).
- Sections: Identity, Quality, Reliability, Speed, Strengths/Weaknesses, Examples, Auto-summary.

### UI framework
- Static HTML + vanilla JS, or minimal Astro/Vite. No heavy framework.

---

## 3. FINAL TEST PLAN

### A. Generator / Data Tests

1. `build_snapshot.py` exits 0 and emits all 4 required artifacts + `results.jsonl`.
2. All 5 files share the same `snapshot_id` and `benchmark_scope`.
3. Every enum field contains only allowed values (parameter_bucket, ram_fit_class, subject, badge values, selection_reason).
4. Every rate field is in [0.0, 1.0].
5. `latency_p95_ms >= latency_p50_ms` for every model.
6. `item_count > 0` for every model.
7. Every `model_id` in leaderboard has a matching entry in model_cards.
8. Every `example_id` in model_cards exists in examples.json.
9. `manifest.artifacts` keys point to files that exist in the bundle.
10. Running `build_snapshot.py` twice produces identical output (byte-identical or JSON-normalized identical).
11. `questions_per_min` recomputed from batch-aggregated latencies matches to 4 decimal places.
12. `correct_per_min` recomputed from batch-aggregated correct counts and latencies matches to 4 decimal places.
13. `balanced_quality_score` recomputed from `(thai_score_rate + math_score_rate) / 2` matches to 4 decimal places.
14. `overall_score_rate` recomputed from `total_correct / item_count` matches.
15. `latency_p50_ms` and `latency_p95_ms` match actual percentiles of all per-question latencies across batch runs.
16. `results.jsonl` row count equals `item_count` for the canonical run × number of models.

### B. Ranking / Badge Tests

17. Rank 1 has the highest `balanced_quality_score`.
18. Ranks are monotonically non-decreasing as score decreases.
19. Tie-breakers applied in order: parseable_rate → compliance_rate → p50 latency.
20. Models tied on all tie-breakers share the same rank.
21. `Best Quality` badge goes to the model(s) with highest `balanced_quality_score`.
22. `Best Thai` badge goes to the model(s) with highest `thai_score_rate`.
23. `Best Math` badge goes to the model(s) with highest `math_score_rate`.
24. `Fastest on Testbed` badge goes to the model(s) with lowest `latency_p50_ms`.
25. `Best Small Model` badge is restricted to `fits_comfortably_16gb` models only.
26. `Best Small Model` ≠ `Best Quality` when the quality winner is `fits_tightly_16gb`.
27. No model has a badge it shouldn't have.
28. Re-running the generator produces identical ranks and badges.

### C. Examples / Transparency Tests

29. Each model has exactly 2 good + 2 bad examples.
30. Good examples have `is_correct = true`. Bad examples have `is_correct = false`.
31. Good examples come from skill tags with `n >= 2` items.
32. Bad examples come from skill tags with `n >= 2` items.
33. `raw_output_truncated` is the first 200 chars of `raw_output_full` (or equal if shorter).
34. Every example has a valid `selection_reason`.
35. Running the generator twice produces identical example selections.
36. The downloadable zip contains all 4 artifacts + `results.jsonl`.
37. All files in the zip are valid JSON/JSONL.
38. `results.jsonl` rows reference only models present in the leaderboard.

### D. UI Acceptance Tests

39. Leaderboard page loads and shows all models.
40. Leaderboard sorted by `balanced_quality_score` descending.
41. All 13 leaderboard columns are visible (including `correct_per_min`).
42. Badge labels render next to correct models.
43. Benchmark label "NT P3 MCQ Text-Only" visible in header.
44. Public caveat about excluded item types visible.
45. Snapshot timestamp and testbed label visible.
46. Clicking a model row navigates to model detail page.
47. Model detail page shows all 7 sections (Identity, Quality, Reliability, Speed, Strengths/Weaknesses, Examples, Auto-summary).
48. Example outputs truncated by default; expand shows full output.
49. Raw JSON bundle download link works and downloads a valid zip.
50. Site deploys and loads at public URL.

---

## 4. FINAL IMPLEMENTATION PLAN

### Phase 0: Lock config (prerequisite, < 1 day)

- Create `config/models.json` with 4 models: model_id, model_family, parameter_bucket, ram_fit_class.
  - gemma4:e4b → fits_tightly_16gb
  - gemma4:e2b → fits_comfortably_16gb
  - typhoon → fits_comfortably_16gb
  - qwen3.5:9b → verify loaded size, assign accordingly
- Create `config/testbed.json`.
- No other config files needed. Example rules and phrase thresholds are hardcoded in the generator.

**Output:** 2 config files committed.

### Phase 1: Snapshot generator (`build_snapshot.py`) — V1

Single Python script that:
1. Reads raw JSONL rows from all runs in the batch.
2. Reads repeat_summary for cross-run data.
3. Reads `config/models.json` and `config/testbed.json`.
4. Normalizes rows (derive `answer_only_compliant`, `output_length_chars`, attach metadata).
5. Aggregates metrics per model:
   - Quality scores from any single run (deterministic — all runs identical on accuracy).
   - Speed metrics (`questions_per_min`, `correct_per_min`) aggregated across **all** batch runs.
   - Latency percentiles (`p50`, `p95`) computed across **all** batch runs.
6. Assigns badges.
7. Selects canonical run (run index 0). Picks 2 good + 2 bad examples per model from canonical run.
8. Generates auto-summaries from phrase templates.
9. Emits: `manifest.json`, `leaderboard.json`, `model_cards.json`, `examples.json`.
10. Emits `results.jsonl` from canonical run rows (normalized, per JSON schema spec Section 9).
11. Zips the output directory into a downloadable bundle.

**Output:** One script. Reads existing data. Produces the complete publishable bundle.

### Phase 2: Snapshot validator (`validate_snapshot.py`) — V1

Single Python script that runs all tests from Section 3A–3C above.

**Output:** One script. Exits 0 or prints errors.

### Phase 3: Generate + validate real snapshot — V1

1. Run `build_snapshot.py` against `mini-r10-20260409` batch.
2. Run `validate_snapshot.py`.
3. Manually verify rankings and badges against known benchmark results.
4. Fix bugs.

**Output:** Validated snapshot directory ready for the UI.

### Phase 4: Static dashboard UI — V1

Minimal static site:
1. Leaderboard page: load `leaderboard.json` + `manifest.json`, render table with all 13 columns, badges, header, caveat.
2. Model detail page: load `model_cards.json` + `examples.json`, render all 7 sections.
3. Download link for raw zip bundle.
4. No subject filter. No search. No sorting toggles beyond default rank order.

**Output:** Working static site.

### Phase 5: Deploy to GitHub Pages — V1

1. Place snapshot bundle in `site/public/data/`.
2. Configure GitHub Pages.
3. Verify live.

**Output:** Public URL.

### Deferred (post-V1)

- Subject filter on leaderboard
- `publish_snapshot.py` as a separate workflow tool
- GitHub Actions automation (snapshot-pr.yml, benchmark-run.yml)
- Self-hosted runner setup
- Registry files and compatibility gates
- Multi-snapshot trend UI
- Sorting/filtering controls on leaderboard
- `correct_per_gb_min` or size-aware efficiency metrics
- `config/example_rules.json` as external config

---

## 5. NON-NEGOTIABLE RISKS TO CLOSE BEFORE BUILDING

- **Verify `ram_fit_class` for qwen3.5:9b.** If its loaded size is ≤ 8 GB, it's `fits_comfortably_16gb` and the Best Small Model badge pool changes. Check actual Ollama memory footprint before writing `models.json`.
- **Confirm repeat_summary structure has per-run latency data needed for batch-aggregated speed metrics.** If per-run latency breakdowns are not in repeat_summary, the generator must read raw JSONL from all 10 runs directly. Verify which source has the data before coding.
- **Confirm canonical run selection is stable.** If run index 0 is used, verify that `benchmark_responses/` files have a reliable run index or ordering. If filenames don't encode run order, define the selection rule based on available metadata.
- **Verify all 4 models have complete runs in the batch.** Any model with a partial or failed run must be flagged before the generator is built, since the aggregation logic assumes `run_status = success` for all.
