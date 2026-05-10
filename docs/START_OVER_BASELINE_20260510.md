# Panya Manee start-over baseline — 2026-05-10

Batch/snapshot baseline: `ntp3-pub-r1-20260510`

## Decision

This is the new publication baseline for future NT P3 text-only comparisons.

Official publication default is now **1 run per model**. Earlier multi-run batches were exactly deterministic under the fixed local setup: same benchmark, same prompt, same dataset, same temperature, same Mac mini/Ollama testbed. Runtime can vary, but accuracy did not vary in those historical repeats.

## Active future roster

Future active roster removes:

- `mistral:7b`
- `phi4-mini:3.8b`
- `falcon3:7b`
- `granite3.3:8b`
- `olmo2:7b`

Future active roster adds:

- `granite4.1:8b`

Canonical active roster is maintained in `registry/active_roster.json`.

Historical `dist/` artifacts are not deleted; removed models are excluded only from future active snapshots.

## Current JSON policy

The dashboard should expose one canonical data entrypoint:

- `site/data/latest/current.json`

`current.json` is generated from one or more compatible snapshot bundles. Later inputs replace earlier rows for duplicate `model_id` values. That means future snapshots can keep `ntp3-pub-r1-20260510` as the base and append only new model results while preserving comparability across time.

If an existing model is retested with the same benchmark, prompt, and dataset contract, the retest replaces that model in the next `current.json`; it does not appear as a second leaderboard row.
