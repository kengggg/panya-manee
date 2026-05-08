# Panya Manee Dashboard Suggested Folder Structure

## Goal

Keep dashboard work separate from the benchmark repo clone while making the implementation path obvious.

---

## Recommended structure

```text
panya-manee-dashboard/
├── README.md
├── v1-dashboard-prd.md
├── implementation-checklist.md
├── json-schema-spec.md
├── generator-task-plan.md
├── folder-structure.md
├── benchmark-context.md
│
├── config/
│   ├── models.json
│   ├── testbed.json
│   └── example-rules.json
│
├── data/
│   ├── sources/
│   │   ├── repeat_summary_mini-r10-20260409.json
│   │   └── raw_runs/
│   ├── snapshots/
│   │   └── nt-p3-mcq-text-only-2026-04-09/
│   │       ├── manifest.json
│   │       ├── leaderboard.json
│   │       ├── model_cards.json
│   │       ├── examples.json
│   │       └── raw/
│   └── exports/
│       └── nt-p3-mcq-text-only-2026-04-09.zip
│
├── scripts/
│   ├── build_snapshot.py
│   ├── validate_snapshot.py
│   └── publish_snapshot.py
│
├── docs/
│   ├── badge-rules.md
│   ├── metric-definitions.md
│   └── example-selection-rules.md
│
└── ui/
    ├── app/
    ├── public/
    └── components/
```

---

## Folder responsibilities

### Root docs

Use the root for product and planning docs only:
- PRD
- schema spec
- implementation checklist
- generator plan
- benchmark notes

### `config/`

Static hand-maintained configuration.

Recommended files:
- `models.json`
  - model family
  - parameter bucket
  - RAM-fit class
  - display label
- `testbed.json`
  - host label
  - chip
  - RAM
  - backend
- `example-rules.json`
  - deterministic example selection rules
  - truncation rules

### `data/sources/`

Immutable source inputs copied from the benchmark workflow.

Recommended contents:
- repeat summaries
- raw run files copied from the benchmark repo when needed

### `data/snapshots/`

Generated publishable snapshot bundles.

Each snapshot folder should contain:
- `manifest.json`
- `leaderboard.json`
- `model_cards.json`
- `examples.json`
- optional `raw/` folder for transparency

### `data/exports/`

Compressed deliverables.

Examples:
- full snapshot zip
- raw transparency bundle zip

### `scripts/`

Deterministic build scripts only.

Suggested split:
- `build_snapshot.py`
- `validate_snapshot.py`
- `publish_snapshot.py`

### `docs/`

Supporting internal rules that are too detailed for the PRD.

Examples:
- exact badge tie logic
- metric threshold wording
- deterministic example selection policy

### `ui/`

Frontend app only.

Keep UI code separate from benchmark and snapshot generation logic.

---

## Recommended near-term cleanup from current state

Current state is intentionally light, but if you want the cleaner implementation shape, next moves should be:

1. create `config/`
2. create `scripts/`
3. move current repeat summary into `data/sources/`
4. reserve `data/snapshots/` for generated public bundles
5. reserve `data/exports/` for zip deliverables

---

## Minimal V1 folder structure if you want less overhead

If you want a smaller version first:

```text
panya-manee-dashboard/
├── README.md
├── v1-dashboard-prd.md
├── implementation-checklist.md
├── json-schema-spec.md
├── generator-task-plan.md
├── benchmark-context.md
├── data/
│   ├── repeat_summary_mini-r10-20260409.json
│   └── snapshots/
├── scripts/
└── ui/
```

This is enough to start implementation without over-structuring.
