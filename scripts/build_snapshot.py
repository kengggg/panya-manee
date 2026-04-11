#!/usr/bin/env python3
from __future__ import annotations

"""
Build a dashboard snapshot bundle from benchmark batch data.

Reads JSONL response files + repeat_summary + config, then emits:
  manifest.json, leaderboard.json, model_cards.json, examples.json,
  repeat_summary.json (transparency), and a downloadable zip bundle.

Usage:
  python scripts/build_snapshot.py --batch-id mini-r10-20260409
  python scripts/build_snapshot.py --batch-id mini-r10-20260409 --out ./dist/my-snapshot
"""

import argparse
import json
import re
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"
REGISTRY_DIR = PROJECT_ROOT / "registry"
NT_TESTS_DIR = PROJECT_ROOT / "nt-tests"
MODELS_CONFIG = REGISTRY_DIR / "models.json"
MACHINE_PROFILES_CONFIG = REGISTRY_DIR / "machine_profiles.json"
COMPATIBILITY_CONFIG = REGISTRY_DIR / "compatibility.json"

BENCHMARK_SCOPE = "mcq_text_only_v1"
SUITE_IDS = [
    "thai_mcq_text_only_all",
    "math_mcq_text_only_all",
    "overall_mcq_text_only_all",
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_answer_only_compliant(raw_output: str) -> bool:
    """Compliant if raw output is exactly one digit 1-4 with no extra text."""
    return raw_output.strip() in {"1", "2", "3", "4"}


def percentile(values: list[float], pct: float) -> float:
    """Compute percentile using the 'nearest rank' / interpolation method."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    # Use linear interpolation matching Python's statistics approach
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    # Rank-based index
    k = (pct / 100.0) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def round4(x: float) -> float:
    return round(x, 4)


def round1(x: float) -> float:
    return round(x, 1)


def suite_id_for_subject(subject: str) -> str:
    if subject == "thai":
        return "thai_mcq_text_only_all"
    if subject == "math":
        return "math_mcq_text_only_all"
    return "overall_mcq_text_only_all"


def normalize_raw_row(row: dict, snapshot_id: str, testbed: dict, model_meta: dict) -> dict:
    meta = resolve_model_meta(row["model_id"], model_meta)
    raw_output = row.get("raw_output", "") or ""
    error_type = None
    if isinstance(raw_output, str) and raw_output.startswith("ERROR:"):
        error_type = "runtime_error"

    return {
        "snapshot_id": snapshot_id,
        "run_id": row["run_id"],
        "benchmark_scope": BENCHMARK_SCOPE,
        "benchmark_suite_id": suite_id_for_subject(row["subject"]),
        "run_status": "success" if error_type is None else "failed",
        "host_label": testbed["host_label"],
        "model_id": row["model_id"],
        "model_family": meta.get("model_family", "unknown"),
        "parameter_bucket": meta.get("parameter_bucket", "unknown"),
        "ram_fit_class": meta.get("ram_fit_class", "unknown"),
        "subject": row["subject"],
        "question_id": row["question_id"],
        "skill_tag": row.get("skill_tag") or [],
        "curriculum_standard": row.get("curriculum_standard"),
        "raw_output": raw_output,
        "parsed_answer": row.get("parsed_answer"),
        "correct_answer": row.get("correct_answer"),
        "is_parseable": row.get("is_parseable", False),
        "answer_only_compliant": is_answer_only_compliant(raw_output),
        "is_correct": row.get("is_correct", False),
        "latency_ms": row.get("latency_ms", 0),
        "prompt_tokens": row.get("prompt_tokens", 0),
        "eval_tokens": row.get("eval_tokens", 0),
        "output_length_chars": len(raw_output),
        "error_type": error_type,
    }


def common_failure_types(rows: list[dict], top_n: int = 3) -> list[str]:
    counts = defaultdict(int)
    for row in rows:
        raw_output = row.get("raw_output", "") or ""
        if isinstance(raw_output, str) and raw_output.startswith("ERROR:"):
            counts["runtime_error"] += 1
        elif not row.get("is_parseable", False):
            counts["unparseable_output"] += 1
        elif not is_answer_only_compliant(raw_output):
            counts["non_answer_only_output"] += 1
        elif not row.get("is_correct", False):
            counts["wrong_answer"] += 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _count in ordered[:top_n]]


def model_lookup_candidates(model_id: str, extra_aliases: list[str] | None = None) -> list[str]:
    candidates = []

    def add(value: str | None):
        if value and value not in candidates:
            candidates.append(value)

    add(model_id)
    if model_id and model_id.endswith(":latest"):
        add(model_id[:-7])
    for alias in extra_aliases or []:
        add(alias)
        if alias.endswith(":latest"):
            add(alias[:-7])
    return candidates


# ── Model metadata lookup ───────────────────────────────────────────────────


def load_model_meta(config_path: Path) -> dict[str, dict]:
    """Returns {model_id: {model_family, parameter_bucket, ram_fit_class, ...}}"""
    data = load_json(config_path)
    return {m["model_id"]: m for m in data["models"]}


def resolve_model_meta(model_id: str, model_meta: dict[str, dict]) -> dict:
    if model_id in model_meta:
        return model_meta[model_id]

    candidates = model_lookup_candidates(model_id)
    for meta in model_meta.values():
        aliases = model_lookup_candidates(meta["model_id"], meta.get("aliases") or [])
        if any(candidate in aliases for candidate in candidates):
            return meta
    return {}


def load_question_bank() -> dict[tuple[str, int], dict]:
    bank = {}
    if not NT_TESTS_DIR.exists():
        return bank

    for path in sorted(NT_TESTS_DIR.glob("*.json")):
        items = load_json(path)
        if not isinstance(items, list):
            continue
        for item in items:
            exam_id = item.get("exam_id")
            question_id = item.get("question_id")
            if exam_id is None or question_id is None:
                continue
            bank[(str(exam_id), int(question_id))] = item
    return bank


def resolve_question_item(row: dict, question_bank: dict[tuple[str, int], dict] | None = None) -> dict:
    if question_bank:
        exam_id = row.get("exam_id")
        question_id = row.get("question_id")
        if exam_id is not None and question_id is not None:
            item = question_bank.get((str(exam_id), int(question_id)))
            if item:
                return item
        if question_id is None:
            return {}

        candidates = [
            item for (candidate_exam_id, candidate_qid), item in question_bank.items()
            if candidate_qid == int(question_id)
        ]

        subject = row.get("subject")
        if subject is not None:
            subject_matches = [item for item in candidates if item.get("subject") == subject]
            if subject_matches:
                candidates = subject_matches

        def filter_candidates(items: list[dict], field: str) -> list[dict]:
            value = row.get(field)
            if value in (None, "", []):
                return items
            matches = [item for item in items if item.get(field) == value]
            return matches or items

        candidates = filter_candidates(candidates, "curriculum_standard")
        candidates = filter_candidates(candidates, "correct_answer")
        candidates = filter_candidates(candidates, "prompt_text")
        candidates = filter_candidates(candidates, "year_buddhist")

        skill_tags = row.get("skill_tag") or []
        if skill_tags:
            skill_matches = [item for item in candidates if (item.get("skill_tag") or []) == skill_tags]
            if skill_matches:
                candidates = skill_matches

        if len(candidates) == 1:
            return candidates[0]
    return {}


# ── Core aggregation ────────────────────────────────────────────────────────


def aggregate_model(model_id: str, all_rows: list[dict], model_meta: dict) -> dict:
    """Compute all metrics for one model from all its batch rows."""
    meta = resolve_model_meta(model_id, model_meta)

    thai_rows = [r for r in all_rows if r["subject"] == "thai"]
    math_rows = [r for r in all_rows if r["subject"] == "math"]

    # Items per run (all runs have the same items, so use total / num_runs)
    run_ids = sorted(set(r["run_id"] for r in all_rows))
    num_runs = len(run_ids)
    items_per_run = len(all_rows) // num_runs if num_runs > 0 else len(all_rows)

    # Score rates: consistent across runs (stdev=0 in this batch), but compute
    # from all rows to be correct in general.
    thai_correct = sum(1 for r in thai_rows if r["is_correct"])
    math_correct = sum(1 for r in math_rows if r["is_correct"])
    total_correct = sum(1 for r in all_rows if r["is_correct"])
    total_items = len(all_rows)
    total_parseable = sum(1 for r in all_rows if r["is_parseable"])
    total_compliant = sum(1 for r in all_rows if is_answer_only_compliant(r.get("raw_output", "")))
    avg_output_length_chars = (
        sum(len((r.get("raw_output", "") or "")) for r in all_rows) / total_items
        if total_items else 0.0
    )

    thai_total = len(thai_rows)
    math_total = len(math_rows)

    thai_score_rate = thai_correct / thai_total if thai_total else 0.0
    math_score_rate = math_correct / math_total if math_total else 0.0
    balanced_quality_score = (thai_score_rate + math_score_rate) / 2.0
    overall_score_rate = total_correct / total_items if total_items else 0.0
    parseable_rate = total_parseable / total_items if total_items else 0.0
    compliance_rate = total_compliant / total_items if total_items else 0.0

    # Latency: across ALL individual question latencies from all runs
    all_latencies = [r["latency_ms"] for r in all_rows if r.get("latency_ms", 0) > 0]
    latency_p50 = round(percentile(all_latencies, 50)) if all_latencies else 0
    latency_p95 = round(percentile(all_latencies, 95)) if all_latencies else 0

    # Speed: batch-aggregated across all runs
    total_latency_ms = sum(r.get("latency_ms", 0) for r in all_rows)
    total_latency_min = total_latency_ms / 60000.0 if total_latency_ms > 0 else 0.0
    questions_per_min = total_items / total_latency_min if total_latency_min > 0 else 0.0
    correct_per_min = total_correct / total_latency_min if total_latency_min > 0 else 0.0

    # Throughput (optional)
    total_eval_tokens = sum(r.get("eval_tokens", 0) for r in all_rows)
    total_eval_ms = sum(r.get("eval_duration_ms", 0) for r in all_rows)
    throughput_toks_per_sec = (total_eval_tokens / (total_eval_ms / 1000.0)) if total_eval_ms > 0 else None

    # Per-run item count (for reporting)
    item_count = items_per_run

    # Per-run correct count (for model card total_correct field)
    correct_per_run = total_correct // num_runs if num_runs > 0 else total_correct

    # Round sub-rates first, then compute bqs from rounded values so that
    # bqs == (stored_thai + stored_math) / 2 holds exactly after rounding.
    thai_score_rate_r = round4(thai_score_rate)
    math_score_rate_r = round4(math_score_rate)
    balanced_quality_score_r = round4((thai_score_rate_r + math_score_rate_r) / 2.0)

    return {
        "model_id": model_id,
        "model_family": meta.get("model_family", "unknown"),
        "parameter_bucket": meta.get("parameter_bucket", "unknown"),
        "ram_fit_class": meta.get("ram_fit_class", "unknown"),
        "balanced_quality_score": balanced_quality_score_r,
        "thai_score_rate": thai_score_rate_r,
        "math_score_rate": math_score_rate_r,
        "overall_score_rate": round4(overall_score_rate),
        "parseable_rate": round4(parseable_rate),
        "answer_only_compliance_rate": round4(compliance_rate),
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "questions_per_min": round1(questions_per_min),
        "correct_per_min": round1(correct_per_min),
        "throughput_toks_per_sec": round1(throughput_toks_per_sec) if throughput_toks_per_sec else None,
        "item_count": item_count,
        "total_correct": correct_per_run,
        "average_output_length_chars": round1(avg_output_length_chars),
    }


# ── Skill tag analysis ──────────────────────────────────────────────────────


def compute_skill_stats(all_rows: list[dict]) -> dict[str, dict]:
    """Compute per-skill-tag {correct, total, score_rate} across all runs."""
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_rows:
        for tag in (r.get("skill_tag") or []):
            stats[tag]["total"] += 1
            if r["is_correct"]:
                stats[tag]["correct"] += 1
    for tag in stats:
        t = stats[tag]["total"]
        c = stats[tag]["correct"]
        stats[tag]["score_rate"] = round4(c / t) if t > 0 else 0.0
    return dict(stats)


def pick_strengths_weaknesses(skill_stats: dict, top_n: int = 3, min_n: int = 2):
    """Pick top-N strengths and weaknesses with min_n filter and tie-breaking.

    Tie-break: higher item count first, then alphabetical by tag name.
    """
    qualified = {tag: s for tag, s in skill_stats.items() if s["total"] >= min_n}

    def strength_key(tag):
        s = qualified[tag]
        # Sort: highest score_rate, then highest total, then alphabetical
        return (-s["score_rate"], -s["total"], tag)

    def weakness_key(tag):
        s = qualified[tag]
        # Sort: lowest score_rate, then highest total, then alphabetical
        return (s["score_rate"], -s["total"], tag)

    sorted_strong = sorted(qualified.keys(), key=strength_key)
    sorted_weak = sorted(qualified.keys(), key=weakness_key)

    strengths = []
    for tag in sorted_strong[:top_n]:
        s = qualified[tag]
        strengths.append({
            "skill_tag": tag,
            "score_rate": s["score_rate"],
            "correct": s["correct"],
            "total": s["total"],
        })

    weaknesses = []
    for tag in sorted_weak[:top_n]:
        s = qualified[tag]
        weaknesses.append({
            "skill_tag": tag,
            "score_rate": s["score_rate"],
            "correct": s["correct"],
            "total": s["total"],
        })

    return strengths, weaknesses


# ── Badge assignment ────────────────────────────────────────────────────────


def assign_badges(model_aggs: list[dict]) -> dict[str, list[str]]:
    """Assign badges. Ties allowed."""
    badges = defaultdict(list)

    # Best Quality: highest balanced_quality_score
    best_bqs = max(m["balanced_quality_score"] for m in model_aggs)
    for m in model_aggs:
        if m["balanced_quality_score"] == best_bqs:
            badges[m["model_id"]].append("Best Quality")

    # Best Thai
    best_thai = max(m["thai_score_rate"] for m in model_aggs)
    for m in model_aggs:
        if m["thai_score_rate"] == best_thai:
            badges[m["model_id"]].append("Best Thai")

    # Best Math
    best_math = max(m["math_score_rate"] for m in model_aggs)
    for m in model_aggs:
        if m["math_score_rate"] == best_math:
            badges[m["model_id"]].append("Best Math")

    # Fastest on Testbed: lowest latency_p50_ms
    best_lat = min(m["latency_p50_ms"] for m in model_aggs)
    for m in model_aggs:
        if m["latency_p50_ms"] == best_lat:
            badges[m["model_id"]].append("Fastest on Testbed")

    # Best Small Model: highest balanced_quality_score among fits_comfortably_16gb
    comfortable = [m for m in model_aggs if m["ram_fit_class"] == "fits_comfortably_16gb"]
    if comfortable:
        best_small_bqs = max(m["balanced_quality_score"] for m in comfortable)
        for m in comfortable:
            if m["balanced_quality_score"] == best_small_bqs:
                badges[m["model_id"]].append("Best Small Model")

    return dict(badges)


# ── Ranking ─────────────────────────────────────────────────────────────────


def rank_models(model_aggs: list[dict]) -> list[dict]:
    """Sort by balanced_quality_score desc, then tie-breakers.

    Tie-break order: parseable_rate desc, compliance_rate desc, latency_p50 asc.
    Tied models share the same rank.
    """
    def sort_key(m):
        return (
            -m["balanced_quality_score"],
            -m["parseable_rate"],
            -m["answer_only_compliance_rate"],
            m["latency_p50_ms"],
        )

    ranked = sorted(model_aggs, key=sort_key)

    # Assign ranks — ties share rank
    result = []
    prev_key = None
    prev_rank = 0
    for i, m in enumerate(ranked):
        k = sort_key(m)
        if k != prev_key:
            prev_rank = i + 1
            prev_key = k
        m_copy = dict(m)
        m_copy["rank"] = prev_rank
        result.append(m_copy)
    return result


# ── Example selection ────────────────────────────────────────────────────────


def select_examples(
    canonical_rows: list[dict],
    strengths: list[dict],
    weaknesses: list[dict],
    model_id: str,
    question_bank: dict[tuple[str, int], dict] | None = None,
    n_good: int = 2,
    n_bad: int = 2,
) -> list[dict]:
    """Deterministic example selection from canonical run.

    Good examples: one per subject when available, chosen from strongest skill tags.
    Bad examples: one per subject when available, chosen from weakest skill tags.
    Stable tie-break: subject, question_id.
    """
    strong_tags = [s["skill_tag"] for s in strengths]
    weak_tags = [w["skill_tag"] for w in weaknesses]

    def subject_rank(subject: str) -> tuple[int, str]:
        order = {"thai": 0, "math": 1}
        return (order.get(subject, 99), subject)

    def example_sort_key(row, preferred_tags):
        """Lower value = higher priority for selection."""
        tags = set(row.get("skill_tag") or [])
        # Priority: has a preferred tag (lower index = higher priority)
        tag_priority = len(preferred_tags)  # default: no match
        for i, t in enumerate(preferred_tags):
            if t in tags:
                tag_priority = i
                break
        return (tag_priority, -row.get("latency_ms", 0), row.get("subject", ""), row.get("question_id", 0))

    def select_subject_diverse_rows(rows: list[dict], preferred_tags: list[str], limit: int) -> list[dict]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row.get("subject", "unknown")].append(row)

        selected = []
        for subject in sorted(grouped.keys(), key=subject_rank):
            subject_rows = sorted(grouped[subject], key=lambda r: example_sort_key(r, preferred_tags))
            if subject_rows:
                selected.append(subject_rows[0])
            if len(selected) >= limit:
                break
        return selected

    # Good: correct items
    correct_rows = [r for r in canonical_rows if r.get("is_correct")]
    good = select_subject_diverse_rows(correct_rows, strong_tags, n_good)

    # Bad: incorrect items (parseable but wrong, or unparseable)
    incorrect_rows = [r for r in canonical_rows if not r.get("is_correct")]
    bad = select_subject_diverse_rows(incorrect_rows, weak_tags, n_bad)

    examples = []
    for i, row in enumerate(good):
        examples.append(_make_example(row, model_id, f"good_{i}", "good_top_skill", question_bank))
    for i, row in enumerate(bad):
        examples.append(_make_example(row, model_id, f"bad_{i}", "bad_weak_skill", question_bank))
    return examples


def _make_example(
    row: dict,
    model_id: str,
    suffix: str,
    reason: str,
    question_bank: dict[tuple[str, int], dict] | None = None,
) -> dict:
    raw = row.get("raw_output", "")
    item = resolve_question_item(row, question_bank)
    choices = row.get("choices") or item.get("choices") or {}
    parsed_answer = row.get("parsed_answer")
    model_answer_text = None
    if isinstance(choices, dict) and parsed_answer is not None:
        model_answer_text = choices.get(str(parsed_answer))

    # Stable example_id: model + subject + question_id + suffix
    safe_model = re.sub(r'[^a-zA-Z0-9]', '_', model_id)
    example_id = f"ex_{safe_model}_{row.get('subject', '')}_{row.get('question_id', 0)}_{suffix}"
    return {
        "example_id": example_id,
        "model_id": model_id,
        "exam_id": row.get("exam_id") or item.get("exam_id"),
        "year_buddhist": row.get("year_buddhist") or item.get("year_buddhist"),
        "subject": row.get("subject", ""),
        "question_id": row.get("question_id", 0),
        "question_no": row.get("question_no") or item.get("question_no"),
        "skill_tag": row.get("skill_tag") or [],
        "curriculum_standard": row.get("curriculum_standard"),
        "is_correct": row.get("is_correct", False),
        "correct_answer": str(row.get("correct_answer", "")),
        "correct_answer_text": row.get("correct_answer_text") or item.get("correct_answer_text"),
        "parsed_answer": row.get("parsed_answer"),
        "model_answer_text": model_answer_text,
        "stimulus_text": row.get("stimulus_text") or item.get("stimulus_text"),
        "prompt_text": row.get("prompt_text") or item.get("prompt_text"),
        "choices": choices,
        "raw_output_truncated": raw[:200],
        "raw_output_full": raw,
        "latency_ms": row.get("latency_ms", 0),
        "selection_reason": reason,
    }


# ── Auto-summary ────────────────────────────────────────────────────────────


def generate_auto_summary(
    model_id: str,
    agg: dict,
    strengths: list[dict],
    weaknesses: list[dict],
    speed_rank: int,
    total_models: int,
) -> str:
    """Deterministic phrase-template summary. No LLM generation."""
    thai = agg["thai_score_rate"]
    math = agg["math_score_rate"]

    # Strength phrase
    if thai >= 0.6 and math >= 0.6:
        strength_phrase = "well across both Thai and Math"
    elif thai - math > 0.15:
        strength_phrase = "stronger on Thai than Math"
    elif math - thai > 0.15:
        strength_phrase = "stronger on Math than Thai"
    else:
        strength_phrase = "at a balanced level across subjects"

    # Reliability phrase
    compliance = agg["answer_only_compliance_rate"]
    if compliance >= 0.95:
        reliability_phrase = "very reliable"
    elif compliance >= 0.70:
        reliability_phrase = "mostly reliable"
    else:
        reliability_phrase = "format-unstable"

    # Speed phrase
    if total_models <= 1:
        speed_phrase = "the only model tested"
    elif speed_rank == 1:
        speed_phrase = "among the fastest"
    elif speed_rank <= total_models * 0.5:
        speed_phrase = "mid-pack on speed"
    else:
        speed_phrase = "slower than peers"

    # Skill phrases
    top_skills = ", ".join(s["skill_tag"] for s in strengths[:3]) if strengths else "no standout areas"
    weak_skills = ", ".join(w["skill_tag"] for w in weaknesses[:3]) if weaknesses else "no clear weaknesses"

    return (
        f"{model_id} performs {strength_phrase}. "
        f"It is {reliability_phrase} on answer-only formatting. "
        f"Its strongest areas are {top_skills}. "
        f"Its weakest areas are {weak_skills}. "
        f"On the current Mac mini testbed, it is {speed_phrase}."
    )


# ── File discovery ──────────────────────────────────────────────────────────


def find_batch_files(batch_id: str, responses_dir: Path) -> dict[str, list[Path]]:
    """Find all JSONL files for a batch, grouped by model_id.

    Uses repeat_summary to get authoritative file list and model mapping.
    Falls back to glob pattern matching if repeat_summary not available.
    """
    summary_path = responses_dir / f"repeat_summary_{batch_id}.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        model_files = defaultdict(list)
        for run in summary["runs"]:
            model_id = run["model_id"]
            # output_file may be absolute path — resolve it
            fpath = Path(run["output_file"])
            if not fpath.exists():
                # Try relative to responses_dir
                fpath = responses_dir / fpath.name
            if fpath.exists():
                model_files[model_id].append((run["run_index"], fpath))
        # Sort by run_index for each model
        result = {}
        for model_id, entries in model_files.items():
            entries.sort(key=lambda x: x[0])
            result[model_id] = [e[1] for e in entries]
        return result

    # Fallback: glob
    pattern = f"responses_{batch_id}-*.jsonl"
    files = sorted(responses_dir.glob(pattern))
    model_files = defaultdict(list)
    for f in files:
        rows = load_jsonl(f)
        if rows:
            model_files[rows[0]["model_id"]].append(f)
    return dict(model_files)


def get_canonical_run_id(batch_id: str, model_id: str, responses_dir: Path) -> str | None:
    """Get the run_id of the canonical run (lowest run_index) from repeat_summary."""
    summary_path = responses_dir / f"repeat_summary_{batch_id}.json"
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    model_runs = [r for r in summary["runs"] if r["model_id"] == model_id]
    if not model_runs:
        return None
    model_runs.sort(key=lambda r: r["run_index"])
    return model_runs[0]["run_id"]


def load_verification_report(batch_id: str, responses_dir: Path) -> dict | None:
    """Load canonical+shadow verification report for a batch if present."""
    path = responses_dir / f"verification_report_{batch_id}.json"
    if not path.exists():
        return None
    return load_json(path)


# ── Main build ──────────────────────────────────────────────────────────────


def load_testbed(config_path: Path, profile_id: str = "macmini-m4-16gb-ollama") -> dict:
    """Load testbed info from machine_profiles.json by profile id."""
    data = load_json(config_path)
    for profile in data["machine_profiles"]:
        if profile["machine_profile"] == profile_id:
            return {
                "host_label": profile["host_label"],
                "chip": profile["chip"],
                "ram_gb": profile["ram_gb"],
                "backend": profile["backend"],
                "machine_profile": profile["machine_profile"],
            }
    raise RuntimeError(f"Machine profile '{profile_id}' not found in {config_path}")


def load_compatibility(config_path: Path) -> dict:
    """Load compatibility gate definition."""
    return load_json(config_path)


def resolve_compatibility_values(compat: dict, rows: list[dict], testbed: dict) -> dict:
    """Resolve actual compatibility field values from the batch data and testbed.

    For fields present in raw data, extract them deterministically.
    For fields not in raw data, use the V1 expected values when they are
    the only contract value for the current setup.
    """
    expected = compat.get("v1_expected_values", {})

    # Fields derivable from raw JSONL rows
    eval_splits = set()
    prompt_versions = set()
    think_modes = set()
    for r in rows:
        if "eval_split" in r:
            eval_splits.add(r["eval_split"])
        if "prompt_version" in r:
            prompt_versions.add(r["prompt_version"])
        think_enabled = r.get("think_enabled", False)
        think_modes.add("on" if think_enabled else "off")

    values = {
        "benchmark_scope": expected["benchmark_scope"],
        "dataset_version": expected["dataset_version"],
        "eval_split": eval_splits.pop() if len(eval_splits) == 1 else expected["eval_split"],
        "prompt_version": prompt_versions.pop() if len(prompt_versions) == 1 else expected["prompt_version"],
        "scoring_version": expected["scoring_version"],
        "machine_profile": testbed["machine_profile"],
        "think_mode": think_modes.pop() if len(think_modes) == 1 else expected["think_mode"],
    }
    return values


def build_snapshot(batch_id: str, snapshot_id: str | None = None, out_dir: Path | None = None):
    if snapshot_id is None:
        snapshot_id = f"nt-p3-mcq-text-only-{batch_id}"
    if out_dir is None:
        out_dir = PROJECT_ROOT / "dist" / snapshot_id

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    model_meta = load_model_meta(MODELS_CONFIG)
    testbed = load_testbed(MACHINE_PROFILES_CONFIG)
    compat = load_compatibility(COMPATIBILITY_CONFIG)
    question_bank = load_question_bank()

    # Find and load all batch files
    model_files = find_batch_files(batch_id, RESPONSES_DIR)
    if not model_files:
        raise RuntimeError(f"No response files found for batch '{batch_id}' in {RESPONSES_DIR}")

    print(f"Found {len(model_files)} models: {', '.join(sorted(model_files.keys()))}")

    # Load all rows per model
    verification_report = load_verification_report(batch_id, RESPONSES_DIR)
    verified_models: dict[str, dict] = {}
    if verification_report and verification_report.get("status") == "pass":
        verified_models = {
            model["model_id"]: model for model in verification_report.get("models", [])
            if model.get("deterministic")
        }

    model_all_rows = {}
    model_canonical_rows = {}
    model_public_rows = {}
    for model_id, files in model_files.items():
        all_rows = []
        for f in files:
            all_rows.extend(load_jsonl(f))
        model_all_rows[model_id] = all_rows

        # Canonical run = first file (lowest run_index, already sorted)
        canonical_run_id = get_canonical_run_id(batch_id, model_id, RESPONSES_DIR)
        if canonical_run_id:
            model_canonical_rows[model_id] = [r for r in all_rows if r["run_id"] == canonical_run_id]
        else:
            # Fallback: use first file
            model_canonical_rows[model_id] = load_jsonl(files[0])

        if model_id in verified_models:
            verified_canonical_run_id = verified_models[model_id].get("canonical_run_id")
            if not verified_canonical_run_id:
                raise RuntimeError(
                    f"Verification report missing canonical_run_id for verified model '{model_id}'"
                )
            public_rows = [r for r in all_rows if r["run_id"] == verified_canonical_run_id]
            if not public_rows:
                raise RuntimeError(
                    f"Verification report canonical_run_id '{verified_canonical_run_id}' for model '{model_id}' "
                    f"did not match any rows in batch '{batch_id}'"
                )
            model_public_rows[model_id] = public_rows
        else:
            model_public_rows[model_id] = all_rows

    # Resolve compatibility values from actual data
    all_rows_flat = [r for rows in model_public_rows.values() for r in rows]
    compat_values = resolve_compatibility_values(compat, all_rows_flat, testbed)

    # Aggregate metrics
    model_aggs = []
    for model_id, rows in model_public_rows.items():
        agg = aggregate_model(model_id, rows, model_meta)
        model_aggs.append(agg)

    # Badge assignment
    badge_map = assign_badges(model_aggs)
    for agg in model_aggs:
        agg["badges"] = badge_map.get(agg["model_id"], [])

    # Ranking
    ranked = rank_models(model_aggs)

    # Speed ranking for auto-summary (by latency_p50_ms ascending)
    speed_sorted = sorted(model_aggs, key=lambda m: m["latency_p50_ms"])
    speed_rank_map = {}
    for i, m in enumerate(speed_sorted):
        speed_rank_map[m["model_id"]] = i + 1

    # Build per-model details
    all_examples = []
    model_cards = []
    for agg in ranked:
        model_id = agg["model_id"]
        rows = model_public_rows[model_id]
        canonical = model_canonical_rows[model_id]

        skill_stats = compute_skill_stats(rows)
        strengths, weaknesses = pick_strengths_weaknesses(skill_stats)
        failure_types = common_failure_types(rows)

        examples = select_examples(canonical, strengths, weaknesses, model_id, question_bank=question_bank)
        all_examples.extend(examples)

        auto_summary = generate_auto_summary(
            model_id, agg, strengths, weaknesses,
            speed_rank_map[model_id], len(model_aggs),
        )

        card = {
            "model_id": model_id,
            "model_family": agg["model_family"],
            "parameter_bucket": agg["parameter_bucket"],
            "ram_fit_class": agg["ram_fit_class"],
            "testbed": {"host_label": testbed["host_label"]},
            "metrics": {
                "balanced_quality_score": agg["balanced_quality_score"],
                "thai_score_rate": agg["thai_score_rate"],
                "math_score_rate": agg["math_score_rate"],
                "overall_score_rate": agg["overall_score_rate"],
                "parseable_rate": agg["parseable_rate"],
                "answer_only_compliance_rate": agg["answer_only_compliance_rate"],
                "latency_p50_ms": agg["latency_p50_ms"],
                "latency_p95_ms": agg["latency_p95_ms"],
                "questions_per_min": agg["questions_per_min"],
                "correct_per_min": agg["correct_per_min"],
                "item_count": agg["item_count"],
                "total_correct": agg["total_correct"],
                "average_output_length_chars": agg["average_output_length_chars"],
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "common_failure_types": failure_types,
            "badges": agg["badges"],
            "auto_summary": auto_summary,
            "example_ids": {
                "good": [e["example_id"] for e in examples if e["is_correct"]],
                "bad": [e["example_id"] for e in examples if not e["is_correct"]],
            },
        }
        if agg["throughput_toks_per_sec"] is not None:
            card["metrics"]["throughput_toks_per_sec"] = agg["throughput_toks_per_sec"]
        model_cards.append(card)

    # ── Emit artifacts ───────────────────────────────────────────────────

    now = datetime.now(timezone(timedelta(hours=7)))

    manifest = {
        "snapshot_id": snapshot_id,
        "published_at": now.isoformat(timespec="seconds"),
        "benchmark_name": "NT P3 Local LLM Benchmark",
        "benchmark_label": "NT P3 MCQ Text-Only",
        "benchmark_scope": BENCHMARK_SCOPE,
        "suite_ids": SUITE_IDS,
        "testbed": testbed,
        "compatibility": {
            "required_match_fields": compat["required_match_fields"],
            "values": compat_values,
        },
        "snapshot_notes": {
            "public_outputs_truncated": True,
            "badges_enabled": True,
            "auto_model_summary": True,
            "ranking_excludes_image_required": True,
            "ranking_excludes_human_checked": True,
            "raw_bundle_downloadable": True,
        },
        "artifacts": {
            "leaderboard": "leaderboard.json",
            "model_cards": "model_cards.json",
            "examples": "examples.json",
            "repeat_summary": "repeat_summary.json",
            "results": "results.jsonl",
        },
    }

    if verification_report and verification_report.get("status") == "pass":
        manifest["publication"] = {
            "mode": "verified_single_run",
            "verification_protocol": verification_report.get("protocol"),
            "screening_batch_id": verification_report.get("screening_batch_id"),
            "canonical_run_index": verification_report.get("canonical_run_index", 1),
            "shadow_run_index": verification_report.get("shadow_run_index", 2),
        }
        manifest["snapshot_notes"]["published_metrics_use_canonical_run_only"] = True
        manifest["snapshot_notes"]["shadow_run_kept_for_verification_only"] = True
        manifest["artifacts"]["verification_report"] = "verification_report.json"

    leaderboard = {
        "snapshot_id": snapshot_id,
        "benchmark_scope": BENCHMARK_SCOPE,
        "rows": [
            {
                "rank": m["rank"],
                "model_id": m["model_id"],
                "model_family": m["model_family"],
                "parameter_bucket": m["parameter_bucket"],
                "ram_fit_class": m["ram_fit_class"],
                "balanced_quality_score": m["balanced_quality_score"],
                "thai_score_rate": m["thai_score_rate"],
                "math_score_rate": m["math_score_rate"],
                "overall_score_rate": m["overall_score_rate"],
                "parseable_rate": m["parseable_rate"],
                "answer_only_compliance_rate": m["answer_only_compliance_rate"],
                "latency_p50_ms": m["latency_p50_ms"],
                "latency_p95_ms": m["latency_p95_ms"],
                "questions_per_min": m["questions_per_min"],
                "correct_per_min": m["correct_per_min"],
                "item_count": m["item_count"],
                "badges": m["badges"],
            }
            for m in ranked
        ],
    }

    examples_json = {
        "snapshot_id": snapshot_id,
        "benchmark_scope": BENCHMARK_SCOPE,
        "examples": all_examples,
    }

    model_cards_json = {
        "snapshot_id": snapshot_id,
        "benchmark_scope": BENCHMARK_SCOPE,
        "models": model_cards,
    }

    def write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  wrote {path}")

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "leaderboard.json", leaderboard)
    write_json(out_dir / "model_cards.json", model_cards_json)
    write_json(out_dir / "examples.json", examples_json)

    # Transparency: combined normalized row-level source rows across the published metric rows.
    # For verified single-run publication, this is canonical-only. The shadow run remains in
    # raw/ plus verification_report.json for auditability.
    results_path = out_dir / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for model_id in sorted(model_public_rows):
            for row in model_public_rows[model_id]:
                normalized = normalize_raw_row(row, snapshot_id, testbed, model_meta)
                f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    print(f"  wrote {results_path}")

    # Transparency: copy repeat_summary
    repeat_src = RESPONSES_DIR / f"repeat_summary_{batch_id}.json"
    if repeat_src.exists():
        shutil.copy2(repeat_src, out_dir / "repeat_summary.json")
        print(f"  copied {out_dir / 'repeat_summary.json'}")

    verification_src = RESPONSES_DIR / f"verification_report_{batch_id}.json"
    if verification_src.exists():
        shutil.copy2(verification_src, out_dir / "verification_report.json")
        print(f"  copied {out_dir / 'verification_report.json'}")

    # Transparency: copy underlying per-run source files
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    copied = 0
    for files in model_files.values():
        for fpath in files:
            shutil.copy2(fpath, raw_dir / fpath.name)
            copied += 1
    print(f"  copied {copied} raw run files into {raw_dir}")

    # Zip bundle
    zip_path = out_dir.parent / f"{snapshot_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(out_dir.rglob("*")):
            if fpath.is_file():
                zf.write(fpath, f"{snapshot_id}/{fpath.relative_to(out_dir)}")
    print(f"  zip bundle: {zip_path}")

    print(f"\nSnapshot '{snapshot_id}' built in {out_dir}")
    return out_dir


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Build dashboard snapshot bundle")
    parser.add_argument("--batch-id", required=True, help="Batch id, e.g. mini-r10-20260409")
    parser.add_argument("--snapshot-id", default=None, help="Override snapshot id")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else None
    build_snapshot(args.batch_id, args.snapshot_id, out_dir)


if __name__ == "__main__":
    main()
