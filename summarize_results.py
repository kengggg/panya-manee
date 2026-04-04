"""Summarize benchmark results across models."""

import json
import glob
from collections import defaultdict

from config import RESPONSES_DIR


def summarize(run_id=None, responses_dir=None):
    responses_dir = responses_dir or RESPONSES_DIR

    if run_id:
        pattern = str(responses_dir / f"responses_{run_id}_*.jsonl")
    else:
        pattern = str(responses_dir / "responses_*.jsonl")

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files found matching: {pattern}")
        return

    model_results = {}
    for f in files:
        rows = [json.loads(line) for line in open(f)]
        if rows:
            model = rows[0]["model_id"]
            model_results[model] = rows

    if not model_results:
        print("No results to summarize.")
        return

    # Compute item counts dynamically
    all_subjects = defaultdict(int)
    for rows in model_results.values():
        for r in rows:
            all_subjects[r["subject"]] += 1
        break  # counts from first model are representative

    total_items = sum(all_subjects.values())
    subject_desc = " + ".join(f"{v} {k.capitalize()}" for k, v in sorted(all_subjects.items()))

    run_label = run_id or "all runs"
    print("=" * 65)
    print(f"NT ป.3 Benchmark — {run_label}")
    print(f"Tier 1: text_only_core | {total_items} items ({subject_desc})")
    print("=" * 65)

    summary = []
    for model, rows in model_results.items():
        total = len(rows)
        correct = sum(1 for r in rows if r["is_correct"])
        parseable = sum(1 for r in rows if r["is_parseable"])
        thai = [r for r in rows if r["subject"] == "thai"]
        math = [r for r in rows if r["subject"] == "math"]
        thai_c = sum(1 for r in thai if r["is_correct"])
        math_c = sum(1 for r in math if r["is_correct"])
        avg_lat = sum(r["latency_ms"] for r in rows) / total / 1000  # avg ms -> seconds
        summary.append((model, correct, total, thai_c, len(thai), math_c, len(math), parseable, avg_lat))

    summary.sort(key=lambda x: -x[1])

    header = "%-32s %8s %8s %8s %8s %7s" % ("Model", "Overall", "Thai", "Math", "Parse%", "Lat/q")
    print(header)
    print("-" * 65)
    for model, c, t, tc, tl, mc, ml, p, lat in summary:
        row = "%-32s %3d/%2d=%3.0f%% %3d/%2d=%3.0f%% %3d/%2d=%3.0f%% %3.0f%% %6.1fs" % (
            model, c, t, c / t * 100, tc, tl, tc / tl * 100 if tl else 0, mc, ml, mc / ml * 100 if ml else 0, p / t * 100, lat
        )
        print(row)

    print()
    print("Random baseline (guessing '1' every time or uniform random): ~25%")
    print()

    # Parse method distribution
    print("=" * 65)
    print("Parse method distribution")
    print("=" * 65)
    methods = ["direct", "markdown_bold", "thai_keyword", "fallback_digit", None]
    method_labels = ["direct", "md_bold", "keyword", "fallback", "unparse"]
    header_pm = "%-32s" % "Model"
    for label in method_labels:
        header_pm += " %8s" % label
    print(header_pm)
    print("-" * 65)
    for model, *_ in summary:
        rows = model_results[model]
        counts = {m: 0 for m in methods}
        for r in rows:
            pm = r.get("parse_method")
            if pm in counts:
                counts[pm] += 1
            else:
                counts[None] += 1
        row_pm = "%-32s" % model
        for m in methods:
            row_pm += " %8d" % counts[m]
        print(row_pm)
    print()

    # Infra usage
    has_infra = any("eval_tokens" in r for r in next(iter(model_results.values())))
    if has_infra:
        print("=" * 65)
        print("Infra usage (per question average)")
        print("=" * 65)
        header_inf = "%-32s %7s %10s %10s %7s" % ("Model", "Tok/q", "Eval ms/q", "Prmt ms/q", "tok/s")
        print(header_inf)
        print("-" * 65)
        for model, *_ in summary:
            rows = model_results[model]
            total = len(rows)
            total_eval_tok = sum(r.get("eval_tokens", 0) for r in rows)
            total_eval_ms = sum(r.get("eval_duration_ms", 0) for r in rows)
            total_prompt_ms = sum(r.get("prompt_eval_duration_ms", 0) for r in rows)
            avg_tok = total_eval_tok / total
            avg_eval_ms = total_eval_ms / total
            avg_prompt_ms = total_prompt_ms / total
            tok_per_sec = total_eval_tok / (total_eval_ms / 1000) if total_eval_ms > 0 else 0
            row_inf = "%-32s %7.0f %10.0f %10.0f %7.1f" % (model, avg_tok, avg_eval_ms, avg_prompt_ms, tok_per_sec)
            print(row_inf)
        print()

    # Skill breakdown
    print("=" * 65)
    print("Accuracy by skill tag (all models combined)")
    print("=" * 65)
    skill_all = defaultdict(lambda: defaultdict(lambda: {"c": 0, "t": 0}))
    for model, rows in model_results.items():
        for r in rows:
            for tag in r.get("skill_tag") or []:
                skill_all[tag][model]["t"] += 1
                if r["is_correct"]:
                    skill_all[tag][model]["c"] += 1

    models_sorted = [x[0] for x in summary]
    header2 = "%-30s" % "Skill"
    for m in models_sorted:
        short = m.split(":")[0][-10:] + ":" + m.split(":")[1][:4] if ":" in m else m[:14]
        header2 += " %7s" % short
    print(header2)
    print("-" * 65)
    for tag in sorted(skill_all.keys()):
        row2 = "%-30s" % tag[:30]
        for m in models_sorted:
            s = skill_all[tag][m]
            if s["t"] > 0:
                row2 += " %3d/%3d " % (s["c"], s["t"])
            else:
                row2 += "     -  "
        print(row2)
