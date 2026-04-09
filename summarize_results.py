"""Summarize benchmark results across models."""

import json
import glob
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import RESPONSES_DIR

console = Console()


def shorten_model(name: str, max_len: int = 20) -> str:
    """Shorten model name, keeping the most distinctive parts."""
    if len(name) <= max_len:
        return name
    if "/" in name:
        name = name.split("/", 1)[1]
    if name.endswith(":latest"):
        name = name[: -len(":latest")]
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _color_pct(value: float) -> str:
    color = "green" if value >= 70 else "yellow" if value >= 40 else "red"
    return f"[{color}]{value:.0f}%[/{color}]"


def _color_score(c: int, t: int) -> str:
    if t == 0:
        return "[dim]-[/dim]"
    pct = c / t * 100
    color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"
    return f"[{color}]{c}/{t}[/{color}]"


def summarize(run_id=None, responses_dir=None):
    responses_dir = responses_dir or RESPONSES_DIR

    if run_id:
        pattern = str(responses_dir / f"responses_{run_id}_*.jsonl")
    else:
        pattern = str(responses_dir / "responses_*.jsonl")

    files = sorted(glob.glob(pattern))
    if not files:
        console.print(f"[dim]No result files found matching: {pattern}[/dim]")
        return

    model_results = {}
    for f in files:
        rows = [json.loads(line) for line in open(f)]
        if rows:
            model = rows[0]["model_id"]
            model_results[model] = rows

    if not model_results:
        console.print("[dim]No results to summarize.[/dim]")
        return

    # Compute item counts dynamically
    all_subjects = defaultdict(int)
    for rows in model_results.values():
        for r in rows:
            all_subjects[r["subject"]] += 1
        break

    total_items = sum(all_subjects.values())
    subject_desc = " + ".join(f"{v} {k.capitalize()}" for k, v in sorted(all_subjects.items()))
    run_label = run_id or "all runs"

    # ── Leaderboard ──────────────────────────────────────────────────────
    summary = []
    for model, rows in model_results.items():
        total = len(rows)
        correct = sum(1 for r in rows if r["is_correct"])
        parseable = sum(1 for r in rows if r["is_parseable"])
        thai = [r for r in rows if r["subject"] == "thai"]
        math = [r for r in rows if r["subject"] == "math"]
        thai_c = sum(1 for r in thai if r["is_correct"])
        math_c = sum(1 for r in math if r["is_correct"])
        avg_lat = sum(r["latency_ms"] for r in rows) / total / 1000
        summary.append((model, correct, total, thai_c, len(thai), math_c, len(math), parseable, avg_lat))

    summary.sort(key=lambda x: -x[1])

    lb = Table(show_edge=False, pad_edge=False, box=None)
    lb.add_column("Model", style="bold")
    lb.add_column("Overall", justify="right")
    lb.add_column("Thai", justify="right")
    lb.add_column("Math", justify="right")
    lb.add_column("Parse", justify="right")
    lb.add_column("Lat/q", justify="right", style="dim")

    for model, c, t, tc, tl, mc, ml, p, lat in summary:
        lb.add_row(
            shorten_model(model),
            f"{_color_pct(c/t*100)} {c}/{t}",
            f"{_color_pct(tc/tl*100 if tl else 0)} {tc}/{tl}" if tl else "[dim]-[/dim]",
            f"{_color_pct(mc/ml*100 if ml else 0)} {mc}/{ml}" if ml else "[dim]-[/dim]",
            f"{p/t*100:.0f}%",
            f"{lat:.1f}s",
        )

    console.print()
    console.print(Panel(
        lb,
        title=f"[bold]NT ป.3 Benchmark — {run_label}[/bold]",
        subtitle=f"{total_items} items ({subject_desc})",
        border_style="blue",
    ))
    console.print("[dim]Random baseline ≈ 25%[/dim]")

    # ── Parse method distribution ────────────────────────────────────────
    methods = ["direct", "markdown_bold", "thai_keyword", "fallback_digit", "thinking_fallback", None]
    method_labels = ["direct", "md_bold", "keyword", "fallback", "think_fb", "unparse"]

    pm_table = Table(show_edge=False, pad_edge=False, box=None)
    pm_table.add_column("Model", style="bold")
    for label in method_labels:
        pm_table.add_column(label, justify="right")

    for model, *_ in summary:
        rows = model_results[model]
        counts = {m: 0 for m in methods}
        for r in rows:
            pm = r.get("parse_method")
            if pm in counts:
                counts[pm] += 1
            else:
                counts[None] += 1
        cells = []
        for m in methods:
            v = counts[m]
            cells.append(str(v) if v > 0 else "[dim]·[/dim]")
        pm_table.add_row(shorten_model(model), *cells)

    console.print()
    console.print(Panel(pm_table, title="[bold]Parse methods[/bold]", border_style="dim"))

    # ── Infra usage ──────────────────────────────────────────────────────
    has_infra = any("eval_tokens" in r for r in next(iter(model_results.values())))
    if has_infra:
        inf_table = Table(show_edge=False, pad_edge=False, box=None)
        inf_table.add_column("Model", style="bold")
        inf_table.add_column("Tok/q", justify="right")
        inf_table.add_column("Eval ms/q", justify="right")
        inf_table.add_column("Prmt ms/q", justify="right")
        inf_table.add_column("tok/s", justify="right")

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
            inf_table.add_row(
                shorten_model(model),
                f"{avg_tok:.0f}",
                f"{avg_eval_ms:.0f}",
                f"{avg_prompt_ms:.0f}",
                f"{tok_per_sec:.1f}" if tok_per_sec > 0 else "[dim]·[/dim]",
            )

        console.print()
        console.print(Panel(inf_table, title="[bold]Infra usage[/bold] (per question avg)", border_style="dim"))

    # ── Skill breakdown by subject ───────────────────────────────────────
    models_sorted = [x[0] for x in summary]
    subjects_in_data = sorted(set(r["subject"] for rows in model_results.values() for r in rows))

    for subj in subjects_in_data:
        skill_subj = defaultdict(lambda: defaultdict(lambda: {"c": 0, "t": 0}))
        for model, rows in model_results.items():
            for r in rows:
                if r["subject"] != subj:
                    continue
                for tag in r.get("skill_tag") or []:
                    skill_subj[tag][model]["t"] += 1
                    if r["is_correct"]:
                        skill_subj[tag][model]["c"] += 1

        sk_table = Table(show_edge=False, pad_edge=False, box=None)
        sk_table.add_column("Skill", style="dim")
        for m in models_sorted:
            sk_table.add_column(shorten_model(m), justify="right")

        tag_totals = {tag: sum(d["t"] for d in by_model.values()) for tag, by_model in skill_subj.items()}
        for tag in sorted(skill_subj.keys(), key=lambda t: -tag_totals[t]):
            cells = []
            for m in models_sorted:
                s = skill_subj[tag][m]
                cells.append(_color_score(s["c"], s["t"]))
            sk_table.add_row(tag, *cells)

        console.print()
        console.print(Panel(sk_table, title=f"[bold]{subj}[/bold] by skill", border_style="dim"))
