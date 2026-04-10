"""
NT Grade 3 Local LLM Benchmark Runner
Tier 1: text_only_core MCQ only
- Sends ONLY stimulus + question + choices to LLM
- Scores AFTER response using stored ground truth
- Saves responses.jsonl per model run
"""

import json
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import (
    OLLAMA_URL,
    DATASET_FILES,
    RESPONSES_DIR,
    PROMPT_TEMPLATE,
    validate_items,
)

console = Console()


def _color_pct(value: float) -> str:
    """Return a rich-colored percentage string."""
    color = "green" if value >= 70 else "yellow" if value >= 40 else "red"
    return f"[{color}]{value:.1f}%[/{color}]"


def _bar(correct: int, total: int, width: int = 15) -> str:
    """Return a colored bar string."""
    if total == 0:
        return ""
    filled = round(correct / total * width)
    return f"[green]{'█' * filled}[/green][dim]{'░' * (width - filled)}[/dim]"


def _build_live_display(model, think, idx, total, correct, parseable,
                        per_subj_correct, per_subj_total, current_label, last_status):
    """Build the live progress panel."""
    pct = correct / idx * 100 if idx > 0 else 0
    parse_pct = parseable / idx * 100 if idx > 0 else 0

    lines = []
    # Progress bar
    bar_filled = round(idx / total * 20) if total > 0 else 0
    bar = f"[green]{'━' * bar_filled}[/green][dim]{'━' * (20 - bar_filled)}[/dim]"
    lines.append(f"  Progress  {bar}  [bold]{idx}[/bold]/{total}")
    # Accuracy
    subj_parts = []
    for subj in sorted(per_subj_total.keys()):
        st = per_subj_total[subj]
        sc = per_subj_correct[subj]
        sp = sc / st * 100 if st > 0 else 0
        subj_parts.append(f"{subj} {_color_pct(sp)}")
    lines.append(f"  Accuracy  {_color_pct(pct)}  ({' │ '.join(subj_parts)})")
    lines.append(f"  Parseable {parse_pct:.0f}%")
    if current_label:
        status_icon = last_status or "…"
        lines.append(f"  Current   [dim]{current_label}[/dim]  {status_icon}")

    think_label = f"Think: budget={think}" if think else "Think: off"
    title = f"[bold]{model}[/bold] │ {think_label}"
    body = "\n".join(lines)
    return Panel(body, title=title, border_style="blue", width=60)


# ── Helpers ──────────────────────────────────────────────────────────────────
def build_prompt(item: dict) -> str:
    """Build prompt — only stimulus, question, choices. No answer."""
    stimulus = item.get("stimulus_text") or ""
    if stimulus and not stimulus.endswith("\n"):
        stimulus += "\n"
    choices = item.get("choices", {})
    return PROMPT_TEMPLATE.format(
        stimulus=stimulus,
        question=item["prompt_text"],
        c1=choices.get("1", ""),
        c2=choices.get("2", ""),
        c3=choices.get("3", ""),
        c4=choices.get("4", ""),
    )


def parse_answer(raw: str):
    """Extract single digit 1-4 from model output. Returns (digit, method) tuple."""
    raw = raw.strip()
    # Try first char
    if raw and raw[0] in "1234":
        return raw[0], "direct"
    # Handle markdown bold: **3) ...** or **ข้อ 3**
    m = re.search(r'\*\*\s*([1-4])[).]', raw)
    if m:
        return m.group(1), "markdown_bold"
    # Handle "ข้อ 3" or "ตัวเลือก 3"
    m = re.search(r'(?:ข้อ|ตัวเลือก|คำตอบ|answer)[^1-4]*([1-4])', raw, re.IGNORECASE)
    if m:
        return m.group(1), "thai_keyword"
    # Find any digit 1-4 in response
    m = re.search(r'[1-4]', raw)
    if m:
        return m.group(0), "fallback_digit"
    return None, None


def call_ollama(model, prompt, timeout=120, think=None, max_attempts=2):
    """Call Ollama API. Returns (raw_output, thinking_output, latency_ms, metrics).

    Args:
        think: None disables thinking. An int enables thinking with that num_predict budget.
        max_attempts: Total attempts before giving up (default 2).

    Retries once on transient errors (connection reset, HTTP 5xx, timeout).
    """
    num_predict = think if think is not None else 2048
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think is not None,
        "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": max(num_predict + 1024, 4096)},
    }
    body = json.dumps(payload).encode()

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(
                OLLAMA_URL,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            latency = (time.time() - t0) * 1000
            msg = data.get("message", {})
            raw = msg.get("content", "").strip()
            thinking = msg.get("thinking", "").strip()
            metrics = {
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "prompt_eval_duration_ms": data.get("prompt_eval_duration", 0) / 1e6,
                "eval_count": data.get("eval_count", 0),
                "eval_duration_ms": data.get("eval_duration", 0) / 1e6,
                "load_duration_ms": data.get("load_duration", 0) / 1e6,
            }
            return raw, thinking, latency, metrics
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(2)

    # All attempts exhausted — raise with clear context
    raise RuntimeError(
        f"Ollama call failed after {max_attempts} attempts: {type(last_err).__name__}: {last_err}"
    ) from last_err


def check_ollama():
    """Check if Ollama is reachable. Returns True/False."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def load_items(subject: str, eval_split: str = "text_only_core") -> list[dict]:
    """Load items — validates data integrity, then filters by eval_split."""
    path = DATASET_FILES[subject]
    all_items = json.load(open(path, encoding="utf-8"))
    validate_items(all_items, path)
    return [i for i in all_items if i.get("eval_split") == eval_split]


def run_benchmark(model: str, subjects: list[str], run_id: str, dry_run: bool = False, think: int | None = None):
    RESPONSES_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESPONSES_DIR / f"responses_{run_id}_{timestamp}.jsonl"

    all_items = []
    for subject in subjects:
        items = load_items(subject)
        for item in items:
            item["_subject"] = subject
        all_items.extend(items)

    total = len(all_items)
    if dry_run:
        console.print(f"[dim]DRY RUN — no API calls[/dim]\n")

    results = []
    correct = 0
    parseable = 0
    per_subj_correct = defaultdict(int)
    per_subj_total = defaultdict(int)

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        for idx, item in enumerate(all_items, 1):
            qid = item["question_id"]
            year = item.get("year_buddhist", "")
            subject = item["_subject"]
            ground_truth = str(item.get("correct_answer", ""))

            current_label = f"{subject}/{year} ข้อ{qid:02d}"
            live.update(_build_live_display(
                model, think, idx - 1, total, correct, parseable,
                per_subj_correct, per_subj_total, current_label, None))

            prompt = build_prompt(item)

            empty_metrics = {"prompt_eval_count": 0, "prompt_eval_duration_ms": 0, "eval_count": 0, "eval_duration_ms": 0, "load_duration_ms": 0}
            if dry_run:
                raw = "3"
                thinking = ""
                latency = 0.0
                metrics = empty_metrics
            else:
                try:
                    raw, thinking, latency, metrics = call_ollama(model, prompt, think=think)
                except Exception as e:
                    raw = f"ERROR: {e}"
                    thinking = ""
                    latency = -1.0
                    metrics = empty_metrics

            parsed, parse_method = parse_answer(raw)
            think_budget_hit = think is not None and metrics["eval_count"] >= think
            if parsed is None and thinking:
                parsed, parse_method = parse_answer(thinking)
                if parsed is not None:
                    parse_method = "thinking_fallback"
            is_parseable = parsed is not None
            is_correct = parsed == ground_truth if is_parseable else False

            if is_parseable:
                parseable += 1
            if is_correct:
                correct += 1
            per_subj_total[subject] += 1
            if is_correct:
                per_subj_correct[subject] += 1

            result = {
                "model_id": model,
                "run_id": run_id,
                "exam_id": item.get("exam_id"),
                "year_buddhist": year,
                "subject": subject,
                "question_id": qid,
                "question_no": item.get("question_no"),
                "eval_split": item.get("eval_split"),
                "skill_tag": item.get("skill_tag"),
                "curriculum_standard": item.get("curriculum_standard"),
                "stimulus_text": item.get("stimulus_text"),
                "prompt_text": item.get("prompt_text"),
                "choices": item.get("choices"),
                "correct_answer_text": item.get("correct_answer_text"),
                "parse_method": parse_method,
                "prompt_version": "v1_answer_only",
                "raw_output": raw,
                "thinking_output": thinking or None,
                "parsed_answer": parsed,
                "correct_answer": ground_truth,
                "is_parseable": is_parseable,
                "is_correct": is_correct,
                "score": item.get("max_score", 3) if is_correct else 0,
                "max_score": item.get("max_score", 3),
                "latency_ms": round(latency),
                "prompt_tokens": metrics["prompt_eval_count"],
                "eval_tokens": metrics["eval_count"],
                "eval_duration_ms": round(metrics["eval_duration_ms"]),
                "prompt_eval_duration_ms": round(metrics["prompt_eval_duration_ms"]),
                "think_enabled": think is not None,
                "think_budget_hit": think_budget_hit,
            }
            results.append(result)

            status_icon = "[green]✓[/green]" if is_correct else ("[yellow]?[/yellow]" if not is_parseable else "[red]✗[/red]")
            if think_budget_hit:
                status_icon += " [yellow]⚠ BUDGET[/yellow]"
            live.update(_build_live_display(
                model, think, idx, total, correct, parseable,
                per_subj_correct, per_subj_total, current_label, status_icon))

            if not dry_run:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    _print_summary(model, think, results, subjects, out_path, dry_run)
    return results


def _print_summary(model, think, results, subjects, out_path, dry_run):
    """Print the rich summary after a benchmark run."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    parseable = sum(1 for r in results if r["is_parseable"])

    # Subject table
    subj_table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
    subj_table.add_column("Subject", style="bold")
    subj_table.add_column("Score", justify="right")
    subj_table.add_column("%", justify="right")
    subj_table.add_column("", min_width=15)

    for subj in subjects:
        sub_results = [r for r in results if r["subject"] == subj]
        sub_correct = sum(1 for r in sub_results if r["is_correct"])
        pct = sub_correct / len(sub_results) * 100 if sub_results else 0
        subj_table.add_row(
            subj,
            f"{sub_correct}/{len(sub_results)}",
            _color_pct(pct),
            _bar(sub_correct, len(sub_results)),
        )

    overall_pct = correct / total * 100 if total else 0
    think_label = f"think={think}" if think else "no-think"
    header_text = (
        f"[bold]{_color_pct(overall_pct)}[/bold]  {correct}/{total} correct  │  "
        f"{parseable}/{total} parseable  │  {think_label}"
    )

    console.print()
    console.print(Panel(
        Group(header_text, "", subj_table),
        title=f"[bold]{model}[/bold]",
        border_style="blue",
        width=60,
    ))

    # Budget warning
    if think is not None:
        budget_hits = sum(1 for r in results if r.get("think_budget_hit"))
        if budget_hits:
            console.print(Panel(
                f"[bold]{budget_hits}/{total}[/bold] items hit the {think}-token limit.\n"
                f"The model never finished reasoning — results unreliable.\n"
                f"Try [bold]--think {think * 2}[/bold] or [bold]--no-think[/bold].",
                title="⚠ Think budget exceeded",
                border_style="yellow",
                width=60,
            ))

    # Skill tables per subject
    for subj in subjects:
        sub_results = [r for r in results if r["subject"] == subj]
        skill_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in sub_results:
            for tag in (r.get("skill_tag") or []):
                skill_stats[tag]["total"] += 1
                if r["is_correct"]:
                    skill_stats[tag]["correct"] += 1

        skill_table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
        skill_table.add_column("Skill", style="dim", min_width=28)
        skill_table.add_column("Score", justify="right")
        skill_table.add_column("%", justify="right", min_width=6)
        skill_table.add_column("", min_width=15)

        for tag, stat in sorted(skill_stats.items(), key=lambda x: -x[1]["total"]):
            if stat["total"] >= 2:
                pct = stat["correct"] / stat["total"] * 100
                skill_table.add_row(
                    tag,
                    f"{stat['correct']}/{stat['total']}",
                    _color_pct(pct),
                    _bar(stat["correct"], stat["total"]),
                )

        console.print(Panel(skill_table, title=f"[bold]{subj}[/bold] by skill", border_style="dim", width=60))

    if not dry_run:
        console.print(f"\n[dim]Saved: {out_path}[/dim]")
