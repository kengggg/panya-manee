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

from config import (
    OLLAMA_URL,
    DATASET_FILES,
    RESPONSES_DIR,
    PROMPT_TEMPLATE,
    validate_items,
)


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


def call_ollama(model, prompt, timeout=120, think=None):
    """Call Ollama API. Returns (raw_output, thinking_output, latency_ms, metrics).

    Args:
        think: None disables thinking. An int enables thinking with that num_predict budget.
    """
    num_predict = think if think is not None else 2048
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think is not None,
        "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": max(num_predict + 1024, 4096)},
    }
    t0 = time.time()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
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

    print(f"\nModel: {model}")
    print(f"Think: {'enabled (budget={think})'.format(think=think) if think else 'disabled'}")
    print(f"Items: {len(all_items)} ({', '.join(subjects)})")
    print(f"Output: {out_path}")
    if dry_run:
        print("[DRY RUN — no API calls]\n")

    results = []
    correct = 0
    parseable = 0

    for idx, item in enumerate(all_items, 1):
        qid = item["question_id"]
        year = item.get("year_buddhist", "")
        subject = item["_subject"]
        ground_truth = str(item.get("correct_answer", ""))

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

        result = {
            "model_id": model,
            "run_id": run_id,
            "exam_id": item.get("exam_id"),
            "year_buddhist": year,
            "subject": subject,
            "question_id": qid,
            "eval_split": item.get("eval_split"),
            "skill_tag": item.get("skill_tag"),
            "curriculum_standard": item.get("curriculum_standard"),
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
        }
        results.append(result)

        status = "✓" if is_correct else ("?" if not is_parseable else "✗")
        print(f"  [{idx:02d}] {subject}/{year} ข้อ{qid:02d} | raw={repr(raw[:20])} parsed={parsed} ans={ground_truth} {status} [{parse_method}] ({latency:.0f}ms, {metrics['eval_count']}tok)")

        if not dry_run:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Overall accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Parseable rate:   {parseable}/{total} = {parseable/total*100:.1f}%")

    for subj in subjects:
        sub_results = [r for r in results if r["subject"] == subj]
        sub_correct = sum(1 for r in sub_results if r["is_correct"])
        print(f"  {subj}: {sub_correct}/{len(sub_results)} = {sub_correct/len(sub_results)*100:.1f}%")

    # By skill
    skill_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        for tag in (r.get("skill_tag") or []):
            skill_stats[tag]["total"] += 1
            if r["is_correct"]:
                skill_stats[tag]["correct"] += 1

    print("\nBy skill tag:")
    for tag, stat in sorted(skill_stats.items(), key=lambda x: -x[1]["total"]):
        if stat["total"] >= 2:
            pct = stat["correct"] / stat["total"] * 100
            print(f"  {tag}: {stat['correct']}/{stat['total']} ({pct:.0f}%)")

    if not dry_run:
        print(f"\nSaved: {out_path}")

    return results
