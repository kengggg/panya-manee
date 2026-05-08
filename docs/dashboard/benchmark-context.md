# Panya Manee — Local LLM Benchmark

**Created:** 2026-04-09
**Status:** Active
**Context:** Benchmark small local LLMs on Thai NT Grade 3 (ป.3) standardized tests, with the Mac mini as baseline inference machine
**Repo:** `https://github.com/kengggg/panya-manee`

---

## Goal

ประเมินว่า small local LLM models ที่รันบนเครื่องจริงได้ มี proficiency ภาษาไทยและคณิตระดับ NT ป.3 แค่ไหน โดยใช้ชุดข้อสอบมาตรฐานไทยเป็น benchmark หลัก

โจทย์ต่อเนื่อง:
- ออกแบบ automation สำหรับรัน benchmark ซ้ำบน Mac mini
- ออกแบบ dashboard / PRD สำหรับดูผลเทียบหลายโมเดล
- ใช้ Mac mini เป็น baseline machine สำหรับ local inference

---

## Benchmark Harness — Current Understanding

Project `panya-manee` ตอนนี้ทำได้แล้วสำหรับ split `text_only_core`

Current runnable set:
- 93 items total
- Thai: 60
- Math: 33

Current behavior:
- ใช้ Ollama local chat API
- prompt ภาษาไทย บังคับตอบแค่ `1-4`
- parse + score ต่อข้อ
- บันทึก per-question JSONL
- summarize ได้ทั้ง overall / subject / skill / parse method / infra usage

Important runtime note:
- repo target Python `>=3.12`
- บน Mac mini ใช้ `uv run python ...` ได้เรียบร้อย
- `rich` dependency ถูก fix แล้วใน `pyproject.toml`

---

## Baseline Machine

Machine used for local inference benchmark:
- Mac mini
- Apple M4
- 16 GB unified memory
- Ollama installed locally

Reason for choosing this machine:
- เป็น small host ที่ตั้งใจใช้เป็น baseline box สำหรับรัน local LLM benchmark แบบ serial

---

## Models Considered / Fit on This Machine

Checked candidates:
- `qwen3.5:9b`
- `gemma4:e2b`
- `gemma4:e4b`
- `scb10x/typhoon2.1-gemma3-4b:latest`

Empirical fit summary on the 16 GB Mac mini:
- Comfortable: Typhoon 4B, Gemma4 e2b, Qwen3.5 9B
- Runnable but closer to upper edge: Gemma4 e4b
- Not recommended to run in parallel on this box

Approx loaded model footprints observed:
- Typhoon 4B: ~3.5 GB
- Gemma4 e2b: ~7.9 GB
- Qwen3.5 9B: ~8.6 GB
- Gemma4 e4b: ~10.0 GB

---

## Benchmark Results — 2026-04-09

Run ID: `mini-thai-20260409`
Mode: `--no-think`
Split: `text_only_core`

### Final ranking

1. `gemma4:e4b`
- Overall: 54/93 = 58.1%
- Thai: 40/60 = 66.7%
- Math: 14/33 = 42.4%
- Parseable: 100%
- Avg latency: ~0.6s/question

2. `qwen3.5:9b`
- Overall: 49/93 = 52.7%
- Thai: 39/60 = 65.0%
- Math: 10/33 = 30.3%
- Parseable: 100%
- Avg latency: ~1.8s/question

3. `gemma4:e2b`
- Overall: 46/93 = 49.5%
- Thai: 39/60 = 65.0%
- Math: 7/33 = 21.2%
- Parseable: 100%
- Avg latency: ~0.4s/question

4. `scb10x/typhoon2.1-gemma3-4b:latest`
- Overall: 41/93 = 44.1%
- Thai: 31/60 = 51.7%
- Math: 10/33 = 30.3%
- Parseable: 100%
- Avg latency: ~0.5s/question

### Main takeaways

- Best overall on this benchmark and this machine: `gemma4:e4b`
- Best Thai cluster: `gemma4:e4b`, `qwen3.5:9b`, `gemma4:e2b` ค่อนข้างใกล้กัน
- Best Math among tested models: `gemma4:e4b`
- Fastest model: `gemma4:e2b`
- Best current baseline choice for the Mini: `gemma4:e4b`

---

## Repeat Trial Results — 10 Runs per Model

Batch ID: `mini-r10-20260409`
Summary file: `benchmark_responses/repeat_summary_mini-r10-20260409.json`

### Key finding

- ทุกโมเดลให้ **ผลคะแนนเหมือนเดิมทุก run** (range ไม่ขยับเลย)
- ภายใต้ setup นี้ (`temperature 0`, same machine, same prompts, same dataset) benchmark outcome ดู **deterministic** มาก
- noise ที่เห็นจริงคือ **runtime**, ไม่ใช่ accuracy

### 10-run averages

1. `gemma4:e4b`
- Mean accuracy: 54.0/93 = 58.06%
- Range: 54–54
- Thai: 66.67%
- Math: 42.42%
- Avg time: 52.6s/run
- Avg cost: 61.7 correct/min
- Avg size-aware efficiency: 6.2 correct/GB-min

2. `qwen3.5:9b`
- Mean accuracy: 49.0/93 = 52.69%
- Range: 49–49
- Thai: 65.00%
- Math: 30.30%
- Avg time: 170.5s/run
- Avg cost: 17.2 correct/min
- Avg size-aware efficiency: 2.0 correct/GB-min

3. `gemma4:e2b`
- Mean accuracy: 46.0/93 = 49.46%
- Range: 46–46
- Thai: 65.00%
- Math: 21.21%
- Avg time: 32.1s/run
- Avg cost: 86.1 correct/min
- Avg size-aware efficiency: 10.9 correct/GB-min

4. `scb10x/typhoon2.1-gemma3-4b:latest`
- Mean accuracy: 41.0/93 = 44.09%
- Range: 41–41
- Thai: 51.67%
- Math: 30.30%
- Avg time: 44.5s/run
- Avg cost: 55.3 correct/min
- Avg size-aware efficiency: 15.8 correct/GB-min

### Interpretation after repeats

- ถ้าต้องการ **best quality baseline** บน Mac mini: `gemma4:e4b`
- ถ้าต้องการ **best speed-efficiency**: `gemma4:e2b`
- ถ้าต้องการ **best size-aware efficiency**: Typhoon ชนะ เพราะเล็กและเร็ว แม้คะแนนต่ำกว่า
- `qwen3.5:9b` ดูไม่คุ้มเวลาบนเครื่องนี้เมื่อเทียบกับคะแนนที่ได้เพิ่ม

---

## Cost / Efficiency Framing for Dashboard

For local inference, "cost" should not be treated like API billing. Better framing:

Primary dimensions:
- Quality: accuracy / score
- Cost: wall-clock time + memory footprint
- Efficiency: quality per unit time (and optionally per unit memory)

Recommended main reportable rate:
- **Correct answers per minute**

Recommended secondary technical efficiency rate:
- **Correct answers per GB-minute**

Reason:
- output token count is tiny in this benchmark
- real cost is dominated more by machine time and prompt/prefill cost than by generated tokens

Observed rates from current run set:
- Typhoon: 54.7 correct/min
- Qwen3.5 9B: 17.5 correct/min
- Gemma4 e2b: 78.0 correct/min
- Gemma4 e4b: 57.3 correct/min

Approx size-aware rate:
- Typhoon: 15.6 correct/min/GB
- Gemma4 e2b: 9.9 correct/min/GB
- Gemma4 e4b: 5.7 correct/min/GB
- Qwen3.5 9B: 2.0 correct/min/GB

---

## Next Likely Work

- Design automation for serial benchmark runs on the Mac mini
- Define dashboard metrics and artifact schema
- Add run manifests / machine metadata / peak memory capture
- Possibly inspect wrong-answer patterns by model
- Tightened V1 dashboard PRD drafted at `panya-manee-v1-dashboard-prd.md`, with snapshot-first publication model, balanced quality score ranking, static JSON bundle contract, and open implementation questions
- Possibly run repeat trials for stability

---

## Notes to resume later

- User is designing a PRD for the dashboard and plans to share it later
- Automation should fit the current repo shape, not replace it
- This topic should continue as the canonical memory for the benchmark work
- User reaffirmed `จำไว้` after the 10-run repeat batch, so this topic remains active long-term memory
