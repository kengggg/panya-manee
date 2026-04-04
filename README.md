# panya-manee

Local LLM arena that benchmarks small models on Thailand's NT (National Test) Grade 3 exam questions via Ollama. Tests Thai language comprehension and math reasoning.

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) running locally
- At least one model pulled (e.g. `ollama pull qwen3:0.6b`)

## Usage

### Run a benchmark

```bash
python main.py run --model qwen3:0.6b
python main.py run --model gemma3:1b --subjects thai --run-id run002
python main.py run --model llama3.2:1b --dry-run  # no API calls
```

### Summarize results

```bash
python main.py summarize                  # all runs
python main.py summarize --run-id run001  # specific run
```

## Test Data

- **Source**: NT (National Test) Grade 3, Year 2566 (2023)
- **Subjects**: Thai language (19 items) + Math (10 items) in Tier 1 (text_only_core)
- **Format**: Multiple choice, 4 options each
- **Scoring**: 3 points per correct answer

## How it works

1. Loads MCQ items from `nt-tests/` JSON files (Tier 1: text-only questions)
2. Prompts the model in Thai to answer with just a digit (1-4)
3. Parses model output and compares to ground truth
4. Saves detailed JSONL results to `benchmark_responses/`
5. Summarizes accuracy by model, subject, and skill tag
