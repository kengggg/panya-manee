"""Shared configuration for panya-manee benchmark."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "nt-tests"
RESPONSES_DIR = PROJECT_ROOT / "benchmark_responses"

OLLAMA_URL = "http://localhost:11434/api/chat"

DATASET_FILES = {
    "thai": DATA_DIR / "nt_p3_th_all.json",
    "math": DATA_DIR / "nt_p3_math_all.json",
}

PROMPT_TEMPLATE = """คุณกำลังทำข้อสอบระดับประถมศึกษาปีที่ 3 ของประเทศไทย
จงเลือกคำตอบที่ถูกที่สุด และตอบเพียงตัวเลขเดียวเท่านั้น (1, 2, 3 หรือ 4)
ห้ามอธิบายหรือเพิ่มข้อความอื่น

{stimulus}คำถาม: {question}
1) {c1}
2) {c2}
3) {c3}
4) {c4}

คำตอบ:"""


REQUIRED_KEYS = {"exam_id", "question_id", "prompt_text", "correct_answer", "eval_split"}


def validate_items(items: list[dict], source_path: Path) -> None:
    """Sanity-check loaded items. Raises ValueError on dirty data."""
    for idx, item in enumerate(items):
        bom_keys = [k for k in item if "\ufeff" in k]
        if bom_keys:
            raise ValueError(
                f"{source_path}: item {idx} has BOM in keys: {bom_keys}. "
                f"Clean the source file (remove UTF-8 BOM)."
            )
        missing = REQUIRED_KEYS - item.keys()
        if missing:
            raise ValueError(
                f"{source_path}: item {idx} missing required keys: {missing}"
            )
