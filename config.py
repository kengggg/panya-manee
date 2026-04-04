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


def fix_bom_keys(item: dict) -> dict:
    """Strip UTF-8 BOM from JSON keys (common in Excel-exported files)."""
    return {k.lstrip("\ufeff"): v for k, v in item.items()}
