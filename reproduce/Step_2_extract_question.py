"""Step_2: extract questions (rule-based demo; with LLM key uses LLM to gen questions)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = sys.argv[1] if len(sys.argv) > 1 else "mock"
out = Path("caches") / DATA / "questions.json"
qs = [
    {"question": "ما هي أهم موضوعات هذه القصة؟", "ref": "الحكمة، الطب التقليدي، التعاون لبناء العيادة"},
    {"question": "من هم الأشخاص الثلاثة الذين بنوا العيادة؟", "ref": "سليم والحكيم، كريم الحداد، ليلى المعلمة"},
    {"question": "What are the top themes in this story?", "ref": "wisdom, traditional medicine, cooperation"},
]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[Step_2] wrote {len(qs)} questions -> {out}")
