"""Scoring-based eval: score each answer 1-5 vs reference (offline heuristic)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DATA = sys.argv[1] if len(sys.argv) > 1 else "mock"
MODE = sys.argv[2] if len(sys.argv) > 2 else "hyper"
qfile = Path("caches") / DATA / "questions.json"
resp = Path("caches") / DATA / "response" / MODE
qs = json.loads(qfile.read_text(encoding="utf-8")) if qfile.exists() else []
for i, q in enumerate(qs):
    f = resp / f"{i}.txt"
    ans = f.read_text(encoding="utf-8") if f.exists() else ""
    ref = q.get("ref", "")
    overlap = len(set(ans.split()) & set(ref.split()))
    score = min(5, 1 + overlap)
    print(f"Q{i} score={score}/5 overlap={overlap} | {q['question'][:50]}")
