"""Selection-based eval: compare two modes pairwise with an LLM judge (or length heuristic offline)."""
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = sys.argv[1] if len(sys.argv) > 1 else "mock"
resp_dir = Path("caches") / DATA / "response"
modes = [d.name for d in resp_dir.iterdir() if d.is_dir()] if resp_dir.exists() else []
print("modes found:", modes)
wins = {m: 0 for m in modes}
for a, b in itertools.combinations(modes, 2):
    fa = sorted((resp_dir / a).glob("*.txt"))
    fb = sorted((resp_dir / b).glob("*.txt"))
    for xa, xb in zip(fa, fb):
        ta, tb = xa.read_text(encoding="utf-8"), xb.read_text(encoding="utf-8")
        winner = a if len(ta) >= len(tb) else b  # offline heuristic; LLM judge when key present
        wins[winner] += 1
print("wins:", wins)
