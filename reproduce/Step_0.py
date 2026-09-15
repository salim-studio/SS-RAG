"""Step_0: prepare data — put raw .txt files in ./datasets/<data_name>/, chunks preview out."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ssrag.utils import chunking_by_token_size

DATA = sys.argv[1] if len(sys.argv) > 1 else "mock"
src = Path("datasets") / DATA
src.mkdir(parents=True, exist_ok=True)
# seed with mock if empty
if not any(src.glob("*.txt")):
    import shutil
    shutil.copy("examples/mock_data.txt", src / "doc1.txt")
    print(f"[Step_0] seeded {src/'doc1.txt'}")
for f in src.glob("*.txt"):
    text = f.read_text(encoding="utf-8")
    chunks = chunking_by_token_size(text)
    print(f"[Step_0] {f.name}: {len(text)} chars -> {len(chunks)} chunks")
print("[Step_0] done.")
