"""Step_3: answer questions with SS-RAG in all modes (== Hyper-RAG Step_3)."""
import sys, json, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from ssrag import SSRAG, QueryParam
from ssrag.utils import EmbeddingFunc
from ssrag.llm import openai_embedding, openai_complete_if_cache

DATA = sys.argv[1] if len(sys.argv) > 1 else "mock"
MODE = sys.argv[2] if len(sys.argv) > 2 else "hyper"

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(prompt, system_prompt, history_messages, **kwargs)

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(texts)

# shared funcs so service_api can import them (like Hyper-RAG)
async def main():
    try:
        import my_config; dim = int(getattr(my_config, "EMB_DIM", 384))
    except Exception:
        dim = 384
    rag = SSRAG(working_dir=str(Path("caches") / DATA),
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=embedding_func))
    qfile = Path("caches") / DATA / "questions.json"
    qs = json.loads(qfile.read_text(encoding="utf-8")) if qfile.exists() else [{"question": "ما هي أهم موضوعات هذه القصة؟"}]
    outdir = Path("caches") / DATA / "response" / MODE
    outdir.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(qs):
        ans = await rag.aquery(q["question"], param=QueryParam(mode=MODE))
        (outdir / f"{i}.txt").write_text(ans, encoding="utf-8")
        print(f"[{MODE}] Q{i}: {q['question'][:60]} -> {(ans[:120]).replace(chr(10),' ')}...")
    print("[Step_3] done.")

if __name__ == "__main__":
    asyncio.run(main())
