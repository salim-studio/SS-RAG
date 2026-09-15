"""Step_1: build hypergraph index (== Hyper-RAG Step_1)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from ssrag import SSRAG
from ssrag.utils import EmbeddingFunc
from ssrag.llm import openai_embedding, openai_complete_if_cache

DATA = sys.argv[1] if len(sys.argv) > 1 else "mock"

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(prompt, system_prompt, history_messages, **kwargs)

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(texts)

def main():
    try:
        import my_config; dim = int(getattr(my_config, "EMB_DIM", 384))
    except Exception:
        dim = 384
    rag = SSRAG(working_dir=str(Path("caches") / DATA),
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=embedding_func))
    texts = [(Path("datasets") / DATA / f).read_text(encoding="utf-8") for f in __import__("os").listdir(Path("datasets") / DATA) if f.endswith(".txt")]
    rag.insert(texts)
    print("[Step_1] stats:", rag.stats())

if __name__ == "__main__":
    main()
