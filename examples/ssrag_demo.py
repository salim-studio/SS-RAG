"""Demo — mirrors Hyper-RAG's hyperrag_demo.py but for SS-RAG. Works OFFLINE (no API key)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np

from ssrag import SSRAG, QueryParam
from ssrag.utils import EmbeddingFunc
from ssrag.llm import openai_embedding, openai_complete_if_cache, openai_complete_stream_if_cache


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(prompt, system_prompt, history_messages, **kwargs)


async def llm_model_stream_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    async for tok in openai_complete_stream_if_cache(prompt, system_prompt, history_messages, **kwargs):
        yield tok


async def embedding_func(texts: list[str]) -> np.ndarray:
    try:
        import my_config
        dim = int(getattr(my_config, "EMB_DIM", 384))
    except Exception:
        dim = 384
    return await openai_embedding(texts)


def insert_with_retry(rag, texts, retries=3, delay=3):
    for i in range(retries):
        try:
            rag.insert(texts)
            return
        except Exception as e:
            print(f"insert failed ({e}), retry {i+1}/{retries}...")
            time.sleep(delay)
    raise RuntimeError("insert failed after retries")


if __name__ == "__main__":
    WORKING_DIR = Path("caches") / "mock"
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import my_config
        emb_dim = int(getattr(my_config, "EMB_DIM", 384))
    except Exception:
        emb_dim = 384

    rag = SSRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        llm_model_stream_func=llm_model_stream_func,
        embedding_func=EmbeddingFunc(embedding_dim=emb_dim, max_token_size=8192, func=embedding_func),
    )

    text = (Path(__file__).parent / "mock_data.txt").read_text(encoding="utf-8")
    insert_with_retry(rag, text)
    print("STATS:", rag.stats())

    for mode in ["naive", "hyper-lite", "hyper", "llm"]:
        try:
            print(f"\n===== mode={mode} =====")
            print(rag.query("ما هي أهم موضوعات هذه القصة؟ ومن بنى العيادة؟", param=QueryParam(mode=mode)))
        except Exception as e:
            print(f"[{mode}] error: {e}")
