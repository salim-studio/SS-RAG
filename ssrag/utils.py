"""Utilities: hashing, chunking, token counting, local embeddings."""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import wraps
from typing import Any, List

import numpy as np

logger = logging.getLogger("ssrag")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def set_logger(log_file: str):
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(fh)
    except Exception:
        pass


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


def compute_args_hash(*args) -> str:
    return hashlib.md5(str(args).encode("utf-8")).hexdigest()


def limit_async_func_call(max_size: int, waiting_time: float = 0.0001):
    def final_decro(func):
        cur = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal cur
            while cur >= max_size:
                await asyncio.sleep(waiting_time)
            cur += 1
            try:
                return await func(*args, **kwargs)
            finally:
                cur -= 1

        return wait_func

    return final_decro


# ---------- token / chunk helpers ----------

def num_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        import tiktoken  # optional
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def chunking_by_token_size(
    content: str,
    overlap_token_size: int = 100,
    max_token_size: int = 1200,
    tiktoken_model: str = "gpt-4o-mini",
) -> List[dict]:
    """Greedy sentence-aware splitter, fallback-safe without tiktoken."""
    sentences = re.split(r"(?<=[.!?؟۔。\n])\s+", content.strip())
    chunks, cur, cur_tok = [], [], 0
    order = 0
    for s in sentences:
        t = num_tokens(s, tiktoken_model)
        if cur_tok + t > max_token_size and cur:
            text = " ".join(cur)
            chunks.append({"content": text, "tokens": cur_tok, "chunk_order_index": order})
            order += 1
            # overlap: keep tail
            if overlap_token_size > 0:
                tail, tail_tok = [], 0
                for x in reversed(cur):
                    xt = num_tokens(x, tiktoken_model)
                    if tail_tok + xt > overlap_token_size:
                        break
                    tail.insert(0, x)
                    tail_tok += xt
                cur, cur_tok = tail, tail_tok
            else:
                cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += t
    if cur:
        chunks.append({"content": " ".join(cur), "tokens": cur_tok, "chunk_order_index": order})
    return chunks


def truncate_list_by_token_size(data: list, key, max_token_size: int) -> list:
    out, toks = [], 0
    for d in data:
        t = num_tokens(key(d))
        if toks + t > max_token_size:
            break
        out.append(d)
        toks += t
    return out


def clean_str(v: Any) -> str:
    if not isinstance(v, str):
        return v
    s = v.strip()
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", s)


# ---------- offline (no-key) deterministic embedding ----------

def offline_embed_batch(texts: List[str], dim: int = 384) -> np.ndarray:
    """Hashing-based char-trigram embedding, L2-normalised. No network needed."""
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = t.lower()
        trigrams = re.findall(r".{1,3}", t) + re.findall(r"\w+", t)
        if not trigrams:
            continue
        for tri in trigrams:
            h = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16) % dim
            vecs[i, h] += 1.0
        n = np.linalg.norm(vecs[i])
        if n > 0:
            vecs[i] /= n
    return vecs


def cosine_top_k(query_vec: np.ndarray, mat: np.ndarray, k: int) -> List[int]:
    if mat.shape[0] == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    sims = (mat / norms) @ q
    idx = np.argsort(-sims)[:k]
    return idx.tolist()


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
