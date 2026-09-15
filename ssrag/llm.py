"""LLM + embedding backends: OpenAI-compatible when keys exist, offline fallback otherwise."""
from __future__ import annotations
import os
import numpy as np

from .utils import offline_embed_batch, compute_args_hash, logger

# Optional openai import
try:
    from openai import AsyncOpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


def _cfg(key: str, default: str = "") -> str:
    # env wins, then my_config.py if present
    v = os.getenv(key, "")
    if v:
        return v
    try:
        import my_config  # type: ignore

        return str(getattr(my_config, key, default) or default)
    except Exception:
        return default


def llm_configured() -> bool:
    return bool(_cfg("LLM_API_KEY")) and _cfg("LLM_API_KEY") != "xxx"


def emb_configured() -> bool:
    return bool(_cfg("EMB_API_KEY")) and _cfg("EMB_API_KEY") != "xxx"


# ---------- chat completions ----------

async def openai_complete_if_cache(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    hashing_kv=None,
    model: str | None = None,
    **kwargs,
) -> str:
    model = model or _cfg("LLM_MODEL", "gpt-4o-mini")
    base_url = _cfg("LLM_BASE_URL")
    api_key = _cfg("LLM_API_KEY")
    if hashing_kv is not None:
        key = compute_args_hash(model, prompt, system_prompt)
        hit = await hashing_kv.get_by_id(key)
        if hit and isinstance(hit, dict) and hit.get("content"):
            return hit["content"]
    text = await _chat(prompt, system_prompt, history_messages or [], model, base_url, api_key, **kwargs)
    if hashing_kv is not None:
        try:
            await hashing_kv.upsert({key: {"content": text}})
        except Exception:
            pass
    return text


async def _chat(prompt, system_prompt, history, model, base_url, api_key, **kwargs) -> str:
    if not api_key or api_key == "xxx" or not _HAS_OPENAI:
        # offline fallback: extractive echo so demo works without keys
        snippet = (prompt or "")[:1200]
        return (
            "[SS-RAG offline mode — no LLM key configured]\n"
            "Based on retrieved context, key points:\n" + snippet
        )
    client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += history or []
    msgs.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(model=model, messages=msgs, **kwargs)
    return resp.choices[0].message.content or ""


async def openai_complete_stream_if_cache(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    hashing_kv=None,
    model: str | None = None,
    **kwargs,
):
    """Async generator yielding tokens."""
    model = model or _cfg("LLM_MODEL", "gpt-4o-mini")
    base_url = _cfg("LLM_BASE_URL")
    api_key = _cfg("LLM_API_KEY")
    if not api_key or api_key == "xxx" or not _HAS_OPENAI:
        full = await _chat(prompt, system_prompt, history_messages or [], model, base_url, api_key)
        for i in range(0, len(full), 40):
            yield full[i : i + 40]
        return
    client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += history_messages or []
    msgs.append({"role": "user", "content": prompt})
    stream = await client.chat.completions.create(model=model, messages=msgs, stream=True, **kwargs)
    async for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:
            delta = ""
        if delta:
            yield delta


# ---------- embeddings ----------

async def openai_embedding(texts: list[str], model: str | None = None, **kwargs) -> np.ndarray:
    model = model or _cfg("EMB_MODEL", "text-embedding-3-small")
    dim = int(_cfg("EMB_DIM", "384") or 384)
    api_key = _cfg("EMB_API_KEY")
    base_url = _cfg("EMB_BASE_URL")
    if not api_key or api_key == "xxx" or not _HAS_OPENAI:
        return offline_embed_batch(texts, dim=dim)
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        resp = await client.embeddings.create(model=model, input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return arr
    except Exception as e:
        logger.warning(f"embedding API failed ({e}), using offline embeddings")
        return offline_embed_batch(texts, dim=dim)
