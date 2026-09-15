"""FastAPI service — mirrors Hyper-RAG service_api.py. Run: uvicorn service_api:app --port 8000"""
from __future__ import annotations
import os, sys, time, logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import asyncio
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("ssrag_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

DATA_NAME = os.getenv("SSRAG_DATA_NAME", "mock").strip()
MODE = os.getenv("SSRAG_MODE", "hyper").strip()
MAX_QPS = float(os.getenv("SSRAG_MAX_QPS", "3") or "3")
API_KEY = os.getenv("SSRAG_API_KEY", "").strip()

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent
WORKING_DIR = ROOT / "caches" / DATA_NAME
sys.path.insert(0, str(ROOT))

from ssrag import SSRAG, QueryParam
from ssrag.utils import EmbeddingFunc
from ssrag.llm import openai_complete_stream_if_cache, openai_complete_if_cache, openai_embedding
from reproduce.Step_3_response_question import llm_model_func, embedding_func
try:
    from my_config import EMB_DIM, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
except Exception:
    EMB_DIM, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL = 384, "xxx", "", "gpt-4o-mini"

app = FastAPI(title="SS-RAG API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

rag: Optional[SSRAG] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=6000)
    mode: Optional[str] = Field(default=None, description="hyper / hyper-lite / graph / naive / llm")


class QueryResponse(BaseModel):
    answer: str
    mode: str
    latency_ms: int


_rate: Dict[str, Tuple[float, int]] = {}

def _rate_limit(ip: str):
    if MAX_QPS <= 0:
        return
    now = time.time()
    ws, c = _rate.get(ip, (now, 0))
    if now - ws >= 1.0:
        _rate[ip] = (now, 1)
        return
    if c >= int(MAX_QPS):
        raise HTTPException(429, "طلبات كثيرة — حاول لاحقاً (rate limited)")
    _rate[ip] = (ws, c + 1)


@app.on_event("startup")
async def _startup():
    global rag
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    async def stream_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        async for tok in openai_complete_stream_if_cache(
            prompt, system_prompt, history_messages,
            model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, **kwargs):
            yield tok

    rag = SSRAG(working_dir=str(WORKING_DIR), llm_model_func=llm_model_func,
                llm_model_stream_func=stream_func,
                embedding_func=EmbeddingFunc(embedding_dim=int(EMB_DIM), max_token_size=8192, func=embedding_func))
    logger.info(f"SS-RAG ready dir={WORKING_DIR} mode={MODE}")


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "app": "SS-RAG", "data": DATA_NAME, "mode": MODE}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "X-API-Key غير صالح")
    _rate_limit(request.client.host if request.client else "unknown")
    mode = (req.mode or MODE).strip()
    if mode not in {"hyper", "hyper-lite", "graph", "naive", "llm"}:
        raise HTTPException(400, "mode must be hyper/hyper-lite/graph/naive/llm")
    t0 = time.time()
    answer = await rag.aquery(req.question, param=QueryParam(mode=mode))
    return QueryResponse(answer=answer, mode=mode, latency_ms=int((time.time() - t0) * 1000))


@app.post("/query_stream")
async def query_stream(req: QueryRequest, request: Request):
    _rate_limit(request.client.host if request.client else "unknown")
    mode = (req.mode or MODE).strip()
    qp = QueryParam(mode=mode)

    async def gen():
        async for tok in rag.astream_query(req.question, param=qp):
            if tok:
                yield tok
            await asyncio.sleep(0)
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
