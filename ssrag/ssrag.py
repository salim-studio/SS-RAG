"""SSRAG main class — hypergraph-driven RAG (SS-RAG)."""
from __future__ import annotations
import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

from .operate import (
    chunking_by_token_size, extract_entities,
    hyper_query, hyper_query_lite, graph_query, naive_query, llm_query,
    hyper_query_stream, hyper_query_lite_stream, naive_query_stream, llm_query_stream,
)
from .llm import openai_embedding, openai_complete_if_cache
from .storage import JsonKVStorage, NumpyVectorStorage, HypergraphStorage
from .utils import EmbeddingFunc, compute_mdhash_id, limit_async_func_call, logger, set_logger
from .base import BaseKVStorage, BaseVectorStorage, BaseHypergraphStorage, StorageNameSpace, QueryParam


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


@dataclass
class SSRAG:
    working_dir: str = field(default_factory=lambda: f"./SS-RAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")

    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 8
    embedding_func_max_async: int = 16

    llm_model_func: callable = openai_complete_if_cache
    llm_model_name: str = ""
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)
    llm_model_stream_func: callable = None
    use_llm_extract: bool = True

    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NumpyVectorStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    hypergraph_storage_cls: Type[BaseHypergraphStorage] = HypergraphStorage
    enable_llm_cache: bool = True

    def __post_init__(self):
        set_logger(os.path.join(self.working_dir, "SSRAG.log"))
        os.makedirs(self.working_dir, exist_ok=True)
        logger.info(f"SS-RAG working dir: {self.working_dir}")

        self.full_docs = self.key_string_value_json_storage_cls(namespace="full_docs", global_config=asdict(self))
        self.text_chunks = self.key_string_value_json_storage_cls(namespace="text_chunks", global_config=asdict(self))
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(namespace="llm_response_cache", global_config=asdict(self))
            if self.enable_llm_cache else None
        )
        self.chunk_entity_relation_hypergraph = self.hypergraph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.embedding_func)
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities", global_config=asdict(self),
            embedding_func=self.embedding_func, meta_fields={"entity_name"})
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships", global_config=asdict(self),
            embedding_func=self.embedding_func, meta_fields={"id_set"})
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks", global_config=asdict(self), embedding_func=self.embedding_func)
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(self.llm_model_func, hashing_kv=self.llm_response_cache, **self.llm_model_kwargs))

    # ---------- insert ----------
    def insert(self, string_or_strings):
        return always_get_an_event_loop().run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            new_docs = {compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()} for c in string_or_strings}
            fresh = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in fresh}
            if not new_docs:
                logger.warning("All docs already indexed")
                return
            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                for dp in chunking_by_token_size(doc["content"], self.chunk_overlap_token_size,
                                                self.chunk_token_size, self.tiktoken_model_name):
                    cid = compute_mdhash_id(dp["content"], prefix="chunk-")
                    inserting_chunks[cid] = {**dp, "full_doc_id": doc_key}
            fresh_c = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in fresh_c}
            if not inserting_chunks:
                logger.warning("All chunks already indexed")
                return
            logger.info(f"[SS-RAG] inserting {len(inserting_chunks)} chunks")
            await self.chunks_vdb.upsert(inserting_chunks)
            maybe = await extract_entities(inserting_chunks,
                                           hypergraph_inst=self.chunk_entity_relation_hypergraph,
                                           entity_vdb=self.entities_vdb, relationships_vdb=self.relationships_vdb,
                                           global_config={**asdict(self), "llm_model_func": self.llm_model_func,
                                                          "use_llm_extract": self.use_llm_extract})
            if maybe is None:
                logger.warning("No entities found")
                return
            self.chunk_entity_relation_hypergraph = maybe
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_done(self):
        for s in [self.full_docs, self.text_chunks, self.llm_response_cache,
                  self.entities_vdb, self.relationships_vdb, self.chunks_vdb,
                  self.chunk_entity_relation_hypergraph]:
            if s is None:
                continue
            await cast(StorageNameSpace, s).index_done_callback()

    # ---------- query ----------
    def query(self, query: str, param: QueryParam = QueryParam()):
        return always_get_an_event_loop().run_until_complete(self.aquery(query, param))

    def _cfg(self):
        cfg = asdict(self)
        cfg["llm_model_func"] = self.llm_model_func
        cfg["_chunks_vdb"] = self.chunks_vdb
        if self.llm_model_stream_func is not None:
            cfg["llm_model_stream_func"] = self.llm_model_stream_func
        return cfg

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        cfg = self._cfg()
        if param.mode == "hyper":
            r = await hyper_query(query, self.chunk_entity_relation_hypergraph, self.entities_vdb,
                                  self.relationships_vdb, self.text_chunks, param, cfg)
        elif param.mode == "hyper-lite":
            r = await hyper_query_lite(query, self.chunk_entity_relation_hypergraph, self.entities_vdb,
                                       self.text_chunks, param, cfg)
        elif param.mode == "graph":
            r = await graph_query(query, self.chunk_entity_relation_hypergraph, self.entities_vdb,
                                  self.relationships_vdb, self.text_chunks, param, cfg)
        elif param.mode == "naive":
            r = await naive_query(query, self.chunks_vdb, self.text_chunks, param, cfg)
        elif param.mode == "llm":
            r = await llm_query(query, param, cfg)
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return r

    async def astream_query(self, query: str, param: QueryParam = QueryParam()):
        if self.llm_model_stream_func is None:
            # graceful fallback: yield full answer at once
            yield await self.aquery(query, param)
            return
        cfg = self._cfg()
        if param.mode == "hyper":
            async for t in hyper_query_stream(query, self.chunk_entity_relation_hypergraph, self.entities_vdb,
                                              self.relationships_vdb, self.text_chunks, param, cfg):
                yield t
        elif param.mode == "hyper-lite":
            async for t in hyper_query_lite_stream(query, self.chunk_entity_relation_hypergraph, self.entities_vdb,
                                                   self.text_chunks, param, cfg):
                yield t
        elif param.mode == "naive":
            async for t in naive_query_stream(query, self.chunks_vdb, self.text_chunks, param, cfg):
                yield t
        elif param.mode in ("llm", "graph"):
            async for t in llm_query_stream(query, param, cfg):
                yield t
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()

    async def _query_done(self):
        if self.llm_response_cache is not None:
            await cast(StorageNameSpace, self.llm_response_cache).query_done_callback()

    def stats(self) -> dict:
        return {
            "vertices": len(getattr(self.chunk_entity_relation_hypergraph, "vertices", {})),
            "hyperedges": len(getattr(self.chunk_entity_relation_hypergraph, "hyperedges", {})),
            "working_dir": self.working_dir,
        }
