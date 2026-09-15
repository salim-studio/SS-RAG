"""File-backed storage: KV (json), Vector (json+brute-force cosine), Hypergraph (json)."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseKVStorage, BaseVectorStorage, BaseHypergraphStorage
from .utils import EmbeddingFunc, load_json, write_json, cosine_top_k, compute_mdhash_id


class JsonKVStorage(BaseKVStorage):
    def __init__(self, namespace: str, global_config: dict):
        super().__init__(namespace=namespace, global_config=global_config)
        self._file = os.path.join(global_config["working_dir"], f"kv_{namespace}.json")
        self._data: Dict[str, Any] = load_json(self._file) or {}
        self._dirty = False

    async def all_keys(self):
        return list(self._data.keys())

    async def get_by_id(self, id: str):
        return self._data.get(id)

    async def get_by_ids(self, ids: list[str], fields=None):
        out = []
        for i in ids:
            v = self._data.get(i)
            if v is not None and fields:
                v = {k: v[k] for k in fields if k in v} if isinstance(v, dict) else v
            out.append(v)
        return out

    async def filter_keys(self, data: list[str]) -> set[str]:
        return {k for k in data if k not in self._data}

    async def upsert(self, data: dict[str, Any]):
        self._data.update(data)
        self._dirty = True

    async def drop(self):
        self._data = {}
        self._dirty = True

    async def index_done_callback(self):
        if self._dirty:
            write_json(self._data, self._file)
            self._dirty = False

    async def query_done_callback(self):
        await self.index_done_callback()


class NumpyVectorStorage(BaseVectorStorage):
    """Simple persistent vector DB: stores {id: {content, embedding, meta}}."""

    def __init__(self, namespace: str, global_config: dict, embedding_func: EmbeddingFunc, meta_fields: set | None = None):
        super().__init__(namespace=namespace, global_config=global_config,
                         embedding_func=embedding_func, meta_fields=meta_fields or set())
        self._file = os.path.join(global_config["working_dir"], f"vdb_{namespace}.json")
        self._data: Dict[str, dict] = load_json(self._file) or {}
        self._dirty = False

    async def upsert(self, data: dict[str, dict]):
        texts, keys = [], []
        for k, v in data.items():
            if "embedding" not in v:
                texts.append(v.get("content", k))
                keys.append(k)
        if texts:
            embs = await self.embedding_func(texts)
            for k, e in zip(keys, embs):
                data[k]["embedding"] = e.tolist() if isinstance(e, np.ndarray) else list(e)
        self._data.update(data)
        self._dirty = True

    async def query(self, query: str, top_k: int) -> list[dict]:
        if not self._data:
            return []
        q = (await self.embedding_func([query]))[0]
        ids = list(self._data.keys())
        mat = np.array([self._data[i].get("embedding", [0.0]) for i in ids], dtype=np.float32)
        # fix ragged dims
        dim = q.shape[0]
        fixed = np.zeros((len(ids), dim), dtype=np.float32)
        for r, row in enumerate(mat):
            row = np.asarray(row, dtype=np.float32).ravel()[:dim]
            fixed[r, : row.shape[0]] = row
        idx = cosine_top_k(np.asarray(q, dtype=np.float32), fixed, top_k)
        return [{"id": ids[i], **self._data[ids[i]]} for i in idx]

    async def index_done_callback(self):
        if self._dirty:
            write_json(self._data, self._file)
            self._dirty = False

    async def query_done_callback(self):
        pass


class HypergraphStorage(BaseHypergraphStorage):
    """Native hypergraph store for SS-RAG.

    vertices: {v_id: {entity_name, entity_type, description, source_ids}}
    hyperedges: {e_id: {nodes: [...], order: 'low'|'high', keywords, description, source_ids}}
    """

    def __init__(self, namespace: str, global_config: dict):
        super().__init__(namespace=namespace, global_config=global_config)
        self._file = os.path.join(global_config["working_dir"], f"hypergraph_{namespace}.json")
        raw = load_json(self._file) or {}
        self.vertices: Dict[str, dict] = raw.get("vertices", {})
        self.hyperedges: Dict[str, dict] = raw.get("hyperedges", {})
        self._dirty = False

    # -- vertices --
    async def has_vertex(self, v_id) -> bool:
        return str(v_id) in self.vertices

    async def get_vertex(self, v_id: str, default=None):
        return self.vertices.get(str(v_id), default)

    async def get_all_vertices(self):
        return self.vertices

    async def upsert_vertex(self, v_id, v_data: Optional[Dict] = None):
        v_id = str(v_id)
        prev = self.vertices.get(v_id, {})
        merged = {**prev, **(v_data or {})}
        # merge source_ids
        s = set(prev.get("source_ids", [])) | set((v_data or {}).get("source_ids", []))
        merged["source_ids"] = sorted(s)
        self.vertices[v_id] = merged
        self._dirty = True

    # -- hyperedges --
    def _eid(self, nodes: List[str], kind: str) -> str:
        return compute_mdhash_id("|".join(sorted(str(n) for n in nodes)) + "#" + kind, prefix="he-")

    async def has_hyperedge(self, e_id: str) -> bool:
        return str(e_id) in self.hyperedges

    async def get_hyperedge(self, e_id: str, default=None):
        return self.hyperedges.get(str(e_id), default)

    async def get_all_hyperedges(self):
        return self.hyperedges

    async def upsert_hyperedge(self, e_id: str, nodes: List[str], e_data: Optional[Dict] = None):
        e_id = str(e_id)
        prev = self.hyperedges.get(e_id, {})
        merged = {**prev, **(e_data or {})}
        merged["nodes"] = [str(n) for n in nodes]
        s = set(prev.get("source_ids", [])) | set((e_data or {}).get("source_ids", []))
        merged["source_ids"] = sorted(s)
        self.hyperedges[e_id] = merged
        self._dirty = True

    async def get_nbr_e_of_vertex(self, v_id) -> list:
        v_id = str(v_id)
        return [eid for eid, e in self.hyperedges.items() if v_id in e.get("nodes", [])]

    async def get_nbr_v_of_hyperedge(self, e_id: str) -> list:
        e = self.hyperedges.get(str(e_id), {})
        return list(e.get("nodes", []))

    async def index_done_callback(self):
        if self._dirty:
            write_json({"vertices": self.vertices, "hyperedges": self.hyperedges}, self._file)
            self._dirty = False

    async def query_done_callback(self):
        pass
