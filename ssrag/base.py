"""Base dataclasses & storage interfaces for SS-RAG (mirrors Hyper-RAG design)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Literal, Optional, Set, Tuple, TypeVar, Union

from .utils import EmbeddingFunc

T = TypeVar("T")


@dataclass
class QueryParam:
    """Query behaviour switch — same idea as Hyper-RAG."""

    mode: Literal["hyper", "hyper-lite", "graph", "naive", "llm"] = "hyper"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    top_k: int = 8
    max_token_for_text_unit: int = 1600
    max_token_for_entity_context: int = 600
    max_token_for_relation_context: int = 1600
    return_type: Literal["json", "text"] = "text"


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(self, ids: list[str], fields: Union[set[str], None] = None) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseHypergraphStorage(StorageNameSpace):
    """Hypergraph = vertices (entities) + hyperedges (n-ary correlations)."""

    async def has_vertex(self, v_id: Any) -> bool:
        raise NotImplementedError

    async def has_hyperedge(self, e_id: str) -> bool:
        raise NotImplementedError

    async def get_vertex(self, v_id: str, default: Any = None):
        raise NotImplementedError

    async def get_hyperedge(self, e_id: str, default: Any = None):
        raise NotImplementedError

    async def get_all_vertices(self):
        raise NotImplementedError

    async def get_all_hyperedges(self):
        raise NotImplementedError

    async def upsert_vertex(self, v_id: Any, v_data: Optional[Dict] = None):
        raise NotImplementedError

    async def upsert_hyperedge(self, e_id: str, nodes: List[str], e_data: Optional[Dict] = None):
        raise NotImplementedError

    async def get_nbr_e_of_vertex(self, v_id: Any) -> list:
        raise NotImplementedError

    async def get_nbr_v_of_hyperedge(self, e_id: str) -> list:
        raise NotImplementedError
