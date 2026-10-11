# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVVectorStore`: a `llama_index.core.vector_stores.types.BasePydanticVectorStore`
backed by `VSIM.*`, indexing each source document's nodes in a plain HotKV
set so `delete()` can find every node belonging to one document."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import build_metadata_filter_fn, metadata_dict_to_node, node_to_metadata_dict

from ...exceptions import ResponseError

__all__ = ["HotKVVectorStore"]



def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


_OVERFETCH_MIN = 20
_OVERFETCH_FACTOR = 4


class HotKVVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store on top of HotKV's `VSIM.*` similarity index.

    ```python
    from hotkv.integrations.llama_index import HotKVVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    vector_store = HotKVVectorStore(client, index_name="docs", dims=1536)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=StorageContext.from_defaults(vector_store=vector_store)
    )
    ```

    Each node's content and metadata are stored as one JSON blob in a
    plain HotKV string key (via `llama_index.core.vector_stores.utils`'s
    standard `node_to_metadata_dict()`/`metadata_dict_to_node()`, the same
    encoding most official LlamaIndex vector store integrations use), and
    its embedding is stored in the `VSIM.*` index named `index_name`. The ids
    of the nodes of one source document (`ref_doc_id`) are kept in an
    ordinary set, so `delete(ref_doc_id)` finds every node belonging to it
    without a separate id index; a set is a normal key, so it is replicated,
    resharded and failed over with the rest on a cluster (the `TAG.*` index
    this store used before is node-local state that a reshard does not move,
    and `delete()` still reads it for nodes stored that way). `query()` applies
    `MetadataFilters` (`==`/`!=`/`>`/`>=`/`<`/`<=`) client side, after
    `VSIM.SEARCH`, over-fetching candidates so a filtered query can still
    return `similarity_top_k` results.
    """

    stores_text: bool = True
    index_name: str = "llamaindex"
    metric: str = "cosine"
    dims: Optional[int] = None

    _client: Any = PrivateAttr()
    _index_ready: bool = PrivateAttr(default=False)

    def __init__(
        self,
        client: Any,
        index_name: str = "llamaindex",
        *,
        dims: Optional[int] = None,
        metric: str = "cosine",
        **kwargs: Any,
    ) -> None:
        super().__init__(index_name=index_name, metric=metric, dims=dims, **kwargs)
        self._client = client
        self._index_ready = False
        if dims is not None:
            self._ensure_index(dims)

    @property
    def client(self) -> Any:
        return self._client

    def _node_key(self, node_id: str) -> str:
        return f"{self.index_name}:node:{node_id}"

    @staticmethod
    def _ref_tag(ref_doc_id: str) -> str:
        return f"hotkv-li-ref:{ref_doc_id}"

    def _ref_key(self, ref_doc_id: str) -> str:
        return f"{self.index_name}:ref:{ref_doc_id}"

    def _ensure_index(self, dims: int) -> None:
        if self._index_ready:
            return
        try:
            self._client.vsim_createindex(self.index_name, dims, metric=self.metric)
        except ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise
        self._index_ready = True

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        for node in nodes:
            if node.embedding is not None:
                self._ensure_index(len(node.embedding))
                break
        ids: List[str] = []
        for node in nodes:
            node_id = node.node_id
            key = self._node_key(node_id)
            self._client.set(key, json.dumps(node_to_metadata_dict(node)))
            if node.embedding is not None:
                self._client.vsim_set(self.index_name, node_id, node.embedding)
            if node.ref_doc_id:
                self._client.sadd(self._ref_key(node.ref_doc_id), node_id)
            ids.append(node_id)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        ref_key = self._ref_key(ref_doc_id)
        node_ids = {_text(member) for member in self._client.smembers(ref_key)}
        # Nodes added before the ref sets were kept carry a TAG instead.
        try:
            for raw_key in self._client.tag_members(self._ref_tag(ref_doc_id)):
                node_ids.add(_text(raw_key).rsplit(":", 1)[-1])
        except ResponseError:
            pass
        for node_id in node_ids:
            self._client.vsim_del(self.index_name, node_id)
            self._client.delete(self._node_key(node_id))
        self._client.delete(ref_key)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            raise ValueError("HotKVVectorStore.query() requires query.query_embedding")
        top_k = query.similarity_top_k or 1
        has_filters = bool(query.filters and query.filters.filters)
        fetch_k = max(top_k * _OVERFETCH_FACTOR, top_k + _OVERFETCH_MIN) if has_filters else top_k

        raw_hits = self._client.vsim_search(self.index_name, fetch_k, query.query_embedding)
        candidate_ids: List[str] = []
        distances: dict = {}
        node_cache: dict = {}
        for raw_id, distance in raw_hits:
            node_id = raw_id.decode("utf-8") if isinstance(raw_id, (bytes, bytearray)) else raw_id
            raw_meta = self._client.get(self._node_key(node_id))
            if raw_meta is None:
                continue
            node_cache[node_id] = metadata_dict_to_node(json.loads(raw_meta))
            distances[node_id] = distance
            candidate_ids.append(node_id)

        filter_fn = build_metadata_filter_fn(
            lambda node_id: node_cache[node_id].metadata, query.filters
        )

        nodes: List[BaseNode] = []
        similarities: List[float] = []
        ids: List[str] = []
        for node_id in candidate_ids:
            if not filter_fn(node_id):
                continue
            nodes.append(node_cache[node_id])
            similarities.append(distances[node_id])
            ids.append(node_id)
            if len(nodes) >= top_k:
                break

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
