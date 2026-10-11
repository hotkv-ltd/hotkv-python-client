# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVVectorStore`: a `langchain_core.vectorstores.VectorStore` backed by
`VSIM.*`, storing text and metadata alongside each vector in a plain
HotKV string key."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ...exceptions import ResponseError

__all__ = ["HotKVVectorStore"]


class HotKVVectorStore(VectorStore):
    """LangChain vector store on top of HotKV's `VSIM.*` similarity index.

    ```python
    from hotkv.integrations.langchain import HotKVVectorStore

    store = HotKVVectorStore(client, embedding, index_name="docs")
    ids = store.add_texts(["HotKV is a RESP compatible store.", "Cats are mammals."])
    store.similarity_search("what is HotKV?", k=1)
    ```

    Each text is stored twice: its embedding in the `VSIM.*` index named
    `index_name`, and its content plus metadata as a JSON blob in a plain
    HotKV string key (`add_texts()`/`similarity_search()` handle both
    sides of that together, so callers only ever see LangChain
    `Document` objects). The index's vector dimensionality is taken from
    `dims` if given, otherwise from the first embedding `add_texts()`
    computes.
    """

    def __init__(
        self,
        client: Any,
        embedding: Embeddings,
        index_name: str = "langchain",
        *,
        dims: Optional[int] = None,
        metric: str = "cosine",
    ) -> None:
        self._client = client
        self._embedding = embedding
        self.index_name = index_name
        self.metric = metric
        self._index_ready = False
        if dims is not None:
            self._ensure_index(dims)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def _doc_key(self, doc_id: str) -> str:
        return f"{self.index_name}:doc:{doc_id}"

    def _ensure_index(self, dims: int) -> None:
        if self._index_ready:
            return
        try:
            self._client.vsim_createindex(self.index_name, dims, metric=self.metric)
        except ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise
        self._index_ready = True

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed `texts` and store them (with `metadatas`, if given), returning
        the ids they were stored under (`ids`, or newly generated `uuid4`s)."""
        texts = list(texts)
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in texts]
        if not texts:
            return []
        vectors = self._embedding.embed_documents(texts)
        self._ensure_index(len(vectors[0]))
        for doc_id, text, metadata, vector in zip(ids, texts, metadatas, vectors):
            self._client.set(self._doc_key(doc_id), json.dumps({"text": text, "metadata": metadata}))
            self._client.vsim_set(self.index_name, doc_id, vector)
        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return [doc for doc, _score in self.similarity_search_with_score(query, k=k, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Like `similarity_search()`, also returning each hit's `VSIM.SEARCH`
        distance (smaller is closer; the exact scale depends on `metric`)."""
        vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(vector, k=k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return [doc for doc, _score in self.similarity_search_by_vector_with_score(embedding, k=k, **kwargs)]

    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        results = self._client.vsim_search(self.index_name, k, embedding)
        out: List[Tuple[Document, float]] = []
        for raw_id, distance in results:
            doc_id = raw_id.decode("utf-8") if isinstance(raw_id, (bytes, bytearray)) else raw_id
            raw = self._client.get(self._doc_key(doc_id))
            if raw is None:
                continue
            payload = json.loads(raw)
            doc = Document(id=doc_id, page_content=payload["text"], metadata=payload.get("metadata") or {})
            out.append((doc, distance))
        return out

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by id. `ids=None` (delete everything) is not supported:
        `VSIM.*` has no "list every key in an index" command, so there is
        nothing to enumerate from; pass the ids you stored instead."""
        if not ids:
            return None
        removed = False
        for doc_id in ids:
            self._client.vsim_del(self.index_name, doc_id)
            if self._client.delete(self._doc_key(doc_id)):
                removed = True
        return removed

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        ids: Optional[List[str]] = None,
        client: Any = None,
        index_name: str = "langchain",
        metric: str = "cosine",
        **kwargs: Any,
    ) -> "HotKVVectorStore":
        """Build a store and add `texts` to it in one call. `client` (a
        `hotkv.HotKV` instance) is required: unlike an embedded vector
        store, HotKV needs a server to connect to."""
        if client is None:
            raise ValueError(
                "HotKVVectorStore.from_texts() requires a `client=` keyword "
                "argument (a hotkv.HotKV instance)"
            )
        store = cls(client, embedding, index_name, metric=metric)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
