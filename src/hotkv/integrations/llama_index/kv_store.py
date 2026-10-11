# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVKVStore`: a `llama_index.core.storage.kvstore.types.BaseKVStore` on
top of HotKV hash commands, one hash per collection."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore

__all__ = ["HotKVKVStore"]


class HotKVKVStore(BaseKVStore):
    """LlamaIndex key/value store (document store, index store, ...) on HotKV.

    ```python
    from hotkv.integrations.llama_index import HotKVKVStore
    from llama_index.core.storage.docstore import KVDocumentStore

    docstore = KVDocumentStore(HotKVKVStore(client, key_prefix="myapp"))
    ```

    Each `collection` (LlamaIndex's grouping for a docstore/index
    store/ref-doc-info store, typically `"docstore/data"`,
    `"docstore/ref_doc_info"`, and similar) is one HotKV hash, keyed by
    `f"{key_prefix}:{collection}"`; `key` becomes the hash field, and
    `val` (an arbitrary JSON-able dict) is stored as its JSON encoded
    value. That makes `get_all()` a single `HGETALL` and `get`/`put`/
    `delete` a single `HGET`/`HSET`/`HDEL`, regardless of collection size.
    """

    def __init__(self, client: Any, *, key_prefix: str = "hotkv:llamaindex:kv") -> None:
        self._client = client
        self._key_prefix = key_prefix

    def _hash_key(self, collection: str) -> str:
        return f"{self._key_prefix}:{collection}"

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        self._client.hset(self._hash_key(collection), {key: json.dumps(val)})

    async def aput(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        self.put(key, val, collection=collection)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = 1,
    ) -> None:
        mapping = {key: json.dumps(val) for key, val in kv_pairs}
        if mapping:
            self._client.hset(self._hash_key(collection), mapping)

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = 1,
    ) -> None:
        self.put_all(kv_pairs, collection=collection, batch_size=batch_size)

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        raw = self._client.hget(self._hash_key(collection), key)
        return None if raw is None else json.loads(raw)

    async def aget(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        return self.get(key, collection=collection)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        raw = self._client.hgetall(self._hash_key(collection))
        result: Dict[str, dict] = {}
        for field, value in raw.items():
            field_str = field.decode("utf-8") if isinstance(field, (bytes, bytearray)) else field
            result[field_str] = json.loads(value)
        return result

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        return self.get_all(collection=collection)

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        return bool(self._client.hdel(self._hash_key(collection), key))

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        return self.delete(key, collection=collection)
