# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVSemanticCache`: an embedding-similarity `langchain_core.caches.BaseCache`
backed by `SCACHE.*`, one namespace per `llm_string`."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Sequence

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings

from ...exceptions import NotLicensedError, ResponseError
from ._generations import dump_generations, extract_usage, load_generations


def _as_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value

__all__ = ["HotKVSemanticCache"]


def _namespace_suffix(llm_string: str) -> str:
    return hashlib.sha256(llm_string.encode("utf-8")).hexdigest()[:16]


class HotKVSemanticCache(BaseCache):
    """Similarity based LLM response cache for LangChain, on top of `SCACHE.*`.

    ```python
    from langchain_core.globals import set_llm_cache
    from langchain_openai import OpenAIEmbeddings
    import hotkv
    from hotkv.integrations.langchain import HotKVSemanticCache

    client = hotkv.HotKV(host="localhost", port=6379)
    set_llm_cache(HotKVSemanticCache(client, OpenAIEmbeddings(), threshold=0.86, ttl=3600))
    ```

    `lookup()` embeds the prompt with the given `Embeddings` object and
    returns the nearest cached answer above `threshold`, if any;
    `update()` embeds and stores it. Every distinct `llm_string` (which
    encodes the model name and every generation parameter) gets its own
    `SCACHE.*` namespace, derived from a hash of `llm_string`, so a
    semantically similar prompt sent to a different model, or the same
    model with different parameters, never returns another model's
    answer. The namespace's vector dimensionality is fixed by the first
    embedding it sees.

    Needs a server license that includes the semantic-cache feature; on a
    server with no license at all, `lookup()` treats a `NOLICENSE` reply as a
    miss and `update()` silently skips storing, so an unlicensed server
    degrades to "never cached" rather than breaking the caller.
    """

    def __init__(
        self,
        client: Any,
        embedding: Embeddings,
        *,
        namespace: str = "langchain-semantic-cache",
        metric: str = "cosine",
        threshold: float = 0.85,
        ttl: Optional[int] = None,
    ) -> None:
        self._client = client
        self._embedding = embedding
        self._namespace = namespace
        self._metric = metric
        self._threshold = threshold
        self._ttl = ttl
        self._created_namespaces: Dict[str, bool] = {}

    def _namespace_for(self, llm_string: str) -> str:
        return f"{self._namespace}:{_namespace_suffix(llm_string)}"

    def _ensure_namespace(self, namespace: str, dims: int) -> None:
        if self._created_namespaces.get(namespace):
            return
        try:
            self._client.scache_create(
                namespace, dims, metric=self._metric, threshold=self._threshold, ttl=self._ttl
            )
        except ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise
            # "already exists or limit reached": make sure it is really there.
            try:
                self._client.scache_stats(namespace)
            except ResponseError as stats_exc:
                if "not found" in str(stats_exc).lower():
                    raise exc
        self._created_namespaces[namespace] = True

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        try:
            vector = self._embedding.embed_query(prompt)
            hit = self._client.scache_get(self._namespace_for(llm_string), vector, threshold=self._threshold)
        except NotLicensedError:
            return None
        if hit is None:
            return None
        response, _similarity = hit
        return load_generations(_as_text(response))

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Any]) -> None:
        namespace = self._namespace_for(llm_string)
        vector = self._embedding.embed_query(prompt)
        try:
            self._ensure_namespace(namespace, len(vector))
            # See `HotKVCache.update()`: stored only when the response
            # reliably carries a model name and LangChain's usage_metadata.
            self._client.scache_set(
                namespace,
                prompt,
                dump_generations(return_val),
                vector,
                ttl=self._ttl,
                usage=extract_usage(return_val),
            )
        except NotLicensedError:
            return

    def clear(self, **kwargs: Any) -> None:
        for namespace in list(self._created_namespaces):
            try:
                self._client.scache_drop(namespace)
            except NotLicensedError:
                pass
        self._created_namespaces.clear()
