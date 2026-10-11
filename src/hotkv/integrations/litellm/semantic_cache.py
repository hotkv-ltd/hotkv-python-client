# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVSemanticCache`: a similarity based LiteLLM
`litellm.caching.base_cache.BaseCache` backed by `SCACHE.*`."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from litellm.caching.base_cache import BaseCache as LiteLLMBaseCache
from litellm.litellm_core_utils.prompt_templates.common_utils import get_str_from_messages

from ...exceptions import NotLicensedError, ResponseError

__all__ = ["HotKVSemanticCache"]

EmbedFn = Callable[[str], List[float]]


def _default_embed(model: str, text: str) -> List[float]:
    import litellm

    response = litellm.embedding(model=model, input=[text])
    return response.data[0]["embedding"]


async def _default_aembed(model: str, text: str) -> List[float]:
    import litellm

    response = await litellm.aembedding(model=model, input=[text])
    return response.data[0]["embedding"]


class HotKVSemanticCache(LiteLLMBaseCache):
    """Similarity based response cache for LiteLLM, on top of `SCACHE.*`.

    ```python
    import litellm
    from hotkv.integrations.litellm import HotKVSemanticCache
    import hotkv

    client = hotkv.HotKV(host="localhost", port=6379)
    litellm.cache = litellm.Cache()
    litellm.cache.cache = HotKVSemanticCache(client, embedding_model="text-embedding-3-small", threshold=0.86)
    ```

    Unlike exact-key caching, a semantic cache needs the actual prompt
    text, not just LiteLLM's pre-hashed `key`: `set_cache()`/`get_cache()`
    pull it back out of `kwargs["messages"]` (the same way LiteLLM's own
    built-in semantic cache backends do) and embed it with `embedding_model`
    through `litellm.embedding()`/`litellm.aembedding()` (or `embed_fn`/
    `aembed_fn`, if given, to use any other embedding call). Namespaced by
    a hash of `embedding_model` plus every extra keyword in `model_params`
    (so switching models, or a differently configured cache instance,
    never serves another one's answers). `get_cache()` treats a
    `NOLICENSE` reply (no `SemanticCache` license feature) as a miss
    rather than raising, so an unlicensed server just never caches.
    """

    def __init__(
        self,
        client: Any,
        *,
        embedding_model: str,
        namespace: str = "litellm-semantic-cache",
        metric: str = "cosine",
        threshold: float = 0.85,
        default_ttl: int = 3600,
        embed_fn: Optional[EmbedFn] = None,
        aembed_fn: Optional[Any] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(default_ttl=default_ttl)
        self._client = client
        self._embedding_model = embedding_model
        self._embed_fn = embed_fn or (lambda text: _default_embed(embedding_model, text))
        self._aembed_fn = aembed_fn or (lambda text: _default_aembed(embedding_model, text))
        self._metric = metric
        self._threshold = threshold
        namespace_key = json.dumps(
            {"model": embedding_model, "params": model_params or {}}, sort_keys=True
        )
        suffix = hashlib.sha256(namespace_key.encode("utf-8")).hexdigest()[:16]
        self._namespace = f"{namespace}:{suffix}"
        self._namespace_ready = False

    def _ensure_namespace(self, dims: int) -> None:
        if self._namespace_ready:
            return
        try:
            self._client.scache_create(
                self._namespace, dims, metric=self._metric, threshold=self._threshold, ttl=self.default_ttl
            )
        except ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise
            # "already exists or limit reached": make sure it is really there.
            try:
                self._client.scache_stats(self._namespace)
            except ResponseError as stats_exc:
                if "not found" in str(stats_exc).lower():
                    raise exc
        self._namespace_ready = True

    @staticmethod
    def _prompt_from_kwargs(kwargs: Dict[str, Any]) -> Optional[str]:
        messages = kwargs.get("messages")
        if not messages:
            return None
        return get_str_from_messages(messages)

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        prompt = self._prompt_from_kwargs(kwargs)
        if prompt is None:
            return
        vector = self._embed_fn(prompt)
        try:
            self._ensure_namespace(len(vector))
            ttl = self.get_ttl(**kwargs)
            self._client.scache_set(self._namespace, prompt, json.dumps(value), vector, ttl=ttl)
        except NotLicensedError:
            return

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        prompt = self._prompt_from_kwargs(kwargs)
        if prompt is None:
            return
        vector = await self._aembed_fn(prompt)
        try:
            await asyncio.to_thread(self._ensure_namespace, len(vector))
            ttl = self.get_ttl(**kwargs)
            await asyncio.to_thread(
                self._client.scache_set, self._namespace, prompt, json.dumps(value), vector, ttl=ttl
            )
        except NotLicensedError:
            return

    async def async_set_cache_pipeline(self, cache_list: Sequence[Tuple[str, Any]], **kwargs: Any) -> None:
        for key, value in cache_list:
            await self.async_set_cache(key, value, **kwargs)

    def get_cache(self, key: str, **kwargs: Any) -> Optional[Any]:
        prompt = self._prompt_from_kwargs(kwargs)
        if prompt is None:
            return None
        vector = self._embed_fn(prompt)
        try:
            hit = self._client.scache_get(self._namespace, vector, threshold=self._threshold)
        except NotLicensedError:
            return None
        if hit is None:
            return None
        response, _similarity = hit
        text = response.decode("utf-8") if isinstance(response, (bytes, bytearray)) else response
        return json.loads(text)

    async def async_get_cache(self, key: str, **kwargs: Any) -> Optional[Any]:
        prompt = self._prompt_from_kwargs(kwargs)
        if prompt is None:
            return None
        vector = await self._aembed_fn(prompt)
        try:
            hit = await asyncio.to_thread(self._client.scache_get, self._namespace, vector, threshold=self._threshold)
        except NotLicensedError:
            return None
        if hit is None:
            return None
        response, _similarity = hit
        text = response.decode("utf-8") if isinstance(response, (bytes, bytearray)) else response
        return json.loads(text)
