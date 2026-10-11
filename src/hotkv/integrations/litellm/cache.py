# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVCache`: an exact-match LiteLLM `litellm.caching.base_cache.BaseCache`
backed by `PCACHE.*`, falling back to core `GET`/`SET` when unlicensed."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Sequence, Tuple

from litellm.caching.base_cache import BaseCache as LiteLLMBaseCache

from .._exact_cache import ExactCacheBackend

__all__ = ["HotKVCache"]


class HotKVCache(LiteLLMBaseCache):
    """Exact-match response cache for LiteLLM, on top of `PCACHE.*`.

    In code:

    ```python
    import litellm
    from hotkv.integrations.litellm import HotKVCache
    import hotkv

    client = hotkv.HotKV(host="localhost", port=6379)
    litellm.cache = litellm.Cache()
    litellm.cache.cache = HotKVCache(client, namespace="litellm-cache", default_ttl=3600)
    ```

    In the LiteLLM proxy config (`config.yaml`), if your build of the proxy
    supports a `custom_cache` callback, point it at a factory returning
    this class; otherwise set `litellm.cache.cache` as above from a
    startup hook:

    ```yaml
    litellm_settings:
      cache: true
      cache_params:
        type: custom_cache
        custom_cache: "myapp.cache:build_hotkv_cache"  # myapp/cache.py: build_hotkv_cache() -> HotKVCache
    ```

    LiteLLM already hashes the request into `key` before calling
    `get_cache()`/`set_cache()`, so this only needs to store a string by
    that key; see `HotKVSemanticCache` for similarity based caching
    instead of exact-key matching. Falls back to core `GET`/`SET` the
    first time `PCACHE.*` answers `NOLICENSE`, so this works against an
    unlicensed (Starter plan) server too.
    """

    def __init__(self, client: Any, *, namespace: str = "litellm-cache", default_ttl: int = 3600) -> None:
        super().__init__(default_ttl=default_ttl)
        self._backend = ExactCacheBackend(client, namespace, ttl=default_ttl)

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        ttl = self.get_ttl(**kwargs)
        self._backend.set(key, json.dumps(value), ttl=ttl)

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        await asyncio.to_thread(self.set_cache, key, value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list: Sequence[Tuple[str, Any]], **kwargs: Any) -> None:
        for key, value in cache_list:
            await self.async_set_cache(key, value, **kwargs)

    def get_cache(self, key: str, **kwargs: Any) -> Optional[Any]:
        raw = self._backend.get(key)
        if raw is None:
            return None
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        return json.loads(text)

    async def async_get_cache(self, key: str, **kwargs: Any) -> Optional[Any]:
        return await asyncio.to_thread(self.get_cache, key, **kwargs)
