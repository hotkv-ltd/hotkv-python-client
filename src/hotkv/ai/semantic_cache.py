# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`SemanticCache`: an embedding-similarity response cache on top of
`SCACHE.*`. The caller always supplies the embedding vector; this module
never computes embeddings itself or depends on any embedding provider.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from ..usage import TokenUsage
from ._call_result import CallResult, split_usage

__all__ = ["SemanticCache", "AsyncSemanticCache"]


class SemanticCache:
    """Cache responses by embedding similarity rather than exact text match.

    `get_or_call()` looks up the nearest cached answer above the similarity
    threshold and only calls your LLM function on a miss.
    """

    def __init__(
        self,
        client: Any,
        namespace: str,
        dims: int,
        *,
        metric: str = "cosine",
        threshold: float = 0.85,
        ttl: Optional[int] = None,
        create: bool = True,
    ) -> None:
        self._client = client
        self.namespace = namespace
        self.dims = dims
        self.threshold = threshold
        if create:
            client.scache_create(namespace, dims, metric=metric, threshold=threshold, ttl=ttl)

    def get(
        self, embedding: Sequence[float], *, threshold: Optional[float] = None
    ) -> Optional[Tuple[Any, float]]:
        """Return `(response, similarity)` for the nearest cached answer to
        `embedding` above the threshold, or `None`."""
        return self._client.scache_get(self.namespace, embedding, threshold=threshold)

    def set(
        self,
        query_text: str,
        response: Any,
        embedding: Sequence[float],
        *,
        ttl: Optional[int] = None,
        usage: Optional[TokenUsage] = None,
    ) -> bool:
        """Cache `response` to `query_text`, indexed by `embedding`. `usage`,
        if given, is recorded with the entry so a later hit attributes its
        tokens/cost (see `hotkv.TokenUsage`) to `stats()`'s `usage_saved_*`
        counters."""
        return self._client.scache_set(
            self.namespace, query_text, response, embedding, ttl=ttl, usage=usage
        )

    def stats(self) -> Dict[str, Any]:
        """Return this namespace's `scache_stats()`, including the
        `usage_saved_*` savings counters attributed from stored `usage`."""
        return self._client.scache_stats(self.namespace)

    def get_or_call(
        self,
        embedding: Sequence[float],
        fn: Callable[[], CallResult],
        *,
        query_text: str = "",
        threshold: Optional[float] = None,
        ttl: Optional[int] = None,
    ) -> Tuple[Any, bool]:
        """Return `(response, hit)`. Calls `fn()` (your real LLM call) only
        when nothing cached is similar enough to `embedding`.

        `fn` may return either just the response, or a `(response, usage)`
        pair (a `hotkv.TokenUsage`); when it does, `usage` is stored with
        the cache entry (see `set()`) so a later hit attributes its savings.
        """
        hit = self.get(embedding, threshold=threshold)
        if hit is not None:
            response, _similarity = hit
            return response, True
        response, usage = split_usage(fn())
        self.set(query_text, response, embedding, ttl=ttl, usage=usage)
        return response, False


class AsyncSemanticCache:
    """Asyncio equivalent of `SemanticCache`. `fn` passed to `get_or_call()`
    may be a sync or an async zero-argument callable."""

    def __init__(
        self,
        client: Any,
        namespace: str,
        dims: int,
        *,
        metric: str = "cosine",
        threshold: float = 0.85,
        ttl: Optional[int] = None,
    ) -> None:
        self._client = client
        self.namespace = namespace
        self.dims = dims
        self.threshold = threshold
        self._metric = metric
        self._ttl = ttl
        self._created = False

    async def _ensure_created(self) -> None:
        if not self._created:
            await self._client.scache_create(
                self.namespace, self.dims, metric=self._metric, threshold=self.threshold, ttl=self._ttl
            )
            self._created = True

    async def get(
        self, embedding: Sequence[float], *, threshold: Optional[float] = None
    ) -> Optional[Tuple[Any, float]]:
        await self._ensure_created()
        return await self._client.scache_get(self.namespace, embedding, threshold=threshold)

    async def set(
        self,
        query_text: str,
        response: Any,
        embedding: Sequence[float],
        *,
        ttl: Optional[int] = None,
        usage: Optional[TokenUsage] = None,
    ) -> bool:
        await self._ensure_created()
        return await self._client.scache_set(
            self.namespace, query_text, response, embedding, ttl=ttl, usage=usage
        )

    async def stats(self) -> Dict[str, Any]:
        """Return this namespace's `scache_stats()`, including the
        `usage_saved_*` savings counters attributed from stored `usage`."""
        await self._ensure_created()
        return await self._client.scache_stats(self.namespace)

    async def get_or_call(
        self,
        embedding: Sequence[float],
        fn: Callable[[], CallResult],
        *,
        query_text: str = "",
        threshold: Optional[float] = None,
        ttl: Optional[int] = None,
    ) -> Tuple[Any, bool]:
        """Asyncio equivalent of `SemanticCache.get_or_call()`."""
        await self._ensure_created()
        hit = await self.get(embedding, threshold=threshold)
        if hit is not None:
            response, _similarity = hit
            return response, True
        result = fn()
        if inspect.isawaitable(result):
            result = await result
        response, usage = split_usage(result)
        await self.set(query_text, response, embedding, ttl=ttl, usage=usage)
        return response, False
