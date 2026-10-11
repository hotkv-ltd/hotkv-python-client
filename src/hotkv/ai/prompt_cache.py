# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`PromptCache`: a deterministic-key LLM response cache on top of
`PCACHE.*`. Provider neutral: it never imports or depends on any specific
LLM SDK. `get_or_call()` is the main entry point: it only invokes the
caller's LLM call on a cache miss.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..exceptions import NotLicensedError
from ..usage import TokenUsage
from ._call_result import CallResult, split_usage
from ._keys import build_cache_key

__all__ = ["PromptCache", "AsyncPromptCache"]


def _as_text(value: Any) -> Any:
    """A cached response as text, the type the wrapped call returned on the
    miss that stored it; bytes that are not UTF-8 stay bytes."""
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(value)
    return value


class PromptCache:
    """LLM response cache keyed by provider, model, messages and generation
    parameters. Use `get_or_call()` so the wrapped LLM call only happens on
    a cache miss; hits and (optionally) token usage are recorded through
    `LLMSTATS.*` when the server is licensed for it.
    """

    def __init__(
        self,
        client: Any,
        namespace: str = "default",
        *,
        max_entries: Optional[int] = None,
        ttl: Optional[int] = None,
        cost_per_hit: Optional[float] = None,
        create: bool = True,
    ) -> None:
        self._client = client
        self.namespace = namespace
        if create:
            client.pcache_create(
                namespace, max_entries=max_entries, ttl=ttl, cost_per_hit=cost_per_hit
            )

    def build_key(
        self,
        provider: str,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        system: Optional[str] = None,
        **params: Any,
    ) -> str:
        """Deterministic cache key for one call shape; see `hotkv.ai._keys`."""
        return build_cache_key(provider, model, messages, system=system, **params)

    def get(self, key: str) -> Optional[bytes]:
        """Return the cached response for `key`, or `None` on a miss."""
        return self._client.pcache_get(self.namespace, key)

    def set(
        self,
        key: str,
        response: Any,
        *,
        ttl: Optional[int] = None,
        meta: Optional[str] = None,
        usage: Optional[TokenUsage] = None,
    ) -> bool:
        """Cache `response` under `key`. `usage`, if given, is recorded with
        the entry so a later hit attributes its tokens/cost (see
        `hotkv.TokenUsage`) to `stats()`'s `usage_saved_*` counters."""
        return self._client.pcache_set(self.namespace, key, response, ttl=ttl, meta=meta, usage=usage)

    def stats(self) -> Dict[str, Any]:
        """Return this namespace's `pcache_stats()`, including the
        `usage_saved_*` savings counters attributed from stored `usage`."""
        return self._client.pcache_stats(self.namespace)

    def get_or_call(
        self,
        provider: str,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        fn: Callable[[], CallResult],
        *,
        system: Optional[str] = None,
        ttl: Optional[int] = None,
        meta: Optional[str] = None,
        stats_bucket: Optional[str] = None,
        tokens: Tuple[int, int] = (0, 0),
        cost_usd: float = 0.0,
        **params: Any,
    ) -> Tuple[Any, bool]:
        """Return `(response, hit)`. Calls `fn()` (your real LLM call, a
        zero-argument callable) only on a cache miss.

        `fn` may return either just the response, or a `(response, usage)`
        pair (a `hotkv.TokenUsage`); when it does, `usage` is stored with the
        cache entry (see `set()`) so a later hit attributes its savings.

        If `stats_bucket` is given, also records the call through
        `LLMSTATS.RECORD`: `tokens` is `(input_tokens, output_tokens)` for
        the call you made (or would have made); `cached_tokens` is their
        sum on a hit, `0` on a miss. Silently skipped (no exception) if the
        server's license does not include LLM stats.
        """
        key = self.build_key(provider, model, messages, system=system, **params)
        start = time.monotonic()
        cached = self.get(key)
        if cached is not None:
            self._record_stats(stats_bucket, tokens, cost_usd, start, cached_tokens=sum(tokens))
            return _as_text(cached), True
        response, usage = split_usage(fn())
        self.set(key, response, ttl=ttl, meta=meta, usage=usage)
        self._record_stats(stats_bucket, tokens, cost_usd, start, cached_tokens=0)
        return response, False

    def _record_stats(
        self,
        bucket: Optional[str],
        tokens: Tuple[int, int],
        cost_usd: float,
        start: float,
        *,
        cached_tokens: int,
    ) -> None:
        if not bucket:
            return
        latency_ms = (time.monotonic() - start) * 1000.0
        try:
            self._client.llmstats_bucket(bucket)
            self._client.llmstats_record(
                bucket, tokens[0], tokens[1], cached_tokens, cost_usd, latency_ms
            )
        except NotLicensedError:
            pass


class AsyncPromptCache:
    """Asyncio equivalent of `PromptCache`. `fn` passed to `get_or_call()`
    may be a sync or an async zero-argument callable."""

    def __init__(
        self,
        client: Any,
        namespace: str = "default",
        *,
        max_entries: Optional[int] = None,
        ttl: Optional[int] = None,
        cost_per_hit: Optional[float] = None,
    ) -> None:
        self._client = client
        self.namespace = namespace
        self._max_entries = max_entries
        self._ttl = ttl
        self._cost_per_hit = cost_per_hit
        self._created = False

    async def _ensure_created(self) -> None:
        if not self._created:
            await self._client.pcache_create(
                self.namespace,
                max_entries=self._max_entries,
                ttl=self._ttl,
                cost_per_hit=self._cost_per_hit,
            )
            self._created = True

    def build_key(
        self,
        provider: str,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        system: Optional[str] = None,
        **params: Any,
    ) -> str:
        return build_cache_key(provider, model, messages, system=system, **params)

    async def get(self, key: str) -> Optional[bytes]:
        await self._ensure_created()
        return await self._client.pcache_get(self.namespace, key)

    async def set(
        self,
        key: str,
        response: Any,
        *,
        ttl: Optional[int] = None,
        meta: Optional[str] = None,
        usage: Optional[TokenUsage] = None,
    ) -> bool:
        await self._ensure_created()
        return await self._client.pcache_set(
            self.namespace, key, response, ttl=ttl, meta=meta, usage=usage
        )

    async def stats(self) -> Dict[str, Any]:
        """Return this namespace's `pcache_stats()`, including the
        `usage_saved_*` savings counters attributed from stored `usage`."""
        await self._ensure_created()
        return await self._client.pcache_stats(self.namespace)

    async def get_or_call(
        self,
        provider: str,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        fn: Callable[[], CallResult],
        *,
        system: Optional[str] = None,
        ttl: Optional[int] = None,
        meta: Optional[str] = None,
        stats_bucket: Optional[str] = None,
        tokens: Tuple[int, int] = (0, 0),
        cost_usd: float = 0.0,
        **params: Any,
    ) -> Tuple[Any, bool]:
        """Asyncio equivalent of `PromptCache.get_or_call()`."""
        await self._ensure_created()
        key = self.build_key(provider, model, messages, system=system, **params)
        start = time.monotonic()
        cached = await self.get(key)
        if cached is not None:
            await self._record_stats(stats_bucket, tokens, cost_usd, start, cached_tokens=sum(tokens))
            return _as_text(cached), True
        result = fn()
        if inspect.isawaitable(result):
            result = await result
        response, usage = split_usage(result)
        await self.set(key, response, ttl=ttl, meta=meta, usage=usage)
        await self._record_stats(stats_bucket, tokens, cost_usd, start, cached_tokens=0)
        return response, False

    async def _record_stats(
        self,
        bucket: Optional[str],
        tokens: Tuple[int, int],
        cost_usd: float,
        start: float,
        *,
        cached_tokens: int,
    ) -> None:
        if not bucket:
            return
        latency_ms = (time.monotonic() - start) * 1000.0
        try:
            await self._client.llmstats_bucket(bucket)
            await self._client.llmstats_record(
                bucket, tokens[0], tokens[1], cached_tokens, cost_usd, latency_ms
            )
        except NotLicensedError:
            pass
