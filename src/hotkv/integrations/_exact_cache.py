# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Exact-match string storage on top of `PCACHE.*`, falling back to core
`GET`/`SET` (a SHA-256 derived key, with `EX` for the TTL) when the server
replies `NOLICENSE` for `PCACHE.*` (a license without the prompt-cache
feature, such as the Starter plan, or no license at all). Detection happens once per
instance: the first `NOLICENSE` reply flips it to the fallback path for
good, so nothing keeps retrying `PCACHE.*` (and failing) on every call.

Shared by `hotkv.integrations.langchain.HotKVCache` and
`hotkv.integrations.litellm.HotKVCache`; not part of the public API.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from ..exceptions import NotLicensedError, ResponseError
from ..usage import TokenUsage

__all__ = ["ExactCacheBackend", "AsyncExactCacheBackend"]

_FALLBACK_PREFIX = "hotkv:exact-cache-fallback"


def _core_key(namespace: str, key: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{key}".encode("utf-8")).hexdigest()
    return f"{_FALLBACK_PREFIX}:{namespace}:{digest}"


def _is_already_exists(exc: ResponseError) -> bool:
    return "already exists" in str(exc).lower()


def _is_not_found(exc: ResponseError) -> bool:
    return "not found" in str(exc).lower()


class ExactCacheBackend:
    """`get`/`set` a string value by an opaque `key`, on a sync `HotKV`/
    `ClusterHotKV` client. Namespaced (a `PCACHE.*` namespace, or a core
    key prefix in the fallback path) so several backends can share one
    HotKV database without colliding."""

    def __init__(self, client: Any, namespace: str, *, ttl: Optional[int] = None) -> None:
        self._client = client
        self.namespace = namespace
        self._ttl = ttl
        self._licensed: Optional[bool] = None

    def _ensure(self) -> None:
        if self._licensed is not None:
            return
        try:
            self._client.pcache_create(self.namespace, ttl=self._ttl)
            self._licensed = True
        except NotLicensedError:
            self._licensed = False
        except ResponseError as exc:
            if not _is_already_exists(exc):
                raise
            # The server answers "already exists or limit reached": make sure
            # the namespace is really there before using it.
            try:
                self._client.pcache_stats(self.namespace)
            except ResponseError as stats_exc:
                if _is_not_found(stats_exc):
                    raise exc
            self._licensed = True

    def get(self, key: str) -> Optional[bytes]:
        self._ensure()
        if self._licensed:
            try:
                return self._client.pcache_get(self.namespace, key)
            except NotLicensedError:
                self._licensed = False
        return self._client.get(_core_key(self.namespace, key))

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
        usage: Optional[TokenUsage] = None,
    ) -> None:
        """`usage`, when given, is recorded with the entry so a later hit
        attributes it to `PCACHE.STATS`'s `usage_saved_*` counters. It is
        silently dropped in the fallback (unlicensed) path: plain `SET` has
        no usage-tracking concept."""
        self._ensure()
        effective_ttl = self._ttl if ttl is None else ttl
        if self._licensed:
            try:
                self._client.pcache_set(self.namespace, key, value, ttl=effective_ttl, usage=usage)
                return
            except NotLicensedError:
                self._licensed = False
        core_key = _core_key(self.namespace, key)
        if effective_ttl is not None:
            self._client.set(core_key, value, ex=effective_ttl)
        else:
            self._client.set(core_key, value)

    def flush(self) -> None:
        """Best-effort clear. In the fallback path, keys are only found
        through `SCAN` (there is no per-namespace index for plain `SET`
        keys), so this walks the keyspace once."""
        self._ensure()
        if self._licensed:
            try:
                self._client.pcache_flush(self.namespace)
                return
            except NotLicensedError:
                self._licensed = False
        pattern = f"{_FALLBACK_PREFIX}:{self.namespace}:*"
        for found_key in self._client.scan_iter(match=pattern):
            self._client.delete(found_key)


class AsyncExactCacheBackend:
    """Asyncio equivalent of `ExactCacheBackend`, for `hotkv.asyncio.HotKV`."""

    def __init__(self, client: Any, namespace: str, *, ttl: Optional[int] = None) -> None:
        self._client = client
        self.namespace = namespace
        self._ttl = ttl
        self._licensed: Optional[bool] = None

    async def _ensure(self) -> None:
        if self._licensed is not None:
            return
        try:
            await self._client.pcache_create(self.namespace, ttl=self._ttl)
            self._licensed = True
        except NotLicensedError:
            self._licensed = False
        except ResponseError as exc:
            if not _is_already_exists(exc):
                raise
            try:
                await self._client.pcache_stats(self.namespace)
            except ResponseError as stats_exc:
                if _is_not_found(stats_exc):
                    raise exc
            self._licensed = True

    async def get(self, key: str) -> Optional[bytes]:
        await self._ensure()
        if self._licensed:
            try:
                return await self._client.pcache_get(self.namespace, key)
            except NotLicensedError:
                self._licensed = False
        return await self._client.get(_core_key(self.namespace, key))

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
        usage: Optional[TokenUsage] = None,
    ) -> None:
        await self._ensure()
        effective_ttl = self._ttl if ttl is None else ttl
        if self._licensed:
            try:
                await self._client.pcache_set(
                    self.namespace, key, value, ttl=effective_ttl, usage=usage
                )
                return
            except NotLicensedError:
                self._licensed = False
        core_key = _core_key(self.namespace, key)
        if effective_ttl is not None:
            await self._client.set(core_key, value, ex=effective_ttl)
        else:
            await self._client.set(core_key, value)
