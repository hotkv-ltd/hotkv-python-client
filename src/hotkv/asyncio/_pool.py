# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""An asyncio-safe bounded pool of `AsyncConnection` objects.

Same shape and health-check/discard-vs-release rules as
`hotkv._pool.ConnectionPool`, built on `asyncio.Condition` instead of a
thread lock; safe to share across many coroutines/tasks on one event loop
(it is not meant to be shared across event loops).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, AsyncIterator, Dict, Optional, Type

from ..exceptions import ConnectionError, ProtocolError, TimeoutError
from ._connection import AsyncConnection

__all__ = ["AsyncConnectionPool"]


class AsyncConnectionPool:
    """Asyncio-safe pool of `AsyncConnection` instances sharing one set of options."""

    def __init__(
        self,
        connection_kwargs: Dict[str, Any],
        *,
        max_connections: int = 50,
        idle_timeout: Optional[float] = 300.0,
        connection_class: Type[AsyncConnection] = AsyncConnection,
    ) -> None:
        if max_connections <= 0:
            raise ValueError("max_connections must be positive")
        self.connection_kwargs = connection_kwargs
        self.max_connections = max_connections
        self.idle_timeout = idle_timeout
        self.connection_class = connection_class

        self._condition: Any = None  # an asyncio.Condition, created lazily
        self._available: list = []
        self._in_use: set = set()
        self._created = 0
        self._closed = False

    def _cond(self) -> asyncio.Condition:
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    async def get_connection(self, timeout: Optional[float] = None) -> AsyncConnection:
        cond = self._cond()
        deadline = None if timeout is None else time.monotonic() + timeout
        async with cond:
            while True:
                if self._closed:
                    raise ConnectionError("connection pool is closed")
                if self._available:
                    conn, last_used = self._available.pop()
                    if self.idle_timeout and (time.monotonic() - last_used) > self.idle_timeout:
                        try:
                            await conn.ping()
                        except Exception:
                            await conn.disconnect()
                            self._created -= 1
                            continue
                    self._in_use.add(conn)
                    return conn
                if self._created < self.max_connections:
                    conn = self.connection_class(**self.connection_kwargs)
                    self._created += 1
                    self._in_use.add(conn)
                    return conn
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ConnectionError(
                            "connection pool exhausted: no connections available"
                        )
                try:
                    await asyncio.wait_for(cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    if deadline is not None:
                        raise ConnectionError(
                            "connection pool exhausted: no connections available"
                        ) from None

    async def release(self, conn: AsyncConnection) -> None:
        cond = self._cond()
        async with cond:
            self._in_use.discard(conn)
            if self._closed or not conn.connected:
                if conn.connected:
                    await conn.disconnect()
                self._created = max(0, self._created - 1)
            else:
                self._available.append((conn, time.monotonic()))
            cond.notify()

    async def discard(self, conn: AsyncConnection) -> None:
        cond = self._cond()
        async with cond:
            self._in_use.discard(conn)
            await conn.disconnect()
            self._created = max(0, self._created - 1)
            cond.notify()

    @contextlib.asynccontextmanager
    async def connection(self, timeout: Optional[float] = None) -> AsyncIterator[AsyncConnection]:
        conn = await self.get_connection(timeout=timeout)
        transport_failed = False
        try:
            yield conn
        except (ConnectionError, TimeoutError, ProtocolError):
            transport_failed = True
            raise
        finally:
            if transport_failed or not conn.connected:
                await self.discard(conn)
            else:
                await self.release(conn)

    async def close(self) -> None:
        cond = self._cond()
        async with cond:
            self._closed = True
            for conn, _ in self._available:
                await conn.disconnect()
            self._available.clear()
            for conn in list(self._in_use):
                await conn.disconnect()
            self._in_use.clear()
            self._created = 0
            cond.notify_all()

    @property
    def size(self) -> int:
        return self._created

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<hotkv.asyncio.AsyncConnectionPool max={self.max_connections} "
            f"created={self._created} available={len(self._available)}>"
        )
