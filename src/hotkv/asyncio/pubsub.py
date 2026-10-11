# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Asyncio publish/subscribe; see `hotkv.pubsub.PubSub` for the full
behavior, which this mirrors as an async iterator instead of a sync one."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Set

from ..exceptions import ResponseError
from ..pubsub import Message, _ACK_KINDS, _norm
from ._pool import AsyncConnectionPool

__all__ = ["AsyncPubSub", "Message"]


class AsyncPubSub:
    """Asyncio publish/subscribe listener."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool
        self._conn = None
        self.channels: Set[bytes] = set()
        self.patterns: Set[bytes] = set()
        self.shard_channels: Set[bytes] = set()

    async def _ensure_conn(self):
        if self._conn is None:
            self._conn = await self._pool.get_connection()
        return self._conn

    async def _do(self, command: str, names, tracked: Set[bytes], subscribing: bool) -> None:
        conn = await self._ensure_conn()
        names = list(names)
        await conn.send_command(command, *names)
        expected = len(names) if names else max(len(tracked), 1)
        for _ in range(expected):
            ack = await conn.read_response()
            if isinstance(ack, ResponseError):
                raise ack
            channel = ack[1]
            if channel is not None:
                normalized = _norm(channel)
                if subscribing:
                    tracked.add(normalized)
                else:
                    tracked.discard(normalized)
        if names and subscribing:
            tracked.update(_norm(n) for n in names)
        elif names and not subscribing:
            for n in names:
                tracked.discard(_norm(n))

    async def subscribe(self, *channels: Any) -> None:
        """Subscribe to `channels`, awaiting every ack."""
        await self._do("SUBSCRIBE", channels, self.channels, subscribing=True)

    async def psubscribe(self, *patterns: Any) -> None:
        """Subscribe to `patterns` (glob-style)."""
        await self._do("PSUBSCRIBE", patterns, self.patterns, subscribing=True)

    async def ssubscribe(self, *channels: Any) -> None:
        """Subscribe to shard channels `channels` (`SSUBSCRIBE`)."""
        await self._do("SSUBSCRIBE", channels, self.shard_channels, subscribing=True)

    async def unsubscribe(self, *channels: Any) -> None:
        """Unsubscribe from `channels`, or every channel if none are given."""
        await self._do("UNSUBSCRIBE", channels, self.channels, subscribing=False)

    async def punsubscribe(self, *patterns: Any) -> None:
        """Unsubscribe from `patterns`, or every pattern if none are given."""
        await self._do("PUNSUBSCRIBE", patterns, self.patterns, subscribing=False)

    async def sunsubscribe(self, *channels: Any) -> None:
        """Unsubscribe from shard channels, or all of them if none are given."""
        await self._do("SUNSUBSCRIBE", channels, self.shard_channels, subscribing=False)

    @property
    def subscribed(self) -> bool:
        """Whether this listener currently has any active subscription."""
        return bool(self.channels or self.patterns or self.shard_channels)

    async def get_message(
        self, timeout: Optional[float] = 0.0, ignore_subscribe_messages: bool = False
    ) -> Optional[Message]:
        """Return the next message, or `None` if `timeout` elapses first.
        `timeout=None` waits until one arrives; `timeout=0` never waits."""
        if self._conn is None:
            return None
        if not await self._conn.can_read(timeout=timeout):
            return None
        raw = await self._conn.read_response()
        return self._process(raw, ignore_subscribe_messages)

    def _process(self, raw: Any, ignore_subscribe_messages: bool) -> Optional[Message]:
        kind_raw = raw[0]
        kind = kind_raw.decode("utf-8", "replace") if isinstance(kind_raw, (bytes, bytearray)) else kind_raw
        if kind == "pmessage":
            message = Message(kind, raw[2], raw[1], raw[3])
        elif kind in ("message", "smessage"):
            message = Message(kind, raw[1], None, raw[2])
        else:
            message = Message(kind, raw[1], None, int(raw[2]) if raw[2] is not None else 0)
        if ignore_subscribe_messages and kind in _ACK_KINDS:
            return None
        return message

    async def listen(self) -> AsyncIterator[Message]:
        """Wait forever, yielding each message as it arrives."""
        while self._conn is not None:
            message = await self.get_message(timeout=None, ignore_subscribe_messages=True)
            if message is not None:
                yield message

    def __aiter__(self) -> AsyncIterator[Message]:
        return self.listen()

    async def close(self) -> None:
        """Release the underlying connection (discarded, not pooled for reuse)."""
        if self._conn is not None:
            conn, self._conn = self._conn, None
            await self._pool.discard(conn)
        self.channels.clear()
        self.patterns.clear()
        self.shard_channels.clear()

    async def __aenter__(self) -> "AsyncPubSub":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
