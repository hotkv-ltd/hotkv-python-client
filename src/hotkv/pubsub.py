# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Publish/subscribe: a dedicated connection, iterated for messages.

`PubSub` holds one connection for as long as it has active subscriptions
(RESP3 push frames and RESP2 message arrays are both handled). Iterate over
it directly, or use `get_message()` for a non-blocking poll and `listen()`
for a blocking generator.
"""

from __future__ import annotations

from typing import Any, Iterator, NamedTuple, Optional, Set

from ._pool import ConnectionPool
from .exceptions import ResponseError

__all__ = ["PubSub", "Message"]

_ACK_KINDS = frozenset(
    {"subscribe", "unsubscribe", "psubscribe", "punsubscribe", "ssubscribe", "sunsubscribe"}
)


class Message(NamedTuple):
    """One pub/sub frame."""

    type: str
    """"message"/"pmessage"/"smessage", or a "*subscribe" acknowledgement."""
    channel: Any
    pattern: Optional[Any]
    """Set only for "pmessage"."""
    data: Any
    """The published payload, or, for a "*subscribe" ack, the new subscription count."""


def _norm(value: Any) -> bytes:
    return value.encode("utf-8") if isinstance(value, str) else bytes(value)


class PubSub:
    """Sync publish/subscribe listener."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool
        self._conn = None
        self.channels: Set[bytes] = set()
        self.patterns: Set[bytes] = set()
        self.shard_channels: Set[bytes] = set()

    def _ensure_conn(self):
        if self._conn is None:
            self._conn = self._pool.get_connection()
        return self._conn

    def _do(self, command: str, names, tracked: Set[bytes], subscribing: bool) -> None:
        conn = self._ensure_conn()
        names = list(names)
        conn.send_command(command, *names)
        if names:
            expected = len(names)
        else:
            expected = max(len(tracked), 1)
        for _ in range(expected):
            ack = conn.read_response()
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

    def subscribe(self, *channels: Any) -> None:
        """Subscribe to `channels`, blocking until every ack is received."""
        self._do("SUBSCRIBE", channels, self.channels, subscribing=True)

    def psubscribe(self, *patterns: Any) -> None:
        """Subscribe to `patterns` (glob-style)."""
        self._do("PSUBSCRIBE", patterns, self.patterns, subscribing=True)

    def ssubscribe(self, *channels: Any) -> None:
        """Subscribe to shard channels `channels` (`SSUBSCRIBE`)."""
        self._do("SSUBSCRIBE", channels, self.shard_channels, subscribing=True)

    def unsubscribe(self, *channels: Any) -> None:
        """Unsubscribe from `channels`, or every channel if none are given."""
        self._do("UNSUBSCRIBE", channels, self.channels, subscribing=False)

    def punsubscribe(self, *patterns: Any) -> None:
        """Unsubscribe from `patterns`, or every pattern if none are given."""
        self._do("PUNSUBSCRIBE", patterns, self.patterns, subscribing=False)

    def sunsubscribe(self, *channels: Any) -> None:
        """Unsubscribe from shard channels, or all of them if none are given."""
        self._do("SUNSUBSCRIBE", channels, self.shard_channels, subscribing=False)

    @property
    def subscribed(self) -> bool:
        """Whether this listener currently has any active subscription."""
        return bool(self.channels or self.patterns or self.shard_channels)

    def get_message(
        self, timeout: Optional[float] = 0.0, ignore_subscribe_messages: bool = False
    ) -> Optional[Message]:
        """Return the next message, or `None` if `timeout` elapses first.
        `timeout=None` blocks until one arrives; `timeout=0` never blocks."""
        if self._conn is None:
            return None
        if not self._conn.can_read(timeout=timeout):
            return None
        raw = self._conn.read_response()
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

    def listen(self) -> Iterator[Message]:
        """Block forever, yielding each message as it arrives."""
        while self._conn is not None:
            message = self.get_message(timeout=None, ignore_subscribe_messages=True)
            if message is not None:
                yield message

    def __iter__(self) -> Iterator[Message]:
        return self.listen()

    def close(self) -> None:
        """Release the underlying connection. It is discarded (not pooled
        for reuse as a plain command connection), since a connection that
        has been in subscribe mode should not silently go back into the
        general pool."""
        if self._conn is not None:
            conn, self._conn = self._conn, None
            self._pool.discard(conn)
        self.channels.clear()
        self.patterns.clear()
        self.shard_channels.clear()

    def __enter__(self) -> "PubSub":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
