# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""A bounded, thread safe pool of `Connection` objects.

Connections are created lazily up to `max_connections`. A connection that
has sat idle longer than `idle_timeout` is health checked with `PING`
before being handed out again; if the check fails it is dropped and a
replacement is created. `pool.connection()` is the usual entry point: it
borrows a connection, and on a clean exit (including a `ResponseError`
raised for a single command, which leaves the connection perfectly usable)
returns it to the pool, but discards it on a real transport failure
(`ConnectionError`, `TimeoutError`, `ProtocolError`).
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Dict, Iterator, Optional, Type

from ._connection import Connection
from .exceptions import ConnectionError, ProtocolError, TimeoutError

__all__ = ["ConnectionPool"]


class ConnectionPool:
    """Thread safe pool of `Connection` instances sharing one set of options."""

    def __init__(
        self,
        connection_kwargs: Dict[str, Any],
        *,
        max_connections: int = 50,
        idle_timeout: Optional[float] = 300.0,
        connection_class: Type[Connection] = Connection,
    ) -> None:
        if max_connections <= 0:
            raise ValueError("max_connections must be positive")
        self.connection_kwargs = connection_kwargs
        self.max_connections = max_connections
        self.idle_timeout = idle_timeout
        self.connection_class = connection_class

        self._cond = threading.Condition(threading.Lock())
        self._available: list = []  # list[tuple[Connection, float]]
        self._in_use: set = set()
        self._created = 0
        self._closed = False

    def get_connection(self, timeout: Optional[float] = None) -> Connection:
        """Borrow a connection, creating one if the pool has room.

        Blocks (up to `timeout` seconds, or forever if `timeout` is `None`)
        when the pool is at `max_connections` and all are in use.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if self._closed:
                    raise ConnectionError("connection pool is closed")
                if self._available:
                    conn, last_used = self._available.pop()
                    if self.idle_timeout and (time.monotonic() - last_used) > self.idle_timeout:
                        try:
                            conn.ping()
                        except Exception:
                            conn.disconnect()
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
                signalled = self._cond.wait(timeout=remaining)
                if not signalled and deadline is not None:
                    raise ConnectionError("connection pool exhausted: no connections available")

    def release(self, conn: Connection) -> None:
        """Return a healthy connection to the pool for reuse."""
        with self._cond:
            self._in_use.discard(conn)
            if self._closed or not conn.connected:
                if conn.connected:
                    conn.disconnect()
                self._created = max(0, self._created - 1)
            else:
                self._available.append((conn, time.monotonic()))
            self._cond.notify()

    def discard(self, conn: Connection) -> None:
        """Drop a connection that is no longer usable (transport failure)."""
        with self._cond:
            self._in_use.discard(conn)
            conn.disconnect()
            self._created = max(0, self._created - 1)
            self._cond.notify()

    @contextlib.contextmanager
    def connection(self, timeout: Optional[float] = None) -> Iterator[Connection]:
        """Borrow a connection for the duration of the `with` block."""
        conn = self.get_connection(timeout=timeout)
        transport_failed = False
        try:
            yield conn
        except (ConnectionError, TimeoutError, ProtocolError):
            transport_failed = True
            raise
        finally:
            if transport_failed or not conn.connected:
                self.discard(conn)
            else:
                self.release(conn)

    def close(self) -> None:
        """Close every pooled and in-use connection. The pool stays unusable."""
        with self._cond:
            self._closed = True
            for conn, _ in self._available:
                conn.disconnect()
            self._available.clear()
            for conn in list(self._in_use):
                conn.disconnect()
            self._in_use.clear()
            self._created = 0
            self._cond.notify_all()

    @property
    def size(self) -> int:
        """Number of connections currently created (idle plus in use)."""
        with self._cond:
            return self._created

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<hotkv.ConnectionPool max={self.max_connections} "
            f"created={self._created} available={len(self._available)}>"
        )
