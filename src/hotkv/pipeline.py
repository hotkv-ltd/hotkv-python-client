# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Pipelining and transactions.

`Pipeline` covers both features with one class: with `transaction=True` (the
default) commands are wrapped in `MULTI`/`EXEC` for atomicity; with
`transaction=False` they are just batched into a single round trip
("pipelining") with no atomicity guarantee, which is faster.

Every queued command is a plain `def` from the shared `CoreCommands`
mixins, so this same class, unmodified, also backs `hotkv.asyncio`'s
`AsyncPipeline` (see `hotkv/asyncio/pipeline.py`), which only overrides the
handful of methods that actually touch the network.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from . import commands
from ._pool import ConnectionPool
from ._protocol import build_args, encode_command
from .exceptions import ConnectionError, HotKVError, ProtocolError, ResponseError, TimeoutError, WatchError

__all__ = ["Pipeline"]

_TRANSPORT_ERRORS = (ConnectionError, TimeoutError, ProtocolError)


class Pipeline(commands.CoreCommands):
    """Batch of commands sent as one round trip, optionally atomic.

    Typical simple pipelining::

        with client.pipeline(transaction=False) as pipe:
            pipe.set("a", 1)
            pipe.incr("counter")
            results = pipe.execute()

    Optimistic transaction (mirrors `WATCH`/read/`MULTI`/queue/`EXEC`)::

        pipe = client.pipeline()
        pipe.watch("balance")
        current = pipe.get("balance")          # runs immediately (pre-MULTI)
        pipe.multi()                           # switch to queueing mode
        pipe.set("balance", int(current) - 10) # queued
        pipe.execute()                         # MULTI + queued + EXEC

    `client.transaction()` wraps exactly this pattern with automatic retry
    on `WatchError`.
    """

    RESPONSE_CALLBACKS = commands.RESPONSE_CALLBACKS

    def __init__(self, pool: ConnectionPool, *, transaction: bool = True) -> None:
        self._pool = pool
        self.transaction = transaction
        self._command_stack: List[Tuple[tuple, dict]] = []
        self._conn = None
        self._watching = False
        self._immediate = False

    def _ensure_conn(self):
        if self._conn is None:
            self._conn = self._pool.get_connection()
        return self._conn

    def execute_command(self, *args: Any, **options: Any) -> Any:
        if self._immediate:
            conn = self._ensure_conn()
            conn.send_command(*args)
            reply = conn.read_response()
            if isinstance(reply, ResponseError):
                raise reply
            callback = self.RESPONSE_CALLBACKS.get(args[0])
            return callback(reply, **options) if callback is not None else reply
        self._command_stack.append((args, options))
        return self

    def watch(self, *keys: Any) -> bool:
        """Watch `keys` for changes; must be called before any queued command.
        Switches the pipeline to immediate mode until `multi()` is called."""
        if self._command_stack:
            raise HotKVError("watch() must be called before queuing pipeline commands")
        conn = self._ensure_conn()
        conn.send_command("WATCH", *keys)
        reply = conn.read_response()
        if isinstance(reply, ResponseError):
            raise reply
        self._watching = True
        self._immediate = True
        return True

    def unwatch(self) -> bool:
        """Forget all watched keys on this pipeline's connection."""
        if self._conn is not None and self._watching:
            self._conn.send_command("UNWATCH")
            reply = self._conn.read_response()
            if isinstance(reply, ResponseError):
                raise reply
        self._watching = False
        return True

    def multi(self) -> "Pipeline":
        """Switch from immediate mode (used for pre-transaction reads after
        `watch()`) to queueing mode. The wire `MULTI` is sent by `execute()`."""
        self._immediate = False
        return self

    def __len__(self) -> int:
        return len(self._command_stack)

    def execute(self, raise_on_error: bool = True) -> List[Any]:
        """Send every queued command in one round trip and return the results
        in order. With `raise_on_error=False`, a per-command `ResponseError`
        is returned in place rather than raised, so the rest of the batch's
        results are still available."""
        if not self._command_stack:
            self.reset()
            return []
        conn = self._ensure_conn()
        queued = self._command_stack
        self._command_stack = []
        use_txn = self.transaction
        try:
            parts = []
            if use_txn:
                parts.append(encode_command(["MULTI"]))
            for args, _options in queued:
                parts.append(encode_command(build_args(*args)))
            if use_txn:
                parts.append(encode_command(["EXEC"]))
            conn.send_packed_command(b"".join(parts))

            if use_txn:
                raw_replies = self._read_transaction_replies(conn, len(queued))
            else:
                raw_replies = [conn.read_response() for _ in queued]

            results: List[Any] = []
            for (args, options), raw in zip(queued, raw_replies):
                if isinstance(raw, ResponseError):
                    if raise_on_error:
                        raise raw
                    results.append(raw)
                    continue
                callback = self.RESPONSE_CALLBACKS.get(args[0])
                results.append(callback(raw, **options) if callback is not None else raw)
            return results
        finally:
            self.reset()

    def _read_transaction_replies(self, conn, count: int) -> List[Any]:
        # Always fully drain the MULTI reply, one ack per queued command and
        # the final EXEC reply, no matter what: the server already has all
        # of that in flight, and leaving any of it unread would desync the
        # next command read on this connection.
        multi_reply = conn.read_response()
        queue_errors = []
        for _ in range(count):
            ack = conn.read_response()
            if isinstance(ack, ResponseError):
                queue_errors.append(ack)
        exec_reply = conn.read_response()

        if isinstance(multi_reply, ResponseError):
            raise multi_reply
        if queue_errors:
            raise queue_errors[0]
        if exec_reply is None:
            raise WatchError("transaction aborted: a watched key changed")
        return exec_reply

    def reset(self) -> None:
        """Discard any queued commands and release the held connection."""
        self._command_stack = []
        self._immediate = False
        if self._conn is not None:
            conn, self._conn = self._conn, None
            if self._watching:
                try:
                    conn.send_command("UNWATCH")
                    conn.read_response()
                except _TRANSPORT_ERRORS:
                    pass
            self._watching = False
            if conn.connected:
                self._pool.release(conn)
            else:
                self._pool.discard(conn)

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.reset()
