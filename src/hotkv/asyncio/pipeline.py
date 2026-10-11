# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Asyncio pipelining and transactions; see `hotkv.pipeline.Pipeline` for
the full behavior, which this mirrors with `await` added at each network
touch point.

Important difference from the sync `Pipeline`: because `execute_command()`
is a coroutine function here, every queued call must be awaited too, even
though queuing itself does no I/O, for example::

    async with client.pipeline(transaction=False) as pipe:
        await pipe.set("a", 1)
        await pipe.incr("counter")
        results = await pipe.execute()

Forgetting the `await` on a queuing call silently does nothing (it just
creates a coroutine object without running it), which is a common mistake.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from .. import commands
from .._protocol import build_args, encode_command
from ..exceptions import ResponseError, WatchError
from ._pool import AsyncConnectionPool

__all__ = ["AsyncPipeline"]


class AsyncPipeline(commands.CoreCommands):
    """Asyncio equivalent of `hotkv.Pipeline`."""

    RESPONSE_CALLBACKS = commands.RESPONSE_CALLBACKS

    def __init__(self, pool: AsyncConnectionPool, *, transaction: bool = True) -> None:
        self._pool = pool
        self.transaction = transaction
        self._command_stack: List[Tuple[tuple, dict]] = []
        self._conn = None
        self._watching = False
        self._immediate = False

    async def _ensure_conn(self):
        if self._conn is None:
            self._conn = await self._pool.get_connection()
        return self._conn

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        if self._immediate:
            conn = await self._ensure_conn()
            await conn.send_command(*args)
            reply = await conn.read_response()
            if isinstance(reply, ResponseError):
                raise reply
            callback = self.RESPONSE_CALLBACKS.get(args[0])
            return callback(reply, **options) if callback is not None else reply
        self._command_stack.append((args, options))
        return self

    async def watch(self, *keys: Any) -> bool:
        """Watch `keys` for changes; switches to immediate mode until `multi()`."""
        if self._command_stack:
            raise RuntimeError("watch() must be called before queuing pipeline commands")
        conn = await self._ensure_conn()
        await conn.send_command("WATCH", *keys)
        reply = await conn.read_response()
        if isinstance(reply, ResponseError):
            raise reply
        self._watching = True
        self._immediate = True
        return True

    async def unwatch(self) -> bool:
        """Forget all watched keys on this pipeline's connection."""
        if self._conn is not None and self._watching:
            await self._conn.send_command("UNWATCH")
            reply = await self._conn.read_response()
            if isinstance(reply, ResponseError):
                raise reply
        self._watching = False
        return True

    def multi(self) -> "AsyncPipeline":
        """Switch from immediate mode to queueing mode."""
        self._immediate = False
        return self

    def __len__(self) -> int:
        return len(self._command_stack)

    async def execute(self, raise_on_error: bool = True) -> List[Any]:
        """Send every queued command in one round trip; see `Pipeline.execute()`."""
        if not self._command_stack:
            await self.reset()
            return []
        conn = await self._ensure_conn()
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
            await conn.send_packed_command(b"".join(parts))

            if use_txn:
                raw_replies = await self._read_transaction_replies(conn, len(queued))
            else:
                raw_replies = [await conn.read_response() for _ in queued]

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
            await self.reset()

    async def _read_transaction_replies(self, conn, count: int) -> List[Any]:
        multi_reply = await conn.read_response()
        queue_errors = []
        for _ in range(count):
            ack = await conn.read_response()
            if isinstance(ack, ResponseError):
                queue_errors.append(ack)
        exec_reply = await conn.read_response()

        if isinstance(multi_reply, ResponseError):
            raise multi_reply
        if queue_errors:
            raise queue_errors[0]
        if exec_reply is None:
            raise WatchError("transaction aborted: a watched key changed")
        return exec_reply

    async def reset(self) -> None:
        """Discard any queued commands and release the held connection."""
        self._command_stack = []
        self._immediate = False
        if self._conn is not None:
            conn, self._conn = self._conn, None
            if self._watching:
                try:
                    await conn.send_command("UNWATCH")
                    await conn.read_response()
                except Exception:
                    pass
            self._watching = False
            if conn.connected:
                await self._pool.release(conn)
            else:
                await self._pool.discard(conn)

    async def __aenter__(self) -> "AsyncPipeline":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.reset()
