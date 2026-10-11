# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""The asyncio HotKV client.

`HotKV` here mixes in the exact same `hotkv.commands.CoreCommands` used by
the sync client: every typed command method is a plain `def` returning
`self.execute_command(...)`, and since `execute_command` is `async def` on
this class, calling e.g. `client.get("k")` returns a coroutine that
`await` runs to completion, without the mixin itself needing to know
whether it is being used synchronously or asynchronously.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, AsyncIterator, Callable, Dict, Optional, Tuple

from .. import commands
from .._url import parse_url
from ..exceptions import ConnectionError, NoScriptError, ProtocolError, ResponseError, TimeoutError, WatchError
from ._pool import AsyncConnectionPool
from .pipeline import AsyncPipeline
from .pubsub import AsyncPubSub

__all__ = ["HotKV"]

_RETRY_EXCEPTIONS = (ConnectionError, TimeoutError, ProtocolError)


class HotKV(commands.CoreCommands):
    """Asyncio HotKV client. See `hotkv.HotKV` for the sync equivalent; the
    two share every option and typed command method."""

    RESPONSE_CALLBACKS = commands.RESPONSE_CALLBACKS
    READONLY_COMMANDS = commands.READONLY_COMMANDS

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        *,
        unix_socket_path: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        db: int = 0,
        client_name: Optional[str] = None,
        connect_timeout: Optional[float] = 5.0,
        command_timeout: Optional[float] = None,
        tls: bool = False,
        tls_ca_file: Optional[str] = None,
        tls_certfile: Optional[str] = None,
        tls_keyfile: Optional[str] = None,
        tls_server_hostname: Optional[str] = None,
        tls_insecure_skip_verify: bool = False,
        resp3: bool = True,
        decode_responses: bool = False,
        encoding: str = "utf-8",
        encoding_errors: str = "strict",
        max_connections: int = 50,
        idle_timeout: Optional[float] = 300.0,
        retry_writes: bool = False,
    ) -> None:
        self.connection_kwargs: Dict[str, Any] = dict(
            host=host,
            port=port,
            unix_socket_path=unix_socket_path,
            username=username,
            password=password,
            db=db,
            client_name=client_name,
            connect_timeout=connect_timeout,
            command_timeout=command_timeout,
            tls=tls,
            tls_ca_file=tls_ca_file,
            tls_certfile=tls_certfile,
            tls_keyfile=tls_keyfile,
            tls_server_hostname=tls_server_hostname,
            tls_insecure_skip_verify=tls_insecure_skip_verify,
            resp3=resp3,
            decode_responses=decode_responses,
            encoding=encoding,
            encoding_errors=encoding_errors,
        )
        self.retry_writes = retry_writes
        self._pool = AsyncConnectionPool(
            self.connection_kwargs, max_connections=max_connections, idle_timeout=idle_timeout
        )

    @classmethod
    def from_url(cls, url: str, **overrides: Any) -> "HotKV":
        """Build a client from a connection URL; see `hotkv._url.parse_url()`."""
        options = parse_url(url)
        options.update(overrides)
        return cls(**options)

    # -- command execution --------------------------------------------------------

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        command_name = args[0]
        retryable = self.retry_writes or command_name in self.READONLY_COMMANDS
        try:
            reply = await self._execute_once(args)
        except _RETRY_EXCEPTIONS:
            if not retryable:
                raise
            reply = await self._execute_once(args)
        if isinstance(reply, ResponseError):
            raise reply
        callback = self.RESPONSE_CALLBACKS.get(command_name)
        return callback(reply, **options) if callback is not None else reply

    async def _execute_once(self, args: Tuple[Any, ...]) -> Any:
        async with self._pool.connection() as conn:
            await conn.send_command(*args)
            return await conn.read_response()

    async def execute(self, *args: Any, **options: Any) -> Any:
        """Escape hatch: send any command, typed wrapper or not."""
        return await self.execute_command(*args, **options)

    command = execute

    # -- lifecycle --------------------------------------------------------------

    async def close(self) -> None:
        """Close every pooled connection."""
        await self._pool.close()

    async def __aenter__(self) -> "HotKV":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<hotkv.asyncio.HotKV pool={self._pool!r}>"

    # -- pipelines, transactions and pub/sub ----------------------------------------

    def pipeline(self, transaction: bool = True) -> AsyncPipeline:
        """Return a new `AsyncPipeline`; see `hotkv.Pipeline` for the semantics."""
        return AsyncPipeline(self._pool, transaction=transaction)

    async def transaction(
        self, func: Callable[[AsyncPipeline], Any], *watch_keys: Any, max_retries: int = 5
    ) -> Any:
        """Asyncio equivalent of `hotkv.HotKV.transaction()`. `func` may be a
        sync or an async callable; its result is awaited if it is one."""
        for _ in range(max_retries):
            pipe = self.pipeline(transaction=True)
            try:
                if watch_keys:
                    await pipe.watch(*watch_keys)
                result = func(pipe)
                if inspect.isawaitable(result):
                    result = await result
                if len(pipe):
                    await pipe.execute()
                return result
            except WatchError:
                continue
            finally:
                await pipe.reset()
        raise WatchError(f"transaction() did not succeed after {max_retries} retries")

    def pubsub(self) -> AsyncPubSub:
        """Return a new `AsyncPubSub` listener (its own dedicated connection)."""
        return AsyncPubSub(self._pool)

    # -- scripting convenience ----------------------------------------------------

    async def eval_auto(self, script: str, keys: Any = (), args: Any = ()) -> Any:
        """Asyncio equivalent of `hotkv.HotKV.eval_auto()`."""
        sha1 = hashlib.sha1(script.encode("utf-8")).hexdigest()
        try:
            return await self.evalsha(sha1, keys, args)
        except NoScriptError:
            return await self.eval(script, keys, args)

    # -- SCAN family cursor iteration --------------------------------------------

    async def scan_iter(
        self, match: Optional[str] = None, count: Optional[int] = None, type: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Iterate over every key, transparently paging through `scan()`."""
        cursor = 0
        while True:
            cursor, keys = await self.scan(cursor, match=match, count=count, type=type)
            for key in keys:
                yield key
            if cursor == 0:
                return

    async def sscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> AsyncIterator[bytes]:
        """Iterate over every member of the set at `key`."""
        cursor = 0
        while True:
            cursor, members = await self.sscan(key, cursor, match=match, count=count)
            for member in members:
                yield member
            if cursor == 0:
                return

    async def hscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> AsyncIterator[Tuple[Any, Any]]:
        """Iterate over every `(field, value)` pair of the hash at `key`."""
        cursor = 0
        while True:
            cursor, page = await self.hscan(key, cursor, match=match, count=count)
            for item in page.items():
                yield item
            if cursor == 0:
                return

    async def zscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> AsyncIterator[Tuple[Any, float]]:
        """Iterate over every `(member, score)` pair of the sorted set at `key`."""
        cursor = 0
        while True:
            cursor, page = await self.zscan(key, cursor, match=match, count=count)
            for item in page:
                yield item
            if cursor == 0:
                return
