# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""The synchronous HotKV client."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from . import commands
from ._pool import ConnectionPool
from ._url import parse_url
from .exceptions import ConnectionError, NoScriptError, ProtocolError, ResponseError, TimeoutError, WatchError
from .pipeline import Pipeline
from .pubsub import PubSub

__all__ = ["HotKV"]

_RETRY_EXCEPTIONS = (ConnectionError, TimeoutError, ProtocolError)


class HotKV(commands.CoreCommands):
    """Synchronous HotKV client.

    Connections are pooled and created lazily; the constructor never blocks
    on the network. Use `HotKV.from_url()` to build one from a
    `hotkv://`/`hotkvs://`/`redis://`/`rediss://`/`unix://` URL.
    """

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
        self._pool = ConnectionPool(
            self.connection_kwargs, max_connections=max_connections, idle_timeout=idle_timeout
        )

    @classmethod
    def from_url(cls, url: str, **overrides: Any) -> "HotKV":
        """Build a client from a connection URL; see `hotkv._url.parse_url()`
        for the accepted schemes and query parameters. Keyword arguments
        override anything the URL also specifies."""
        options = parse_url(url)
        options.update(overrides)
        return cls(**options)

    # -- command execution --------------------------------------------------------

    def execute_command(self, *args: Any, **options: Any) -> Any:
        command_name = args[0]
        retryable = self.retry_writes or command_name in self.READONLY_COMMANDS
        try:
            reply = self._execute_once(args)
        except _RETRY_EXCEPTIONS:
            if not retryable:
                raise
            reply = self._execute_once(args)
        if isinstance(reply, ResponseError):
            raise reply
        callback = self.RESPONSE_CALLBACKS.get(command_name)
        return callback(reply, **options) if callback is not None else reply

    def _execute_once(self, args: Tuple[Any, ...]) -> Any:
        with self._pool.connection() as conn:
            conn.send_command(*args)
            return conn.read_response()

    def execute(self, *args: Any, **options: Any) -> Any:
        """Escape hatch: send any command, typed wrapper or not."""
        return self.execute_command(*args, **options)

    command = execute

    # -- lifecycle --------------------------------------------------------------

    def close(self) -> None:
        """Close every pooled connection."""
        self._pool.close()

    def __enter__(self) -> "HotKV":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<hotkv.HotKV pool={self._pool!r}>"

    # -- pipelines, transactions and pub/sub ----------------------------------------

    def pipeline(self, transaction: bool = True) -> Pipeline:
        """Return a new `Pipeline`. `transaction=True` (the default) wraps
        the batch in `MULTI`/`EXEC`; `transaction=False` just batches the
        round trip without atomicity."""
        return Pipeline(self._pool, transaction=transaction)

    def transaction(
        self, func: Callable[[Pipeline], Any], *watch_keys: Any, max_retries: int = 5
    ) -> Any:
        """Run `func(pipe)` as an optimistic transaction watching `watch_keys`.

        `func` should read any state it needs from `pipe` (executed
        immediately, before `pipe.multi()`), call `pipe.multi()`, then queue
        the writes; this method calls `pipe.execute()` itself. On
        `WatchError` (a watched key changed), it retries up to `max_retries`
        times before re-raising.
        """
        for _ in range(max_retries):
            pipe = self.pipeline(transaction=True)
            try:
                if watch_keys:
                    pipe.watch(*watch_keys)
                result = func(pipe)
                if len(pipe):
                    pipe.execute()
                return result
            except WatchError:
                continue
            finally:
                pipe.reset()
        raise WatchError(f"transaction() did not succeed after {max_retries} retries")

    def pubsub(self) -> PubSub:
        """Return a new `PubSub` listener (its own dedicated connection)."""
        return PubSub(self._pool)

    # -- scripting convenience ----------------------------------------------------

    def eval_auto(self, script: str, keys: Any = (), args: Any = ()) -> Any:
        """Run `script`, using `EVALSHA` when the server already has it
        cached (computed locally) and transparently falling back to `EVAL`
        (which also (re)caches it) on a `NOSCRIPT` reply."""
        sha1 = hashlib.sha1(script.encode("utf-8")).hexdigest()
        try:
            return self.evalsha(sha1, keys, args)
        except NoScriptError:
            return self.eval(script, keys, args)

    # -- SCAN family cursor iteration --------------------------------------------

    def scan_iter(
        self, match: Optional[str] = None, count: Optional[int] = None, type: Optional[str] = None
    ) -> Iterator[bytes]:
        """Iterate over every key, transparently paging through `scan()`."""
        cursor = 0
        while True:
            cursor, keys = self.scan(cursor, match=match, count=count, type=type)
            yield from keys
            if cursor == 0:
                return

    def sscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> Iterator[bytes]:
        """Iterate over every member of the set at `key`."""
        cursor = 0
        while True:
            cursor, members = self.sscan(key, cursor, match=match, count=count)
            yield from members
            if cursor == 0:
                return

    def hscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> Iterator[Tuple[Any, Any]]:
        """Iterate over every `(field, value)` pair of the hash at `key`."""
        cursor = 0
        while True:
            cursor, page = self.hscan(key, cursor, match=match, count=count)
            yield from page.items()
            if cursor == 0:
                return

    def zscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> Iterator[Tuple[Any, float]]:
        """Iterate over every `(member, score)` pair of the sorted set at `key`."""
        cursor = 0
        while True:
            cursor, page = self.zscan(key, cursor, match=match, count=count)
            yield from page
            if cursor == 0:
                return
