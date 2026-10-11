# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""A single asyncio TCP/Unix/TLS connection to a HotKV server.

Mirrors `hotkv._connection.Connection` method for method (same handshake,
same primitives), built on `asyncio` streams instead of a blocking socket,
and sharing the same `hotkv._protocol.Reader` incremental parser.
"""

from __future__ import annotations

import asyncio
import ssl
from typing import Any, Optional

from .._protocol import NEED_MORE, Reader, build_args, decode_value, encode_command
from ..exceptions import AuthenticationError, ConnectionError, ResponseError, TimeoutError

__all__ = ["AsyncConnection"]

_RECV_SIZE = 65536


class AsyncConnection:
    """One asyncio connection to a HotKV server. Not safe for concurrent use."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
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
        readonly: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.unix_socket_path = unix_socket_path
        self.username = username
        self.password = password
        self.db = db
        self.client_name = client_name
        self.connect_timeout = connect_timeout
        self.command_timeout = command_timeout
        self.tls = tls
        self.tls_ca_file = tls_ca_file
        self.tls_certfile = tls_certfile
        self.tls_keyfile = tls_keyfile
        self.tls_server_hostname = tls_server_hostname
        self.tls_insecure_skip_verify = tls_insecure_skip_verify
        self.resp3 = resp3
        self.decode_responses = decode_responses
        self.encoding = encoding
        self.encoding_errors = encoding_errors
        # Send READONLY after the handshake, so a cluster replica serves reads
        # on this connection (the cluster client's replica pools).
        self.readonly = readonly

        self.protocol_version = 2
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._parser: Optional[Reader] = None

    @property
    def connected(self) -> bool:
        return self._writer is not None

    def _target(self) -> str:
        if self.unix_socket_path:
            return f"unix:{self.unix_socket_path}"
        return f"{self.host}:{self.port}"

    async def connect(self) -> None:
        if self._writer is not None:
            return
        await self._open()
        self._parser = Reader()
        try:
            await self._handshake()
        except BaseException:
            await self.disconnect()
            raise

    async def _open(self) -> None:
        extra: dict = {}
        if self.tls:
            extra["ssl"] = self._build_tls_context()
            extra["server_hostname"] = self.tls_server_hostname or (
                None if self.unix_socket_path else self.host
            )
        try:
            if self.unix_socket_path:
                coro = asyncio.open_unix_connection(self.unix_socket_path, **extra)
            else:
                coro = asyncio.open_connection(self.host, self.port, **extra)
            self._reader, self._writer = await asyncio.wait_for(coro, timeout=self.connect_timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"timed out connecting to {self._target()}") from exc
        except (OSError, ssl.SSLError) as exc:
            raise ConnectionError(f"could not connect to {self._target()}: {exc}") from exc
        if not self.unix_socket_path:
            sock = self._writer.get_extra_info("socket")
            if sock is not None:
                try:
                    import socket as _socket

                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                except OSError:
                    pass

    def _build_tls_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if self.tls_insecure_skip_verify:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        elif self.tls_ca_file:
            context.load_verify_locations(cafile=self.tls_ca_file)
        else:
            context.load_default_certs()
        if self.tls_certfile:
            context.load_cert_chain(certfile=self.tls_certfile, keyfile=self.tls_keyfile)
        return context

    async def disconnect(self) -> None:
        writer, self._writer = self._writer, None
        self._reader = None
        self._parser = None
        self.protocol_version = 2
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    close = disconnect

    async def __aenter__(self) -> "AsyncConnection":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.disconnect()

    async def _handshake(self) -> None:
        self.protocol_version = 2
        negotiated_resp3 = False
        if self.resp3:
            hello_args = ["HELLO", "3"]
            if self.username or self.password:
                hello_args += ["AUTH", self.username or "default", self.password or ""]
            if self.client_name:
                hello_args += ["SETNAME", self.client_name]
            await self.send_command(*hello_args)
            reply = await self.read_response(disable_decoding=True)
            if not isinstance(reply, ResponseError):
                self.protocol_version = 3
                negotiated_resp3 = True
        if not negotiated_resp3:
            if self.username or self.password:
                auth_args = ["AUTH"]
                if self.username:
                    auth_args.append(self.username)
                auth_args.append(self.password or "")
                await self.send_command(*auth_args)
                reply = await self.read_response(disable_decoding=True)
                if isinstance(reply, ResponseError):
                    raise AuthenticationError(reply.message)
            if self.client_name:
                await self.send_command("CLIENT SETNAME", self.client_name)
                reply = await self.read_response(disable_decoding=True)
                if isinstance(reply, ResponseError):
                    raise ConnectionError(reply.message)
        if self.db:
            await self.send_command("SELECT", str(self.db))
            reply = await self.read_response(disable_decoding=True)
            if isinstance(reply, ResponseError):
                raise ConnectionError(reply.message)
        if self.readonly:
            await self.send_command("READONLY")
            reply = await self.read_response(disable_decoding=True)
            if isinstance(reply, ResponseError):
                raise ConnectionError(reply.message)

    async def send_command(self, command_name: str, *args: Any) -> None:
        await self.send_packed_command(encode_command(build_args(command_name, *args)))

    async def send_packed_command(self, payload: bytes) -> None:
        if self._writer is None:
            await self.connect()
        assert self._writer is not None
        try:
            self._writer.write(payload)
            await asyncio.wait_for(self._writer.drain(), timeout=self.command_timeout)
        except asyncio.TimeoutError as exc:
            await self.disconnect()
            raise TimeoutError(f"timed out writing to {self._target()}") from exc
        except OSError as exc:
            await self.disconnect()
            raise ConnectionError(f"error writing to {self._target()}: {exc}") from exc

    async def read_response(self, disable_decoding: bool = False) -> Any:
        if self._reader is None or self._parser is None:
            raise ConnectionError("not connected")
        while True:
            value = self._parser.gets()
            if value is not NEED_MORE:
                break
            await self._fill_buffer()
        if self.decode_responses and not disable_decoding:
            value = decode_value(value, self.encoding, self.encoding_errors)
        return value

    async def _fill_buffer(self) -> None:
        assert self._reader is not None and self._parser is not None
        try:
            chunk = await asyncio.wait_for(self._reader.read(_RECV_SIZE), timeout=self.command_timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"timed out reading from {self._target()}") from exc
        except OSError as exc:
            await self.disconnect()
            raise ConnectionError(f"error reading from {self._target()}: {exc}") from exc
        if not chunk:
            await self.disconnect()
            raise ConnectionError(f"connection closed by {self._target()}")
        self._parser.feed(chunk)

    async def can_read(self, timeout: Optional[float] = 0.0) -> bool:
        if self._reader is None or self._parser is None:
            return False
        if self._parser.has_next():
            return True
        try:
            chunk = await asyncio.wait_for(self._reader.read(_RECV_SIZE), timeout=timeout)
        except (asyncio.TimeoutError, OSError):
            return False
        if not chunk:
            await self.disconnect()
            return False
        self._parser.feed(chunk)
        return self._parser.has_next()

    async def ping(self) -> bool:
        await self.send_command("PING")
        reply = await self.read_response(disable_decoding=True)
        if isinstance(reply, ResponseError):
            raise reply
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "connected" if self.connected else "disconnected"
        return f"<hotkv.asyncio.AsyncConnection {self._target()} proto={self.protocol_version} {state}>"
