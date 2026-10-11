# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""A single blocking TCP/Unix/TLS connection to a HotKV server.

`Connection` owns one socket and one incremental `Reader`. It performs the
`HELLO 3` handshake (falling back to RESP2 plus plain `AUTH` when the server
rejects `HELLO`), optional `SELECT`, and exposes `send_command()` /
`read_response()` primitives that the sync client, pipeline, pub/sub and
cluster client build on. It does not retry on its own; retry policy lives
in the client, which is where idempotency can be judged.
"""

from __future__ import annotations

import selectors
import socket
import ssl
from typing import Any, Optional

from ._protocol import NEED_MORE, Reader, build_args, decode_value, encode_command
from .exceptions import AuthenticationError, ConnectionError, ResponseError, TimeoutError

__all__ = ["Connection"]

_RECV_SIZE = 65536


class Connection:
    """One blocking connection to a HotKV server. Not thread safe."""

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
        self._sock: Optional[socket.socket] = None
        self._reader: Optional[Reader] = None

    # -- lifecycle ------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = self._open_socket()
        self._sock = sock
        self._reader = Reader()
        try:
            self._handshake()
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        sock, self._sock = self._sock, None
        self._reader = None
        self.protocol_version = 2
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    close = disconnect

    def __enter__(self) -> "Connection":
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.disconnect()

    def _target(self) -> str:
        if self.unix_socket_path:
            return f"unix:{self.unix_socket_path}"
        return f"{self.host}:{self.port}"

    def _open_socket(self) -> socket.socket:
        if self.unix_socket_path:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            try:
                sock.connect(self.unix_socket_path)
            except OSError as exc:
                raise ConnectionError(
                    f"could not connect to unix socket {self.unix_socket_path}: {exc}"
                ) from exc
        else:
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.connect_timeout
                )
            except socket.timeout as exc:
                raise TimeoutError(f"timed out connecting to {self._target()}") from exc
            except OSError as exc:
                raise ConnectionError(f"could not connect to {self._target()}: {exc}") from exc
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        if self.tls:
            sock = self._wrap_tls(sock)
        sock.settimeout(self.command_timeout)
        return sock

    def _wrap_tls(self, sock: socket.socket) -> socket.socket:
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
        server_hostname = self.tls_server_hostname or self.host or None
        try:
            return context.wrap_socket(sock, server_hostname=server_hostname)
        except ssl.SSLError as exc:
            sock.close()
            raise ConnectionError(f"TLS handshake with {self._target()} failed: {exc}") from exc

    def _handshake(self) -> None:
        self.protocol_version = 2
        negotiated_resp3 = False
        if self.resp3:
            hello_args = ["HELLO", "3"]
            if self.username or self.password:
                hello_args += ["AUTH", self.username or "default", self.password or ""]
            if self.client_name:
                hello_args += ["SETNAME", self.client_name]
            self.send_command(*hello_args)
            reply = self.read_response(disable_decoding=True)
            if not isinstance(reply, ResponseError):
                self.protocol_version = 3
                negotiated_resp3 = True
        if not negotiated_resp3:
            if self.username or self.password:
                auth_args = ["AUTH"]
                if self.username:
                    auth_args.append(self.username)
                auth_args.append(self.password or "")
                self.send_command(*auth_args)
                reply = self.read_response(disable_decoding=True)
                if isinstance(reply, ResponseError):
                    raise AuthenticationError(reply.message)
            if self.client_name:
                self.send_command("CLIENT SETNAME", self.client_name)
                reply = self.read_response(disable_decoding=True)
                if isinstance(reply, ResponseError):
                    raise ConnectionError(reply.message)
        if self.db:
            self.send_command("SELECT", str(self.db))
            reply = self.read_response(disable_decoding=True)
            if isinstance(reply, ResponseError):
                raise ConnectionError(reply.message)
        if self.readonly:
            self.send_command("READONLY")
            reply = self.read_response(disable_decoding=True)
            if isinstance(reply, ResponseError):
                raise ConnectionError(reply.message)

    # -- request / reply --------------------------------------------------------

    def send_command(self, command_name: str, *args: Any) -> None:
        """Encode and write one command. Connects lazily if needed."""
        self.send_packed_command(encode_command(build_args(command_name, *args)))

    def send_packed_command(self, payload: bytes) -> None:
        """Write already RESP-encoded bytes (used by pipelines to batch a
        whole round of commands into a single write)."""
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        try:
            self._sock.sendall(payload)
        except socket.timeout as exc:
            self.disconnect()
            raise TimeoutError(f"timed out writing to {self._target()}") from exc
        except OSError as exc:
            self.disconnect()
            raise ConnectionError(f"error writing to {self._target()}: {exc}") from exc

    def read_response(self, disable_decoding: bool = False) -> Any:
        """Read and parse the next reply. Returns error replies as
        `ResponseError` instances rather than raising them; callers decide."""
        if self._sock is None or self._reader is None:
            raise ConnectionError("not connected")
        while True:
            value = self._reader.gets()
            if value is not NEED_MORE:
                break
            self._fill_buffer()
        if self.decode_responses and not disable_decoding:
            value = decode_value(value, self.encoding, self.encoding_errors)
        return value

    def _fill_buffer(self) -> None:
        assert self._sock is not None and self._reader is not None
        try:
            chunk = self._sock.recv(_RECV_SIZE)
        except socket.timeout as exc:
            raise TimeoutError(f"timed out reading from {self._target()}") from exc
        except OSError as exc:
            self.disconnect()
            raise ConnectionError(f"error reading from {self._target()}: {exc}") from exc
        if not chunk:
            self.disconnect()
            raise ConnectionError(f"connection closed by {self._target()}")
        self._reader.feed(chunk)

    def can_read(self, timeout: Optional[float] = 0.0) -> bool:
        """Non-blocking (or short-timeout) check for a fully buffered reply.
        Used by pub/sub polling and pool idle checks."""
        if self._sock is None or self._reader is None:
            return False
        if self._reader.has_next():
            return True
        selector = selectors.DefaultSelector()
        try:
            selector.register(self._sock, selectors.EVENT_READ)
            if not selector.select(timeout):
                return False
        finally:
            selector.close()
        try:
            chunk = self._sock.recv(_RECV_SIZE)
        except (OSError, socket.timeout):
            return False
        if not chunk:
            self.disconnect()
            return False
        self._reader.feed(chunk)
        return self._reader.has_next()

    def ping(self) -> bool:
        self.send_command("PING")
        reply = self.read_response(disable_decoding=True)
        if isinstance(reply, ResponseError):
            raise reply
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "connected" if self.connected else "disconnected"
        return f"<hotkv.Connection {self._target()} proto={self.protocol_version} {state}>"
