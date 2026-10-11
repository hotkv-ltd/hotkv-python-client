# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Connection URL parsing, shared by the sync and asyncio `from_url()`.

Supported schemes: `hotkv://` and `hotkvs://` (TLS); `redis://` and
`rediss://` are accepted as well so a Redis connection string can be reused
as-is when migrating. `unix://` addresses a Unix domain socket.
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Dict

__all__ = ["parse_url"]

_TLS_SCHEMES = {"hotkvs", "rediss"}
_UNIX_SCHEMES = {"unix"}
_TCP_SCHEMES = {"hotkv", "redis"}
_ALL_SCHEMES = _TLS_SCHEMES | _UNIX_SCHEMES | _TCP_SCHEMES

_DEFAULT_PORT = 6379
_TRUE_STRINGS = {"1", "true", "yes", "on"}


def _as_bool(value: str) -> bool:
    return value.strip().lower() in _TRUE_STRINGS


def parse_url(url: str) -> Dict[str, Any]:
    """Parse a connection URL into client constructor keyword arguments.

    Recognized query parameters: `db`, `username`, `password`,
    `client_name`, `connect_timeout`, `command_timeout`, `tls_ca_file`,
    `tls_certfile`, `tls_keyfile`, `tls_server_hostname`,
    `tls_insecure_skip_verify`. Raises `ValueError` for an unknown scheme or
    a non-numeric database index.
    """
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _ALL_SCHEMES:
        raise ValueError(
            f"unsupported HotKV URL scheme {parts.scheme!r}; expected one of "
            f"{sorted(_ALL_SCHEMES)}"
        )

    query = {k: v[-1] for k, v in urllib.parse.parse_qs(parts.query).items()}
    options: Dict[str, Any] = {}

    if scheme in _UNIX_SCHEMES:
        options["unix_socket_path"] = urllib.parse.unquote(parts.path)
    else:
        options["host"] = parts.hostname or "localhost"
        options["port"] = parts.port or _DEFAULT_PORT
        options["tls"] = scheme in _TLS_SCHEMES
        db_part = parts.path.lstrip("/") if parts.path else ""
        if db_part:
            try:
                options["db"] = int(db_part)
            except ValueError as exc:
                raise ValueError(f"invalid database index in URL path: {db_part!r}") from exc

    if parts.username:
        options["username"] = urllib.parse.unquote(parts.username)
    if parts.password:
        options["password"] = urllib.parse.unquote(parts.password)

    if "db" in query:
        options["db"] = int(query["db"])
    if "username" in query:
        options["username"] = query["username"]
    if "password" in query:
        options["password"] = query["password"]
    if "client_name" in query:
        options["client_name"] = query["client_name"]
    if "connect_timeout" in query:
        options["connect_timeout"] = float(query["connect_timeout"])
    if "command_timeout" in query:
        options["command_timeout"] = float(query["command_timeout"])
    if "tls_ca_file" in query:
        options["tls_ca_file"] = query["tls_ca_file"]
        options["tls"] = True
    if "tls_certfile" in query:
        options["tls_certfile"] = query["tls_certfile"]
        options["tls"] = True
    if "tls_keyfile" in query:
        options["tls_keyfile"] = query["tls_keyfile"]
        options["tls"] = True
    if "tls_server_hostname" in query:
        options["tls_server_hostname"] = query["tls_server_hostname"]
    if "tls_insecure_skip_verify" in query:
        options["tls_insecure_skip_verify"] = _as_bool(query["tls_insecure_skip_verify"])

    return options
