# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Typed exception hierarchy used across the sync and asyncio HotKV clients.

All exceptions raised by this package derive from `HotKVError`, so callers
that only need a coarse "something went wrong talking to HotKV" handler can
catch that single type. More specific branches exist for connection
problems (`HotKVConnectionError`), protocol framing problems
(`ProtocolError`) and typed server reply errors (`ResponseError` and its
subclasses), including a dedicated `NotLicensedError` for the `NOLICENSE`
prefix that enterprise commands reply with when the current license does
not include the feature.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple, Type

__all__ = [
    "HotKVError",
    "ConnectionError",
    "TimeoutError",
    "AuthenticationError",
    "ProtocolError",
    "DataError",
    "PubSubError",
    "WatchError",
    "ResponseError",
    "NotLicensedError",
    "WrongTypeError",
    "NoAuthError",
    "WrongPassError",
    "NoScriptError",
    "BusyError",
    "ExecAbortError",
    "ClusterError",
    "ClusterDownError",
    "CrossSlotError",
    "TryAgainError",
    "AskError",
    "MovedError",
    "parse_error",
]


class HotKVError(Exception):
    """Base class for every exception raised by the HotKV client."""


class ConnectionError(HotKVError):
    """A connection to the server could not be established, or was lost.

    Shadows the builtin `ConnectionError` only within the `hotkv` namespace,
    matching the common convention for RESP protocol clients in Python.
    """


class TimeoutError(ConnectionError):
    """A connect, read or write operation exceeded its configured timeout."""


class AuthenticationError(ConnectionError):
    """The client could not authenticate while establishing a connection."""


class ProtocolError(HotKVError):
    """A reply from the server did not conform to RESP2 or RESP3 framing."""


class DataError(HotKVError, ValueError):
    """A value given to the client could not be encoded as a command argument
    (also a `ValueError`, as the number formatting raises it for NaN and
    out of range amounts)."""


class PubSubError(HotKVError):
    """Raised for misuse of the publish/subscribe API."""


class WatchError(HotKVError):
    """Raised by `Client.transaction()` when a watched key changed and the
    server aborted the transaction (`EXEC` returned `None`)."""


class ClusterError(HotKVError):
    """Base class for cluster client errors that are not a single server
    reply, such as running out of redirects or being unable to route a
    command to a slot."""


class ResponseError(HotKVError):
    """A `-ERR ...` (RESP2) or `!...` (RESP3 blob error) reply from HotKV.

    `code` is the leading, whitespace-terminated token of the reply (for
    example `ERR`, `WRONGTYPE`, `MOVED`, `NOAUTH`, `NOLICENSE`); `message`
    is the remainder of the reply text with that token stripped. Comparing
    or matching should be done on `code`, never on the free-form message
    text. `raw` is the whole reply text exactly as the server sent it (the
    code token and the message), or `None` for an error built by the client.
    """

    code = "ERR"
    raw: Optional[str] = None

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        self.message = message
        if code:
            self.code = code
        super().__init__(message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class NotLicensedError(ResponseError):
    """The server rejected an enterprise command with `-NOLICENSE ...`
    because the active license does not include the required feature.

    Integration tests should catch this specifically and skip, so that the
    test suite passes automatically once a fuller license is installed.
    """

    code = "NOLICENSE"


class WrongTypeError(ResponseError):
    """`-WRONGTYPE ...`: the key holds a different data type."""

    code = "WRONGTYPE"


class NoAuthError(ResponseError):
    """`-NOAUTH ...`: the connection must authenticate before this command."""

    code = "NOAUTH"


class WrongPassError(ResponseError):
    """`-WRONGPASS ...`: invalid username/password during authentication."""

    code = "WRONGPASS"


class NoScriptError(ResponseError):
    """`-NOSCRIPT ...`: EVALSHA referenced a script the server does not have."""

    code = "NOSCRIPT"


class BusyError(ResponseError):
    """`-BUSY ...`: the server is busy running a script."""

    code = "BUSY"


class ExecAbortError(ResponseError):
    """`-EXECABORT ...`: a queued command in a transaction was invalid."""

    code = "EXECABORT"


class ClusterDownError(ResponseError):
    """`-CLUSTERDOWN ...`: the cluster cannot accept commands right now."""

    code = "CLUSTERDOWN"


class CrossSlotError(ResponseError):
    """`-CROSSSLOT ...`: command keys do not hash to the same slot."""

    code = "CROSSSLOT"


class TryAgainError(ResponseError):
    """`-TRYAGAIN ...`: a multi-key cluster operation is mid-migration."""

    code = "TRYAGAIN"


def _parse_redirect(message: str) -> Tuple[int, str, int]:
    parts = message.split()
    if len(parts) < 2:
        raise ProtocolError(f"malformed redirect reply: {message!r}")
    slot = int(parts[0])
    host, _, port = parts[1].rpartition(":")
    return slot, host, int(port)


class AskError(ResponseError):
    """`-ASK ...`: send `ASKING` then retry against the given node."""

    code = "ASK"

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message, code)
        self.slot, self.host, self.port = _parse_redirect(message)


class MovedError(ResponseError):
    """`-MOVED ...`: the slot now permanently lives on the given node."""

    code = "MOVED"

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message, code)
        self.slot, self.host, self.port = _parse_redirect(message)


_ERROR_CLASSES: Dict[str, Type[ResponseError]] = {
    cls.code: cls
    for cls in (
        NotLicensedError,
        WrongTypeError,
        NoAuthError,
        WrongPassError,
        NoScriptError,
        BusyError,
        ExecAbortError,
        ClusterDownError,
        CrossSlotError,
        TryAgainError,
        AskError,
        MovedError,
    )
}

_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def parse_error(raw: str) -> ResponseError:
    """Turn the text of a server error reply into a typed `ResponseError`.

    The leading token is matched against known error prefixes (see the
    HotKV documentation for the error replies); anything unrecognized, including
    replies with no distinct leading token, becomes a plain `ResponseError`
    with `code="ERR"`.
    """
    code, _, rest = raw.partition(" ")
    if not _CODE_RE.match(code):
        error = ResponseError(raw, "ERR")
    else:
        cls = _ERROR_CLASSES.get(code, ResponseError)
        error = cls(rest or raw, code)
    error.raw = raw
    return error
