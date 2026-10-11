# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""RESP2/RESP3 wire encoding and an incremental, streaming reply parser.

`Reader` is fed raw bytes as they arrive (`feed()`) and yields fully parsed
replies one at a time (`gets()`), returning the `NEED_MORE` sentinel when
the buffered bytes do not yet contain a complete reply. This lets the exact
same parser drive a blocking socket connection and an asyncio stream
connection: both simply loop "read a chunk, feed it, drain with `gets()`".

`encode_command()` turns an already flattened argument list into the RESP
array wire format used to send commands; `build_args()` is a small helper
that splits a (possibly two-word, e.g. "CONFIG GET") command name into its
wire tokens ahead of the rest of the arguments.
"""

from __future__ import annotations

import numbers
from typing import Any, Dict, List, Optional, Sequence

from ._numbers import format_float
from .exceptions import DataError, ProtocolError, ResponseError, parse_error

__all__ = [
    "NEED_MORE",
    "Reader",
    "PushMessage",
    "VerbatimString",
    "build_args",
    "encode_command",
    "encode_arg",
    "decode_value",
]

_MAX_DEPTH = 64
_MAX_BULK = 512 * 1024 * 1024  # 512 MiB, matches the server's own limit.
_MAX_ELEMENTS = 1 << 20


class _NeedMore(Exception):
    """Internal control-flow signal: the buffer does not hold a full reply."""


NEED_MORE = object()
"""Sentinel returned by `Reader.gets()` when a reply is not fully buffered."""


class PushMessage(list):
    """A RESP3 out-of-band push frame (wire type `>`), used for pub/sub."""


class VerbatimString(bytes):
    """A RESP3 verbatim string (wire type `=`): bytes plus a format tag.

    `format` holds the 3 character encoding hint the server sent (for
    example "txt" or "mkd"); the instance itself is the content with that
    prefix stripped.
    """

    def __new__(cls, fmt: str, content: bytes) -> "VerbatimString":
        obj = super().__new__(cls, content)
        obj.format = fmt
        return obj


class Reader:
    """Incremental RESP2/RESP3 reply parser. Not thread safe; one per connection."""

    __slots__ = ("_buf", "_pos")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._pos = 0

    def feed(self, data: bytes) -> None:
        """Append freshly read bytes to the internal buffer."""
        if data:
            self._buf += data

    def gets(self) -> Any:
        """Return the next parsed reply, or `NEED_MORE` if not enough is buffered.

        On `NEED_MORE`, no bytes are consumed: feed more data and call
        `gets()` again.
        """
        start = self._pos
        try:
            value = self._read(0)
        except _NeedMore:
            self._pos = start
            return NEED_MORE
        if self._pos > 65536:
            del self._buf[: self._pos]
            self._pos = 0
        return value

    def has_next(self) -> bool:
        """Non-destructively check whether a full reply is already buffered.

        Used for non-blocking polling (pub/sub, pool idle checks) without
        consuming the reply; a later `gets()` still returns it.
        """
        start = self._pos
        try:
            self._read(0)
        except _NeedMore:
            self._pos = start
            return False
        self._pos = start
        return True

    # -- low level buffer helpers -------------------------------------------

    def _need(self, n: int) -> None:
        if self._pos + n > len(self._buf):
            raise _NeedMore()

    def _line(self) -> bytes:
        end = self._buf.find(b"\r\n", self._pos)
        if end == -1:
            raise _NeedMore()
        line = bytes(self._buf[self._pos : end])
        self._pos = end + 2
        return line

    def _blob(self, length: int) -> bytes:
        if length < 0 or length > _MAX_BULK:
            raise ProtocolError(f"bulk length {length} out of range")
        self._need(length + 2)
        data = bytes(self._buf[self._pos : self._pos + length])
        if bytes(self._buf[self._pos + length : self._pos + length + 2]) != b"\r\n":
            raise ProtocolError("missing CRLF terminator after bulk payload")
        self._pos += length + 2
        return data

    # -- type dispatch --------------------------------------------------------

    def _read(self, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            raise ProtocolError("RESP reply nested too deep")
        self._need(1)
        marker = bytes(self._buf[self._pos : self._pos + 1])
        self._pos += 1
        handler = _HANDLERS.get(marker)
        if handler is None:
            raise ProtocolError(f"unrecognized RESP type byte {marker!r}")
        return handler(self, depth)

    def _simple_string(self, depth: int) -> str:
        return self._line().decode("utf-8", "replace")

    def _error(self, depth: int) -> ResponseError:
        return parse_error(self._line().decode("utf-8", "replace"))

    def _integer(self, depth: int) -> int:
        return int(self._line())

    def _bulk_string(self, depth: int) -> Optional[bytes]:
        length = int(self._line())
        if length == -1:
            return None
        return self._blob(length)

    def _array(self, depth: int) -> Optional[list]:
        count = int(self._line())
        if count == -1:
            return None
        if count < 0 or count > _MAX_ELEMENTS:
            raise ProtocolError(f"array length {count} out of range")
        return [self._read(depth + 1) for _ in range(count)]

    def _null(self, depth: int) -> None:
        self._line()
        return None

    def _double(self, depth: int) -> float:
        raw = self._line().decode("ascii")
        if raw == "inf":
            return float("inf")
        if raw == "-inf":
            return float("-inf")
        if raw == "nan":
            return float("nan")
        return float(raw)

    def _boolean(self, depth: int) -> bool:
        raw = self._line()
        if raw == b"t":
            return True
        if raw == b"f":
            return False
        raise ProtocolError(f"invalid RESP3 boolean {raw!r}")

    def _blob_error(self, depth: int) -> ResponseError:
        length = int(self._line())
        return parse_error(self._blob(length).decode("utf-8", "replace"))

    def _verbatim_string(self, depth: int) -> VerbatimString:
        length = int(self._line())
        data = self._blob(length)
        if len(data) >= 4 and data[3:4] == b":":
            return VerbatimString(data[:3].decode("ascii"), data[4:])
        return VerbatimString("txt", data)

    def _big_number(self, depth: int) -> int:
        return int(self._line())

    def _map(self, depth: int) -> dict:
        count = int(self._line())
        if count < 0 or count > _MAX_ELEMENTS:
            raise ProtocolError(f"map length {count} out of range")
        result: Dict[Any, Any] = {}
        for _ in range(count):
            key = self._read(depth + 1)
            result[key] = self._read(depth + 1)
        return result

    def _set(self, depth: int) -> set:
        count = int(self._line())
        if count < 0 or count > _MAX_ELEMENTS:
            raise ProtocolError(f"set length {count} out of range")
        return {self._read(depth + 1) for _ in range(count)}

    def _attribute(self, depth: int) -> Any:
        count = int(self._line())
        if count < 0 or count > _MAX_ELEMENTS:
            raise ProtocolError(f"attribute length {count} out of range")
        for _ in range(count):
            self._read(depth + 1)
            self._read(depth + 1)
        # An attribute precedes the real reply; parse and return that one.
        return self._read(depth)

    def _push(self, depth: int) -> PushMessage:
        count = int(self._line())
        if count < 0 or count > _MAX_ELEMENTS:
            raise ProtocolError(f"push length {count} out of range")
        return PushMessage(self._read(depth + 1) for _ in range(count))


_HANDLERS = {
    b"+": Reader._simple_string,
    b"-": Reader._error,
    b":": Reader._integer,
    b"$": Reader._bulk_string,
    b"*": Reader._array,
    b"_": Reader._null,
    b",": Reader._double,
    b"#": Reader._boolean,
    b"!": Reader._blob_error,
    b"=": Reader._verbatim_string,
    b"(": Reader._big_number,
    b"%": Reader._map,
    b"~": Reader._set,
    b"|": Reader._attribute,
    b">": Reader._push,
}


def build_args(command_name: str, *args: Any) -> List[Any]:
    """Split a command name on spaces (so "CONFIG GET" becomes two wire
    tokens) and append the remaining arguments."""
    if " " in command_name:
        tokens: List[Any] = command_name.split(" ")
        tokens.extend(args)
        return tokens
    return [command_name, *args]


def encode_arg(arg: Any) -> bytes:
    if isinstance(arg, (bytes, bytearray, memoryview)):
        return bytes(arg)
    if isinstance(arg, str):
        return arg.encode("utf-8")
    if isinstance(arg, bool):
        raise DataError(
            f"refusing to encode bool {arg!r} as a command argument; pass an "
            "explicit 0/1 integer or the matching keyword token instead"
        )
    if isinstance(arg, int):
        return b"%d" % arg
    if isinstance(arg, float):
        # Plain decimal, never an exponent; an infinity goes out as +inf/-inf
        # (a sorted set takes one) and the server refuses it where it must be finite.
        return format_float(arg, allow_inf=True).encode("ascii")
    if isinstance(arg, numbers.Integral):
        return b"%d" % int(arg)
    if isinstance(arg, numbers.Real):  # numpy floats and the like
        return format_float(arg, allow_inf=True).encode("ascii")
    raise DataError(f"unsupported command argument type: {type(arg).__name__}")


def encode_command(args: Sequence[Any]) -> bytes:
    """Encode already-flattened command arguments as a RESP array."""
    parts = [b"*", str(len(args)).encode("ascii"), b"\r\n"]
    for arg in args:
        payload = encode_arg(arg)
        parts.append(b"$")
        parts.append(str(len(payload)).encode("ascii"))
        parts.append(b"\r\n")
        parts.append(payload)
        parts.append(b"\r\n")
    return b"".join(parts)


def decode_value(value: Any, encoding: str, errors: str) -> Any:
    """Recursively decode the `bytes` leaves of a parsed reply to `str`.

    Used when the client is configured with `decode_responses=True`. List
    and push-message structure is preserved; dict keys and values, and set
    members, are decoded as well.
    """
    if isinstance(value, (bytes, bytearray)):
        return value.decode(encoding, errors)
    if isinstance(value, list):
        items = [decode_value(item, encoding, errors) for item in value]
        return PushMessage(items) if isinstance(value, PushMessage) else items
    if isinstance(value, dict):
        return {
            decode_value(k, encoding, errors): decode_value(v, encoding, errors)
            for k, v in value.items()
        }
    if isinstance(value, set):
        return {decode_value(item, encoding, errors) for item in value}
    return value
