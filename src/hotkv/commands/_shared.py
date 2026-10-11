# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Small, shared building blocks for reply post-processing callbacks.

Every command mixin module builds a `CALLBACKS` dict (command name -> a
`callback(reply, **options)` function) and a `READONLY` set of command
names that are safe to retry once after a dropped connection. These helpers
keep that repetitive plumbing short and consistent across mixins.

Field name coercions here are deliberately opinionated: keys and small
fixed-vocabulary values (an "algo" or "metric" name, an entry count) in a
structured stats/info reply are always normalized to `str`/`int`/`float`
because they are protocol schema, not user data. Actual user content
(a GET value, a cached response, a tag name) is left exactly as the
connection decoded it, respecting `decode_responses`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..exceptions import DataError

__all__ = [
    "as_int",
    "as_float",
    "as_bool",
    "force_text",
    "pairs_to_dict",
    "flat_to_pairs",
    "named_fields",
    "expiry_args",
    "cb_bool",
    "cb_float",
    "cb_ok",
    "cb_pairs_dict",
    "CallbackMap",
    "ReadonlySet",
]

CallbackMap = Dict[str, Callable[..., Any]]
ReadonlySet = frozenset


def as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(value)


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    return float(value)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return bool(as_int(value))


def force_text(value: Any) -> Any:
    """Decode bytes to str unconditionally; used for protocol schema
    tokens (field names, enum values) that are never arbitrary binary data."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return value


def pairs_to_dict(flat: Any) -> Dict[Any, Any]:
    """`['a', 1, 'b', 2]` -> `{'a': 1, 'b': 2}`, keys and values untouched.

    Also accepts a reply already shaped as a native RESP3 map (a `dict`,
    e.g. `HGETALL`/`XINFO STREAM` under `HELLO 3`) and returns it as-is
    (copied), so callers never need to know which wire shape the
    connection's negotiated protocol version produced.
    """
    if isinstance(flat, dict):
        return dict(flat)
    if not flat:
        return {}
    it = iter(flat)
    return dict(zip(it, it))


def flat_to_pairs(flat: Any) -> List[Tuple[Any, Any]]:
    """`['a', 1, 'b', 2]` -> `[('a', 1), ('b', 2)]`, order preserved.

    Also accepts a reply already shaped as RESP3's native "array of
    pairs" (e.g. `ZRANGE ... WITHSCORES`/`HRANDFIELD ... WITHVALUES`
    under `HELLO 3`, each element already a 2-element list/tuple) and
    normalizes it the same way, so callers never need to know which wire
    shape the connection's negotiated protocol version produced.
    """
    items = list(flat) if flat else []
    if items and isinstance(items[0], (list, tuple)):
        return [(pair[0], pair[1]) for pair in items]
    it = iter(items)
    return list(zip(it, it))


def named_fields(
    flat: Any,
    *,
    floats: Iterable[str] = (),
    texts: Iterable[str] = (),
) -> Dict[str, Any]:
    """Build a `{field_name: value}` dict from a flat reply of alternating
    `name, value` pairs. `name` is always force-decoded to `str`. Fields
    listed in `floats` are coerced with `as_float`, fields in `texts` are
    force-decoded to `str`; every other scalar field defaults to `as_int`
    (the common shape for these fixed-schema STATS/INFO replies). A value
    that is already a list/dict/set (nested structure) is left untouched.

    A field name this call does not know about (an unlisted, unreleased
    server field) is never dropped, but is also never forced through
    `as_int`: if that would raise, the raw value is kept as-is instead, so
    an unknown non-numeric field cannot break parsing. This is what makes
    every `named_fields()` based reply forward compatible with a server
    that has added fields: callers still see the
    new field (by name), just without a coercion guess for it.

    Also accepts a reply already shaped as a native RESP3 map (a `dict`,
    e.g. `XINFO STREAM`/`XINFO GROUPS` under `HELLO 3`).
    """
    floats_set = set(floats)
    texts_set = set(texts)
    result: Dict[str, Any] = {}
    if not flat:
        return result
    if isinstance(flat, dict):
        items = flat.items()
    else:
        it = iter(flat)
        items = zip(it, it)
    for raw_key, value in items:
        key = force_text(raw_key)
        if key in floats_set:
            value = as_float(value)
        elif key in texts_set:
            value = force_text(value)
        elif not isinstance(value, (list, dict, set)):
            try:
                value = as_int(value)
            except (TypeError, ValueError):
                pass
        result[key] = value
    return result


def expiry_args(ttl: Optional[int], pxat: Optional[int]) -> List[Any]:
    """The `TTL seconds` or `PXAT unix-ms` option of an AI engine write: `ttl` is
    seconds from now, `pxat` the absolute Unix time in milliseconds the entry
    expires at (a time in the past is accepted and the entry is already
    expired; more than 100 years ahead is refused by the server). The server
    refuses both at once, so the client does too."""
    if ttl is not None and pxat is not None:
        raise DataError("ttl and pxat cannot both be given")
    if ttl is not None:
        return ["TTL", ttl]
    if pxat is not None:
        return ["PXAT", pxat]
    return []


def cb_bool(reply: Any, **_options: Any) -> bool:
    return as_bool(reply)


def cb_float(reply: Any, **_options: Any) -> Optional[float]:
    return as_float(reply)


def cb_ok(reply: Any, **_options: Any) -> bool:
    return reply == "OK" or reply is True


def cb_pairs_dict(reply: Any, **_options: Any) -> Dict[Any, Any]:
    return pairs_to_dict(reply)
