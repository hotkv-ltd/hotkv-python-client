# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""SCAN family: one page per call. Cursor-looping `*_iter()` convenience
generators live on the concrete sync/asyncio clients (a `for`/`async for`
loop cannot be shared through the plain-`def` mixin trick)."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .._typing import KeyT
from ._shared import CallbackMap, ReadonlySet, as_float, flat_to_pairs, pairs_to_dict

__all__ = ["ScanCommands", "CALLBACKS", "READONLY"]


def _cb_scan(reply: Any, **_options: Any) -> Tuple[int, List[bytes]]:
    cursor, items = reply
    return (int(cursor), items)


def _cb_hscan(reply: Any, novalues: bool = False, **_options: Any) -> Tuple[int, Any]:
    cursor, items = reply
    return (int(cursor), items if novalues else pairs_to_dict(items))


def _cb_zscan(reply: Any, **_options: Any) -> Tuple[int, List[Tuple[Any, float]]]:
    cursor, items = reply
    return (int(cursor), [(m, as_float(s)) for m, s in flat_to_pairs(items)])


CALLBACKS: CallbackMap = {
    "SCAN": _cb_scan,
    "SSCAN": _cb_scan,
    "HSCAN": _cb_hscan,
    "ZSCAN": _cb_zscan,
}

READONLY: ReadonlySet = frozenset({"SCAN", "SSCAN", "HSCAN", "ZSCAN"})


class ScanCommands:
    """Typed wrappers for one page of SCAN/HSCAN/SSCAN/ZSCAN."""

    def scan(
        self,
        cursor: int = 0,
        *,
        match: Optional[str] = None,
        count: Optional[int] = None,
        type: Optional[str] = None,
    ) -> Tuple[int, List[bytes]]:
        """Scan the keyspace. Returns `(next_cursor, keys)`; `next_cursor == 0` means done."""
        args: List[Any] = [cursor]
        if match is not None:
            args += ["MATCH", match]
        if count is not None:
            args += ["COUNT", count]
        if type is not None:
            args += ["TYPE", type]
        return self.execute_command("SCAN", *args)

    def sscan(
        self,
        key: KeyT,
        cursor: int = 0,
        *,
        match: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Tuple[int, List[bytes]]:
        """Scan the set at `key`. Returns `(next_cursor, members)`."""
        args: List[Any] = [key, cursor]
        if match is not None:
            args += ["MATCH", match]
        if count is not None:
            args += ["COUNT", count]
        return self.execute_command("SSCAN", *args)

    def hscan(
        self,
        key: KeyT,
        cursor: int = 0,
        *,
        match: Optional[str] = None,
        count: Optional[int] = None,
        novalues: bool = False,
    ) -> Tuple[int, Any]:
        """Scan the hash at `key`. Returns `(next_cursor, {field: value})`,
        or `(next_cursor, [field, ...])` when `novalues=True`."""
        args: List[Any] = [key, cursor]
        if match is not None:
            args += ["MATCH", match]
        if count is not None:
            args += ["COUNT", count]
        if novalues:
            args.append("NOVALUES")
        return self.execute_command("HSCAN", *args, novalues=novalues)

    def zscan(
        self,
        key: KeyT,
        cursor: int = 0,
        *,
        match: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Tuple[int, List[Tuple[Any, float]]]:
        """Scan the sorted set at `key`. Returns `(next_cursor, [(member, score), ...])`."""
        args: List[Any] = [key, cursor]
        if match is not None:
            args += ["MATCH", match]
        if count is not None:
            args += ["COUNT", count]
        return self.execute_command("ZSCAN", *args)
