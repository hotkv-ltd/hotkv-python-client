# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""List commands, including the blocking pop/move variants."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from .._typing import KeyT, ValueT
from ._shared import CallbackMap, ReadonlySet, cb_ok

__all__ = ["ListCommands", "CALLBACKS", "READONLY"]

CALLBACKS: CallbackMap = {"LSET": cb_ok, "LTRIM": cb_ok}

READONLY: ReadonlySet = frozenset(
    {"LLEN", "LRANGE", "LINDEX", "LPOS"}
)


class ListCommands:
    """Typed wrappers for list commands."""

    def lpush(self, key: KeyT, *values: ValueT) -> int:
        """Push `values` onto the head of the list at `key`, returning the new length."""
        return self.execute_command("LPUSH", key, *values)

    def rpush(self, key: KeyT, *values: ValueT) -> int:
        """Push `values` onto the tail of the list at `key`, returning the new length."""
        return self.execute_command("RPUSH", key, *values)

    def lpushx(self, key: KeyT, *values: ValueT) -> int:
        """Like `lpush()`, but only if `key` already exists and is a list."""
        return self.execute_command("LPUSHX", key, *values)

    def rpushx(self, key: KeyT, *values: ValueT) -> int:
        """Like `rpush()`, but only if `key` already exists and is a list."""
        return self.execute_command("RPUSHX", key, *values)

    def lpop(self, key: KeyT, count: Optional[int] = None) -> Union[bytes, List[bytes], None]:
        """Pop from the head: one value (or `None`), or up to `count` values."""
        args: List[Any] = [key] if count is None else [key, count]
        return self.execute_command("LPOP", *args)

    def rpop(self, key: KeyT, count: Optional[int] = None) -> Union[bytes, List[bytes], None]:
        """Pop from the tail: one value (or `None`), or up to `count` values."""
        args: List[Any] = [key] if count is None else [key, count]
        return self.execute_command("RPOP", *args)

    def llen(self, key: KeyT) -> int:
        """Length of the list at `key` (0 if unset)."""
        return self.execute_command("LLEN", key)

    def lrange(self, key: KeyT, start: int, stop: int) -> List[bytes]:
        """Return elements between `start` and `stop` (inclusive, may be negative)."""
        return self.execute_command("LRANGE", key, start, stop)

    def lindex(self, key: KeyT, index: int) -> Optional[bytes]:
        """Return the element at `index`, or `None` if out of range."""
        return self.execute_command("LINDEX", key, index)

    def lset(self, key: KeyT, index: int, value: ValueT) -> bool:
        """Set the element at `index` to `value`."""
        return self.execute_command("LSET", key, index, value)

    def linsert(self, key: KeyT, where: str, pivot: ValueT, value: ValueT) -> int:
        """Insert `value` before or after `pivot` (`where` is "BEFORE"/"AFTER").

        Returns the new length, `0` if `key` does not exist, or `-1` if
        `pivot` was not found.
        """
        return self.execute_command("LINSERT", key, where.upper(), pivot, value)

    def lrem(self, key: KeyT, count: int, value: ValueT) -> int:
        """Remove occurrences of `value`: `count > 0` head to tail, `< 0` tail
        to head, `0` all of them. Returns the number removed."""
        return self.execute_command("LREM", key, count, value)

    def ltrim(self, key: KeyT, start: int, stop: int) -> bool:
        """Trim the list to only the elements between `start` and `stop`."""
        return self.execute_command("LTRIM", key, start, stop)

    def rpoplpush(self, source: KeyT, destination: KeyT) -> Optional[bytes]:
        """Pop the tail of `source` and push it onto the head of `destination`."""
        return self.execute_command("RPOPLPUSH", source, destination)

    def lmove(
        self, source: KeyT, destination: KeyT, source_side: str, dest_side: str
    ) -> Optional[bytes]:
        """Move one element between lists. `source_side`/`dest_side` are "LEFT"/"RIGHT"."""
        return self.execute_command(
            "LMOVE", source, destination, source_side.upper(), dest_side.upper()
        )

    def lpos(
        self,
        key: KeyT,
        element: ValueT,
        *,
        rank: Optional[int] = None,
        count: Optional[int] = None,
        maxlen: Optional[int] = None,
    ) -> Union[int, List[int], None]:
        """Find the index (or, with `count`, up to `count` indexes) of `element`."""
        args: List[Any] = [key, element]
        if rank is not None:
            args += ["RANK", rank]
        if count is not None:
            args += ["COUNT", count]
        if maxlen is not None:
            args += ["MAXLEN", maxlen]
        return self.execute_command("LPOS", *args)

    # -- blocking variants ------------------------------------------------------

    def blpop(self, keys: Any, timeout: float) -> Optional[List[bytes]]:
        """Block up to `timeout` seconds (0 = forever) for a head element on
        any of `keys`. Returns `[key, value]`, or `None` on timeout."""
        if isinstance(keys, (str, bytes)):
            keys = [keys]
        return self.execute_command("BLPOP", *keys, timeout)

    def brpop(self, keys: Any, timeout: float) -> Optional[List[bytes]]:
        """Like `blpop()`, popping from the tail."""
        if isinstance(keys, (str, bytes)):
            keys = [keys]
        return self.execute_command("BRPOP", *keys, timeout)

    def blmove(
        self,
        source: KeyT,
        destination: KeyT,
        source_side: str,
        dest_side: str,
        timeout: float,
    ) -> Optional[bytes]:
        """Blocking version of `lmove()`."""
        return self.execute_command(
            "BLMOVE", source, destination, source_side.upper(), dest_side.upper(), timeout
        )
