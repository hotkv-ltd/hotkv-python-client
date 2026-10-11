# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""String commands (GET/SET and friends).

Every method here just builds a command and returns `self.execute_command(...)`
with no `async`/`await`: `execute_command` is sync on the sync client and a
coroutine function on the asyncio client, so this exact same mixin serves
both (the asyncio client's method call returns the coroutine object, which
the caller awaits).
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Union

from .._typing import KeyT, KeysT, ValueT
from ._shared import CallbackMap, ReadonlySet, as_float, cb_ok

__all__ = ["StringCommands", "CALLBACKS", "READONLY"]


def _cb_set(reply: Any, get: bool = False, **_options: Any) -> Union[bool, Optional[bytes]]:
    # With GET, the reply is the previous value (or nil if unset), never
    # "OK". Without it, NX/XX reply null (not OK) when the condition was
    # not met.
    if get:
        return reply
    return reply is not None and (reply == "OK" or reply is True)


CALLBACKS: CallbackMap = {
    "SET": _cb_set,
    "SETNX": lambda reply, **_: bool(reply),
    "SETEX": cb_ok,
    "PSETEX": cb_ok,
    "MSET": cb_ok,
    "MSETNX": lambda reply, **_: bool(reply),
    "INCRBYFLOAT": lambda reply, **_: as_float(reply),
}

READONLY: ReadonlySet = frozenset(
    {"GET", "MGET", "STRLEN", "GETRANGE", "SUBSTR"}
)


class StringCommands:
    """Typed wrappers for the GET/SET family of commands."""

    def get(self, key: KeyT) -> Optional[bytes]:
        """Return the value of `key`, or `None` if it does not exist."""
        return self.execute_command("GET", key)

    def set(
        self,
        key: KeyT,
        value: ValueT,
        *,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        exat: Optional[int] = None,
        pxat: Optional[int] = None,
        keepttl: bool = False,
        nx: bool = False,
        xx: bool = False,
        get: bool = False,
    ) -> Union[bool, Optional[bytes]]:
        """Set `key` to `value`.

        Returns `True` on success, `False` when `nx`/`xx` prevented the
        write; with `get=True`, instead returns the previous value (or
        `None` if `key` did not exist), regardless of whether the write
        itself happened. `ex`/`px`/`exat`/`pxat` set an expiry (relative
        seconds/milliseconds, or an absolute Unix timestamp in seconds/
        milliseconds); `keepttl` keeps the key's current TTL instead of
        clearing it (mutually exclusive with the other four). `nx`/`xx`
        are mutually exclusive (only write if the key is absent/present).
        """
        args: List[Any] = [key, value]
        if ex is not None:
            args += ["EX", ex]
        if px is not None:
            args += ["PX", px]
        if exat is not None:
            args += ["EXAT", exat]
        if pxat is not None:
            args += ["PXAT", pxat]
        if keepttl:
            args.append("KEEPTTL")
        if nx:
            args.append("NX")
        if xx:
            args.append("XX")
        if get:
            args.append("GET")
        return self.execute_command("SET", *args, get=get)

    def setnx(self, key: KeyT, value: ValueT) -> bool:
        """Set `key` to `value` only if `key` does not already exist."""
        return self.execute_command("SETNX", key, value)

    def setex(self, key: KeyT, seconds: int, value: ValueT) -> bool:
        """Set `key` to `value` with an expiry of `seconds`."""
        return self.execute_command("SETEX", key, seconds, value)

    def psetex(self, key: KeyT, milliseconds: int, value: ValueT) -> bool:
        """Set `key` to `value` with an expiry of `milliseconds`."""
        return self.execute_command("PSETEX", key, milliseconds, value)

    def getset(self, key: KeyT, value: ValueT) -> Optional[bytes]:
        """Set `key` to `value`, returning the previous value."""
        return self.execute_command("GETSET", key, value)

    def getdel(self, key: KeyT) -> Optional[bytes]:
        """Return the value of `key` and delete it in one round trip."""
        return self.execute_command("GETDEL", key)

    def getex(
        self,
        key: KeyT,
        *,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        exat: Optional[int] = None,
        pxat: Optional[int] = None,
        persist: bool = False,
    ) -> Optional[bytes]:
        """Return the value of `key`, optionally updating its expiry."""
        args: List[Any] = [key]
        if ex is not None:
            args += ["EX", ex]
        elif px is not None:
            args += ["PX", px]
        elif exat is not None:
            args += ["EXAT", exat]
        elif pxat is not None:
            args += ["PXAT", pxat]
        elif persist:
            args.append("PERSIST")
        return self.execute_command("GETEX", *args)

    def append(self, key: KeyT, value: ValueT) -> int:
        """Append `value` to `key`, returning the new length."""
        return self.execute_command("APPEND", key, value)

    def strlen(self, key: KeyT) -> int:
        """Return the length of the string stored at `key` (0 if unset)."""
        return self.execute_command("STRLEN", key)

    def incr(self, key: KeyT) -> int:
        """Increment the integer value of `key` by one."""
        return self.execute_command("INCR", key)

    def decr(self, key: KeyT) -> int:
        """Decrement the integer value of `key` by one."""
        return self.execute_command("DECR", key)

    def incrby(self, key: KeyT, amount: int) -> int:
        """Increment the integer value of `key` by `amount`."""
        return self.execute_command("INCRBY", key, amount)

    def decrby(self, key: KeyT, amount: int) -> int:
        """Decrement the integer value of `key` by `amount`."""
        return self.execute_command("DECRBY", key, amount)

    def incrbyfloat(self, key: KeyT, amount: float) -> float:
        """Increment the float value of `key` by `amount`."""
        return self.execute_command("INCRBYFLOAT", key, amount)

    def mget(self, keys: KeysT) -> List[Optional[bytes]]:
        """Return the values of several keys, `None` for any that are unset."""
        return self.execute_command("MGET", *keys)

    def mset(self, mapping: Mapping[KeyT, ValueT]) -> bool:
        """Set several key/value pairs atomically."""
        args: List[Any] = []
        for key, value in mapping.items():
            args += [key, value]
        return self.execute_command("MSET", *args)

    def msetnx(self, mapping: Mapping[KeyT, ValueT]) -> bool:
        """Set several key/value pairs only if none of the keys already exist."""
        args: List[Any] = []
        for key, value in mapping.items():
            args += [key, value]
        return self.execute_command("MSETNX", *args)

    def setrange(self, key: KeyT, offset: int, value: ValueT) -> int:
        """Overwrite part of a string starting at `offset`, returning the new length."""
        return self.execute_command("SETRANGE", key, offset, value)

    def getrange(self, key: KeyT, start: int, end: int) -> bytes:
        """Return the substring of `key`'s value between `start` and `end` (inclusive)."""
        return self.execute_command("GETRANGE", key, start, end)
