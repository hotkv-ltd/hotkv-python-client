# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Generic key commands: existence, type, renaming, and expiry."""

from __future__ import annotations

from typing import Any, List, Optional

from .._typing import KeyT
from ._shared import CallbackMap, ReadonlySet, cb_bool, cb_ok

__all__ = ["KeyCommands", "CALLBACKS", "READONLY"]

CALLBACKS: CallbackMap = {
    "EXPIRE": cb_bool,
    "PEXPIRE": cb_bool,
    "EXPIREAT": cb_bool,
    "PEXPIREAT": cb_bool,
    "PERSIST": cb_bool,
    "RENAMENX": cb_bool,
    "COPY": cb_bool,
    "MOVE": cb_bool,
    "RENAME": cb_ok,
    "RESTORE": cb_ok,
}

READONLY: ReadonlySet = frozenset(
    {
        "EXISTS",
        "TYPE",
        "TTL",
        "PTTL",
        "EXPIRETIME",
        "PEXPIRETIME",
        "KEYS",
        "RANDOMKEY",
        "DUMP",
    }
)


class KeyCommands:
    """Typed wrappers for generic key management and expiry commands."""

    def delete(self, *keys: KeyT) -> int:
        """Delete one or more keys, returning the number actually removed."""
        return self.execute_command("DEL", *keys)

    def unlink(self, *keys: KeyT) -> int:
        """Like `delete()`, but the server reclaims memory asynchronously."""
        return self.execute_command("UNLINK", *keys)

    def exists(self, *keys: KeyT) -> int:
        """Count how many of `keys` exist (a key repeated twice counts twice)."""
        return self.execute_command("EXISTS", *keys)

    def type(self, key: KeyT) -> str:
        """Return the type name stored at `key` ("string", "list", ... or "none")."""
        return self.execute_command("TYPE", key)

    def touch(self, *keys: KeyT) -> int:
        """Update the last-access time of `keys`, returning how many exist."""
        return self.execute_command("TOUCH", *keys)

    def rename(self, key: KeyT, new_key: KeyT) -> bool:
        """Rename `key` to `new_key`. Raises if `key` does not exist."""
        return self.execute_command("RENAME", key, new_key)

    def renamenx(self, key: KeyT, new_key: KeyT) -> bool:
        """Rename `key` to `new_key` only if `new_key` does not already exist."""
        return self.execute_command("RENAMENX", key, new_key)

    def copy(
        self, source: KeyT, destination: KeyT, *, db: Optional[int] = None, replace: bool = False
    ) -> bool:
        """Copy `source` to `destination`."""
        args: List[Any] = [source, destination]
        if db is not None:
            args += ["DB", db]
        if replace:
            args.append("REPLACE")
        return self.execute_command("COPY", *args)

    def move(self, key: KeyT, db: int) -> bool:
        """Move `key` to database index `db`."""
        return self.execute_command("MOVE", key, db)

    def keys(self, pattern: KeyT = "*") -> List[bytes]:
        """Return all keys matching `pattern`. Prefer `scan_iter()` on large keyspaces."""
        return self.execute_command("KEYS", pattern)

    def randomkey(self) -> Optional[bytes]:
        """Return a random key, or `None` if the keyspace is empty."""
        return self.execute_command("RANDOMKEY")

    def dump(self, key: KeyT) -> Optional[bytes]:
        """Return a serialized representation of `key`'s value, or `None`."""
        return self.execute_command("DUMP", key)

    def restore(
        self,
        key: KeyT,
        ttl: int,
        serialized_value: bytes,
        *,
        replace: bool = False,
        absttl: bool = False,
        idletime: Optional[int] = None,
        freq: Optional[int] = None,
    ) -> bool:
        """Recreate `key` from a `dump()` payload; `ttl` is milliseconds (0 = none)."""
        args: List[Any] = [key, ttl, serialized_value]
        if replace:
            args.append("REPLACE")
        if absttl:
            args.append("ABSTTL")
        if idletime is not None:
            args += ["IDLETIME", idletime]
        if freq is not None:
            args += ["FREQ", freq]
        return self.execute_command("RESTORE", *args)

    # -- expiry ---------------------------------------------------------------

    def _expire_args(
        self, key: KeyT, when: int, *, nx: bool, xx: bool, gt: bool, lt: bool
    ) -> List[Any]:
        args: List[Any] = [key, when]
        if nx:
            args.append("NX")
        elif xx:
            args.append("XX")
        elif gt:
            args.append("GT")
        elif lt:
            args.append("LT")
        return args

    def expire(
        self,
        key: KeyT,
        seconds: int,
        *,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> bool:
        """Set a TTL of `seconds` on `key`."""
        return self.execute_command(
            "EXPIRE", *self._expire_args(key, seconds, nx=nx, xx=xx, gt=gt, lt=lt)
        )

    def pexpire(
        self,
        key: KeyT,
        milliseconds: int,
        *,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> bool:
        """Set a TTL of `milliseconds` on `key`."""
        return self.execute_command(
            "PEXPIRE", *self._expire_args(key, milliseconds, nx=nx, xx=xx, gt=gt, lt=lt)
        )

    def expireat(
        self,
        key: KeyT,
        unix_time: int,
        *,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> bool:
        """Expire `key` at the Unix timestamp `unix_time` (seconds)."""
        return self.execute_command(
            "EXPIREAT", *self._expire_args(key, unix_time, nx=nx, xx=xx, gt=gt, lt=lt)
        )

    def pexpireat(
        self,
        key: KeyT,
        unix_time_ms: int,
        *,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> bool:
        """Expire `key` at the Unix timestamp `unix_time_ms` (milliseconds)."""
        return self.execute_command(
            "PEXPIREAT", *self._expire_args(key, unix_time_ms, nx=nx, xx=xx, gt=gt, lt=lt)
        )

    def ttl(self, key: KeyT) -> int:
        """Remaining TTL of `key` in seconds; `-1` no expiry, `-2` no key."""
        return self.execute_command("TTL", key)

    def pttl(self, key: KeyT) -> int:
        """Remaining TTL of `key` in milliseconds; `-1` no expiry, `-2` no key."""
        return self.execute_command("PTTL", key)

    def persist(self, key: KeyT) -> bool:
        """Remove the TTL from `key`, making it persistent."""
        return self.execute_command("PERSIST", key)

    def expiretime(self, key: KeyT) -> int:
        """Absolute expiry of `key` as a Unix timestamp in seconds."""
        return self.execute_command("EXPIRETIME", key)

    def pexpiretime(self, key: KeyT) -> int:
        """Absolute expiry of `key` as a Unix timestamp in milliseconds."""
        return self.execute_command("PEXPIRETIME", key)
