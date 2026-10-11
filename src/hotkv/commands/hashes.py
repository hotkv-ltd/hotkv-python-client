# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Hash commands, including the per-field TTL family (HEXPIRE/HTTL/...)."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Union

from .._typing import FieldT, KeyT, ValueT
from ._shared import CallbackMap, ReadonlySet, cb_bool, flat_to_pairs, pairs_to_dict

__all__ = ["HashCommands", "CALLBACKS", "READONLY"]


def _cb_hrandfield(reply: Any, withvalues: bool = False, **_options: Any) -> Any:
    if withvalues and isinstance(reply, list):
        return flat_to_pairs(reply)
    return reply


CALLBACKS: CallbackMap = {
    "HSETNX": cb_bool,
    "HEXISTS": cb_bool,
    "HGETALL": lambda reply, **_: pairs_to_dict(reply),
    "HINCRBYFLOAT": lambda reply, **_: float(reply),
    "HRANDFIELD": _cb_hrandfield,
}

READONLY: ReadonlySet = frozenset(
    {
        "HGET",
        "HMGET",
        "HGETALL",
        "HLEN",
        "HKEYS",
        "HVALS",
        "HEXISTS",
        "HSTRLEN",
        "HRANDFIELD",
        "HTTL",
        "HPTTL",
        "HEXPIRETIME",
        "HPEXPIRETIME",
    }
)


class HashCommands:
    """Typed wrappers for hash commands, including per-field expiry."""

    def hset(self, key: KeyT, mapping: Mapping[FieldT, ValueT]) -> int:
        """Set one or more fields on the hash at `key`, returning the number added."""
        args: List[Any] = []
        for field, value in mapping.items():
            args += [field, value]
        return self.execute_command("HSET", key, *args)

    def hsetnx(self, key: KeyT, field: FieldT, value: ValueT) -> bool:
        """Set `field` only if it does not already exist in the hash."""
        return self.execute_command("HSETNX", key, field, value)

    def hget(self, key: KeyT, field: FieldT) -> Optional[bytes]:
        """Return the value of `field`, or `None`."""
        return self.execute_command("HGET", key, field)

    def hmget(self, key: KeyT, fields: Sequence[FieldT]) -> List[Optional[bytes]]:
        """Return the values of several fields, `None` for any that are unset."""
        return self.execute_command("HMGET", key, *fields)

    def hgetall(self, key: KeyT) -> dict:
        """Return the whole hash at `key` as a `{field: value}` dict."""
        return self.execute_command("HGETALL", key)

    def hdel(self, key: KeyT, *fields: FieldT) -> int:
        """Delete one or more fields, returning the number removed."""
        return self.execute_command("HDEL", key, *fields)

    def hexists(self, key: KeyT, field: FieldT) -> bool:
        """Whether `field` exists in the hash at `key`."""
        return self.execute_command("HEXISTS", key, field)

    def hlen(self, key: KeyT) -> int:
        """Number of fields in the hash at `key`."""
        return self.execute_command("HLEN", key)

    def hkeys(self, key: KeyT) -> List[bytes]:
        """All field names in the hash at `key`."""
        return self.execute_command("HKEYS", key)

    def hvals(self, key: KeyT) -> List[bytes]:
        """All values in the hash at `key`."""
        return self.execute_command("HVALS", key)

    def hincrby(self, key: KeyT, field: FieldT, amount: int) -> int:
        """Increment the integer value of `field` by `amount`."""
        return self.execute_command("HINCRBY", key, field, amount)

    def hincrbyfloat(self, key: KeyT, field: FieldT, amount: float) -> float:
        """Increment the float value of `field` by `amount`."""
        return self.execute_command("HINCRBYFLOAT", key, field, amount)

    def hstrlen(self, key: KeyT, field: FieldT) -> int:
        """Length of the value stored in `field` (0 if unset)."""
        return self.execute_command("HSTRLEN", key, field)

    def hrandfield(
        self, key: KeyT, count: Optional[int] = None, *, withvalues: bool = False
    ) -> Union[Optional[bytes], List[Any]]:
        """Return one random field, or `count` fields (negative allows repeats).

        With `withvalues=True` (requires `count`), returns `(field, value)` pairs.
        """
        args: List[Any] = [key]
        if count is not None:
            args.append(count)
            if withvalues:
                args.append("WITHVALUES")
        return self.execute_command("HRANDFIELD", *args, withvalues=withvalues)

    # -- per-field TTL (HEXPIRE / HTTL family) ---------------------------------

    def _hfields_args(
        self,
        key: KeyT,
        when: int,
        fields: Sequence[FieldT],
        *,
        nx: bool,
        xx: bool,
        gt: bool,
        lt: bool,
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
        args += ["FIELDS", len(fields), *fields]
        return args

    def hexpire(
        self,
        key: KeyT,
        seconds: int,
        fields: Sequence[FieldT],
        *,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> List[int]:
        """Set a TTL of `seconds` on each of `fields`.

        Returns one status code per field: `1` set, `0` condition not met,
        `-2` no such field, `2` the field was deleted (`seconds <= 0`).
        """
        return self.execute_command(
            "HEXPIRE", *self._hfields_args(key, seconds, fields, nx=nx, xx=xx, gt=gt, lt=lt)
        )

    def hpexpire(
        self,
        key: KeyT,
        milliseconds: int,
        fields: Sequence[FieldT],
        *,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> List[int]:
        """Set a TTL of `milliseconds` on each of `fields`."""
        return self.execute_command(
            "HPEXPIRE",
            *self._hfields_args(key, milliseconds, fields, nx=nx, xx=xx, gt=gt, lt=lt),
        )

    def hexpireat(
        self,
        key: KeyT,
        unix_time: int,
        fields: Sequence[FieldT],
        *,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> List[int]:
        """Expire each of `fields` at the Unix timestamp `unix_time` (seconds)."""
        return self.execute_command(
            "HEXPIREAT", *self._hfields_args(key, unix_time, fields, nx=nx, xx=xx, gt=gt, lt=lt)
        )

    def hpexpireat(
        self,
        key: KeyT,
        unix_time_ms: int,
        fields: Sequence[FieldT],
        *,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> List[int]:
        """Expire each of `fields` at the Unix timestamp `unix_time_ms` (milliseconds)."""
        return self.execute_command(
            "HPEXPIREAT",
            *self._hfields_args(key, unix_time_ms, fields, nx=nx, xx=xx, gt=gt, lt=lt),
        )

    def hpersist(self, key: KeyT, fields: Sequence[FieldT]) -> List[int]:
        """Remove the TTL from each of `fields`."""
        return self.execute_command("HPERSIST", key, "FIELDS", len(fields), *fields)

    def httl(self, key: KeyT, fields: Sequence[FieldT]) -> List[int]:
        """Remaining TTL in seconds for each of `fields` (`-1` none, `-2` no field)."""
        return self.execute_command("HTTL", key, "FIELDS", len(fields), *fields)

    def hpttl(self, key: KeyT, fields: Sequence[FieldT]) -> List[int]:
        """Remaining TTL in milliseconds for each of `fields`."""
        return self.execute_command("HPTTL", key, "FIELDS", len(fields), *fields)

    def hexpiretime(self, key: KeyT, fields: Sequence[FieldT]) -> List[int]:
        """Absolute expiry (Unix seconds) for each of `fields`."""
        return self.execute_command("HEXPIRETIME", key, "FIELDS", len(fields), *fields)

    def hpexpiretime(self, key: KeyT, fields: Sequence[FieldT]) -> List[int]:
        """Absolute expiry (Unix milliseconds) for each of `fields`."""
        return self.execute_command("HPEXPIRETIME", key, "FIELDS", len(fields), *fields)
