# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Set commands."""

from __future__ import annotations

from typing import Any, List, Optional, Set, Union

from .._typing import KeyT, KeysT, ValueT
from ._shared import CallbackMap, ReadonlySet, cb_bool

__all__ = ["SetCommands", "CALLBACKS", "READONLY"]


def _cb_set_result(reply: Any, **_options: Any) -> Set[Any]:
    return set(reply) if reply is not None else set()


def _cb_smismember(reply: Any, **_options: Any) -> List[bool]:
    return [bool(v) for v in (reply or [])]


def _cb_spop(reply: Any, count: Optional[int] = None, **_options: Any) -> Any:
    if count is None:
        return reply
    return set(reply) if reply is not None else set()


CALLBACKS: CallbackMap = {
    "SISMEMBER": cb_bool,
    "SMOVE": cb_bool,
    "SMEMBERS": _cb_set_result,
    "SUNION": _cb_set_result,
    "SINTER": _cb_set_result,
    "SDIFF": _cb_set_result,
    "SMISMEMBER": _cb_smismember,
    "SPOP": _cb_spop,
}

READONLY: ReadonlySet = frozenset(
    {
        "SISMEMBER",
        "SMISMEMBER",
        "SMEMBERS",
        "SCARD",
        "SRANDMEMBER",
        "SUNION",
        "SINTER",
        "SDIFF",
        "SINTERCARD",
    }
)


class SetCommands:
    """Typed wrappers for set commands."""

    def sadd(self, key: KeyT, *members: ValueT) -> int:
        """Add `members` to the set at `key`, returning the number newly added."""
        return self.execute_command("SADD", key, *members)

    def srem(self, key: KeyT, *members: ValueT) -> int:
        """Remove `members` from the set at `key`, returning the number removed."""
        return self.execute_command("SREM", key, *members)

    def sismember(self, key: KeyT, member: ValueT) -> bool:
        """Whether `member` is in the set at `key`."""
        return self.execute_command("SISMEMBER", key, member)

    def smismember(self, key: KeyT, members: Any) -> List[bool]:
        """Whether each of `members` is in the set at `key`."""
        return self.execute_command("SMISMEMBER", key, *members)

    def smembers(self, key: KeyT) -> Set[Any]:
        """Return all members of the set at `key`."""
        return self.execute_command("SMEMBERS", key)

    def scard(self, key: KeyT) -> int:
        """Number of members in the set at `key`."""
        return self.execute_command("SCARD", key)

    def spop(self, key: KeyT, count: Optional[int] = None) -> Union[bytes, Set[Any], None]:
        """Remove and return one random member, or up to `count` distinct members."""
        args: List[Any] = [key] if count is None else [key, count]
        return self.execute_command("SPOP", *args, count=count)

    def srandmember(
        self, key: KeyT, count: Optional[int] = None
    ) -> Union[bytes, List[bytes], None]:
        """Return one random member, or `count` members (negative allows repeats)."""
        args: List[Any] = [key] if count is None else [key, count]
        return self.execute_command("SRANDMEMBER", *args)

    def smove(self, source: KeyT, destination: KeyT, member: ValueT) -> bool:
        """Atomically move `member` from `source` to `destination`."""
        return self.execute_command("SMOVE", source, destination, member)

    def sunion(self, *keys: KeyT) -> Set[Any]:
        """Union of all sets in `keys`."""
        return self.execute_command("SUNION", *keys)

    def sunionstore(self, destination: KeyT, *keys: KeyT) -> int:
        """Store the union of `keys` into `destination`, returning its size."""
        return self.execute_command("SUNIONSTORE", destination, *keys)

    def sinter(self, *keys: KeyT) -> Set[Any]:
        """Intersection of all sets in `keys`."""
        return self.execute_command("SINTER", *keys)

    def sinterstore(self, destination: KeyT, *keys: KeyT) -> int:
        """Store the intersection of `keys` into `destination`, returning its size."""
        return self.execute_command("SINTERSTORE", destination, *keys)

    def sdiff(self, *keys: KeyT) -> Set[Any]:
        """Members in the first set of `keys` but none of the rest."""
        return self.execute_command("SDIFF", *keys)

    def sdiffstore(self, destination: KeyT, *keys: KeyT) -> int:
        """Store the difference of `keys` into `destination`, returning its size."""
        return self.execute_command("SDIFFSTORE", destination, *keys)

    def sintercard(self, keys: KeysT, *, limit: Optional[int] = None) -> int:
        """Cardinality of the intersection of `keys`, without materializing it."""
        keys = list(keys)
        args: List[Any] = [len(keys), *keys]
        if limit is not None:
            args += ["LIMIT", limit]
        return self.execute_command("SINTERCARD", *args)
