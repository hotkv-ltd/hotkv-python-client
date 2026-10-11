# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Core (license-free) HotKV exclusive commands: TAG.*, nanosecond TTLs
(NEXPIRE/NTTL/NEXPIREAT), HOTKEYS, GCRA.* and VSIM.* (including its
persistence sub-commands, which do require a full license; the server, not
the client, enforces that and replies with `NOLICENSE`).
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

from .._numbers import format_vector
from .._typing import KeyT, ValueT
from ._shared import CallbackMap, ReadonlySet, as_float, cb_bool, cb_ok, flat_to_pairs, named_fields

__all__ = ["HotKVCoreCommands", "GCRAResult", "CALLBACKS", "READONLY"]


class GCRAResult(NamedTuple):
    """Reply of `gcra_limit()`."""

    denied: bool
    limit: int
    remaining: int
    retry_after_ms: int
    reset_after_ms: int

    @property
    def allowed(self) -> bool:
        return not self.denied


def _cb_hotkeys_top(reply: Any, **_options: Any) -> List[Tuple[Any, int]]:
    return [(key, int(count)) for key, count in reply]


def _cb_gcra_limit(reply: Any, **_options: Any) -> GCRAResult:
    denied, limit, remaining, retry_after_ms, reset_after_ms = reply
    return GCRAResult(
        bool(denied), int(limit), int(remaining), int(retry_after_ms), int(reset_after_ms)
    )


def _cb_vsim_get(reply: Any, **_options: Any) -> Optional[List[float]]:
    if reply is None:
        return None
    return [as_float(v) for v in reply]


def _cb_vsim_search(reply: Any, **_options: Any) -> List[Tuple[Any, float]]:
    return [(key, as_float(distance)) for key, distance in flat_to_pairs(reply)]


def _cb_vsim_info(reply: Any, with_index: bool = False, **_options: Any) -> Any:
    if not with_index:
        return reply
    return named_fields(reply, texts={"name", "metric"})


CALLBACKS: CallbackMap = {
    "TAG.EXISTS": cb_bool,
    "NEXPIRE": cb_bool,
    "NEXPIREAT": cb_bool,
    "HOTKEYS RESET": cb_ok,
    "HOTKEYS TOP": _cb_hotkeys_top,
    "GCRA.LIMIT": _cb_gcra_limit,
    "VSIM.CREATEINDEX": cb_ok,
    "VSIM.DROPINDEX": cb_ok,
    "VSIM.SET": cb_bool,
    "VSIM.GET": _cb_vsim_get,
    "VSIM.DEL": cb_bool,
    "VSIM.SEARCH": _cb_vsim_search,
    "VSIM.INFO": _cb_vsim_info,
    "VSIM.DROP": cb_bool,
}

READONLY: ReadonlySet = frozenset(
    {
        "TAG.GET",
        "TAG.MEMBERS",
        "TAG.INTER",
        "TAG.COUNT",
        "TAG.EXISTS",
        "NTTL",
        "HOTKEYS TOP",
        "VSIM.GET",
        "VSIM.SEARCH",
        "VSIM.INFO",
        "VSIM.LOAD",
        "VSIM.LIST",
    }
)


class HotKVCoreCommands:
    """Typed wrappers for TAG.*, nanosecond TTLs, HOTKEYS, GCRA.* and VSIM.*."""

    # -- TAG.* --------------------------------------------------------------

    def tag_set(self, key: KeyT, *tags: ValueT) -> int:
        """Add `tags` to `key`. Raises if `key` does not exist."""
        return self.execute_command("TAG.SET", key, *tags)

    def tag_rem(self, key: KeyT, *tags: ValueT) -> int:
        """Remove `tags` from `key`, returning the number actually removed."""
        return self.execute_command("TAG.REM", key, *tags)

    def tag_get(self, key: KeyT) -> List[bytes]:
        """Return every tag on `key`."""
        return self.execute_command("TAG.GET", key)

    def tag_members(self, tag: ValueT) -> List[bytes]:
        """Return every key carrying `tag`."""
        return self.execute_command("TAG.MEMBERS", tag)

    def tag_inter(self, *tags: ValueT) -> List[bytes]:
        """Return keys that carry every tag in `tags`."""
        return self.execute_command("TAG.INTER", *tags)

    def tag_count(self, key: KeyT) -> int:
        """Number of tags on `key`."""
        return self.execute_command("TAG.COUNT", key)

    def tag_exists(self, key: KeyT, tag: ValueT) -> bool:
        """Whether `key` carries `tag`."""
        return self.execute_command("TAG.EXISTS", key, tag)

    # -- nanosecond TTL -------------------------------------------------------

    def nexpire(self, key: KeyT, nanoseconds: int) -> bool:
        """Set a relative TTL of `nanoseconds` on `key`."""
        return self.execute_command("NEXPIRE", key, nanoseconds)

    def nttl(self, key: KeyT) -> int:
        """Remaining TTL in nanoseconds; `-1` no expiry, `-2` no key."""
        return self.execute_command("NTTL", key)

    def nexpireat(self, key: KeyT, unix_time_ns: int) -> bool:
        """Expire `key` at the absolute Unix timestamp `unix_time_ns` (nanoseconds)."""
        return self.execute_command("NEXPIREAT", key, unix_time_ns)

    # -- HOTKEYS ----------------------------------------------------------------

    def hotkeys_top(self, count: Optional[int] = None) -> List[Tuple[Any, int]]:
        """Return the `count` (default 10) most frequently accessed keys, as
        `(key, access_count)` pairs, most frequent first."""
        args: List[Any] = [] if count is None else [count]
        return self.execute_command("HOTKEYS TOP", *args)

    def hotkeys_reset(self) -> bool:
        """Reset hot key access counters."""
        return self.execute_command("HOTKEYS RESET")

    # -- GCRA.* (rate limiter primitive) -----------------------------------------

    def gcra_limit(
        self, key: KeyT, period_ms: int, burst: int, count: int = 1
    ) -> GCRAResult:
        """Check (and consume `count` tokens from) a GCRA rate limiter."""
        return self.execute_command("GCRA.LIMIT", key, period_ms, burst, count)

    def gcra_reset(self, *keys: KeyT) -> int:
        """Reset GCRA limiter state for `keys`, returning how many had state."""
        return self.execute_command("GCRA.RESET", *keys)

    # -- VSIM.* (vector similarity index) ----------------------------------------

    def vsim_createindex(
        self,
        name: ValueT,
        dims: int,
        *,
        metric: str = "cosine",
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        ef_search: Optional[int] = None,
    ) -> bool:
        """Create a vector index `name` with `dims` dimensions."""
        args: List[Any] = [name, dims, "METRIC", metric]
        if m is not None:
            args += ["M", m]
        if ef_construction is not None:
            args += ["EF_CONSTRUCTION", ef_construction]
        if ef_search is not None:
            args += ["EF_SEARCH", ef_search]
        return self.execute_command("VSIM.CREATEINDEX", *args)

    def vsim_dropindex(self, name: ValueT) -> bool:
        """Drop vector index `name`."""
        return self.execute_command("VSIM.DROPINDEX", name)

    def vsim_set(self, index: ValueT, key: KeyT, vector: Sequence[float]) -> bool:
        """Store `vector` for `key` in `index`. Returns `True` if newly inserted."""
        return self.execute_command("VSIM.SET", index, key, *format_vector(vector))

    def vsim_get(self, index: ValueT, key: KeyT) -> Optional[List[float]]:
        """Return the vector stored for `key` in `index`, or `None`."""
        return self.execute_command("VSIM.GET", index, key)

    def vsim_del(self, index: ValueT, key: KeyT) -> bool:
        """Remove `key` from `index`."""
        return self.execute_command("VSIM.DEL", index, key)

    def vsim_search(
        self,
        index: ValueT,
        k: int,
        vector: Sequence[float],
        *,
        filter_tags: Optional[Sequence[ValueT]] = None,
        ef: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """Find the `k` nearest neighbours of `vector` in `index`, nearest first."""
        args: List[Any] = [index, k, *format_vector(vector)]
        if filter_tags:
            args += ["FILTER", *filter_tags]
        if ef is not None:
            args += ["EF", ef]
        return self.execute_command("VSIM.SEARCH", *args)

    def vsim_info(self, index: Optional[ValueT] = None) -> Any:
        """Without `index`, list all index names; with it, return that index's stats."""
        args: List[Any] = [] if index is None else [index]
        return self.execute_command("VSIM.INFO", *args, with_index=index is not None)

    def vsim_save(self, snapshot_name: ValueT, blob: bytes) -> int:
        """Persist a snapshot `blob` under `snapshot_name`. Requires a full license."""
        return self.execute_command("VSIM.SAVE", snapshot_name, blob)

    def vsim_load(self, snapshot_name: ValueT) -> Optional[bytes]:
        """Load a snapshot blob by name, or `None`. Requires a full license."""
        return self.execute_command("VSIM.LOAD", snapshot_name)

    def vsim_list(self) -> List[bytes]:
        """List stored snapshot names. Requires a full license."""
        return self.execute_command("VSIM.LIST")

    def vsim_drop(self, snapshot_name: ValueT) -> bool:
        """Delete a stored snapshot. Requires a full license."""
        return self.execute_command("VSIM.DROP", snapshot_name)
