# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Stream commands (XADD/XREAD/consumer groups/XINFO)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .._typing import FieldT, KeyT, ValueT
from ._shared import CallbackMap, ReadonlySet, named_fields, pairs_to_dict

__all__ = ["StreamCommands", "CALLBACKS", "READONLY"]


def _parse_entries(entries: Optional[Sequence[Any]]) -> List[Tuple[Any, Dict[Any, Any]]]:
    return [(entry_id, pairs_to_dict(fields)) for entry_id, fields in (entries or [])]


def _cb_xrange(reply: Any, **_options: Any) -> List[Tuple[Any, Dict[Any, Any]]]:
    return _parse_entries(reply)


def _cb_xread(reply: Any, **_options: Any) -> Optional[Dict[Any, Any]]:
    if reply is None:
        return None
    items = reply.items() if isinstance(reply, dict) else reply
    return {name: _parse_entries(entries) for name, entries in items}


def _cb_xclaim(reply: Any, justid: bool = False, **_options: Any) -> Any:
    if justid:
        return reply
    return _parse_entries(reply)


def _cb_xautoclaim(reply: Any, justid: bool = False, **_options: Any) -> Tuple[Any, Any, list]:
    cursor, entries = reply[0], reply[1]
    deleted = reply[2] if len(reply) > 2 else []
    return (cursor, entries if justid else _parse_entries(entries), deleted)


def _cb_xpending_summary(reply: Any, extended: bool = False, **_options: Any) -> Any:
    if extended:
        return [(pid, consumer, idle, deliveries) for pid, consumer, idle, deliveries in reply]
    count, min_id, max_id, per_consumer = reply
    consumers = [(name, int(n)) for name, n in (per_consumer or [])]
    return (count, min_id, max_id, consumers)


def _cb_xinfo_stream(reply: Any, **_options: Any) -> Dict[str, Any]:
    return named_fields(
        reply,
        texts={
            "name",
            "last-generated-id",
            "max-deleted-entry-id",
            "recorded-first-entry-id",
            "first-entry",
            "last-entry",
        },
    )


def _cb_xinfo_groups(reply: Any, **_options: Any) -> List[Dict[str, Any]]:
    return [named_fields(g, texts={"name", "last-delivered-id"}) for g in (reply or [])]


def _cb_xinfo_consumers(reply: Any, **_options: Any) -> List[Dict[str, Any]]:
    return [named_fields(c, texts={"name"}) for c in (reply or [])]


CALLBACKS: CallbackMap = {
    "XRANGE": _cb_xrange,
    "XREVRANGE": _cb_xrange,
    "XREAD": _cb_xread,
    "XREADGROUP": _cb_xread,
    "XCLAIM": _cb_xclaim,
    "XAUTOCLAIM": _cb_xautoclaim,
    "XPENDING": _cb_xpending_summary,
    "XGROUP CREATE": lambda reply, **_: reply == "OK" or reply is True,
    "XGROUP SETID": lambda reply, **_: reply == "OK" or reply is True,
    "XGROUP CREATECONSUMER": lambda reply, **_: bool(reply),
    "XGROUP DESTROY": lambda reply, **_: bool(reply),
    "XINFO STREAM": _cb_xinfo_stream,
    "XINFO GROUPS": _cb_xinfo_groups,
    "XINFO CONSUMERS": _cb_xinfo_consumers,
}

READONLY: ReadonlySet = frozenset(
    {
        "XRANGE",
        "XREVRANGE",
        "XLEN",
        "XREAD",
        "XPENDING",
        "XINFO STREAM",
        "XINFO GROUPS",
        "XINFO CONSUMERS",
    }
)


class StreamCommands:
    """Typed wrappers for stream commands, including consumer groups."""

    def xadd(
        self,
        key: KeyT,
        fields: Mapping[FieldT, ValueT],
        *,
        id: Union[str, bytes] = "*",
        nomkstream: bool = False,
        maxlen: Optional[int] = None,
        minid: Optional[ValueT] = None,
        approx: bool = False,
        limit: Optional[int] = None,
    ) -> bytes:
        """Append an entry, returning its stream ID."""
        args: List[Any] = [key]
        if nomkstream:
            args.append("NOMKSTREAM")
        if maxlen is not None:
            args += ["MAXLEN", "~" if approx else "=", maxlen]
        elif minid is not None:
            args += ["MINID", "~" if approx else "=", minid]
        if limit is not None:
            args += ["LIMIT", limit]
        args.append(id)
        for field, value in fields.items():
            args += [field, value]
        return self.execute_command("XADD", *args)

    def xlen(self, key: KeyT) -> int:
        """Number of entries in the stream at `key`."""
        return self.execute_command("XLEN", key)

    def xrange(
        self, key: KeyT, start: ValueT = "-", end: ValueT = "+", *, count: Optional[int] = None
    ) -> List[Tuple[Any, Dict[Any, Any]]]:
        """Entries from `start` to `end` (stream IDs, or "-"/"+" for the ends)."""
        args: List[Any] = [key, start, end]
        if count is not None:
            args += ["COUNT", count]
        return self.execute_command("XRANGE", *args)

    def xrevrange(
        self, key: KeyT, end: ValueT = "+", start: ValueT = "-", *, count: Optional[int] = None
    ) -> List[Tuple[Any, Dict[Any, Any]]]:
        """Like `xrange()`, newest entries first."""
        args: List[Any] = [key, end, start]
        if count is not None:
            args += ["COUNT", count]
        return self.execute_command("XREVRANGE", *args)

    def xdel(self, key: KeyT, *ids: ValueT) -> int:
        """Delete entries by ID, returning the number removed."""
        return self.execute_command("XDEL", key, *ids)

    def xtrim(
        self,
        key: KeyT,
        *,
        maxlen: Optional[int] = None,
        minid: Optional[ValueT] = None,
        approx: bool = False,
        limit: Optional[int] = None,
    ) -> int:
        """Trim the stream, returning the number of entries removed."""
        args: List[Any] = [key]
        if maxlen is not None:
            args += ["MAXLEN", "~" if approx else "=", maxlen]
        elif minid is not None:
            args += ["MINID", "~" if approx else "=", minid]
        if limit is not None:
            args += ["LIMIT", limit]
        return self.execute_command("XTRIM", *args)

    def xread(
        self,
        streams: Mapping[KeyT, ValueT],
        *,
        count: Optional[int] = None,
        block: Optional[int] = None,
    ) -> Optional[Dict[Any, List[Tuple[Any, Dict[Any, Any]]]]]:
        """Read new entries from several streams; `streams` maps key to the
        last-seen ID (use "$" for "only new entries"). Returns `None` if
        `block` (milliseconds) elapses with nothing new."""
        args: List[Any] = []
        if count is not None:
            args += ["COUNT", count]
        if block is not None:
            args += ["BLOCK", block]
        args.append("STREAMS")
        keys = list(streams.keys())
        args += keys
        args += [streams[k] for k in keys]
        return self.execute_command("XREAD", *args)

    def xgroup_create(
        self,
        key: KeyT,
        group: ValueT,
        id: Union[str, bytes] = "$",
        *,
        mkstream: bool = False,
    ) -> bool:
        """Create consumer group `group` on `key` starting at `id`."""
        args: List[Any] = [key, group, id]
        if mkstream:
            args.append("MKSTREAM")
        return self.execute_command("XGROUP CREATE", *args)

    def xgroup_setid(self, key: KeyT, group: ValueT, id: Union[str, bytes] = "$") -> bool:
        """Reposition consumer group `group`'s last-delivered ID."""
        return self.execute_command("XGROUP SETID", key, group, id)

    def xgroup_destroy(self, key: KeyT, group: ValueT) -> bool:
        """Destroy consumer group `group`."""
        return self.execute_command("XGROUP DESTROY", key, group)

    def xgroup_createconsumer(self, key: KeyT, group: ValueT, consumer: ValueT) -> bool:
        """Explicitly create `consumer` in `group`."""
        return self.execute_command("XGROUP CREATECONSUMER", key, group, consumer)

    def xgroup_delconsumer(self, key: KeyT, group: ValueT, consumer: ValueT) -> int:
        """Remove `consumer`, returning the number of pending entries it had."""
        return self.execute_command("XGROUP DELCONSUMER", key, group, consumer)

    def xreadgroup(
        self,
        group: ValueT,
        consumer: ValueT,
        streams: Mapping[KeyT, ValueT],
        *,
        count: Optional[int] = None,
        block: Optional[int] = None,
        noack: bool = False,
    ) -> Optional[Dict[Any, List[Tuple[Any, Dict[Any, Any]]]]]:
        """Read as `consumer` in `group`; `streams` maps key to ID (usually ">")."""
        args: List[Any] = ["GROUP", group, consumer]
        if count is not None:
            args += ["COUNT", count]
        if block is not None:
            args += ["BLOCK", block]
        if noack:
            args.append("NOACK")
        args.append("STREAMS")
        keys = list(streams.keys())
        args += keys
        args += [streams[k] for k in keys]
        return self.execute_command("XREADGROUP", *args)

    def xack(self, key: KeyT, group: ValueT, *ids: ValueT) -> int:
        """Acknowledge processed entries, returning the number acknowledged."""
        return self.execute_command("XACK", key, group, *ids)

    def xpending(
        self,
        key: KeyT,
        group: ValueT,
        *,
        start: Optional[ValueT] = None,
        end: Optional[ValueT] = None,
        count: Optional[int] = None,
        consumer: Optional[ValueT] = None,
        idle: Optional[int] = None,
    ) -> Any:
        """Summary form (no `start`/`end`/`count`): `(count, min_id, max_id,
        [(consumer, count), ...])`. Extended form: a list of
        `(id, consumer, idle_ms, deliveries)` tuples."""
        extended = count is not None
        args: List[Any] = [key, group]
        if extended:
            if idle is not None:
                args += ["IDLE", idle]
            args += [start if start is not None else "-", end if end is not None else "+", count]
            if consumer is not None:
                args.append(consumer)
        return self.execute_command("XPENDING", *args, extended=extended)

    def xclaim(
        self,
        key: KeyT,
        group: ValueT,
        consumer: ValueT,
        min_idle_time: int,
        ids: Sequence[ValueT],
        *,
        idle: Optional[int] = None,
        time: Optional[int] = None,
        retrycount: Optional[int] = None,
        force: bool = False,
        justid: bool = False,
    ) -> Any:
        """Reassign pending entries to `consumer`."""
        args: List[Any] = [key, group, consumer, min_idle_time, *ids]
        if idle is not None:
            args += ["IDLE", idle]
        if time is not None:
            args += ["TIME", time]
        if retrycount is not None:
            args += ["RETRYCOUNT", retrycount]
        if force:
            args.append("FORCE")
        if justid:
            args.append("JUSTID")
        return self.execute_command("XCLAIM", *args, justid=justid)

    def xautoclaim(
        self,
        key: KeyT,
        group: ValueT,
        consumer: ValueT,
        min_idle_time: int,
        start: ValueT = "0-0",
        *,
        count: int = 100,
        justid: bool = False,
    ) -> Tuple[Any, Any, list]:
        """Like `xclaim()`, scanning from `start` instead of listing IDs.

        Returns `(next_cursor, entries_or_ids, deleted_ids)`. `count`
        defaults to 100 and, on this server, must be sent: its command
        dispatch requires at least 6 arguments for `XAUTOCLAIM`.
        """
        args: List[Any] = [key, group, consumer, min_idle_time, start, "COUNT", count]
        if justid:
            args.append("JUSTID")
        return self.execute_command("XAUTOCLAIM", *args, justid=justid)

    def xinfo_stream(self, key: KeyT, *, full: bool = False) -> Dict[str, Any]:
        """Return general information about the stream at `key`."""
        args: List[Any] = [key]
        if full:
            args.append("FULL")
        return self.execute_command("XINFO STREAM", *args)

    def xinfo_groups(self, key: KeyT) -> List[Dict[str, Any]]:
        """Return information about each consumer group on `key`."""
        return self.execute_command("XINFO GROUPS", key)

    def xinfo_consumers(self, key: KeyT, group: ValueT) -> List[Dict[str, Any]]:
        """Return information about each consumer in `group`."""
        return self.execute_command("XINFO CONSUMERS", key, group)
