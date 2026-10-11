# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Routing rules shared by the sync and asyncio cluster clients: which
argument of a command is its routing key, which commands have to be asked of
every primary because the server answers them per node, and how those
per-node answers are merged. Pure functions, no I/O.
"""

from __future__ import annotations

import ssl
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .exceptions import AuthenticationError

__all__ = [
    "Address",
    "command_name",
    "routing_key",
    "fanout_kind",
    "merge_replies",
    "parse_slots",
    "is_permanent_failure",
    "FIRST",
    "UNION",
    "SUM",
    "ALL",
]

Address = Tuple[str, int]

# How the per-node answers of a fanned-out command are merged.
UNION = "union"  # arrays: concatenated, a repeated element kept once
SUM = "sum"  # integers: added up
ALL = "all"  # the command must run on every primary; the first reply is returned
FIRST = "first"  # the first reply that is not nil (a value only one node may hold)

# Commands the server answers from the node's own state, not from a slot. A
# cluster client sends them to every primary and merges: on a single node they
# list only what that node holds.
_FANOUT: Dict[str, str] = {
    "KEYS": UNION,
    "DBSIZE": SUM,
    "FLUSHALL": ALL,
    "FLUSHDB": ALL,
    "TAG.MEMBERS": UNION,
    "TAG.INTER": UNION,
    "TS.LIST": UNION,
    "SCRIPT LOAD": ALL,
    "SCRIPT FLUSH": ALL,
    # A function library has to be on every node for FCALL to find it there.
    "FUNCTION LOAD": ALL,
    "FUNCTION DELETE": ALL,
    "FUNCTION FLUSH": ALL,
    "FUNCTION RESTORE": ALL,
    # A vector snapshot sits on the node it was saved on: asked of every
    # primary, so one VSIM.LIST shows is found and dropped wherever it is.
    "VSIM.LOAD": FIRST,
    "VSIM.DROP": SUM,
}
# <family>.LIST of the AI and side-store engines: one name per cache namespace,
# index, bucket, budget, limiter, template, pipeline or feature group, listed
# from the node the command lands on.
for _family in (
    "SCACHE",
    "PCACHE",
    "NCACHE",
    "TOOL.CACHE",
    "GUARD",
    "RERANK",
    "LLMSTATS",
    "LLMBUDGET",
    "RATELIMIT",
    "PROMPT",
    "EMBED.DEDUP",
    "FEATURE",
    "RAG",
    "VSIM",
    "AMEM.SUMMARY",
):
    _FANOUT[_family + ".LIST"] = UNION

# Commands whose first argument is not a key: they run on any node.
_KEYLESS = frozenset(
    {
        "PING",
        "ECHO",
        "INFO",
        "TIME",
        "LASTSAVE",
        "SAVE",
        "BGSAVE",
        "BGREWRITEAOF",
        "CONFIG",
        "CLIENT",
        "COMMAND",
        "CLUSTER",
        "READONLY",
        "READWRITE",
        "HELLO",
        "AUTH",
        "SCRIPT",
        "FUNCTION",
        "SLOWLOG",
        "LATENCY",
        "MODULE",
        "ACL",
        "WAIT",
        "PUBLISH",
        "PUBSUB",
        "SELECT",
        "RANDOMKEY",
        "SCAN",
        "LLMSTATS.PRICE",
        "OBSERVE",
        "ALERTS",
        "HOTKEYS",
        "REPLICATION.INFO",
    }
)
_KEYLESS_PREFIXES = ("TENANT.", "IAM.", "AUDIT.", "ENCRYPTION.")

# <command> numkeys key [key ...]
_NUMKEYS_AT_1 = frozenset(
    {"ZUNION", "ZINTER", "ZDIFF", "ZINTERCARD", "SINTERCARD", "LMPOP", "ZMPOP"}
)
# <command> timeout numkeys key [key ...] and <command> arg numkeys key [key ...]
_NUMKEYS_AT_2 = frozenset(
    {"BLMPOP", "BZMPOP", "EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO", "FCALL", "FCALL_RO"}
)
# <command> <subcommand> key ...
_KEY_AT_2 = frozenset({"OBJECT", "XINFO", "XGROUP", "DEBUG"})


def command_name(first: Any) -> str:
    """The upper-case name of a command from its first argument."""
    if isinstance(first, (bytes, bytearray)):
        return bytes(first).decode("utf-8", "replace").upper()
    return str(first).upper()


def _upper(value: Any) -> str:
    return command_name(value)


def routing_key(name: str, args: Sequence[Any]) -> Optional[Any]:
    """The argument of `args` (a full command, name first) whose hash slot
    decides the node, or None when the command has no key and can run on any
    node. A command that names several keys (they must share one slot) is
    routed by its first key.

    A command whose name carries its subcommand in one word ("CONFIG GET",
    "XGROUP CREATE", "ALERTS SETURL", as the typed wrappers send them) is
    routed as if the subcommand were the next argument."""
    if " " in name:
        head, _, subcommand = name.partition(" ")
        args = (head, *subcommand.split(), *args[1:])
        name = head
    n = len(args)
    if name in _KEYLESS or name.startswith(_KEYLESS_PREFIXES):
        return None
    if name in _FANOUT or (name == "VSIM.INFO" and n == 1):
        return None
    if name in _NUMKEYS_AT_1:
        return args[2] if n > 2 and _numkeys(args[1]) > 0 else None
    if name in _NUMKEYS_AT_2:
        return args[3] if n > 3 and _numkeys(args[2]) > 0 else None
    if name == "MEMORY":
        return args[2] if n > 2 and _upper(args[1]) == "USAGE" else None
    if name in _KEY_AT_2:
        return args[2] if n > 2 and _upper(args[1]) not in ("HELP",) else None
    if name == "BITOP":
        return args[2] if n > 2 else None
    if name in ("XREAD", "XREADGROUP"):
        for i in range(1, n - 1):
            if _upper(args[i]) == "STREAMS":
                return args[i + 1]
        return None
    return args[1] if n > 1 else None


def _numkeys(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def fanout_kind(name: str, args: Sequence[Any]) -> Optional[str]:
    """How `args` must be fanned out to every primary, or None for a command
    that goes to one node."""
    if name in ("SCRIPT", "FUNCTION") and len(args) > 1:
        return _FANOUT.get(name + " " + _upper(args[1]))
    if name == "VSIM.INFO" and len(args) == 1:
        # Without an index name: the names of the indexes the node holds.
        return UNION
    return _FANOUT.get(name)


def is_permanent_failure(exc: BaseException) -> bool:
    """Whether a connection failure will not go away by trying again: the node
    refused the credentials, or a certificate was refused. A node that went
    away while the connection opened is not one: that is a failover."""
    return isinstance(exc, AuthenticationError) or isinstance(exc.__cause__, ssl.SSLCertVerificationError)


def merge_replies(kind: str, replies: List[Any]) -> Any:
    """Merge the raw per-node replies of a fanned-out command."""
    if not replies:
        return [] if kind == UNION else 0 if kind == SUM else None
    if kind == SUM:
        return sum(int(r) for r in replies)
    if kind == ALL:
        return replies[0]
    if kind == FIRST:
        return next((r for r in replies if r is not None), None)
    merged: List[Any] = []
    seen = set()
    for reply in replies:
        if reply is None:
            continue
        for item in reply:
            marker = bytes(item) if isinstance(item, (bytes, bytearray)) else repr(item)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(item)
    return merged


def _host(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = bytes(value).decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)
    return text or fallback


def parse_slots(reply: Any, asked_host: str) -> Tuple[List[Optional[Address]], List[List[Address]]]:
    """Slot tables from a `CLUSTER SLOTS` reply: the master of each of the
    16384 slots and its replicas. A node that announces no host (an empty or
    nil host) is reached at `asked_host`, the node that was asked."""
    masters: List[Optional[Address]] = [None] * 16384
    replicas: List[List[Address]] = [[] for _ in range(16384)]
    for entry in reply:
        start, end = int(entry[0]), int(entry[1])
        master = (_host(entry[2][0], asked_host), int(entry[2][1]))
        replica_addrs = [(_host(r[0], asked_host), int(r[1])) for r in entry[3:]]
        for slot in range(start, end + 1):
            masters[slot] = master
            replicas[slot] = replica_addrs
    return masters, replicas
