# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Combines every command family mixin into one `CoreCommands` class, and
merges each family's `CALLBACKS` and `READONLY` tables into the two maps
`execute_command()` consults: `RESPONSE_CALLBACKS` and `READONLY_COMMANDS`.

Every mixin method is a plain `def` that ends in
`return self.execute_command(...)` with no `async`/`await`, so the exact
same `CoreCommands` class is mixed into the sync client, the asyncio
client, the pipeline/transaction and both cluster clients: only
`execute_command()` itself differs between sync and async.
"""

from __future__ import annotations

from typing import FrozenSet

from . import (
    admin,
    ai_caches,
    ai_memory,
    bitmaps,
    geo,
    hashes,
    hotkv_core,
    hyperloglog,
    keys,
    lists,
    ops,
    scan,
    scripting,
    server,
    sets,
    streams,
    strings,
    zsets,
)
from ._shared import CallbackMap

__all__ = ["CoreCommands", "RESPONSE_CALLBACKS", "READONLY_COMMANDS"]

_MODULES = (
    strings,
    keys,
    hashes,
    lists,
    sets,
    zsets,
    streams,
    geo,
    bitmaps,
    hyperloglog,
    server,
    scripting,
    scan,
    hotkv_core,
    ai_caches,
    ai_memory,
    ops,
    admin,
)


class CoreCommands(
    strings.StringCommands,
    keys.KeyCommands,
    hashes.HashCommands,
    lists.ListCommands,
    sets.SetCommands,
    zsets.SortedSetCommands,
    streams.StreamCommands,
    geo.GeoCommands,
    bitmaps.BitmapCommands,
    hyperloglog.HyperLogLogCommands,
    server.ServerCommands,
    scripting.ScriptingCommands,
    scan.ScanCommands,
    hotkv_core.HotKVCoreCommands,
    ai_caches.AICacheCommands,
    ai_memory.AIMemoryCommands,
    ops.OpsCommands,
    admin.AdminCommands,
):
    """Every typed command wrapper, combined.

    Mixed into `hotkv.HotKV`, `hotkv.asyncio.HotKV`, `Pipeline`, and both
    cluster clients. Requires the concrete class to provide
    `execute_command(*args, **options)`.
    """


RESPONSE_CALLBACKS: CallbackMap = {}
READONLY_COMMANDS: FrozenSet[str] = frozenset()
for _module in _MODULES:
    RESPONSE_CALLBACKS.update(_module.CALLBACKS)
    READONLY_COMMANDS = READONLY_COMMANDS | _module.READONLY
del _module
