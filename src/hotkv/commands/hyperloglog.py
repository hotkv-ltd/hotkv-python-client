# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""HyperLogLog commands."""

from __future__ import annotations

from .._typing import KeyT, ValueT
from ._shared import CallbackMap, ReadonlySet, cb_bool, cb_ok

__all__ = ["HyperLogLogCommands", "CALLBACKS", "READONLY"]

CALLBACKS: CallbackMap = {"PFADD": cb_bool, "PFMERGE": cb_ok}

READONLY: ReadonlySet = frozenset({"PFCOUNT"})


class HyperLogLogCommands:
    """Typed wrappers for HyperLogLog commands."""

    def pfadd(self, key: KeyT, *elements: ValueT) -> bool:
        """Add `elements`. Returns `True` if the estimated cardinality changed."""
        return self.execute_command("PFADD", key, *elements)

    def pfcount(self, *keys: KeyT) -> int:
        """Approximate cardinality of the union of `keys`."""
        return self.execute_command("PFCOUNT", *keys)

    def pfmerge(self, destination: KeyT, *source_keys: KeyT) -> bool:
        """Merge `source_keys` into `destination`."""
        return self.execute_command("PFMERGE", destination, *source_keys)
