# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Bitmap commands, including BITFIELD."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .._typing import KeyT
from ._shared import CallbackMap, ReadonlySet

__all__ = ["BitmapCommands", "CALLBACKS", "READONLY"]

CALLBACKS: CallbackMap = {}

READONLY: ReadonlySet = frozenset({"GETBIT", "BITCOUNT", "BITPOS"})


class BitmapCommands:
    """Typed wrappers for bitmap commands."""

    def setbit(self, key: KeyT, offset: int, value: int) -> int:
        """Set the bit at `offset` to `0` or `1`, returning the previous bit."""
        return self.execute_command("SETBIT", key, offset, value)

    def getbit(self, key: KeyT, offset: int) -> int:
        """Return the bit at `offset` (0 if `key` or the offset is unset)."""
        return self.execute_command("GETBIT", key, offset)

    def bitcount(
        self,
        key: KeyT,
        start: Optional[int] = None,
        end: Optional[int] = None,
        *,
        bit: bool = False,
    ) -> int:
        """Count set bits, optionally within `start`..`end` (bytes, or bits if `bit=True`)."""
        args: List[Any] = [key]
        if start is not None and end is not None:
            args += [start, end]
            if bit:
                args.append("BIT")
        return self.execute_command("BITCOUNT", *args)

    def bitpos(
        self,
        key: KeyT,
        bit_value: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        *,
        bit: bool = False,
    ) -> int:
        """Find the first bit set to `bit_value`, or `-1` if none is found."""
        args: List[Any] = [key, bit_value]
        if start is not None:
            args.append(start)
            if end is not None:
                args.append(end)
                if bit:
                    args.append("BIT")
        return self.execute_command("BITPOS", *args)

    def bitop(self, operation: str, destination: KeyT, *keys: KeyT) -> int:
        """Apply a bitwise op ("AND"/"OR"/"XOR"/"NOT") across `keys` into `destination`."""
        return self.execute_command("BITOP", operation.upper(), destination, *keys)

    def bitfield(self, key: KeyT, operations: Sequence[Sequence[Any]]) -> List[Optional[int]]:
        """Run a batch of BITFIELD sub-operations, for example
        `[("SET", "u8", 0, 255), ("GET", "u8", 0)]` or
        `[("OVERFLOW", "SAT"), ("INCRBY", "i8", 0, 10)]`.
        """
        args: List[Any] = [key]
        for op in operations:
            args.extend(op)
        return self.execute_command("BITFIELD", *args)
