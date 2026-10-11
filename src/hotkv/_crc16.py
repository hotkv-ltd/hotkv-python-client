# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""CRC16/XMODEM and cluster key slot computation.

HotKV's cluster mode uses the same 16384 slot, CRC16/XMODEM hashing scheme
as the RESP protocol it is compatible with, including the `{hash tag}`
convention that lets callers force related keys onto the same slot.
"""

from __future__ import annotations

from typing import Union

__all__ = ["crc16", "key_slot", "TOTAL_SLOTS"]

TOTAL_SLOTS = 16384

_POLY = 0x1021


def _build_table() -> tuple:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_TABLE = _build_table()


def crc16(data: bytes) -> int:
    """Raw CRC16/XMODOM checksum of `data` (no slot reduction)."""
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc & 0xFFFF


def key_slot(key: Union[str, bytes]) -> int:
    """Return the cluster slot (0..16383) that `key` hashes to.

    If `key` contains a `{...}` hash tag with at least one character between
    the braces, only that substring is hashed, so all keys sharing a tag
    land on the same slot. An empty tag (`{}`) is not a hash tag and the
    whole key is hashed, matching standard cluster semantics.
    """
    if isinstance(key, str):
        key = key.encode("utf-8")
    start = key.find(b"{")
    if start != -1:
        end = key.find(b"}", start + 1)
        if end != -1 and end != start + 1:
            key = key[start + 1 : end]
    return crc16(key) % TOTAL_SLOTS
