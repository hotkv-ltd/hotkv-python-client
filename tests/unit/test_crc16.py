# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""CRC16/XMODEM and cluster key slot vectors, matching `CLUSTER KEYSLOT`."""

from __future__ import annotations

import pytest

from hotkv._crc16 import TOTAL_SLOTS, crc16, key_slot


def test_raw_crc16_xmodem_vector():
    assert crc16(b"123456789") == 0x31C3


@pytest.mark.parametrize(
    "key, expected_slot",
    [
        ("foo", 12182),
        ("bar", 5061),
        ("hello", 866),
        ("{user1000}.following", 3443),
        ("{user1000}.followers", 3443),
        ("{}foo", 9500),
        ("foo{}{bar}", 8363),
        ("foo{{bar}}zap", 4015),
        ("foo{bar}{zap}", 5061),
    ],
)
def test_key_slot_vectors(key, expected_slot):
    assert key_slot(key) == expected_slot


def test_key_slot_accepts_bytes_and_str_identically():
    assert key_slot("foo") == key_slot(b"foo")


def test_hash_tag_colocates_related_keys():
    assert key_slot("{user1000}.following") == key_slot("{user1000}.followers")


def test_slot_always_in_range():
    for key in ("a", "b", "some-longer-key-name", "{tag}rest", ""):
        slot = key_slot(key)
        assert 0 <= slot < TOTAL_SLOTS
