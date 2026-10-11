# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for the core (license-free) HotKV exclusive commands:
TAG.*, nanosecond TTLs, HOTKEYS, GCRA.* and VSIM.* (persistence sub-commands
need a license that includes them and are skipped when it does not).
"""

from __future__ import annotations

import hotkv


def test_tag_set_get_rem_count_exists(client, prefix):
    key = prefix + "tagged"
    client.set(key, "v")
    assert client.tag_set(key, "red", "blue") == 2
    assert set(client.tag_get(key)) == {b"red", b"blue"}
    assert client.tag_count(key) == 2
    assert client.tag_exists(key, "red") is True
    assert client.tag_exists(key, "green") is False
    assert client.tag_rem(key, "red") == 1
    assert client.tag_get(key) == [b"blue"]
    client.delete(key)


def test_tag_members_and_inter(client, prefix):
    k1, k2 = prefix + "t1", prefix + "t2"
    shared_tag = prefix + "shared"  # unique per test run: tags are not key-scoped
    client.set(k1, "v")
    client.set(k2, "v")
    client.tag_set(k1, shared_tag, "onlyone")
    client.tag_set(k2, shared_tag, "onlytwo")
    members = set(client.tag_members(shared_tag))
    assert members == {k1.encode(), k2.encode()}
    assert client.tag_inter(shared_tag, "onlyone") == [k1.encode()]
    client.delete(k1, k2)


def test_tag_set_requires_existing_key(client, prefix):
    key = prefix + "notexist"
    import pytest

    with pytest.raises(hotkv.ResponseError):
        client.tag_set(key, "x")


def test_nexpire_nttl_nexpireat(client, prefix):
    key = prefix + "nano"
    client.set(key, "v")
    assert client.nttl(key) == -1
    assert client.nexpire(key, 60_000_000_000) is True
    ttl_ns = client.nttl(key)
    assert 0 < ttl_ns <= 60_000_000_000
    assert client.nexpireat(key, (2 ** 62)) is True
    assert client.nttl(key) > 0
    client.delete(key)
    assert client.nttl(key) == -2


def test_hotkeys(client, prefix):
    key = prefix + "hot"
    client.set(key, "v")
    for _ in range(5):
        client.get(key)
    top = client.hotkeys_top(50)
    assert isinstance(top, list)
    if top:
        sample_key, sample_count = top[0]
        assert isinstance(sample_count, int)
    assert client.hotkeys_reset() is True
    client.delete(key)


def test_gcra_limit_and_reset(client, prefix):
    key = prefix + "gcra"
    result = client.gcra_limit(key, 60_000, 2)
    assert result.denied is False
    assert result.allowed is True
    assert result.limit == 2
    result2 = client.gcra_limit(key, 60_000, 2)
    result3 = client.gcra_limit(key, 60_000, 2)
    assert result3.denied is True
    assert result3.retry_after_ms >= 0
    assert client.gcra_reset(key) == 1


def test_vsim_index_lifecycle(client, prefix):
    index = prefix + "vsimidx"
    assert client.vsim_createindex(index, 4, metric="l2") is True
    assert client.vsim_set(index, "a", [1.0, 0.0, 0.0, 0.0]) is True
    assert client.vsim_set(index, "b", [0.0, 1.0, 0.0, 0.0]) is True
    vec = client.vsim_get(index, "a")
    assert vec is not None and len(vec) == 4
    results = client.vsim_search(index, 2, [1.0, 0.0, 0.0, 0.0])
    assert results[0][0] == b"a"
    info = client.vsim_info(index)
    assert info["dimensions"] == 4
    assert info["metric"] == "l2"
    all_indexes = client.vsim_info()
    assert index.encode() in all_indexes
    assert client.vsim_del(index, "b") is True
    assert client.vsim_dropindex(index) is True


def test_vsim_search_with_filter(client, prefix):
    index = prefix + "vsimfilt"
    client.vsim_createindex(index, 2)
    client.vsim_set(index, "x", [1.0, 0.0])
    client.vsim_set(index, "y", [0.9, 0.1])
    client.set(index + ":x", "marker")
    client.tag_set(index + ":x", "keep")
    # FILTER matches against TAG.INTER, so tag the vector's *key* is not
    # directly relevant here; just confirm the FILTER argument is accepted.
    results = client.vsim_search(index, 5, [1.0, 0.0], filter_tags=["nonexistent-tag"])
    assert results == []
    client.vsim_dropindex(index)
    client.delete(index + ":x")


def test_vsim_persistence_requires_full_license(client, prefix):
    # Snapshot names take letters, digits and a few separators only.
    name = "".join(ch for ch in prefix if ch.isalnum()) + "snap"
    try:
        client.vsim_save(name, b"blob-data")
    except hotkv.NotLicensedError as exc:
        import pytest

        pytest.skip(f"VSIM persistence not licensed: {exc}")
    else:
        assert client.vsim_load(name) == b"blob-data"
        assert name.encode() in b"".join(client.vsim_list())
        assert client.vsim_drop(name) is True
