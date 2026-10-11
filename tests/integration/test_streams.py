# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for stream commands and consumer groups."""

from __future__ import annotations


def test_xadd_xlen_xrange(client, prefix):
    key = prefix + "stream"
    id1 = client.xadd(key, {"field": "1"})
    id2 = client.xadd(key, {"field": "2"})
    assert client.xlen(key) == 2
    entries = client.xrange(key)
    assert [eid for eid, _ in entries] == [id1, id2]
    assert entries[0][1] == {b"field": b"1"}
    client.delete(key)


def test_xrevrange(client, prefix):
    key = prefix + "stream-rev"
    client.xadd(key, {"a": "1"})
    client.xadd(key, {"a": "2"})
    entries = client.xrevrange(key)
    assert entries[0][1] == {b"a": b"2"}
    client.delete(key)


def test_xdel_xtrim(client, prefix):
    key = prefix + "stream-del"
    id1 = client.xadd(key, {"a": "1"})
    client.xadd(key, {"a": "2"})
    client.xadd(key, {"a": "3"})
    assert client.xdel(key, id1) == 1
    assert client.xlen(key) == 2
    client.xtrim(key, maxlen=1)
    assert client.xlen(key) == 1
    client.delete(key)


def test_xread(client, prefix):
    key = prefix + "stream-read"
    client.xadd(key, {"a": "1"}, id="1-1")
    client.xadd(key, {"a": "2"}, id="2-1")
    result = client.xread({key: "0"})
    assert result is not None
    entries = result[key.encode()]
    assert [eid for eid, _ in entries] == [b"1-1", b"2-1"]
    client.delete(key)


def test_consumer_group_flow(client, prefix):
    key = prefix + "stream-group"
    group = "g1"
    consumer = "c1"
    client.xadd(key, {"a": "1"}, id="1-1")
    client.xadd(key, {"a": "2"}, id="2-1")
    assert client.xgroup_create(key, group, "0") is True

    result = client.xreadgroup(group, consumer, {key: ">"})
    entries = result[key.encode()]
    assert len(entries) == 2

    ids = [eid for eid, _ in entries]
    assert client.xack(key, group, *ids[:1]) == 1

    summary = client.xpending(key, group)
    count, min_id, max_id, consumers = summary
    assert count == 1

    extended = client.xpending(key, group, count=10)
    assert len(extended) == 1
    pending_id, pending_consumer, idle_ms, deliveries = extended[0]
    assert pending_consumer == consumer.encode()

    claimed = client.xclaim(key, group, "c2", 0, [ids[1]])
    assert claimed[0][0] == ids[1]

    groups_info = client.xinfo_groups(key)
    assert any(g["name"] == group for g in groups_info)

    consumers_info = client.xinfo_consumers(key, group)
    assert len(consumers_info) >= 1

    stream_info = client.xinfo_stream(key)
    assert stream_info["length"] == 2

    assert client.xgroup_destroy(key, group) is True
    client.delete(key)


def test_xautoclaim(client, prefix):
    key = prefix + "stream-autoclaim"
    group = "g1"
    client.xadd(key, {"a": "1"}, id="1-1")
    client.xgroup_create(key, group, "0")
    client.xreadgroup(group, "c1", {key: ">"})
    cursor, entries, deleted = client.xautoclaim(key, group, "c2", 0, "0-0")
    assert len(entries) == 1
    assert entries[0][0] == b"1-1"
    client.delete(key)
