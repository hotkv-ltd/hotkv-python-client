# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for the cluster client's slot map and routing logic, using a
synthetic `CLUSTER SLOTS` reply. No server needed: `refresh_slots()` is
monkeypatched out during construction so this never touches the network.
"""

from __future__ import annotations

from unittest.mock import patch

from hotkv._crc16 import key_slot
from hotkv.cluster import ClusterHotKV

# Two shards, matching the wire shape of a CLUSTER SLOTS reply: for each
# entry, [start, end, [master_host, master_port, node_id], [replica...]].
SAMPLE_SLOTS_REPLY = [
    [0, 8191, [b"10.0.0.1", 7001, b"node-a"], [b"10.0.0.2", 7002, b"node-a-replica"]],
    [8192, 16383, [b"10.0.0.3", 7003, b"node-b"], [b"10.0.0.4", 7004, b"node-b-replica"]],
]


def make_cluster(**kwargs) -> ClusterHotKV:
    with patch.object(ClusterHotKV, "refresh_slots", lambda self: None):
        cluster = ClusterHotKV([("10.0.0.1", 7001)], **kwargs)
    cluster._apply_slots(SAMPLE_SLOTS_REPLY)
    return cluster


def test_apply_slots_builds_full_slot_map():
    cluster = make_cluster()
    assert cluster._slot_masters[0] == ("10.0.0.1", 7001)
    assert cluster._slot_masters[8191] == ("10.0.0.1", 7001)
    assert cluster._slot_masters[8192] == ("10.0.0.3", 7003)
    assert cluster._slot_masters[16383] == ("10.0.0.3", 7003)


def test_masters_property():
    cluster = make_cluster()
    assert cluster.masters == [("10.0.0.1", 7001), ("10.0.0.3", 7003)]


def test_routing_by_first_argument_matches_key_slot():
    cluster = make_cluster()
    for key in ("foo", "bar", "hello", "some:other:key"):
        expected_slot = key_slot(key)
        expected_master = cluster._slot_masters[expected_slot]
        address = cluster._target_address(("GET", key), readonly=True)
        assert address == expected_master


def test_keyless_command_routes_to_a_known_master():
    cluster = make_cluster()
    address = cluster._target_address(("PING",), readonly=True)
    assert address in cluster.masters


def test_read_from_replicas_when_enabled():
    cluster = make_cluster(read_from_replicas=True)
    slot = key_slot("foo")
    address = cluster._target_address(("GET", "foo"), readonly=True)
    assert address == cluster._slot_replicas[slot][0]


def test_read_from_replicas_disabled_by_default():
    cluster = make_cluster()
    slot = key_slot("foo")
    address = cluster._target_address(("GET", "foo"), readonly=True)
    assert address == cluster._slot_masters[slot]


def test_write_always_routes_to_master_even_with_replicas_enabled():
    cluster = make_cluster(read_from_replicas=True)
    slot = key_slot("foo")
    address = cluster._target_address(("SET", "foo"), readonly=False)
    assert address == cluster._slot_masters[slot]


def test_pool_for_reuses_pool_per_address():
    cluster = make_cluster()
    address = ("10.0.0.1", 7001)
    pool1 = cluster._pool_for(address)
    pool2 = cluster._pool_for(address)
    assert pool1 is pool2
