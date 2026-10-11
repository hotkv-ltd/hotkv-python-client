# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration sanity test for `ClusterHotKV` against a standalone server.

A standalone server is a single node, not a real cluster (`CLUSTER SLOTS`
replies with an empty list), so this only exercises what is actually
reachable here: construction (which calls `refresh_slots()`) and the clear,
typed error once a command needs to be routed. Full slot-routing and
redirect-following behavior is covered without a server in
`tests/unit/test_cluster_routing.py`.
"""

from __future__ import annotations

import pytest

import hotkv
from hotkv.cluster import ClusterHotKV


def test_cluster_client_constructs_against_a_single_node(test_url):
    import urllib.parse

    parts = urllib.parse.urlsplit(test_url)
    cluster = ClusterHotKV([(parts.hostname, parts.port)])
    try:
        assert cluster.masters == []  # no slots assigned on a non-cluster node
        with pytest.raises(hotkv.ClusterError):
            cluster.get("some-key")
    finally:
        cluster.close()
