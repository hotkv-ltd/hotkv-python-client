# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for how the sync and asyncio cluster clients ride out redirects,
a failover and node-local commands, with a scripted stand-in for the network
round trip. No server needed."""

from __future__ import annotations

import ssl
import time
from typing import Any, Callable, List, Tuple
from unittest.mock import patch

import pytest

import hotkv.asyncio
from hotkv import exceptions as exc
from hotkv._crc16 import key_slot
from hotkv.cluster import ClusterHotKV

A = ("10.0.0.1", 7001)
B = ("10.0.0.3", 7003)
A_REPLICA = ("10.0.0.2", 7002)
SLOTS = [
    [0, 8191, [b"10.0.0.1", 7001, b"a"], [b"10.0.0.2", 7002, b"a2"]],
    [8192, 16383, [b"10.0.0.3", 7003, b"b"]],
]
# A key whose slot lives on node A.
KEY_A = next(f"k{i}" for i in range(1000) if key_slot(f"k{i}") < 8192)


class Net:
    """Scripted replies: `script` maps a call number to a function of
    (address, args, asking, replica) returning a reply or raising."""

    def __init__(self, handler: Callable[..., Any]) -> None:
        self.handler = handler
        self.calls: List[Tuple[Any, ...]] = []

    def __call__(self, address, args, *, asking=False, replica=False):
        self.calls.append((address, tuple(args), asking, replica))
        return self.handler(len(self.calls), address, tuple(args), asking, replica)


def connection_error(sent: bool) -> exc.ConnectionError:
    err = exc.ConnectionError("connection refused")
    err.sent = sent  # type: ignore[attr-defined]
    return err


def make_sync(handler, **kwargs) -> Tuple[ClusterHotKV, Net]:
    with patch.object(ClusterHotKV, "refresh_slots", lambda self: None):
        cluster = ClusterHotKV([A], retry_timeout=1.0, **kwargs)
    cluster._apply_slots(SLOTS)
    net = Net(handler)
    cluster._round_trip = net  # type: ignore[assignment]
    cluster.refreshes = 0

    def refresh():
        cluster.refreshes += 1

    cluster.refresh_slots = refresh  # type: ignore[assignment]
    return cluster, net


def test_moved_is_followed_to_the_new_owner():
    def handler(n, address, args, asking, replica):
        if n == 1:
            return exc.MovedError(f"{key_slot(KEY_A)} 10.0.0.3:7003")
        return b"value"

    cluster, net = make_sync(handler)
    assert cluster.get(KEY_A) == b"value"
    assert net.calls[0][0] == A
    assert net.calls[1][0] == B
    assert cluster.refreshes == 1
    # The slot now maps to the node that answered MOVED.
    assert cluster._slot_masters[key_slot(KEY_A)] == B


def test_ask_sends_asking_to_the_target_without_remapping():
    def handler(n, address, args, asking, replica):
        if n == 1:
            return exc.AskError(f"{key_slot(KEY_A)} 10.0.0.3:7003")
        return b"migrating"

    cluster, net = make_sync(handler)
    assert cluster.get(KEY_A) == b"migrating"
    assert net.calls[1][0] == B and net.calls[1][2] is True
    assert cluster._slot_masters[key_slot(KEY_A)] == A


def test_a_redirect_loop_gives_up():
    cluster, net = make_sync(lambda *a: exc.AskError(f"{key_slot(KEY_A)} 10.0.0.3:7003"), max_redirects=3)
    with pytest.raises(exc.ClusterError, match="too many redirects"):
        cluster.get(KEY_A)
    assert len(net.calls) == 4


def test_a_dead_primary_is_ridden_out_by_refreshing_and_retrying():
    def handler(n, address, args, asking, replica):
        if n <= 2:
            raise connection_error(sent=False)
        return b"after-failover"

    cluster, net = make_sync(handler)
    assert cluster.get(KEY_A) == b"after-failover"
    assert len(net.calls) == 3
    assert cluster.refreshes == 2


def test_clusterdown_and_tryagain_replies_are_retried():
    def handler(n, address, args, asking, replica):
        if n == 1:
            return exc.ClusterDownError("The cluster is down")
        if n == 2:
            return exc.TryAgainError("Multiple keys request during rehashing of slot")
        return b"ok"

    cluster, net = make_sync(handler)
    assert cluster.get(KEY_A) == b"ok"
    assert len(net.calls) == 3


def test_a_write_to_a_demoted_primary_is_retried_after_a_refresh():
    def handler(n, address, args, asking, replica):
        if n == 1:
            return exc.ResponseError("You can't write against a read only replica.", "READONLY")
        return b"OK"

    cluster, net = make_sync(handler)
    cluster.set(KEY_A, "v")
    assert len(net.calls) == 2


def test_retries_end_with_a_cluster_error_after_the_timeout():
    cluster, net = make_sync(lambda *a: (_ for _ in ()).throw(connection_error(sent=False)))
    cluster.retry_timeout = 0.3
    with pytest.raises(exc.ClusterError, match="giving up"):
        cluster.get(KEY_A)


def test_without_retry_writes_a_write_lost_in_flight_is_not_repeated():
    def handler(n, address, args, asking, replica):
        raise connection_error(sent=True)

    cluster, net = make_sync(handler, retry_writes=False)
    with pytest.raises(exc.ConnectionError):
        cluster.set(KEY_A, "v")
    assert len(net.calls) == 1
    # A read is repeated all the same, and so is a write that never left.
    cluster.retry_timeout = 0.3
    with pytest.raises(exc.ClusterError):
        cluster.get(KEY_A)
    assert len(net.calls) > 2


def test_without_retry_writes_a_write_that_never_left_is_retried():
    def handler(n, address, args, asking, replica):
        if n == 1:
            raise connection_error(sent=False)
        return b"OK"

    cluster, net = make_sync(handler, retry_writes=False)
    cluster.set(KEY_A, "v")
    assert len(net.calls) == 2


def test_reads_go_to_a_replica_and_fall_back_to_the_primary_when_it_is_down():
    def handler(n, address, args, asking, replica):
        if replica:
            raise connection_error(sent=False)
        return b"from-primary"

    cluster, net = make_sync(handler, read_from_replicas=True)
    assert cluster.get(KEY_A) == b"from-primary"
    assert net.calls[0][0] == A_REPLICA and net.calls[0][3] is True
    assert net.calls[1][0] == A and net.calls[1][3] is False


def test_listings_are_asked_of_every_primary_and_merged():
    def handler(n, address, args, asking, replica):
        assert args == ("PCACHE.LIST",)
        return [b"ns-a1", b"shared"] if address == A else [b"shared", b"ns-b1"]

    cluster, net = make_sync(handler)
    assert cluster.pcache_list() == [b"ns-a1", b"shared", b"ns-b1"]
    assert sorted(c[0] for c in net.calls) == sorted([A, B])


def test_counts_are_summed_over_the_primaries():
    cluster, net = make_sync(lambda n, address, *a: 5 if address == A else 7)
    assert cluster.dbsize() == 12


def test_a_listing_restarts_when_a_primary_turns_out_to_be_stale():
    def handler(n, address, args, asking, replica):
        if n == 2:
            return exc.MovedError("5000 10.0.0.9:7009")
        return [b"x"]

    cluster, net = make_sync(handler)
    assert cluster.pcache_list() == [b"x"]
    assert len(net.calls) == 4  # A, B (stale), then A and B again


def test_keyless_commands_go_to_a_primary_and_eval_is_routed_by_its_key():
    cluster, net = make_sync(lambda *a: b"PONG")
    cluster.execute("PING")
    assert net.calls[-1][0] in (A, B)
    cluster.execute("EVAL", "return 1", 1, KEY_A)
    assert net.calls[-1][0] == A


async def _async_cluster(handler, **kwargs):
    from hotkv.asyncio.cluster import ClusterHotKV as AsyncCluster

    cluster = AsyncCluster([A], retry_timeout=1.0, **kwargs)
    cluster._apply_slots(SLOTS)
    cluster._ready = True
    net = Net(handler)

    async def round_trip(address, args, *, asking=False, replica=False):
        return net(address, args, asking=asking, replica=replica)

    async def refresh():
        cluster.refreshes += 1

    cluster.refreshes = 0
    cluster._round_trip = round_trip  # type: ignore[assignment]
    cluster.refresh_slots = refresh  # type: ignore[assignment]
    return cluster, net


async def test_async_moved_ask_and_failover():
    def handler(n, address, args, asking, replica):
        if n == 1:
            return exc.MovedError(f"{key_slot(KEY_A)} 10.0.0.3:7003")
        if n == 2:
            raise connection_error(sent=False)
        if n == 3:
            return exc.AskError(f"{key_slot(KEY_A)} 10.0.0.1:7001")
        return b"done"

    cluster, net = await _async_cluster(handler)
    assert await cluster.get(KEY_A) == b"done"
    assert [c[0] for c in net.calls] == [A, B, B, A]
    assert net.calls[3][2] is True


async def test_async_listings_and_sums():
    def handler(n, address, args, asking, replica):
        if args[0] == "DBSIZE":
            return 4 if address == A else 6
        return [b"a"] if address == A else [b"b"]

    cluster, net = await _async_cluster(handler)
    assert await cluster.scache_list() == [b"a", b"b"]
    assert await cluster.dbsize() == 10


async def test_async_read_from_replicas_falls_back():
    def handler(n, address, args, asking, replica):
        if replica:
            raise connection_error(sent=False)
        return b"v"

    cluster, net = await _async_cluster(handler, read_from_replicas=True)
    assert await cluster.get(KEY_A) == b"v"
    assert net.calls[0][3] is True and net.calls[1][3] is False


def test_async_client_is_exported():
    assert hotkv.asyncio.ClusterHotKV is not None


def test_a_refused_password_is_raised_at_once_not_retried_until_the_timeout():
    def handler(n, address, args, asking, replica):
        err = exc.AuthenticationError("WRONGPASS invalid username-password pair")
        err.sent = False  # type: ignore[attr-defined]
        raise err

    cluster, net = make_sync(handler)
    cluster.retry_timeout = 30.0
    started = time.monotonic()
    with pytest.raises(exc.AuthenticationError):
        cluster.get(KEY_A)
    assert time.monotonic() - started < 5.0
    assert len(net.calls) == 1
    # The same for a command fanned out to every primary.
    with pytest.raises(exc.AuthenticationError):
        cluster.pcache_list()
    assert time.monotonic() - started < 5.0


def test_a_refused_certificate_is_raised_at_once_but_a_dropped_handshake_is_retried():
    def refused(n, address, args, asking, replica):
        err = exc.ConnectionError("TLS handshake with 10.0.0.1:7001 failed")
        err.__cause__ = ssl.SSLCertVerificationError(1, "certificate verify failed")
        err.sent = False  # type: ignore[attr-defined]
        raise err

    cluster, net = make_sync(refused)
    cluster.retry_timeout = 30.0
    with pytest.raises(exc.ConnectionError, match="TLS handshake"):
        cluster.get(KEY_A)
    assert len(net.calls) == 1

    def dropped_once(n, address, args, asking, replica):
        if n == 1:
            err = exc.ConnectionError("TLS handshake with 10.0.0.1:7001 failed")
            err.__cause__ = ssl.SSLEOFError(8, "EOF occurred in violation of protocol")
            err.sent = False  # type: ignore[attr-defined]
            raise err
        return b"v"

    cluster, net = make_sync(dropped_once)
    assert cluster.get(KEY_A) == b"v"
    assert len(net.calls) == 2


def test_a_write_lost_in_flight_is_sent_again_by_default_and_not_with_retry_writes_off():
    """The default is at-least-once delivery (it rides out a failover): a counter
    can then be applied twice when the first copy did reach a live node. With
    retry_writes=False a counter command is never repeated."""

    def lost_once(n, address, args, asking, replica):
        if n == 1:
            raise connection_error(sent=True)
        return 1

    cluster, net = make_sync(lost_once)
    assert cluster.incr(KEY_A) == 1
    assert [c[1][0] for c in net.calls] == ["INCR", "INCR"]

    def always_lost(n, address, args, asking, replica):
        raise connection_error(sent=True)

    cluster, net = make_sync(always_lost, retry_writes=False)
    with pytest.raises(exc.ConnectionError):
        cluster.execute("LLMBUDGET.CONSUME", KEY_A, 100)
    with pytest.raises(exc.ConnectionError):
        cluster.execute("RATELIMIT.CHECK", KEY_A, 1)
    with pytest.raises(exc.ConnectionError):
        cluster.incr(KEY_A)
    assert len(net.calls) == 3  # each sent once, none repeated


def test_a_vector_snapshot_is_saved_by_its_name_and_loaded_from_whichever_primary_has_it():
    def handler(n, address, args, asking, replica):
        if args[0] == "VSIM.LOAD":
            return b"snapshot-bytes" if address == B else None
        if args[0] == "VSIM.DROP":
            return 1 if address == B else 0
        return b"OK"

    cluster, net = make_sync(handler)
    cluster.vsim_save(KEY_A, b"x")
    assert net.calls[-1][0] == A  # by the snapshot's name, like a key
    assert cluster.vsim_load("snap-on-b") == b"snapshot-bytes"
    assert sorted(c[0] for c in net.calls[1:3]) == sorted([A, B])  # asked of every primary
    assert cluster.vsim_drop("snap-on-b") is True


async def test_async_a_refused_password_is_raised_at_once():
    def handler(n, address, args, asking, replica):
        err = exc.AuthenticationError("WRONGPASS invalid username-password pair")
        err.sent = False  # type: ignore[attr-defined]
        raise err

    cluster, net = await _async_cluster(handler)
    cluster.retry_timeout = 30.0
    started = time.monotonic()
    with pytest.raises(exc.AuthenticationError):
        await cluster.get(KEY_A)
    with pytest.raises(exc.AuthenticationError):
        await cluster.pcache_list()
    assert time.monotonic() - started < 5.0
    assert len(net.calls) == 2  # one try each, no retry loop
