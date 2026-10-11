# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for the routing rules the sync and asyncio cluster clients
share (`hotkv._cluster_common`): which argument of a command is its key,
which commands are fanned out to every primary, and how their per-node
answers merge. No server needed."""

from __future__ import annotations

import pytest

from hotkv import _cluster_common as cc


def key_of(*args):
    return cc.routing_key(cc.command_name(args[0]), args)


@pytest.mark.parametrize(
    "args, expected",
    [
        (("GET", "k"), "k"),
        (("get", "k"), "k"),
        (("SET", "k", "v"), "k"),
        (("MGET", "a", "b"), "a"),
        (("PCACHE.GET", "ns", "prompt"), "ns"),
        (("SCACHE.SET", "ns", "q", "r", "0.1"), "ns"),
        (("VSIM.SEARCH", "idx", 3, 0.1, 0.2), "idx"),
        (("VSIM.CREATEINDEX", "idx", 4), "idx"),
        (("LLMBUDGET.CONSUME", "b", 10), "b"),
        (("AMEM.CONV.APPEND", "sess", "user", "hi"), "sess"),
        (("TS.ADD", "series", "*", 1), "series"),
        (("EVAL", "return 1", 2, "k1", "k2"), "k1"),
        (("EVALSHA", "abc", 1, "k1"), "k1"),
        (("FCALL", "fn", 1, "k1"), "k1"),
        (("EVAL", "return 1", 0), None),
        (("ZUNION", 2, "z1", "z2"), "z1"),
        (("ZINTERCARD", 2, "z1", "z2"), "z1"),
        (("LMPOP", 1, "list", "LEFT"), "list"),
        (("BLMPOP", 0, 1, "list", "LEFT"), "list"),
        (("SINTERCARD", 2, "s1", "s2"), "s1"),
        (("OBJECT", "ENCODING", "k"), "k"),
        (("MEMORY", "USAGE", "k"), "k"),
        (("MEMORY", "STATS"), None),
        (("XINFO", "STREAM", "s"), "s"),
        (("XGROUP", "CREATE", "s", "g", "$"), "s"),
        (("BITOP", "AND", "dest", "a", "b"), "dest"),
        (("XREAD", "COUNT", 2, "STREAMS", "s1", "s2", 0, 0), "s1"),
        (("XREADGROUP", "GROUP", "g", "c", "STREAMS", "s1", ">"), "s1"),
        (("PING",), None),
        (("INFO", "keyspace"), None),
        (("CONFIG", "GET", "maxmemory"), None),
        (("CLUSTER", "SLOTS"), None),
        (("SCAN", 0), None),
        (("LLMSTATS.PRICE", "GET", "model"), None),
        (("TENANT.CREATE", "acme"), None),
        (("PCACHE.LIST",), None),
        (("TAG.MEMBERS", "tag"), None),
    ],
)
def test_routing_key(args, expected):
    assert key_of(*args) == expected


@pytest.mark.parametrize(
    "name",
    [
        "SCACHE.LIST",
        "PCACHE.LIST",
        "NCACHE.LIST",
        "TOOL.CACHE.LIST",
        "GUARD.LIST",
        "RERANK.LIST",
        "LLMSTATS.LIST",
        "LLMBUDGET.LIST",
        "RATELIMIT.LIST",
        "PROMPT.LIST",
        "EMBED.DEDUP.LIST",
        "FEATURE.LIST",
        "RAG.LIST",
        "VSIM.LIST",
        "AMEM.SUMMARY.LIST",
        "TS.LIST",
        "TAG.MEMBERS",
        "TAG.INTER",
        "KEYS",
    ],
)
def test_node_local_listings_are_fanned_out_as_unions(name):
    assert cc.fanout_kind(name, (name,)) == cc.UNION


def test_other_fanout_kinds():
    assert cc.fanout_kind("DBSIZE", ("DBSIZE",)) == cc.SUM
    assert cc.fanout_kind("FLUSHALL", ("FLUSHALL",)) == cc.ALL
    assert cc.fanout_kind("SCRIPT", ("SCRIPT", "LOAD", "return 1")) == cc.ALL
    assert cc.fanout_kind("SCRIPT", ("SCRIPT", "EXISTS", "abc")) is None
    assert cc.fanout_kind("FUNCTION", ("FUNCTION", "LOAD", "#!lua name=lib")) == cc.ALL
    assert cc.fanout_kind("FUNCTION", ("FUNCTION", "delete", "lib")) == cc.ALL
    assert cc.fanout_kind("FUNCTION", ("FUNCTION", "LIST")) is None
    assert cc.fanout_kind("GET", ("GET", "k")) is None
    # VSIM.INFO lists the node's indexes without an argument, describes one with it.
    assert cc.fanout_kind("VSIM.INFO", ("VSIM.INFO",)) == cc.UNION
    assert cc.fanout_kind("VSIM.INFO", ("VSIM.INFO", "idx")) is None
    assert key_of("VSIM.INFO", "idx") == "idx"
    assert key_of("VSIM.INFO") is None
    # A keyed engine command is routed by its key, not fanned out.
    assert cc.fanout_kind("AMEM.SEM.LIST", ("AMEM.SEM.LIST", "agent")) is None


def test_merge_union_keeps_each_element_once_in_first_seen_order():
    merged = cc.merge_replies(cc.UNION, [[b"a", b"b"], [b"b", b"c"], None, []])
    assert merged == [b"a", b"b", b"c"]


def test_merge_union_handles_nested_replies():
    merged = cc.merge_replies(cc.UNION, [[[b"s1", 1]], [[b"s1", 1], [b"s2", 2]]])
    assert merged == [[b"s1", 1], [b"s2", 2]]


def test_merge_sum_and_all():
    assert cc.merge_replies(cc.SUM, [3, 4, 5]) == 12
    assert cc.merge_replies(cc.ALL, [b"OK", b"OK"]) == b"OK"
    assert cc.merge_replies(cc.UNION, []) == []
    assert cc.merge_replies(cc.SUM, []) == 0


def test_parse_slots_uses_the_asked_node_for_an_unannounced_host():
    reply = [
        [0, 99, [b"", 7000, b"id-a"], [b"10.0.0.2", 7001, b"id-b"]],
        [100, 16383, [None, 7002, b"id-c"]],
    ]
    masters, replicas = cc.parse_slots(reply, "seed.example")
    assert masters[0] == ("seed.example", 7000)
    assert replicas[50] == [("10.0.0.2", 7001)]
    assert masters[100] == ("seed.example", 7002)
    assert masters[16383] == ("seed.example", 7002)


@pytest.mark.parametrize(
    "args, expected",
    [
        # The typed wrappers send a command with its subcommand in one word.
        (("CONFIG GET", "maxmemory"), None),
        (("CLIENT SETNAME", "worker-1"), None),
        (("ALERTS SETURL", "https://alerts.example/hook"), None),
        (("ALERTS INFO",), None),
        (("LLMSTATS.PRICE SET", "gpt-x", "INPUT", 1, "OUTPUT", 2), None),
        (("HOTKEYS TOP", 10), None),
        (("XGROUP CREATE", "s", "g", "$"), "s"),
        (("XINFO STREAM", "s"), "s"),
        (("MEMORY USAGE", "k"), "k"),
        (("OBJECT ENCODING", "k"), "k"),
        # A snapshot of a vector index is named by the caller and sits where it
        # was saved: saving routes by that name.
        (("VSIM.SAVE", "snap-1", b"blob"), "snap-1"),
    ],
)
def test_a_subcommand_in_the_name_routes_as_one_after_it(args, expected):
    assert key_of(*args) == expected


def test_a_command_with_the_subcommand_in_its_name_is_routed_as_the_split_command_is():
    for split, joined in [
        (("CONFIG", "GET", "maxmemory"), ("CONFIG GET", "maxmemory")),
        (("XGROUP", "CREATE", "s", "g", "$"), ("XGROUP CREATE", "s", "g", "$")),
        (("ALERTS", "SETURL", "u"), ("ALERTS SETURL", "u")),
    ]:
        assert key_of(*split) == key_of(*joined)


def test_a_vector_snapshot_is_loaded_and_dropped_wherever_a_primary_holds_it():
    assert cc.fanout_kind("VSIM.LOAD", ("VSIM.LOAD", "snap-1")) == cc.FIRST
    assert cc.fanout_kind("VSIM.DROP", ("VSIM.DROP", "snap-1")) == cc.SUM
    assert cc.fanout_kind("VSIM.SAVE", ("VSIM.SAVE", "snap-1", b"x")) is None
    assert cc.merge_replies(cc.FIRST, [None, b"payload", b"stale"]) == b"payload"
    assert cc.merge_replies(cc.FIRST, [None, None]) is None
    assert cc.merge_replies(cc.FIRST, []) is None
    assert cc.merge_replies(cc.SUM, [0, 1, 0]) == 1


def test_a_refused_password_or_certificate_is_a_permanent_failure():
    import ssl

    from hotkv import exceptions as exc

    assert cc.is_permanent_failure(exc.AuthenticationError("WRONGPASS invalid username-password pair"))
    verification = exc.ConnectionError("TLS handshake with 10.0.0.1:7001 failed")
    verification.__cause__ = ssl.SSLCertVerificationError(1, "certificate verify failed")
    assert cc.is_permanent_failure(verification)
    # A node that went away, or one that dropped the handshake, is a failover.
    assert not cc.is_permanent_failure(exc.ConnectionError("connection refused"))
    assert not cc.is_permanent_failure(exc.TimeoutError("timed out"))
    dropped = exc.ConnectionError("TLS handshake failed")
    dropped.__cause__ = ssl.SSLEOFError(8, "EOF occurred in violation of protocol")
    assert not cc.is_permanent_failure(dropped)
