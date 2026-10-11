# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Exact wire arguments of typed methods whose engines answer NOLICENSE on an
unlicensed server (so they cannot be checked live there), against the documented
argument order, and the publish / pub-sub introspection methods
(live in tests/integration/test_command_coverage.py). No server needed."""

from __future__ import annotations

import pytest


def test_alerts_arguments(recording_client):
    recording_client.alerts_seturl("https://hooks.example/x")
    assert recording_client.last_call == ("ALERTS SETURL", "https://hooks.example/x")
    recording_client.alerts_test()
    assert recording_client.last_call == ("ALERTS TEST",)
    recording_client.alerts_test("warning", "disk is filling")
    assert recording_client.last_call == ("ALERTS TEST", "warning", "disk is filling")


@pytest.mark.parametrize(
    "prefix, name", [("PCACHE", "pcache"), ("TOOL.CACHE", "toolcache"), ("NCACHE", "ncache"), ("GUARD", "guard"), ("RERANK", "rerank")]
)
def test_cache_flush_arguments(recording_client, prefix, name):
    getattr(recording_client, f"{name}_flush")("ns")
    assert recording_client.last_call == (f"{prefix}.FLUSH", "ns")


def test_embed_dedup_list(recording_client):
    recording_client.embed_dedup_list()
    assert recording_client.last_call == ("EMBED.DEDUP.LIST",)


def test_publish_and_introspection_arguments(recording_client):
    assert recording_client.publish("ch", "msg") is None  # no canned reply queued
    assert recording_client.last_call == ("PUBLISH", "ch", "msg")
    recording_client.spublish("sh", b"\x00\xff")
    assert recording_client.last_call == ("SPUBLISH", "sh", b"\x00\xff")
    recording_client.when("PUBSUB CHANNELS", [b"a"])
    assert recording_client.pubsub_channels() == [b"a"]
    assert recording_client.last_call == ("PUBSUB CHANNELS",)
    recording_client.pubsub_channels("n*")
    assert recording_client.last_call == ("PUBSUB CHANNELS", "n*")
    recording_client.when("PUBSUB NUMSUB", [b"a", 2, b"b", 0])
    assert recording_client.pubsub_numsub("a", "b") == {b"a": 2, b"b": 0}
    assert recording_client.last_call == ("PUBSUB NUMSUB", "a", "b")
    recording_client.when("PUBSUB NUMSUB", {b"a": 3})  # a RESP3 map
    assert recording_client.pubsub_numsub("a") == {b"a": 3}
    recording_client.pubsub_numpat()
    assert recording_client.last_call == ("PUBSUB NUMPAT",)
    recording_client.pubsub_shardchannels("s*")
    assert recording_client.last_call == ("PUBSUB SHARDCHANNELS", "s*")
    recording_client.when("PUBSUB SHARDNUMSUB", [b"s", 1])
    assert recording_client.pubsub_shardnumsub("s") == {b"s": 1}
