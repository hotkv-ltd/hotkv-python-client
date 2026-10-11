# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`pxat=` (the absolute Unix time in milliseconds an entry expires at) next to
`ttl=` on the AI engine writes that take either, checked against the
documented argument order. Wire only: these engines answer
NOLICENSE on an unlicensed server. No server needed."""

from __future__ import annotations

import pytest

import hotkv

AT = 1893456000000


def test_scache_set_pxat(recording_client):
    recording_client.scache_set("ns", "q", "r", [1.0, 0.5], pxat=AT)
    assert recording_client.last_call == ("SCACHE.SET", "ns", "q", "r", "1", "0.5", "PXAT", AT)


def test_pcache_set_pxat_with_meta_and_usage(recording_client):
    recording_client.pcache_set(
        "ns", "p", "r", pxat=AT, meta="{}", usage=hotkv.TokenUsage(model="m", input_tokens=3)
    )
    assert recording_client.last_call == (
        "PCACHE.SET", "ns", "p", "r", "PXAT", AT, "META", "{}", "MODEL", "m", "INPUT", 3
    )


@pytest.mark.parametrize("method", ["toolcache_set", "ncache_set", "guard_set", "rerank_set"])
def test_keyed_cache_set_pxat(recording_client, method):
    getattr(recording_client, method)("ns", "k", "v", pxat=AT)
    assert recording_client.last_call[-2:] == ("PXAT", AT)
    getattr(recording_client, method)("ns", "k", "v", ttl=60)
    assert recording_client.last_call[-2:] == ("TTL", 60)


def test_amem_pxat(recording_client):
    recording_client.amem_conv_create("s", window=5, pxat=AT)
    assert recording_client.last_call == ("AMEM.CONV.CREATE", "s", "WINDOW", 5, "PXAT", AT)
    recording_client.amem_sem_store("a", "k", "c", [1.0, 2.0], pxat=AT)
    assert recording_client.last_call == ("AMEM.SEM.STORE", "a", "k", "c", "1", "2", "PXAT", AT)
    recording_client.amem_work_set("s", "k", "v", pxat=AT)
    assert recording_client.last_call == ("AMEM.WORK.SET", "s", "k", "v", "PXAT", AT)


def test_ttl_still_works_and_both_together_is_refused(recording_client):
    recording_client.pcache_set("ns", "p", "r", ttl=30)
    assert recording_client.last_call == ("PCACHE.SET", "ns", "p", "r", "TTL", 30)
    for call in (
        lambda: recording_client.pcache_set("ns", "p", "r", ttl=30, pxat=AT),
        lambda: recording_client.scache_set("ns", "q", "r", [1.0], ttl=30, pxat=AT),
        lambda: recording_client.amem_work_set("s", "k", "v", ttl=30, pxat=AT),
    ):
        with pytest.raises(hotkv.DataError):
            call()
