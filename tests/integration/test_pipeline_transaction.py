# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for pipelining, MULTI/EXEC transactions, and WATCH."""

from __future__ import annotations

import pytest

import hotkv


def test_plain_pipeline_batches_and_preserves_order(client, prefix):
    k1, k2 = prefix + "p1", prefix + "p2"
    with client.pipeline(transaction=False) as pipe:
        pipe.set(k1, "1")
        pipe.incr(k1)
        pipe.set(k2, "x")
        pipe.get(k2)
        results = pipe.execute()
    assert results == [True, 2, True, b"x"]
    client.delete(k1, k2)


def test_pipeline_error_does_not_abort_the_batch_when_allowed(client, prefix):
    key = prefix + "notanumber"
    client.set(key, "abc")
    with client.pipeline(transaction=False) as pipe:
        pipe.set(prefix + "ok", "v")
        pipe.incr(key)  # ERR value is not an integer
        pipe.get(prefix + "ok")
        results = pipe.execute(raise_on_error=False)
    assert results[0] is True
    assert isinstance(results[1], hotkv.ResponseError)
    assert results[1].code == "ERR"
    assert results[2] == b"v"
    client.delete(key, prefix + "ok")


def test_transaction_wraps_in_multi_exec(client, prefix):
    key = prefix + "txn"
    with client.pipeline(transaction=True) as pipe:
        pipe.set(key, "1")
        pipe.incr(key)
        results = pipe.execute()
    assert results == [True, 2]
    client.delete(key)


def test_watch_transaction_helper_commits(client, prefix):
    key = prefix + "balance"
    client.set(key, 100)

    def txn(pipe):
        current = pipe.get(key)
        pipe.multi()
        pipe.set(key, int(current) - 10)

    client.transaction(txn, key)
    assert client.get(key) == b"90"
    client.delete(key)


def test_watch_transaction_retries_on_conflict(client, prefix, test_url):
    key = prefix + "contested"
    client.set(key, 1)
    attempt_count = {"n": 0}

    def txn(pipe):
        attempt_count["n"] += 1
        current = pipe.get(key)
        if attempt_count["n"] == 1:
            # A concurrent writer changes the watched key before this
            # attempt reaches MULTI, so its EXEC must abort and the
            # transaction() helper must retry.
            interloper = hotkv.from_url(test_url)
            interloper.set(key, 999)
            interloper.close()
        pipe.multi()
        pipe.set(key, int(current) + 1)

    client.transaction(txn, key)
    assert attempt_count["n"] >= 2
    assert client.get(key) == b"1000"
    client.delete(key)


def test_pipeline_reset_releases_connection_for_reuse(client, prefix):
    key = prefix + "reuse"
    pipe = client.pipeline(transaction=False)
    pipe.set(key, "1")
    pipe.execute()
    pipe2 = client.pipeline(transaction=False)
    pipe2.get(key)
    assert pipe2.execute() == [b"1"]
    client.delete(key)
