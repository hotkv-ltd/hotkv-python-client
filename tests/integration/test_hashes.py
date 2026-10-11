# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for hash commands, including per-field TTL."""

from __future__ import annotations

import hotkv
import pytest


def test_hset_hget_hgetall_hdel(client, prefix):
    key = prefix + "h"
    assert client.hset(key, {"a": "1", "b": "2"}) == 2
    assert client.hget(key, "a") == b"1"
    assert client.hgetall(key) == {b"a": b"1", b"b": b"2"}
    assert client.hdel(key, "a") == 1
    assert client.hexists(key, "a") is False
    assert client.hexists(key, "b") is True
    client.delete(key)


def test_hsetnx(client, prefix):
    key = prefix + "hsetnx"
    assert client.hsetnx(key, "f", "1") is True
    assert client.hsetnx(key, "f", "2") is False
    assert client.hget(key, "f") == b"1"
    client.delete(key)


def test_hmget_hkeys_hvals_hlen(client, prefix):
    key = prefix + "hm"
    client.hset(key, {"a": "1", "b": "2", "c": "3"})
    assert client.hmget(key, ["a", "c", "z"]) == [b"1", b"3", None]
    assert set(client.hkeys(key)) == {b"a", b"b", b"c"}
    assert set(client.hvals(key)) == {b"1", b"2", b"3"}
    assert client.hlen(key) == 3
    client.delete(key)


def test_hincrby_hincrbyfloat_hstrlen(client, prefix):
    key = prefix + "hi"
    client.hset(key, {"n": "10"})
    assert client.hincrby(key, "n", 5) == 15
    assert client.hincrbyfloat(key, "n", 0.5) == 15.5
    assert client.hstrlen(key, "n") == len("15.5")
    client.delete(key)


def test_hrandfield(client, prefix):
    key = prefix + "hr"
    client.hset(key, {"a": "1", "b": "2", "c": "3"})
    field = client.hrandfield(key)
    assert field in (b"a", b"b", b"c")
    many = client.hrandfield(key, -10)
    assert len(many) == 10
    with_values = client.hrandfield(key, 3, withvalues=True)
    assert len(with_values) == 3
    for f, v in with_values:
        assert f in (b"a", b"b", b"c")
    client.delete(key)


def test_hash_field_ttl_family(client, prefix):
    key = prefix + "httl"
    client.hset(key, {"a": "1", "b": "2"})
    try:
        result = client.hexpire(key, 100, ["a"])
    except hotkv.ResponseError as exc:
        pytest.skip(f"server does not support hash field TTLs: {exc}")
    assert result == [1]
    ttl = client.httl(key, ["a"])
    assert len(ttl) == 1 and 0 < ttl[0] <= 100
    assert client.httl(key, ["b"]) == [-1]
    assert client.httl(key, ["missing"]) == [-2]
    assert client.hpersist(key, ["a"]) == [1]
    assert client.httl(key, ["a"]) == [-1]
    client.delete(key)


def test_hash_field_ttl_nx_flag(client, prefix):
    key = prefix + "httlnx"
    client.hset(key, {"f": "1"})
    try:
        first = client.hexpire(key, 100, ["f"], nx=True)
    except hotkv.ResponseError as exc:
        pytest.skip(f"server does not support hash field TTLs: {exc}")
    assert first == [1]
    second = client.hexpire(key, 200, ["f"], nx=True)
    assert second == [0]  # NX: a TTL is already set, so this is a no-op
    ttl = client.httl(key, ["f"])
    assert 0 < ttl[0] <= 100
    client.delete(key)
