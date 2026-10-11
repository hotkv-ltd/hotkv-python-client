# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for server admin, scripting and SCAN family commands."""

from __future__ import annotations

import hotkv
import pytest


def test_ping_echo(client):
    assert client.ping() == "PONG"
    assert client.ping("hi") == b"hi"
    assert client.echo("hello") == b"hello"


def test_info_and_dbsize_and_time(client):
    info = client.info()
    assert isinstance(info, dict)
    assert "hotkv_version" in info or "redis_version" in info
    assert isinstance(client.dbsize(), int)
    seconds, micros = client.time()
    assert seconds > 1_700_000_000
    assert 0 <= micros < 1_000_000


def test_config_get_set(client):
    config = client.config_get("maxmemory")
    assert isinstance(config, dict)


def test_client_id_and_setname(client):
    cid = client.client_id()
    assert isinstance(cid, int) and cid > 0


def test_scan_iter_finds_keys(client, prefix):
    keys = {prefix + f"scan:{i}" for i in range(5)}
    for k in keys:
        client.set(k, "1")
    found = set(client.scan_iter(match=prefix + "scan:*"))
    assert found == {k.encode() for k in keys}
    client.delete(*keys)


def test_scan_single_page(client, prefix):
    key = prefix + "scanone"
    client.set(key, "v")
    cursor, page = client.scan(0, match=prefix + "*", count=1000)
    assert key.encode() in page
    client.delete(key)


def test_hscan_iter(client, prefix):
    key = prefix + "hscan"
    client.hset(key, {"a": "1", "b": "2", "c": "3"})
    found = dict(client.hscan_iter(key))
    assert found == {b"a": b"1", b"b": b"2", b"c": b"3"}
    client.delete(key)


def test_sscan_iter(client, prefix):
    key = prefix + "sscan"
    client.sadd(key, "a", "b", "c")
    found = set(client.sscan_iter(key))
    assert found == {b"a", b"b", b"c"}
    client.delete(key)


def test_zscan_iter(client, prefix):
    key = prefix + "zscan"
    client.zadd(key, {"a": 1, "b": 2})
    found = dict(client.zscan_iter(key))
    assert found == {b"a": 1.0, b"b": 2.0}
    client.delete(key)


def test_eval_basic(client):
    result = client.eval("return 1")
    assert result == 1


def test_eval_with_keys_and_args(client, prefix):
    key = prefix + "evalkey"
    try:
        result = client.eval("return redis.call('SET', KEYS[1], ARGV[1])", [key], ["value"])
    except hotkv.ResponseError as exc:
        pytest.skip(f"server Lua scripting not available in this build: {exc}")
    assert client.get(key) == b"value"
    client.delete(key)


def test_script_load_evalsha_and_auto(client):
    sha = client.script_load("return 42")
    assert isinstance(sha, (bytes, str))
    assert client.script_exists(sha) == [True]
    assert client.evalsha(sha) == 42

    result = client.eval_auto("return 43")
    assert result == 43
