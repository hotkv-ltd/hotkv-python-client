# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for string commands and generic key/expiry commands."""

from __future__ import annotations


def test_set_get_delete(client, prefix):
    key = prefix + "a"
    assert client.set(key, "hello") is True
    assert client.get(key) == b"hello"
    assert client.exists(key) == 1
    assert client.delete(key) == 1
    assert client.get(key) is None


def test_set_nx_xx(client, prefix):
    key = prefix + "nx"
    assert client.set(key, "1", nx=True) is True
    assert client.set(key, "2", nx=True) is False
    assert client.get(key) == b"1"
    assert client.set(key, "3", xx=True) is True
    assert client.get(key) == b"3"
    missing = prefix + "nx-missing"
    assert client.set(missing, "v", xx=True) is False
    client.delete(key)


def test_set_with_expiry_options(client, prefix):
    key = prefix + "ex"
    client.set(key, "v", ex=100)
    ttl = client.ttl(key)
    assert 0 < ttl <= 100
    client.set(key, "v2", px=50000)
    assert client.get(key) == b"v2"
    assert 0 < client.pttl(key) <= 50000
    client.delete(key)


def test_set_get_option(client, prefix):
    key = prefix + "setget"
    # GET on a key that does not exist yet: nil, but the write still happens.
    assert client.set(key, "v1", get=True) is None
    assert client.get(key) == b"v1"
    # GET returns the previous value, and the new one is still written.
    assert client.set(key, "v2", get=True) == b"v1"
    assert client.get(key) == b"v2"
    # NX + GET: the write is skipped (key exists), but GET still reports
    # the current value rather than True/False.
    assert client.set(key, "v3", nx=True, get=True) == b"v2"
    assert client.get(key) == b"v2"
    client.delete(key)


def test_set_exat_pxat_keepttl(client, prefix):
    import time

    key = prefix + "setexat"
    future = int(time.time()) + 100
    client.set(key, "v1", exat=future)
    assert 0 < client.ttl(key) <= 100
    # KEEPTTL: the value changes, the TTL from EXAT survives.
    client.set(key, "v2", keepttl=True)
    assert client.get(key) == b"v2"
    assert 0 < client.ttl(key) <= 100
    # A plain SET with no TTL option clears it.
    client.set(key, "v3")
    assert client.ttl(key) == -1
    future_ms = (int(time.time()) + 100) * 1000
    client.set(key, "v4", pxat=future_ms)
    assert 0 < client.pttl(key) <= 100_000
    client.delete(key)


def test_setnx_setex_psetex(client, prefix):
    key = prefix + "b"
    assert client.setnx(key, "one") is True
    assert client.setnx(key, "two") is False
    client.delete(key)
    assert client.setex(key, 60, "v") is True
    assert 0 < client.ttl(key) <= 60
    key2 = prefix + "c"
    assert client.psetex(key2, 60000, "v") is True
    assert 0 < client.pttl(key2) <= 60000
    client.delete(key, key2)


def test_getset_getdel_getex(client, prefix):
    key = prefix + "d"
    client.set(key, "orig")
    assert client.getset(key, "new") == b"orig"
    assert client.get(key) == b"new"
    assert client.getex(key, ex=100) == b"new"
    assert client.ttl(key) > 0
    assert client.getdel(key) == b"new"
    assert client.get(key) is None


def test_append_strlen_getrange_setrange(client, prefix):
    key = prefix + "e"
    client.set(key, "Hello")
    assert client.append(key, " World") == 11
    assert client.strlen(key) == 11
    assert client.getrange(key, 0, 4) == b"Hello"
    client.setrange(key, 6, "HotKV")
    assert client.get(key) == b"Hello HotKV"
    client.delete(key)


def test_incr_decr_family(client, prefix):
    key = prefix + "counter"
    assert client.incr(key) == 1
    assert client.incrby(key, 9) == 10
    assert client.decr(key) == 9
    assert client.decrby(key, 4) == 5
    assert client.incrbyfloat(key, 0.5) == 5.5
    client.delete(key)


def test_mget_mset_msetnx(client, prefix):
    k1, k2, k3 = prefix + "1", prefix + "2", prefix + "3"
    assert client.mset({k1: "a", k2: "b"}) is True
    assert client.mget([k1, k2, k3]) == [b"a", b"b", None]
    assert client.msetnx({k3: "c", k1: "z"}) is False  # k1 already exists
    assert client.get(k3) is None
    assert client.msetnx({k3: "c"}) is True
    client.delete(k1, k2, k3)


def test_binary_safe_values(client, prefix):
    key = prefix + "binary"
    payload = bytes(range(256))
    client.set(key, payload)
    assert client.get(key) == payload
    client.delete(key)


def test_type_touch_randomkey_keys(client, prefix):
    key = prefix + "typed"
    client.set(key, "v")
    assert client.type(key) == "string"
    assert client.touch(key) == 1
    assert key.encode() in client.keys(prefix + "*")
    assert client.randomkey() is not None
    client.delete(key)


def test_rename_renamenx_copy_move(client, prefix):
    k1, k2, k3 = prefix + "src", prefix + "dst", prefix + "dst2"
    client.set(k1, "v")
    assert client.rename(k1, k2) is True
    assert client.get(k2) == b"v"
    client.set(k1, "other")
    assert client.renamenx(k1, k2) is False  # k2 exists
    assert client.copy(k1, k3) is True
    assert client.get(k3) == b"other"
    client.delete(k1, k2, k3)


def test_dump_restore(client, prefix):
    key, key2 = prefix + "d1", prefix + "d2"
    client.set(key, "dumpme")
    payload = client.dump(key)
    assert payload is not None
    assert client.restore(key2, 0, payload) is True
    assert client.get(key2) == b"dumpme"
    client.delete(key, key2)


def test_expire_family_and_persist(client, prefix):
    key = prefix + "expiring"
    client.set(key, "v")
    assert client.expire(key, 100) is True
    assert 0 < client.ttl(key) <= 100
    assert client.persist(key) is True
    assert client.ttl(key) == -1
    assert client.pexpire(key, 100000) is True
    assert 0 < client.pttl(key) <= 100000
    client.delete(key)


def test_expireat_pexpireat_and_expiretime(client, prefix):
    import time

    key = prefix + "at"
    client.set(key, "v")
    future = int(time.time()) + 100
    assert client.expireat(key, future) is True
    # Only check the contract the client guarantees: an int is passed
    # through unchanged, and it is positive once a TTL is set.
    assert isinstance(client.expiretime(key), int)
    assert client.expiretime(key) > 0
    assert 0 < client.ttl(key) <= 100
    client.delete(key)
    assert client.expiretime(key) == -2


def test_ttl_on_missing_key(client, prefix):
    key = prefix + "missing"
    assert client.ttl(key) == -2
    assert client.pttl(key) == -2


def test_unlink(client, prefix):
    key = prefix + "unlink"
    client.set(key, "v")
    assert client.unlink(key) == 1
    assert client.exists(key) == 0
