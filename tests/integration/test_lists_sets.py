# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for list and set commands."""

from __future__ import annotations


def test_lpush_rpush_lrange_llen(client, prefix):
    key = prefix + "list"
    assert client.rpush(key, "a", "b", "c") == 3
    assert client.lpush(key, "z") == 4
    assert client.lrange(key, 0, -1) == [b"z", b"a", b"b", b"c"]
    assert client.llen(key) == 4
    client.delete(key)


def test_lpop_rpop_single(client, prefix):
    key = prefix + "pop"
    client.rpush(key, "a", "b", "c", "d")
    assert client.lpop(key) == b"a"
    assert client.rpop(key) == b"d"
    assert client.lrange(key, 0, -1) == [b"b", b"c"]
    client.delete(key)
    assert client.lpop(key) is None


def test_lpop_rpop_with_count(client, prefix):
    key = prefix + "popcount"
    client.rpush(key, "a", "b", "c", "d")
    assert client.lpop(key, count=2) == [b"a", b"b"]
    assert client.rpop(key, count=2) == [b"d", b"c"]
    # The list is now empty (and gone): a further count pop is a miss (nil).
    assert client.lpop(key, count=5) is None
    client.delete(key)


def test_lindex_lset_linsert_lrem_ltrim(client, prefix):
    key = prefix + "misc"
    client.rpush(key, "a", "b", "c")
    assert client.lindex(key, 1) == b"b"
    assert client.lset(key, 1, "B") is True
    assert client.lindex(key, 1) == b"B"
    assert client.linsert(key, "BEFORE", "B", "X") == 4
    assert client.lrange(key, 0, -1) == [b"a", b"X", b"B", b"c"]
    client.rpush(key, "c")
    assert client.lrem(key, 1, "c") == 1
    assert client.ltrim(key, 0, 1) is True
    assert client.lrange(key, 0, -1) == [b"a", b"X"]
    client.delete(key)


def test_lpushx_rpushx(client, prefix):
    key = prefix + "pushx"
    assert client.lpushx(key, "a") == 0
    client.rpush(key, "seed")
    assert client.rpushx(key, "b") == 2
    client.delete(key)


def test_rpoplpush_and_lmove(client, prefix):
    src, dst = prefix + "src", prefix + "dst"
    client.rpush(src, "a", "b", "c")
    assert client.rpoplpush(src, dst) == b"c"
    assert client.lmove(src, dst, "LEFT", "RIGHT") == b"a"
    assert client.lrange(dst, 0, -1) == [b"c", b"a"]
    client.delete(src, dst)


def test_lpos(client, prefix):
    key = prefix + "pos"
    client.rpush(key, "a", "b", "c", "b")
    assert client.lpos(key, "b") == 1
    assert client.lpos(key, "b", count=2) == [1, 3]
    client.delete(key)


def test_blpop_brpop_immediate(client, prefix):
    key = prefix + "blocking"
    client.rpush(key, "only")
    assert client.blpop(key, 1) == [key.encode(), b"only"]
    assert client.brpop(key, 0.1) is None
    client.delete(key)


def test_sadd_smembers_scard_sismember(client, prefix):
    key = prefix + "set"
    assert client.sadd(key, "a", "b", "c") == 3
    assert client.smembers(key) == {b"a", b"b", b"c"}
    assert client.scard(key) == 3
    assert client.sismember(key, "a") is True
    assert client.sismember(key, "z") is False
    assert client.smismember(key, ["a", "z", "b"]) == [True, False, True]
    client.delete(key)


def test_srem_spop_srandmember(client, prefix):
    key = prefix + "setops"
    client.sadd(key, "a", "b", "c", "d")
    assert client.srem(key, "a") == 1
    popped = client.spop(key)
    assert popped in (b"b", b"c", b"d")
    member = client.srandmember(key)
    assert member is not None
    client.delete(key)


def test_spop_srandmember_with_count(client, prefix):
    key = prefix + "popcount"
    client.sadd(key, "a", "b", "c")
    popped = client.spop(key, count=2)
    assert isinstance(popped, set) and len(popped) == 2
    assert popped <= {b"a", b"b", b"c"}
    remaining = client.srandmember(key, count=5)  # fewer members than asked
    assert len(remaining) == 1
    many = client.srandmember(key, count=-5)  # negative: repeats allowed
    assert len(many) == 5
    # A count-pop past exhaustion (key now gone) is an empty set, not nil.
    client.spop(key, count=1)
    assert client.spop(key, count=1) == set()
    client.delete(key)


def test_smove_and_set_algebra(client, prefix):
    a, b, dest = prefix + "a", prefix + "b", prefix + "dest"
    client.sadd(a, "x", "y", "z")
    client.sadd(b, "y", "z", "w")
    assert client.smove(a, b, "x") is True
    assert client.sismember(b, "x") is True
    assert client.sunion(a, b) == {b"y", b"z", b"w", b"x"}
    assert client.sinter(a, b) == {b"y", b"z"}
    assert client.sdiff(b, a) == {b"w", b"x"}
    assert client.sunionstore(dest, a, b) == 4
    assert client.sinterstore(dest, a, b) == 2
    assert client.sdiffstore(dest, b, a) == 2
    assert client.sintercard([a, b]) == 2
    assert client.sintercard([a, b], limit=1) == 1
    client.delete(a, b, dest)
