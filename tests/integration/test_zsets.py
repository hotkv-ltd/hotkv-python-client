# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for sorted set commands."""

from __future__ import annotations


def test_zadd_zscore_zcard_zincrby(client, prefix):
    key = prefix + "z"
    assert client.zadd(key, {"a": 1, "b": 2, "c": 3}) == 3
    assert client.zscore(key, "b") == 2.0
    assert client.zcard(key) == 3
    assert client.zincrby(key, 5, "a") == 6.0
    client.delete(key)


def test_zadd_plain_upsert(client, prefix):
    key = prefix + "zflags"
    assert client.zadd(key, {"a": 1}) == 1
    assert client.zadd(key, {"a": 5}) == 0  # not newly added, just updated
    assert client.zscore(key, "a") == 5.0
    client.delete(key)


def test_zadd_nx_xx(client, prefix):
    key = prefix + "znxxx"
    client.zadd(key, {"m": 5})
    # NX: member already exists, so this is a no-op.
    assert client.zadd(key, {"m": 10}, nx=True) == 0
    assert client.zscore(key, "m") == 5.0
    # XX: member exists, so the update goes through.
    assert client.zadd(key, {"m": 10}, xx=True) == 0  # not newly added
    assert client.zscore(key, "m") == 10.0
    # XX on a member that does not exist is a no-op (never added).
    assert client.zadd(key, {"missing": 1}, xx=True) == 0
    assert client.zscore(key, "missing") is None
    client.delete(key)


def test_zadd_gt_lt_ch(client, prefix):
    key = prefix + "zgtlt"
    client.zadd(key, {"m": 10})
    # GT: only updates when the new score is greater; CH reports it.
    assert client.zadd(key, {"m": 5}, gt=True, ch=True) == 0
    assert client.zscore(key, "m") == 10.0
    assert client.zadd(key, {"m": 20}, gt=True, ch=True) == 1
    assert client.zscore(key, "m") == 20.0
    # LT: only updates when the new score is lower.
    assert client.zadd(key, {"m": 25}, lt=True, ch=True) == 0
    assert client.zscore(key, "m") == 20.0
    assert client.zadd(key, {"m": 1}, lt=True, ch=True) == 1
    assert client.zscore(key, "m") == 1.0
    client.delete(key)


def test_zadd_incr(client, prefix):
    key = prefix + "zincr"
    assert client.zadd(key, {"m": 5}, incr=True) == 5.0
    assert client.zadd(key, {"m": 2.5}, incr=True) == 7.5
    # NX+INCR on an existing member aborts and replies nil.
    assert client.zadd(key, {"m": 1}, nx=True, incr=True) is None
    assert client.zscore(key, "m") == 7.5
    client.delete(key)


def test_zrange_withscores_and_rev(client, prefix):
    key = prefix + "zr"
    client.zadd(key, {"a": 1, "b": 2, "c": 3})
    assert client.zrange(key, 0, -1) == [b"a", b"b", b"c"]
    assert client.zrange(key, 0, -1, withscores=True) == [(b"a", 1.0), (b"b", 2.0), (b"c", 3.0)]
    assert client.zrevrange(key, 0, -1) == [b"c", b"b", b"a"]
    client.delete(key)


def test_zrangebyscore_and_zrevrangebyscore(client, prefix):
    key = prefix + "zrbs"
    client.zadd(key, {"a": 1, "b": 2, "c": 3})
    assert client.zrangebyscore(key, 2, 3) == [b"b", b"c"]
    assert client.zrevrangebyscore(key, 3, 2) == [b"c", b"b"]
    assert client.zcount(key, 1, 2) == 2
    client.delete(key)




def test_zrank_zrevrank(client, prefix):
    key = prefix + "zrank"
    client.zadd(key, {"a": 1, "b": 2, "c": 3})
    assert client.zrank(key, "b") == 1
    assert client.zrevrank(key, "b") == 1
    assert client.zrank(key, "missing") is None
    client.delete(key)


def test_zrank_zrevrank_withscore(client, prefix):
    key = prefix + "zrankscore"
    client.zadd(key, {"a": 1, "b": 2, "c": 3})
    assert client.zrank(key, "b", withscore=True) == (1, 2.0)
    assert client.zrevrank(key, "b", withscore=True) == (1, 2.0)
    assert client.zrank(key, "missing", withscore=True) is None
    client.delete(key)


def test_zrange_byscore_bylex_rev_limit(client, prefix):
    key = prefix + "zbyscore"
    client.zadd(key, {"a": 1, "b": 2, "c": 3})
    assert client.zrange(key, 1, 2, byscore=True) == [b"a", b"b"]
    assert client.zrange(key, "(1", 3, byscore=True) == [b"b", b"c"]
    assert client.zrange(key, 3, 1, byscore=True, rev=True) == [b"c", b"b", b"a"]
    assert client.zrange(key, "-", "+", bylex=True) == [b"a", b"b", b"c"]
    assert client.zrange(key, 0, -1, rev=True) == [b"c", b"b", b"a"]
    assert client.zrange(key, 1, 3, byscore=True, offset=1, count=1) == [b"b"]
    assert client.zrange(key, 0, -1, withscores=True) == [(b"a", 1.0), (b"b", 2.0), (b"c", 3.0)]
    client.delete(key)


def test_zrangestore_byscore(client, prefix):
    source, dest = prefix + "zrs-src", prefix + "zrs-dest"
    client.zadd(source, {"a": 1, "b": 2, "c": 3})
    assert client.zrangestore(dest, source, 1, 2, byscore=True) == 2
    assert client.zrange(dest, 0, -1, withscores=True) == [(b"a", 1.0), (b"b", 2.0)]
    client.delete(source, dest)


def test_zpopmin_zpopmax_single(client, prefix):
    key = prefix + "zpop"
    client.zadd(key, {"a": 1, "b": 2, "c": 3})
    assert client.zpopmin(key) == [(b"a", 1.0)]
    assert client.zpopmax(key) == [(b"c", 3.0)]
    client.delete(key)


def test_zremrangebyscore_and_byrank(client, prefix):
    key = prefix + "zrem"
    client.zadd(key, {"a": 1, "b": 2, "c": 3, "d": 4})
    assert client.zremrangebyscore(key, 1, 2) == 2
    assert client.zrange(key, 0, -1) == [b"c", b"d"]
    client.zadd(key, {"e": 5})
    assert client.zremrangebyrank(key, 0, 0) == 1
    client.delete(key)


def test_zunionstore_zinterstore_zdiffstore(client, prefix):
    a, b, dest = prefix + "za", prefix + "zb", prefix + "zdest"
    client.zadd(a, {"x": 1, "y": 2})
    client.zadd(b, {"y": 10, "z": 3})
    assert client.zunionstore(dest, [a, b]) == 3
    assert client.zscore(dest, "y") == 12.0
    assert client.zinterstore(dest, [a, b]) == 1
    assert client.zscore(dest, "y") == 12.0
    assert client.zdiffstore(dest, [a, b]) == 1
    assert client.zscore(dest, "x") == 1.0
    client.delete(a, b, dest)


def test_zunionstore_zinterstore_weights_aggregate(client, prefix):
    a, b, dest = prefix + "zwa", prefix + "zwb", prefix + "zwdest"
    client.zadd(a, {"m": 1, "only_a": 1})
    client.zadd(b, {"m": 1})
    assert client.zunionstore(dest, [a, b], weights=[2, 3], aggregate="MAX") == 2
    assert client.zscore(dest, "m") == 3.0  # max(2*1, 3*1)
    assert client.zunionstore(dest, [a, b], weights=[2, 3], aggregate="SUM") == 2
    assert client.zscore(dest, "m") == 5.0  # 2*1 + 3*1
    assert client.zinterstore(dest, [a, b], weights=[2, 3], aggregate="MIN") == 1
    assert client.zscore(dest, "m") == 2.0  # min(2*1, 3*1)
    client.delete(a, b, dest)


def test_zunion_zinter_zdiff_no_store(client, prefix):
    a, b = prefix + "za2", prefix + "zb2"
    client.zadd(a, {"x": 1, "y": 2})
    client.zadd(b, {"y": 10, "z": 3})
    assert set(client.zunion([a, b])) == {b"x", b"y", b"z"}
    assert client.zinter([a, b]) == [b"y"]
    assert client.zdiff([a, b]) == [b"x"]
    client.delete(a, b)


def test_zmscore(client, prefix):
    key = prefix + "zms"
    client.zadd(key, {"a": 1, "b": 2})
    assert client.zmscore(key, ["a", "b", "missing"]) == [1.0, 2.0, None]
    client.delete(key)


def test_zpopmin_zpopmax_with_count(client, prefix):
    key = prefix + "zpopcount"
    client.zadd(key, {"a": 1, "b": 2, "c": 3, "d": 4})
    assert client.zpopmin(key, count=2) == [(b"a", 1.0), (b"b", 2.0)]
    assert client.zpopmax(key, count=2) == [(b"d", 4.0), (b"c", 3.0)]
    assert client.zpopmin(key, count=5) == []
    client.delete(key)


def test_zrandmember(client, prefix):
    key = prefix + "zrand"
    client.zadd(key, {"a": 1, "b": 2, "c": 3})
    member = client.zrandmember(key)
    assert member in (b"a", b"b", b"c")
    many = client.zrandmember(key, -5)
    assert len(many) == 5
    withscores = client.zrandmember(key, 3, withscores=True)
    assert len(withscores) == 3
    client.delete(key)
