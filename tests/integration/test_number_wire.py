# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Floats on the wire against a real server: tiny, huge, integral, negative
and infinite values in the core commands that take a number. Python's
`repr(1e-05)` is `'1e-05'`; the client sends plain decimals instead, and these
tests hold the server to reading every one of them back as the same number."""

from __future__ import annotations

import math

import pytest

import hotkv
from hotkv import format_float

TINY = [1e-05, 1e-300, 5e-324, 2.2250738585072014e-308]
HUGE = [1e21, 1.7976931348623157e308, -1.7976931348623157e308]
ODD = [3.0, -3.0, 0.1, 0.30000000000000004, 123456789.125, -2.5e-10]


def test_zadd_scores_survive_the_wire(client, prefix):
    key = prefix + "z"
    scores = {f"m{i}": value for i, value in enumerate(TINY + HUGE + ODD)}
    assert client.zadd(key, scores) == len(scores)
    for member, value in scores.items():
        assert client.zscore(key, member) == value, member
    assert client.zscore(key, "nope") is None
    client.delete(key)


def test_zadd_infinite_scores_and_bounds(client, prefix):
    key = prefix + "zinf"
    client.zadd(key, {"low": float("-inf"), "mid": 0.0, "high": float("inf")})
    assert client.zscore(key, "low") == float("-inf")
    assert client.zscore(key, "high") == float("inf")
    assert client.zrangebyscore(key, "-inf", "+inf") == [b"low", b"mid", b"high"]
    assert client.zrangebyscore(key, float("-inf"), float("inf")) == [b"low", b"mid", b"high"]
    assert client.zrangebyscore(key, float("-inf"), -1e300) == [b"low"]
    assert client.zrangebyscore(key, 1e300, float("inf")) == [b"high"]
    client.delete(key)


def test_score_bounds_as_floats_and_exclusive_bounds(client, prefix):
    key = prefix + "zb"
    client.zadd(key, {"a": 1e-05, "b": 2e-05, "c": 3e-05, "d": 1e21})
    assert client.zrangebyscore(key, 1e-05, 2e-05) == [b"a", b"b"]
    assert client.zrangebyscore(key, "(" + format_float(1e-05), 3e-05) == [b"b", b"c"]
    assert client.zrangebyscore(key, 0.0, 1e20) == [b"a", b"b", b"c"]
    assert client.zcount(key, 1e-05, 1e21) == 4
    assert client.zrevrangebyscore(key, 1e21, "(" + format_float(2e-05)) == [b"d", b"c"]
    assert client.zrange(key, 1e-05, 2e-05, byscore=True) == [b"a", b"b"]
    client.delete(key)


def test_zincrby_and_zadd_incr_with_tiny_and_huge_amounts(client, prefix):
    key = prefix + "zi"
    assert client.zincrby(key, 1e-05, "m") == 1e-05
    assert client.zincrby(key, 2e-05, "m") == pytest.approx(3e-05)
    assert client.zadd(key, {"h": 1e21}, incr=True) == 1e21
    assert client.zincrby(key, 1.7976931348623157e308, "big") == 1.7976931348623157e308
    client.delete(key)


def test_weights_in_zunionstore(client, prefix):
    a, b, dest = prefix + "{w}a", prefix + "{w}b", prefix + "{w}d"
    client.zadd(a, {"x": 1.0})
    client.zadd(b, {"x": 2.0})
    client.zunionstore(dest, [a, b], weights=[1e-05, 1e21])
    assert client.zscore(dest, "x") == pytest.approx(1.0 * 1e-05 + 2.0 * 1e21)
    for k in (a, b, dest):
        client.delete(k)


def test_incrbyfloat_and_hincrbyfloat_with_tiny_and_huge_amounts(client, prefix):
    key = prefix + "f"
    assert client.incrbyfloat(key, 1e-05) == pytest.approx(1e-05)
    assert client.incrbyfloat(key, 1e21) == pytest.approx(1e21)
    assert client.incrbyfloat(key, -3.0) == pytest.approx(1e21)
    client.delete(key)
    # The result is written in fixed notation with 17 decimals: 1e-17 is
    # kept, and a smaller amount is read (the plain 300 digit text parses) but adds nothing.
    assert client.incrbyfloat(key, 1e-17) == 1e-17
    client.delete(key)
    assert client.incrbyfloat(key, 1e-300) == 0.0
    client.delete(key)
    assert client.incrbyfloat(key, 1.7976931348623157e308) == 1.7976931348623157e308
    client.delete(key)
    hkey = prefix + "h"
    assert client.hincrbyfloat(hkey, "f", 1e-05) == pytest.approx(1e-05)
    assert client.hincrbyfloat(hkey, "f", 1e21) == pytest.approx(1e21)
    assert client.hincrbyfloat(hkey, "g", 3.0) == 3.0
    assert client.hget(hkey, "g") == b"3"
    client.delete(hkey)


def test_infinity_and_nan_for_incrbyfloat(client, prefix):
    key = prefix + "fi"
    with pytest.raises(hotkv.ResponseError):
        client.incrbyfloat(key, float("inf"))
    with pytest.raises(hotkv.DataError):  # refused by the client, never sent
        client.incrbyfloat(key, float("nan"))
    assert client.get(key) is None


def test_geo_coordinates_and_radius(client, prefix):
    key = prefix + "geo"
    assert client.geoadd(key, {"p1": (13.361389, 38.115556), "p2": (15.087269, 37.502669), "z": (1e-05, 1e-05)}) == 3
    assert client.geoadd(key, {"p1": (13.361389, 38.115556)}) == 0
    pos = client.geopos(key, "z")[0]
    assert pos == pytest.approx((1e-05, 1e-05), abs=1e-5)
    dist = client.geodist(key, "p1", "p2", "km")
    assert dist == pytest.approx(166.2742, abs=0.01)
    found = client.geosearch(key, longitude=15.0, latitude=37.0, radius=2e2, unit="km")
    assert set(found) == {b"p1", b"p2"}
    assert client.geosearch(key, longitude=1e-05, latitude=1e-05, radius=5, unit="m") == [b"z"]
    box = client.geosearch(key, longitude=14.0, latitude=37.8, width=3e2, height=3e2, unit="km")
    assert set(box) == {b"p1", b"p2"}
    client.delete(key)


def test_vsim_with_tiny_and_negative_components(client, prefix):
    index = prefix + "vnum"
    client.vsim_createindex(index, 3, metric="cosine")
    vectors = {
        "a": [1e-05, 0.0, 1.0],
        "b": [1.0, -1e-05, 0.0],
        "c": [1e-30, 1e-30, 1.0],  # a small float32
        "d": [1e-300, 0.0, 1.0],  # below float32: sent as 0
        "e": [3.0, -3.0, 0.5],
    }
    for name, vector in vectors.items():
        assert client.vsim_set(index, name, vector) is True
    got = client.vsim_get(index, "a")
    assert got == pytest.approx([1e-05, 0.0, 1.0], rel=1e-6)
    got = client.vsim_get(index, "d")
    assert got == [0.0, 0.0, 1.0]
    norm = math.sqrt(3.0 * 3.0 + 3.0 * 3.0 + 0.5 * 0.5)  # a cosine index stores unit vectors
    assert client.vsim_get(index, "e") == pytest.approx([3.0 / norm, -3.0 / norm, 0.5 / norm], rel=1e-6)
    hits = client.vsim_search(index, 2, [1e-05, 0.0, 1.0])
    assert hits[0][0] in (b"a", b"c", b"d")
    assert all(math.isfinite(score) for _, score in hits)
    # the edges of the float32 range
    assert client.vsim_set(index, "edge", [7.1e-46, 3.40282356e38, 7.0e-46]) is True
    assert client.vsim_set(index, "edge", [7.0e-46, 1e-46, 3.4028235e38]) is False  # 0, 0 and a large one
    with pytest.raises(hotkv.DataError):
        client.vsim_set(index, "big", [1e39, 0.0, 0.0])
    with pytest.raises(hotkv.DataError):
        client.vsim_set(index, "big", [3.4028236e38, 0.0, 0.0])
    with pytest.raises(hotkv.DataError):
        client.vsim_set(index, "nan", [float("nan"), 0.0, 0.0])
    client.vsim_dropindex(index)


def test_a_float_key_or_member_is_sent_as_plain_decimal(client, prefix):
    # the key text and the cluster routing of a float key must agree
    key = prefix + "k"
    client.set(key, 1e-05)
    assert client.get(key) == b"0.00001"
    client.set(key, 3.0)
    assert client.get(key) == b"3"
    client.delete(key)


def test_generic_command_takes_floats_too(client, prefix):
    key = prefix + "g"
    assert client.execute("ZADD", key, 1e-300, "m") == 1
    assert float(client.execute("ZSCORE", key, "m")) == 1e-300
    client.delete(key)
