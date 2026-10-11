# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for geospatial, bitmap and HyperLogLog commands."""

from __future__ import annotations


def test_geoadd_geopos_geodist(client, prefix):
    key = prefix + "geo"
    client.geoadd(key, {"palermo": (13.361389, 38.115556), "catania": (15.087269, 37.502669)})
    pos = client.geopos(key, "palermo")
    assert pos[0] is not None
    lon, lat = pos[0]
    assert 13.0 < lon < 14.0
    assert 38.0 < lat < 38.3
    dist = client.geodist(key, "palermo", "catania", "km")
    assert dist is not None and 150 < dist < 200
    client.delete(key)


def test_geohash(client, prefix):
    key = prefix + "geohash"
    client.geoadd(key, {"palermo": (13.361389, 38.115556)})
    hashes = client.geohash(key, "palermo")
    assert len(hashes) == 1 and hashes[0] is not None
    client.delete(key)


def test_geosearch(client, prefix):
    key = prefix + "geosearch"
    client.geoadd(
        key,
        {
            "palermo": (13.361389, 38.115556),
            "catania": (15.087269, 37.502669),
        },
    )
    results = client.geosearch(key, longitude=15, latitude=37, radius=200, unit="km")
    assert b"catania" in results
    with_dist = client.geosearch(
        key, longitude=15, latitude=37, radius=200, unit="km", withdist=True
    )
    names = {r.member for r in with_dist}
    assert b"catania" in names
    client.delete(key)


def test_geosearchstore(client, prefix):
    source, dest = prefix + "geosrc", prefix + "geodst"
    client.geoadd(source, {"a": (13.0, 38.0), "b": (13.1, 38.1)})
    stored = client.geosearchstore(dest, source, longitude=13.0, latitude=38.0, radius=50, unit="km")
    assert stored == 2
    assert set(client.zrange(dest, 0, -1)) == {b"a", b"b"}
    client.delete(source, dest)


def test_setbit_getbit_bitcount_bitpos(client, prefix):
    key = prefix + "bits"
    assert client.setbit(key, 7, 1) == 0
    assert client.getbit(key, 7) == 1
    assert client.bitcount(key) == 1
    assert client.bitpos(key, 1) == 7
    client.delete(key)


def test_bitfield(client, prefix):
    key = prefix + "bitfield"
    result = client.bitfield(key, [("SET", "u8", 0, 255), ("GET", "u8", 0)])
    assert result == [0, 255]
    result2 = client.bitfield(key, [("INCRBY", "u8", 0, 10), ("GET", "u8", 0)])
    assert result2 == [9, 9]  # wraps: 255 + 10 = 265 mod 256 = 9
    client.delete(key)


def test_bitop(client, prefix):
    k1, k2, dest = prefix + "b1", prefix + "b2", prefix + "bdest"
    client.set(k1, "\xff")
    client.set(k2, "\x0f")
    client.setbit(k1, 0, 1)
    client.setbit(k2, 0, 0)
    result_len = client.bitop("AND", dest, k1, k2)
    assert result_len >= 1
    client.delete(k1, k2, dest)


def test_pfadd_pfcount_pfmerge(client, prefix):
    k1, k2, dest = prefix + "hll1", prefix + "hll2", prefix + "hlldest"
    assert client.pfadd(k1, "a", "b", "c") is True
    assert client.pfadd(k2, "c", "d", "e") is True
    count1 = client.pfcount(k1)
    assert 2 <= count1 <= 4
    assert client.pfmerge(dest, k1, k2) is True
    merged_count = client.pfcount(dest)
    assert 4 <= merged_count <= 6
    client.delete(k1, k2, dest)
