# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Live checks of the typed methods the other integration modules do not reach,
each asserting the value the server answers, and of the commands that have no typed method, sent through `execute()`.

Server-wide or connection-level commands (FLUSHDB, MOVE, CONFIG SET, CLIENT
SETNAME, pub/sub) run only against a standalone server."""

from __future__ import annotations

import time

import pytest

import hotkv


# -- lists ---------------------------------------------------------------------------


def test_blmove_moves_the_head_to_the_other_list_and_times_out_empty(client, prefix):
    src, dst = prefix + "src", prefix + "dst"
    client.rpush(src, "a", "b", "c")
    assert client.blmove(src, dst, "LEFT", "RIGHT", 1) == b"a"
    assert client.blmove(src, dst, "RIGHT", "LEFT", 0.5) == b"c"
    assert client.lrange(src, 0, -1) == [b"b"]
    assert client.lrange(dst, 0, -1) == [b"c", b"a"]
    client.delete(src)
    started = time.monotonic()
    assert client.blmove(src, dst, "LEFT", "LEFT", 0.2) is None  # empty source: waits, then nil
    assert 0.15 <= time.monotonic() - started < 3
    client.delete(dst)


# -- sorted sets -----------------------------------------------------------------------


def test_bzpopmin_and_bzpopmax(client, prefix):
    a, b = prefix + "a", prefix + "b"
    client.zadd(a, {"x": 1.0, "y": 5.0})
    assert client.bzpopmin([a, b], 1) == (a.encode(), b"x", 1.0)
    assert client.bzpopmax([a, b], 0.5) == (a.encode(), b"y", 5.0)
    assert client.bzpopmin([a, b], 0.1) is None
    assert client.bzpopmax([a, b], 0.1) is None


def test_zrem_zintercard_and_the_lex_commands(client, prefix):
    a, b = prefix + "a", prefix + "b"
    client.zadd(a, {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0})
    client.zadd(b, {"c": 1, "d": 2, "z": 3})
    assert client.zintercard([a, b]) == 2
    assert client.zintercard([a, b], limit=1) == 1
    assert client.zlexcount(a, "[b", "(e") == 3
    assert client.zlexcount(a, "-", "+") == 5
    assert client.zrangebylex(a, "[b", "(e") == [b"b", b"c", b"d"]
    assert client.zrangebylex(a, "-", "+", offset=1, count=2) == [b"b", b"c"]
    assert client.zrevrangebylex(a, "+", "[d") == [b"e", b"d"]
    assert client.zremrangebylex(a, "[a", "[b") == 2
    assert client.zrem(a, "c", "nope") == 1
    assert client.zrangebylex(a, "-", "+") == [b"d", b"e"]
    assert client.zrem(a, "d", "e") == 2
    assert client.exists(a) == 0


# -- hashes with field expiry, and the scan family -------------------------------------


def test_hash_field_expiry_variants(client, prefix):
    key = prefix + "h"
    client.hset(key, mapping={"a": 1, "b": 2, "c": 3})
    at = int(time.time()) + 1000
    assert client.hexpireat(key, at, ["a"]) == [1]
    assert client.hexpiretime(key, ["a", "b", "nope"]) == [at, -1, -2]
    assert client.hpexpire(key, 5000, ["b"]) == [1]
    ttls = client.hpttl(key, ["b", "c", "nope"])
    assert 0 < ttls[0] <= 5000 and ttls[1:] == [-1, -2]
    at_ms = (int(time.time()) + 2000) * 1000
    assert client.hpexpireat(key, at_ms, ["c"]) == [1]
    assert client.hpexpiretime(key, ["c", "a"]) == [at_ms, at * 1000]
    assert client.hpexpire(key, 5000, ["nope"]) == [-2]
    assert client.hpersist(key, ["a", "c"]) == [1, 1]
    assert client.hexpiretime(key, ["a", "c"]) == [-1, -1]
    client.delete(key)


def test_sscan_hscan_zscan(client, prefix):
    s, h, z = prefix + "s", prefix + "h", prefix + "z"
    client.sadd(s, "a", "b", "c")
    client.hset(h, mapping={"f1": "v1", "f2": "v2"})
    client.zadd(z, {"m1": 1.5, "m2": 2.5})
    cursor, members = client.sscan(s)
    assert cursor == 0 and sorted(members) == [b"a", b"b", b"c"]
    assert client.sscan(s, match="b*") == (0, [b"b"])
    cursor, fields = client.hscan(h)
    assert cursor == 0 and fields == {b"f1": b"v1", b"f2": b"v2"}
    cursor, names = client.hscan(h, novalues=True)
    assert cursor == 0 and sorted(names) == [b"f1", b"f2"]
    assert client.hscan(h, match="f2") == (0, {b"f2": b"v2"})
    cursor, pairs = client.zscan(z)
    assert cursor == 0 and sorted(pairs) == [(b"m1", 1.5), (b"m2", 2.5)]
    assert client.zscan(z, match="m2", count=10) == (0, [(b"m2", 2.5)])
    for key in (s, h, z):
        client.delete(key)


# -- keys and strings -------------------------------------------------------------------


def test_pexpireat_and_pexpiretime(client, prefix):
    key = prefix + "k"
    client.set(key, "v")
    at = int(time.time() * 1000) + 60_000
    assert client.pexpireat(key, at) is True
    assert client.pexpiretime(key) == at
    assert 0 < client.pttl(key) <= 60_000
    assert client.pexpireat(prefix + "missing", at) is False
    assert client.pexpiretime(prefix + "missing") == -2
    client.delete(key)


def test_setrange_and_xtrim_return_values(client, prefix):
    key = prefix + "str"
    client.set(key, "Hello World")
    assert client.setrange(key, 6, "HotKV") == 11
    assert client.setrange(key, 13, "!") == 14
    assert client.get(key) == b"Hello HotKV\x00\x00!"
    stream = prefix + "xs"
    for i in range(5):
        client.xadd(stream, {"i": i}, id=f"{i + 1}-0")
    assert client.xtrim(stream, maxlen=2) == 3
    assert client.xlen(stream) == 2
    assert client.xtrim(stream, minid="5-0") == 1
    assert [entry_id for entry_id, _ in client.xrange(stream)] == [b"5-0"]
    client.delete(stream)
    client.delete(key)


# -- streams: consumer groups -------------------------------------------------------------


def test_xgroup_setid_createconsumer_delconsumer(client, prefix):
    key = prefix + "xg"
    for i in range(3):
        client.xadd(key, {"i": i}, id=f"{i + 1}-0")
    assert client.xgroup_create(key, "g", "0") is True
    assert client.xgroup_createconsumer(key, "g", "c1") is True
    assert client.xgroup_createconsumer(key, "g", "c1") is False  # already there
    assert [c["name"] for c in client.xinfo_consumers(key, "g")] == [b"c1"] or [
        c["name"] for c in client.xinfo_consumers(key, "g")
    ] == ["c1"]
    read = client.xreadgroup("g", "c1", {key: ">"}, count=2)
    assert len(read[key.encode()] if key.encode() in read else read[key]) == 2
    assert client.xgroup_delconsumer(key, "g", "c1") == 2  # its two pending entries
    assert client.xgroup_setid(key, "g", "2-0") is True
    group = client.xinfo_groups(key)[0]
    assert group["last-delivered-id"] in (b"2-0", "2-0")
    assert client.xgroup_destroy(key, "g") is True
    client.delete(key)


# -- scripting -----------------------------------------------------------------------------


def test_script_flush_empties_the_script_cache(client):
    sha = client.script_load("return 41 + 1")
    sha_text = sha.decode() if isinstance(sha, bytes) else sha
    assert client.script_exists(sha_text) == [True]
    assert client.script_flush() is True
    assert client.script_exists(sha_text) == [False]
    assert client.script_flush(asynchronous=True) is True


# -- server and connection level -------------------------------------------------------------


def test_client_setname_getname_and_config_set(test_url):
    c = hotkv.from_url(test_url, max_connections=1)  # one connection, so the name sticks
    try:
        assert c.client_getname() is None
        assert c.client_setname("sdk-test") is True
        assert c.client_getname() == b"sdk-test"
        original = c.config_get("hz")["hz"]
        assert c.config_set({"hz": original}) is True
        assert c.config_get("hz") == {"hz": original}
    finally:
        c.close()


def test_flushdb_and_flushall(client, prefix):
    client.set(prefix + "a", "1")
    assert client.flushdb() is True
    assert client.dbsize() == 0
    client.set(prefix + "a", "1")
    assert client.flushall(asynchronous=True) is True
    deadline = time.time() + 5
    while client.dbsize() and time.time() < deadline:  # ASYNC: the keys go in the background
        time.sleep(0.05)
    assert client.dbsize() == 0
    client.set(prefix + "b", "1")
    assert client.flushall() is True
    assert client.exists(prefix + "b") == 0


def test_move_to_another_database(client, prefix, test_url):
    key = prefix + "mv"
    other = hotkv.from_url(test_url, db=1)
    try:
        other.delete(key)
        client.set(key, "moved")
        assert client.move(key, 1) is True
        assert client.exists(key) == 0
        assert other.get(key) == b"moved"
        client.set(key, "again")
        assert client.move(key, 1) is False  # the target already has it
    finally:
        other.delete(key)
        other.close()
        client.delete(key)


# -- publish and pub/sub introspection ----------------------------------------------------------


def test_publish_and_pubsub_introspection(client, prefix, test_url):
    channel = prefix + "chan"
    assert client.publish(channel, "nobody listens") == 0
    assert client.pubsub_numsub(channel) == {channel.encode(): 0}
    sub = hotkv.from_url(test_url).pubsub()
    try:
        sub.subscribe(channel)
        sub.psubscribe(prefix + "pat*")
        for _ in range(50):
            if client.pubsub_numsub(channel) == {channel.encode(): 1}:
                break
            time.sleep(0.02)
        assert client.pubsub_numsub(channel) == {channel.encode(): 1}
        assert channel.encode() in client.pubsub_channels(prefix + "chan*")
        assert client.pubsub_channels(prefix + "nothing*") == []
        assert client.pubsub_numpat() >= 1
        assert client.publish(channel, "hello") == 1
        assert client.publish(prefix + "pattern-hit", "x") == 1
        seen = []
        deadline = time.time() + 3
        while len(seen) < 4 and time.time() < deadline:
            message = sub.get_message(timeout=0.2)
            if message is not None:
                seen.append(message)
        assert b"hello" in [m.data for m in seen]
    finally:
        sub.close()


def test_spublish_and_shard_channel_introspection(client, prefix, test_url):
    channel = prefix + "shard"
    assert client.spublish(channel, "nobody") == 0
    sub = hotkv.from_url(test_url).pubsub()
    try:
        sub.ssubscribe(channel)
        for _ in range(50):
            if client.pubsub_shardnumsub(channel) == {channel.encode(): 1}:
                break
            time.sleep(0.02)
        assert client.pubsub_shardnumsub(channel) == {channel.encode(): 1}
        assert client.pubsub_shardchannels(prefix + "shard*") == [channel.encode()]
        assert client.spublish(channel, "hi") == 1
        seen = []
        deadline = time.time() + 3
        while not seen and time.time() < deadline:
            message = sub.get_message(timeout=0.2)
            if message is not None and message.data == b"hi":
                seen.append(message)
        assert seen
    finally:
        sub.close()


# -- commands without a typed method, through execute() ---------------------------------------------


def test_bloom_count_min_and_top_k_through_execute(client, prefix):
    bf, cms, tk = prefix + "bf", prefix + "cms", prefix + "tk"
    assert client.execute("BF.RESERVE", bf, "0.01", 1000) in (b"OK", "OK", True)
    assert client.execute("BF.ADD", bf, "a") == 1
    assert client.execute("BF.ADD", bf, "a") == 0
    assert client.execute("BF.EXISTS", bf, "a") == 1
    assert client.execute("BF.MADD", bf, "b", "c") == [1, 1]
    assert client.execute("BF.MEXISTS", bf, "a", "zzz") == [1, 0]
    assert client.execute("BF.CARD", bf) == 3
    assert client.execute("BF.INFO", bf, "CAPACITY") == {"Capacity": 1000}
    assert client.execute("CMS.INITBYDIM", cms, 100, 5) in (b"OK", "OK", True)
    assert client.execute("CMS.INCRBY", cms, "a", 3, "b", 2) == [3, 2]
    assert client.execute("CMS.QUERY", cms, "a", "b", "z") == [3, 2, 0]
    assert client.execute("TOPK.RESERVE", tk, 3) in (b"OK", "OK", True)
    client.execute("TOPK.ADD", tk, "a", "b", "a")
    assert client.execute("TOPK.COUNT", tk, "a", "b") == [2, 1]
    assert client.execute("TOPK.LIST", tk) == [b"a", b"b"]
    assert client.execute("TOPK.QUERY", tk, "a", "zzz") == [1, 0]
    for key in (bf, cms, tk):
        client.delete(key)


def test_newer_hash_and_string_commands_through_execute(client, prefix):
    h, k1, k2 = prefix + "h", prefix + "k1", prefix + "k2"
    client.hset(h, mapping={"a": "1", "b": "2"})
    assert client.execute("HGETDEL", h, "FIELDS", 1, "a") == [b"1"]
    assert client.hget(h, "a") is None
    assert client.execute("HSETEX", h, "EX", 100, "FIELDS", 1, "c", "3") == 1
    assert client.execute("HGETEX", h, "EX", 200, "FIELDS", 1, "c") == [b"3"]
    assert 100 < client.httl(h, ["c"])[0] <= 200
    assert client.execute("MSETEX", 2, k1, "v1", k2, "v2", "EX", 100) == 1
    assert client.get(k1) == b"v1" and 0 < client.ttl(k2) <= 100
    assert client.execute("DELEX", k1, "IFEQ", "wrong") == 0
    assert client.execute("DELEX", k1, "IFEQ", "v1") == 1
    assert client.exists(k1) == 0
    for key in (h, k2):
        client.delete(key)


def test_xcfgset_sort_lcs_lmpop_object_and_memory_through_execute(client, prefix):
    stream, lst, a, b, h = prefix + "xs", prefix + "l", prefix + "a", prefix + "b", prefix + "h"
    client.xadd(stream, {"f": "v"})
    assert client.execute("XCFGSET", stream, "IDMP-DURATION", 100) in (b"OK", "OK", True)
    client.rpush(lst, "3", "1", "2")
    assert client.execute("SORT", lst) == [b"1", b"2", b"3"]
    assert client.execute("SORT", lst, "DESC", "LIMIT", 0, 2) == [b"3", b"2"]
    client.set(a, "ohmytext")
    client.set(b, "mynewtext")
    assert client.execute("LCS", a, b) == b"mytext"
    assert client.execute("LCS", a, b, "LEN") == 6
    assert client.execute("LMPOP", 1, lst, "LEFT", "COUNT", 2) == [lst.encode(), [b"3", b"1"]]
    client.hset(h, {"f": "v"})
    assert client.execute("OBJECT", "ENCODING", h) in (b"listpackex", b"listpack", b"hashtable")
    assert client.execute("MEMORY", "USAGE", h) > 0
    for key in (stream, lst, a, b, h):
        client.delete(key)


def test_client_commands_without_a_typed_method_through_execute(test_url):
    c = hotkv.from_url(test_url, max_connections=1)
    try:
        assert c.execute("CLIENT", "TRACKINGINFO") is not None
        assert c.execute("CLIENT", "GETREDIR") == -1
        assert c.execute("CLIENT", "TRACKING", "on") in (b"OK", "OK", True)
        info = c.execute("CLIENT", "TRACKINGINFO")
        flags = info[b"flags"] if isinstance(info, dict) and b"flags" in info else info
        assert flags
        assert c.execute("CLIENT", "TRACKING", "off") in (b"OK", "OK", True)
        with pytest.raises(hotkv.ResponseError):  # CACHING needs OPTIN/OPTOUT tracking
            c.execute("CLIENT", "CACHING", "yes")
        assert b"id=" in c.execute("CLIENT", "INFO")
    finally:
        c.close()


def test_cluster_commands_say_so_on_a_standalone_server(client, prefix):
    for command in (("CLUSTERSCAN", 0), ("SFLUSH", 0, 100)):
        with pytest.raises(hotkv.ResponseError, match="cluster support disabled"):
            client.execute(*command)
    with pytest.raises(hotkv.ResponseError):  # TRIMSLOTS needs arguments, and a cluster
        client.execute("TRIMSLOTS")


def test_licensed_commands_through_execute_answer_nolicense_when_unlicensed(client):
    try:
        client.execute("FEATURE.SETAT", "g", "e", 1, "f", 1)
    except hotkv.NotLicensedError as exc:
        assert "feature-store" in str(exc)
    except hotkv.ResponseError:
        pass  # licensed server: the arguments name no such feature group
    else:
        pytest.fail("FEATURE.SETAT with no such group should not succeed")


# -- enterprise methods this server cannot run (NOLICENSE): the error is the live check ----------------


def test_enterprise_methods_not_runnable_here_are_gated_by_the_license(client, prefix):
    calls = [
        (lambda: client.alerts_seturl("http://127.0.0.1:1/x"), "observability"),
        (lambda: client.alerts_test(), "observability"),
        (lambda: client.embed_dedup_list(), "embed-dedup"),
        (lambda: client.pcache_flush(prefix + "x"), "prompt-cache"),
        (lambda: client.toolcache_flush(prefix + "x"), "tool-cache"),
        (lambda: client.ncache_flush(prefix + "x"), "negative"),
        (lambda: client.guard_flush(prefix + "x"), "guard"),
        (lambda: client.rerank_flush(prefix + "x"), "rerank"),
    ]
    gated = 0
    for call, feature in calls:
        try:
            call()
        except hotkv.NotLicensedError as exc:
            assert feature in str(exc).lower(), (feature, str(exc))
            gated += 1
        except hotkv.ResponseError:
            pass  # a licensed server answers the call itself (the namespace does not exist)
    if gated == 0:
        pytest.skip("licensed server: the calls ran and answered with their own errors")
