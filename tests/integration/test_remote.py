# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""The client as a customer uses it: over the internet, to a TLS and password
protected instance, licensed for the AI features. Skipped unless
`HOTKV_REMOTE_HOST` is set.

Environment (nothing here is ever written to a file or printed):

    HOTKV_REMOTE_HOST       host or IP of the instance
    HOTKV_REMOTE_PORT       its TLS port (default 6379)
    HOTKV_REMOTE_PASSWORD   the password (the user name defaults to "default";
                            HOTKV_REMOTE_USER overrides it)
    HOTKV_REMOTE_CA         a CA (or the self-signed certificate) to verify the
                            server with; without it the system roots are used
    HOTKV_REMOTE_PLAIN=1    the instance speaks plain TCP (a local rehearsal)

The instance is shared, so every key and engine entry carries a random prefix
`remote-python-<random>` and is removed at the end, and nothing global is touched
(no FLUSHALL, no CONFIG SET, no ACL, no alert settings, no price table)."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import urllib.parse
import uuid

import pytest

import hotkv
import hotkv.asyncio

HOST = os.environ.get("HOTKV_REMOTE_HOST")
PORT = int(os.environ.get("HOTKV_REMOTE_PORT", "6379"))
CA = os.environ.get("HOTKV_REMOTE_CA")
PLAIN = os.environ.get("HOTKV_REMOTE_PLAIN") == "1"
USER = os.environ.get("HOTKV_REMOTE_USER")

pytestmark = pytest.mark.skipif(not HOST, reason="HOTKV_REMOTE_HOST is not set; skipping the remote tests")


def _password() -> str:
    return os.environ.get("HOTKV_REMOTE_PASSWORD", "")


def _options(**extra):
    opts = dict(host=HOST, port=PORT, password=_password() or None, connect_timeout=15.0, command_timeout=30.0)
    if USER:
        opts["username"] = USER
    if not PLAIN:
        opts["tls"] = True
        if CA:
            opts["tls_ca_file"] = CA
    opts.update(extra)
    return opts


def _url() -> str:
    scheme = "hotkv" if PLAIN else "hotkvs"
    user = urllib.parse.quote(USER or "", safe="")
    secret = urllib.parse.quote(_password(), safe="")
    query = "" if (PLAIN or not CA) else "?tls_ca_file=" + urllib.parse.quote(CA, safe="")
    return f"{scheme}://{user}:{secret}@{HOST}:{PORT}/0{query}"


@pytest.fixture(scope="module")
def tag():
    return "remote-python-" + uuid.uuid4().hex[:10]


@pytest.fixture(scope="module")
def cleanups():
    actions = []
    yield actions
    c = hotkv.HotKV(**_options())
    try:
        for action in reversed(actions):
            try:
                action(c)
            except Exception:  # noqa: BLE001 - best effort, the tests report their own failures
                pass
    finally:
        c.close()


@pytest.fixture()
def r():
    c = hotkv.HotKV(**_options())
    yield c
    c.close()


def later(cleanups, *calls):
    """Cleanup steps, each tried on its own at the end of the module."""
    for call in calls:
        cleanups.append(call)


def keep(cleanups, *keys):
    cleanups.append(lambda c, ks=keys: [c.delete(k) for k in ks])


# -- connecting -------------------------------------------------------------------------------


def test_ping_over_the_options_and_over_the_url(r):
    assert r.ping() in (b"PONG", "PONG")
    from_url = hotkv.from_url(_url(), connect_timeout=15.0)
    try:
        assert from_url.ping() in (b"PONG", "PONG")
        assert from_url.echo("remote") == b"remote"
    finally:
        from_url.close()


def test_a_wrong_password_fails_cleanly():
    if not _password():
        pytest.skip("the instance has no password")
    bad = hotkv.HotKV(**_options(password="not-the-password"))
    try:
        with pytest.raises(hotkv.HotKVError) as caught:
            bad.ping()
        assert isinstance(caught.value, (hotkv.AuthenticationError, hotkv.NoAuthError, hotkv.WrongPassError, hotkv.ConnectionError))
        assert "not-the-password" not in str(caught.value)
    finally:
        bad.close()
    nopass = hotkv.HotKV(**_options(password=None))
    try:
        with pytest.raises(hotkv.HotKVError):
            nopass.set("remote-python-denied", "x")
    finally:
        nopass.close()


@pytest.mark.skipif(PLAIN, reason="TLS only")
def test_tls_with_the_wrong_ca_is_refused():
    other = hotkv.HotKV(**_options(tls_ca_file=None, tls_server_hostname="nobody.example"))
    try:
        with pytest.raises(hotkv.ConnectionError):
            other.ping()
    finally:
        other.close()


@pytest.mark.skipif(PLAIN, reason="TLS only")
def test_insecure_skip_verify_connects_without_the_ca():
    insecure = hotkv.HotKV(**_options(tls_ca_file=None, tls_insecure_skip_verify=True))
    try:
        assert insecure.ping() in (b"PONG", "PONG")
    finally:
        insecure.close()


# -- data types ----------------------------------------------------------------------------------


def test_strings_and_expiry(r, tag, cleanups):
    key = f"{tag}:s"
    keep(cleanups, key)
    assert r.set(key, "value", ex=100) is True
    assert r.get(key) == b"value"
    assert 0 < r.ttl(key) <= 100
    assert r.incr(key + ":n") == 1 and r.incrby(key + ":n", 9) == 10
    keep(cleanups, key + ":n")
    assert r.set(key + ":short", "x", px=300) is True
    keep(cleanups, key + ":short")
    time.sleep(0.8)
    assert r.get(key + ":short") is None
    assert r.expire(key, 50) is True and r.persist(key) is True and r.ttl(key) == -1
    assert r.mset({key + ":a": "1", key + ":b": "2"}) is True
    keep(cleanups, key + ":a", key + ":b")
    assert r.mget([key + ":a", key + ":b", key + ":missing"]) == [b"1", b"2", None]
    assert r.get(key + ":missing") is None


def test_hashes_lists_sets_zsets_streams(r, tag, cleanups):
    h, li, s, z, x = (f"{tag}:{n}" for n in ("hash", "list", "set", "zset", "stream"))
    keep(cleanups, h, li, s, z, x)
    assert r.hset(h, {"a": "1", "b": "2"}) == 2
    assert r.hgetall(h) == {b"a": b"1", b"b": b"2"}
    assert r.hincrby(h, "a", 5) == 6
    assert r.hdel(h, "b") == 1
    assert r.rpush(li, "1", "2", "3") == 3 and r.lpush(li, "0") == 4
    assert r.lrange(li, 0, -1) == [b"0", b"1", b"2", b"3"]
    assert r.lpop(li) == b"0" and r.llen(li) == 3
    assert r.sadd(s, "a", "b", "c") == 3 and r.sismember(s, "b") is True
    assert sorted(r.smembers(s)) == [b"a", b"b", b"c"]
    assert r.zadd(z, {"m1": 1.5, "m2": 1e-05, "m3": 1e21}) == 3
    assert r.zrange(z, 0, -1, withscores=True) == [(b"m2", 1e-05), (b"m1", 1.5), (b"m3", 1e21)]
    assert r.zscore(z, "m1") == 1.5 and r.zrank(z, "m3") == 2
    first = r.xadd(x, {"f": "v"})
    r.xadd(x, {"f": "w"})
    assert r.xlen(x) == 2
    entries = r.xrange(x)
    assert [fields for _, fields in entries] == [{b"f": b"v"}, {b"f": b"w"}]
    assert entries[0][0] == first


def test_a_one_megabyte_value_round_trips(r, tag, cleanups):
    key = f"{tag}:big"
    keep(cleanups, key)
    blob = os.urandom(1024 * 1024)
    assert r.set(key, blob) is True
    assert r.strlen(key) == len(blob)
    assert r.get(key) == blob
    text = ("héllo wörld " * 100_000).encode()
    r.set(key, text)
    assert r.get(key) == text


def test_binary_safe_keys_and_values(r, tag, cleanups):
    key = f"{tag}:bin".encode() + b"\x00\xff\r\n"
    keep(cleanups, key)
    value = bytes(range(256)) * 4
    assert r.set(key, value) is True
    assert r.get(key) == value


# -- pipelines, transactions, pub/sub ------------------------------------------------------------------


def test_pipeline_and_transactions(r, tag, cleanups):
    a, n = f"{tag}:pa", f"{tag}:pn"
    keep(cleanups, a, n)
    with r.pipeline(transaction=False) as pipe:
        pipe.set(a, "1")
        pipe.incr(n)
        pipe.incr(n)
        pipe.get(a)
        assert pipe.execute() == [True, 1, 2, b"1"]
    with r.pipeline() as tx:  # MULTI / EXEC
        tx.set(a, "2")
        tx.incr(n)
        tx.get(a)
        assert tx.execute() == [True, 3, b"2"]

    def bump(pipe):
        current = int(pipe.get(n))
        pipe.multi()
        pipe.set(n, current + 10)

    r.transaction(bump, n)
    assert r.get(n) == b"13"
    with r.pipeline(transaction=False) as pipe:  # one failing command does not abort the batch
        pipe.set(a, "x")
        pipe.lpush(a, "y")
        pipe.get(a)
        results = pipe.execute(raise_on_error=False)
    assert results[0] is True and isinstance(results[1], hotkv.WrongTypeError) and results[2] == b"x"


def test_watch_notices_another_connection(r, tag, cleanups):
    key = f"{tag}:w"
    keep(cleanups, key)
    r.set(key, "1")
    other = hotkv.HotKV(**_options())
    try:
        pipe = r.pipeline()
        pipe.watch(key)
        other.set(key, "changed")
        pipe.multi()
        pipe.set(key, "mine")
        with pytest.raises(hotkv.WatchError):
            pipe.execute()
    finally:
        other.close()
    assert r.get(key) == b"changed"


def test_pubsub_between_separate_connections(tag):
    channel = f"{tag}:chan"
    sub_client, pub = hotkv.HotKV(**_options()), hotkv.HotKV(**_options())
    try:
        sub = sub_client.pubsub()
        sub.subscribe(channel)
        sub.psubscribe(f"{tag}:pat*")
        deadline = time.time() + 10
        while pub.pubsub_numsub(channel).get(channel.encode(), 0) < 1 and time.time() < deadline:
            time.sleep(0.05)
        assert pub.publish(channel, "hello") == 1
        assert pub.publish(f"{tag}:pattern", "p") == 1
        got = {}
        deadline = time.time() + 10
        while len(got) < 2 and time.time() < deadline:
            m = sub.get_message(timeout=0.5)
            if m is not None and m.data in (b"hello", b"p"):
                got[m.data] = m
        assert set(got) == {b"hello", b"p"}
        sub.close()
    finally:
        sub_client.close()
        pub.close()


# -- the connection survives, and many callers share it ------------------------------------------------------


def test_reconnects_after_the_server_closes_the_connection(tag, cleanups):
    key = f"{tag}:rc"
    keep(cleanups, key)
    victim = hotkv.HotKV(**_options(max_connections=1))
    killer = hotkv.HotKV(**_options())
    try:
        assert victim.set(key, "before") is True
        my_id = victim.client_id()
        assert killer.execute("CLIENT", "KILL", "ID", my_id) == 1
        assert victim.get(key) == b"before"  # a read is retried once on a new connection
        killer.execute("CLIENT", "KILL", "ID", victim.client_id())
        with pytest.raises(hotkv.ConnectionError):  # a write is not blindly repeated
            victim.set(key, "lost")
        assert victim.set(key, "after") is True  # and the pool recovered by itself
        assert victim.get(key) == b"after"
    finally:
        victim.close()
        killer.close()


def test_threads_share_one_client(tag, cleanups):
    shared = hotkv.HotKV(**_options(max_connections=8))
    keys = [f"{tag}:th{i}" for i in range(8)]
    keep(cleanups, *keys)
    errors = []

    def work(i):
        try:
            for _ in range(50):
                shared.incr(keys[i])
                shared.set(keys[i] + "x", str(i))
            assert shared.get(keys[i]) == b"50"
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not errors, errors[:3]
        keep(cleanups, *[k + "x" for k in keys])
    finally:
        shared.close()


async def test_asyncio_client_and_gather(tag, cleanups):
    c = hotkv.asyncio.HotKV(**_options(max_connections=10))
    keys = [f"{tag}:as{i}" for i in range(20)]
    keep(cleanups, *keys)
    try:
        assert await c.ping() in (b"PONG", "PONG")
        await asyncio.gather(*(c.set(k, str(i)) for i, k in enumerate(keys)))
        got = await asyncio.gather(*(c.get(k) for k in keys))
        assert got == [str(i).encode() for i in range(20)]
        counts = await asyncio.gather(*(c.incr(keys[0] + "n") for _ in range(30)))
        assert sorted(counts) == list(range(1, 31))
        keep(cleanups, keys[0] + "n")
        async with c.pipeline(transaction=False) as pipe:
            await pipe.set(keys[1], "p")
            await pipe.get(keys[1])
            assert await pipe.execute() == [True, b"p"]
        from_url = hotkv.asyncio.from_url(_url(), connect_timeout=15.0)
        try:
            assert await from_url.ping() in (b"PONG", "PONG")
        finally:
            await from_url.close()
    finally:
        await c.close()


# -- the AI features, through the typed API --------------------------------------------------------------------------


@pytest.fixture()
def ai(r):
    try:
        r.scache_list()
    except hotkv.NotLicensedError as exc:
        pytest.skip(f"the instance is not licensed for the AI features: {exc}")
    return r


def test_semantic_and_prompt_caches(ai, tag, cleanups):
    sc, pc = f"{tag}-sc", f"{tag}-pc"
    later(cleanups, lambda c: c.scache_drop(sc), lambda c: c.pcache_drop(pc))
    assert ai.scache_create(sc, 4, threshold=0.9) is True
    assert ai.scache_set(sc, "what is hotkv", "a fast key value store", [1.0, 0.0, 0.0, 0.0]) is True
    answer, similarity = ai.scache_get(sc, [0.99, 0.05, 0.0, 0.0])
    assert answer == b"a fast key value store" and similarity > 0.95
    assert ai.scache_get(sc, [0.0, 0.0, 1.0, 0.0]) is None
    stats = ai.scache_stats(sc)
    assert stats["entries"] == 1 and stats["hits"] == 1 and stats["misses"] == 1
    assert ai.scache_drop(sc) is True
    assert ai.pcache_create(pc, cost_per_hit=1e-06) is True
    assert ai.pcache_set(pc, "p", "r", usage=hotkv.TokenUsage(model="claude-sonnet-5", input_tokens=100, output_tokens=20)) is True
    assert ai.pcache_get(pc, "p") == b"r" and ai.pcache_get(pc, "other") is None
    stats = ai.pcache_stats(pc)
    assert stats["hits"] == 1 and stats["usage_saved_input_tokens"] >= 100
    assert ai.pcache_drop(pc) is True


def test_keyed_caches(ai, tag, cleanups):
    ns = f"{tag}-kc"
    for drop in (ai.toolcache_drop, ai.ncache_drop, ai.guard_drop, ai.rerank_drop):
        cleanups.append(lambda c, d=drop: d(ns))
    for create, set_, get, delete, stats, drop in (
        (ai.toolcache_create, ai.toolcache_set, ai.toolcache_get, ai.toolcache_del, ai.toolcache_stats, ai.toolcache_drop),
        (ai.ncache_create, ai.ncache_set, ai.ncache_get, ai.ncache_del, ai.ncache_stats, ai.ncache_drop),
        (ai.guard_create, ai.guard_set, ai.guard_get, ai.guard_del, ai.guard_stats, ai.guard_drop),
        (ai.rerank_create, ai.rerank_set, ai.rerank_get, ai.rerank_del, ai.rerank_stats, ai.rerank_drop),
    ):
        assert create(ns, max_entries=100) is True
        assert set_(ns, "k", "v", ttl=60) is True
        assert get(ns, "k") == b"v" and get(ns, "none") is None
        assert stats(ns)["entries"] == 1
        assert delete(ns, "k") == 1
        assert drop(ns) is True


def test_llm_budget_with_float_limits_and_llm_stats(ai, tag, cleanups):
    budget, bucket = f"{tag}-budget", f"{tag}-bucket"
    later(cleanups, lambda c: c.llmbudget_drop(budget), lambda c: c.llmstats_drop(bucket))
    assert ai.llmbudget_set(budget, 1_000_000, 1e-05, 3600) is True
    assert float(ai.llmbudget_status(budget)["cost_limit_usd"]) == pytest.approx(1e-05)
    assert ai.llmbudget_consume(budget, 100, 4e-06) is True
    assert ai.llmbudget_consume(budget, 100, 7e-06) is False
    assert ai.llmbudget_set(budget, 1_000_000, 12.5, 3600) is True
    assert ai.llmbudget_consume(budget, 10, 0.25) is True
    status = ai.llmbudget_status(budget)
    assert status["tokens_used"] == 110 and float(status["cost_used_usd"]) == pytest.approx(0.25, abs=1e-5)
    assert ai.llmbudget_drop(budget) is True
    assert ai.llmstats_bucket(bucket) is True
    assert ai.llmstats_record(bucket, 100, 20, 0, 1e-05, 120.0) is True
    assert ai.llmstats_record(bucket, 50, 10, 0, 0.002, 80.0, error=True) is True
    stats = ai.llmstats_get(bucket)
    assert stats["requests"] == 2 and stats["input_tokens"] == 150
    assert ai.llmstats_drop(bucket) is True


def test_prompt_registry_and_embed_dedup(ai, tag, cleanups):
    name, ns = f"{tag}-prompt", f"{tag}-dedup"
    later(cleanups, lambda c: c.prompt_drop(name), lambda c: c.embed_dedup_drop(ns))
    assert ai.prompt_put(name, "v1 {{x}}") == 1 and ai.prompt_put(name, "v2 {{x}}", meta="{}") == 2
    assert ai.prompt_get(name)[1] == b"v2 {{x}}" and ai.prompt_get(name, 1)[1] == b"v1 {{x}}"
    assert ai.prompt_versions(name) == [1, 2]
    assert ai.prompt_setactive(name, 1) is True
    assert name.encode() in ai.prompt_list()
    assert ai.prompt_drop(name) is True
    assert ai.embed_dedup_put(ns, "h", "m", "0.1,0.2") is True
    assert ai.embed_dedup_put(ns, "h", "m", "0.1,0.2") is False
    assert ai.embed_dedup_get(ns, "h") == b"0.1,0.2"
    assert ai.embed_dedup_stats(ns)["entries"] == 1
    assert ai.embed_dedup_drop(ns) is True


def test_agent_memory_and_rag(ai, tag, cleanups):
    session, agent, rag = f"{tag}-sess", f"{tag}-agent", f"{tag}-rag"
    later(cleanups, lambda c: c.amem_conv_drop(session), lambda c: c.amem_work_clear(session), lambda c: c.amem_epis_clear(agent), lambda c: c.rag_drop(rag))
    assert ai.amem_conv_create(session, window=10, ttl=600) is True
    assert ai.amem_conv_append(session, "user", "hello") is True and ai.amem_conv_append(session, "assistant", "hi") is True
    turns = ai.amem_conv_get(session)
    assert [t["role"] for t in turns] in ([b"user", b"assistant"], ["user", "assistant"])
    assert ai.amem_work_set(session, "k", "v", ttl=60) is True and ai.amem_work_get(session, "k") == b"v"
    assert ai.amem_sem_create(agent, 3) is True
    assert ai.amem_sem_store(agent, "m1", "likes tea", [1.0, 0.0, 0.0]) is True
    recalled = ai.amem_sem_recall(agent, 1, [0.9, 0.1, 0.0])
    assert len(recalled) == 1
    assert ai.amem_epis_log(agent, "tool", "ran search") is True
    assert ai.amem_epis_stats(agent)["total_events"] == 1
    assert ai.rag_create(rag, 3) is True
    assert ai.rag_ingest(rag, "doc1", "HotKV is a fast in memory key value store.", vectors=[[1.0, 0.0, 0.0]]) >= 1
    hits = ai.rag_search(rag, "fast store", k=1, vector=[1.0, 0.0, 0.0])
    assert len(hits) == 1
    assert ai.rag_stats(rag)["totalDocs"] == 1
    assert ai.rag_drop(rag) is True


def test_rate_limits_time_series_features_and_vectors(ai, tag, cleanups):
    rl, ts, group, index, gcra = f"{tag}-rl", f"{tag}:ts", f"{tag}-fs", f"{tag}-vi", f"{tag}:gcra"
    later(cleanups, lambda c: c.ratelimit_drop(rl), lambda c: c.ts_delete(ts), lambda c: c.feature_drop(group), lambda c: c.vsim_dropindex(index), lambda c: c.delete(gcra))
    assert ai.ratelimit_create(rl, 2, 60_000) is True
    assert [ai.ratelimit_check(rl, "u").allowed for _ in range(3)] == [True, True, False]
    assert ai.ratelimit_reset(rl, "u") is True and ai.ratelimit_check(rl, "u").allowed is True
    assert ai.ratelimit_drop(rl) is True
    verdicts = [ai.gcra_limit(gcra, 60_000, 2) for _ in range(4)]
    assert [v.allowed for v in verdicts][:2] == [True, True] and False in [v.allowed for v in verdicts]
    assert verdicts[0].limit == 2 and verdicts[0].remaining == 1
    now = int(time.time() * 1000)
    assert ai.ts_create(ts, retention_ms=3_600_000) is True
    for i, v in enumerate((1.5, 2.5, 3.5)):
        assert ai.ts_add(ts, v, now + i) == now + i
    assert ai.ts_range(ts, now, now + 5) == [(now, 1.5), (now + 1, 2.5), (now + 2, 3.5)]
    assert ai.ts_info(ts)["totalSamples"] == 3
    assert ai.ts_delete(ts) is True
    assert ai.feature_create(group, "d", {"age": "int64", "score": "double"}) is True
    assert ai.feature_set(group, "u1", {"age": 30, "score": 4.5}) is True
    values, updated, _ = ai.feature_get(group, "u1")
    assert values[b"age"] == 30 and float(values[b"score"]) == 4.5 and updated > 0
    assert ai.feature_drop(group) is True
    assert ai.vsim_createindex(index, 3) is True
    assert ai.vsim_set(index, "a", [1.0, 0.0, 0.0]) is True and ai.vsim_set(index, "b", [0.0, 1.0, 0.0]) is True
    hits = ai.vsim_search(index, 2, [0.9, 0.1, 0.0])
    assert hits[0][0] == b"a" and len(hits) == 2
    assert ai.vsim_dropindex(index) is True
