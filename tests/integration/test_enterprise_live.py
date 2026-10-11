# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Live checks of the licensed engines through the typed methods, with the
values the server answers: number formatting on the wire, `pxat`, the methods
the other modules do not reach, snapshots, and reads of the management
families. Every test skips on a server that answers NOLICENSE (the wire
arguments are checked in tests/unit/test_enterprise_wire.py regardless)."""

from __future__ import annotations

import time

import pytest

import hotkv

AT_FAR = (int(time.time()) + 86400 * 30) * 1000
AT_PAST = int(time.time() * 1000) - 60_000


@pytest.fixture()
def licensed(client):
    """The client, after one real enterprise call; skip when the server
    answers NOLICENSE."""
    try:
        client.scache_list()
    except hotkv.NotLicensedError as exc:
        pytest.skip(f"the server is not licensed for the enterprise engines: {exc}")
    return client


# -- number formatting, live ----------------------------------------------------------------


def test_llmbudget_cost_limit_in_fixed_point(licensed, prefix):
    name = prefix + "budget"
    # The exponent form is what the old client sent; the server refuses it.
    with pytest.raises(hotkv.ResponseError):
        licensed.execute("LLMBUDGET.SET", name, 1000, "1e-05", 60)
    for limit, shown in ((1e-05, "0.000010"), (5e-07, "0.000001"), (50, "50.000000"), (0.1 + 0.2, "0.300000"), (1e9, "1000000000.000000")):
        assert licensed.llmbudget_set(name, 1000, limit, 60) is True
        assert licensed.llmbudget_status(name)["cost_limit_usd"] == pytest.approx(float(shown))
    for bad in (-1.0, float("nan"), float("inf"), 1e30):
        with pytest.raises(ValueError):
            licensed.llmbudget_set(name, 1000, bad, 60)
    # a tiny limit is enforced in micro-dollars
    assert licensed.llmbudget_set(name, 0, 1e-05, 60) is True
    assert licensed.llmbudget_consume(name, 0, 4e-06) is True
    assert licensed.llmbudget_consume(name, 0, 5e-06) is True
    assert licensed.llmbudget_consume(name, 0, 2e-06) is False  # 11 micro-dollars of 10
    status = licensed.llmbudget_status(name)
    assert status["cost_used_usd"] == pytest.approx(9e-06, abs=1e-9)
    assert status["requests_rejected"] == 1
    with pytest.raises(ValueError):
        licensed.llmbudget_consume(name, 0, -1.0)
    assert licensed.llmbudget_drop(name) is True


def test_llmstats_prices_and_costs_with_tiny_numbers(licensed, prefix):
    model = prefix + "tiny-model"
    bucket = prefix + "tiny-bucket"
    assert licensed.llmstats_price_set(model, 1e-05, 2.5e-06, cache_read_usd_per_mtok=1e-06) is True
    price = licensed.llmstats_price_get(model)
    assert price.input_usd_per_mtok == pytest.approx(1e-05)
    assert price.output_usd_per_mtok == pytest.approx(3e-06)  # 2.5e-06 rounds half up to 3 micro-dollars
    assert price.cache_read_usd_per_mtok == pytest.approx(1e-06)
    with pytest.raises(ValueError):
        licensed.llmstats_price_set(model, -1, 1)
    assert licensed.llmstats_bucket(bucket) is True
    assert licensed.llmstats_record(bucket, 10, 5, 0, 1e-05, 12.4) is True
    assert licensed.llmstats_record(bucket, 10, 5, 0, 2.5e-06, 1.0) is True
    stats = licensed.llmstats_get(bucket)
    assert stats["requests"] == 2
    assert stats["cost_usd"] == pytest.approx(1.2e-05, abs=1e-9)  # the server records the cost in whole micro-dollars (2.5e-06 is 2)
    assert licensed.llmstats_price_del(model) is True
    assert licensed.llmstats_drop(bucket) is True


def test_thresholds_costs_and_rates_with_tiny_numbers(licensed, prefix):
    sc, pc, rl = prefix + "sc", prefix + "pc", prefix + "rl"
    assert licensed.scache_create(sc, 2, threshold=1e-05) is True
    assert licensed.scache_threshold(sc, 2e-05) is True
    assert licensed.scache_set(sc, "q", "r", [1e-05, 1.0]) is True
    assert licensed.scache_get(sc, [1e-05, 1.0])[0] == b"r"
    assert licensed.scache_get(sc, [1e-05, 1.0], threshold=1e-05) is not None
    assert licensed.scache_set(sc, "q2", "r2", [1e-300, 1.0]) is True  # below float32: stored as 0
    licensed.scache_drop(sc)
    assert licensed.pcache_create(pc, cost_per_hit=1e-06) is True
    assert licensed.pcache_config_set(pc, "COST_PER_HIT", 2e-06) is True
    licensed.pcache_drop(pc)
    assert licensed.ratelimit_create(rl, 5, 1000, algo="token_bucket", refill=1e-05) is True
    assert licensed.ratelimit_check(rl, "k").allowed is True
    licensed.ratelimit_drop(rl)
    with pytest.raises(ValueError):
        licensed.scache_create(sc, 2, threshold=float("inf"))


def test_time_series_and_feature_values_with_tiny_and_huge_numbers(licensed, prefix):
    key, group = prefix + "tsn", prefix + "fn"
    now = int(time.time() * 1000)
    assert licensed.ts_create(key, retention_ms=3_600_000) is True
    for i, value in enumerate((1e-05, 1e21, -2.5, 3.0)):
        assert licensed.ts_add(key, value, now + i) == now + i
    samples = licensed.ts_range(key, now, now + 10)
    assert [t for t, _ in samples] == [now, now + 1, now + 2, now + 3]
    assert samples[1][1] == pytest.approx(1e21)
    assert samples[2][1] == -2.5 and samples[3][1] == 3.0
    assert samples[0][1] == pytest.approx(1e-05, abs=1e-6)  # the server answers 6 decimals
    licensed.ts_delete(key)
    assert licensed.feature_create(group, "numbers", {"x": "double", "n": "int64"}) is True
    assert licensed.feature_set(group, "e", {"x": 1e-05, "n": 3}) is True
    values, _, _ = licensed.feature_get(group, "e")
    assert float(values[b"x"]) == pytest.approx(1e-05, abs=1e-6)  # a double feature comes back as text
    assert values[b"n"] == 3
    licensed.feature_drop(group)


def test_vectors_with_tiny_components_in_the_engines(licensed, prefix):
    agent, rag = prefix + "agent", prefix + "rag"
    assert licensed.amem_sem_create(agent, 2) is True
    assert licensed.amem_sem_store(agent, "k", "content", [1e-05, 1.0]) is True
    found = licensed.amem_sem_recall(agent, 1, [1e-05, 1.0])
    assert len(found) == 1
    assert licensed.amem_sem_forget(agent, "k") is True
    assert licensed.rag_create(rag, 2) is True
    assert licensed.rag_ingest(rag, "d1", "hotkv is a fast key value store", vectors=[[1e-05, 1.0]]) >= 1
    hits = licensed.rag_search(rag, "fast store", k=1, alpha=1e-05, vector=[1e-05, 1.0])
    assert len(hits) == 1
    licensed.rag_drop(rag)


# -- pxat ------------------------------------------------------------------------------------------


def test_pxat_expires_entries_at_an_absolute_time(licensed, prefix):
    pc, sc, kc = prefix + "px-pc", prefix + "px-sc", prefix + "px-kc"
    assert licensed.pcache_create(pc) is True
    assert licensed.pcache_set(pc, "later", "r", pxat=AT_FAR) is True
    assert licensed.pcache_set(pc, "gone", "r", pxat=AT_PAST) is True
    assert licensed.pcache_get(pc, "later") == b"r"
    assert licensed.pcache_get(pc, "gone") is None
    licensed.pcache_drop(pc)
    assert licensed.scache_create(sc, 2) is True
    assert licensed.scache_set(sc, "q", "r", [1.0, 0.0], pxat=AT_FAR) is True
    assert licensed.scache_get(sc, [1.0, 0.0])[0] == b"r"
    assert licensed.scache_set(sc, "q2", "r2", [0.0, 1.0], pxat=AT_PAST) is True
    assert licensed.scache_get(sc, [0.0, 1.0]) is None
    licensed.scache_drop(sc)
    for create, set_, get, drop in (
        (licensed.toolcache_create, licensed.toolcache_set, licensed.toolcache_get, licensed.toolcache_drop),
        (licensed.ncache_create, licensed.ncache_set, licensed.ncache_get, licensed.ncache_drop),
    ):
        assert create(kc) is True
        assert set_(kc, "a", "v", pxat=AT_FAR) is True
        assert set_(kc, "b", "v", pxat=AT_PAST) is True
        assert get(kc, "a") == b"v" and get(kc, "b") is None
        drop(kc)
    with pytest.raises(hotkv.ResponseError):  # more than 100 years ahead
        licensed.execute("PCACHE.SET", pc + "x", "p", "r", "PXAT", AT_FAR * 100)


def test_amem_pxat(licensed, prefix):
    session, agent = prefix + "px-s", prefix + "px-a"
    assert licensed.amem_conv_create(session, window=5, pxat=AT_FAR) is True
    assert licensed.amem_conv_append(session, "user", "hi") is True
    assert licensed.amem_conv_get(session)[0]["content"] in (b"hi", "hi")
    licensed.amem_conv_drop(session)
    assert licensed.amem_work_set(session, "k", "v", pxat=AT_FAR) is True
    assert licensed.amem_work_get(session, "k") == b"v"
    assert licensed.amem_work_set(session, "old", "v", pxat=AT_PAST) is True
    assert licensed.amem_work_get(session, "old") is None
    licensed.amem_work_clear(session)
    assert licensed.amem_sem_create(agent, 2) is True
    assert licensed.amem_sem_store(agent, "k", "c", [1.0, 0.0], pxat=AT_FAR) is True
    assert licensed.amem_sem_store(agent, "gone", "c", [0.0, 1.0], pxat=AT_PAST) is True
    assert b"k" in licensed.amem_sem_list(agent)
    assert b"gone" not in licensed.amem_sem_list(agent)
    licensed.amem_sem_forget(agent, "k")


# -- the methods the other modules do not reach ----------------------------------------------------------


def test_flush_methods_and_embed_dedup_list(licensed, prefix):
    pc, ns = prefix + "fl-pc", prefix + "fl-dedup"
    assert licensed.pcache_create(pc) is True
    for i in range(3):
        licensed.pcache_set(pc, f"p{i}", "r")
    assert licensed.pcache_flush(pc) == 3
    assert licensed.pcache_stats(pc)["entries"] == 0
    licensed.pcache_drop(pc)
    for create, set_, flush, stats, drop in (
        (licensed.toolcache_create, licensed.toolcache_set, licensed.toolcache_flush, licensed.toolcache_stats, licensed.toolcache_drop),
        (licensed.ncache_create, licensed.ncache_set, licensed.ncache_flush, licensed.ncache_stats, licensed.ncache_drop),
        (licensed.guard_create, licensed.guard_set, licensed.guard_flush, licensed.guard_stats, licensed.guard_drop),
        (licensed.rerank_create, licensed.rerank_set, licensed.rerank_flush, licensed.rerank_stats, licensed.rerank_drop),
    ):
        assert create(pc) is True
        set_(pc, "a", "1")
        set_(pc, "b", "2")
        assert flush(pc) == 2
        assert stats(pc)["entries"] == 0
        drop(pc)
    assert licensed.embed_dedup_put(ns, "h1", "model", "0.1,0.2") is True
    assert ns.encode() in licensed.embed_dedup_list()
    assert licensed.embed_dedup_get(ns, "h1") == b"0.1,0.2"
    assert licensed.embed_dedup_put(ns, "h1", "model", "0.1,0.2") is False  # already there: refcount
    assert licensed.embed_dedup_stats(ns)["entries"] == 1
    assert licensed.embed_dedup_release(ns, "h1") is True
    assert licensed.embed_dedup_purge(ns) is True
    assert licensed.embed_dedup_drop(ns) is True


def test_alerts_settings_round_trip(licensed):
    before = licensed.alerts_info()
    try:
        with pytest.raises(hotkv.ResponseError, match="https"):
            licensed.alerts_seturl("http://hooks.invalid/alerts")
        assert licensed.alerts_seturl("https://hooks.invalid/alerts") is True
        assert licensed.alerts_setseverity("critical") is True
        info = licensed.alerts_info()
        assert info["webhook_url"] in (b"https://hooks.invalid/alerts", "https://hooks.invalid/alerts")
        assert info["min_severity"] in (b"critical", "critical")
        assert licensed.alerts_enable() is True
        assert licensed.alerts_test("critical", "sdk test, delivery to an unresolvable host fails quietly") is True
        assert licensed.alerts_disable() is True
    finally:
        licensed.alerts_seturl("")
        licensed.alerts_setseverity(before["min_severity"].decode() if isinstance(before["min_severity"], bytes) else before["min_severity"])


def test_feature_setat_stamps_the_update_time(licensed, prefix):
    group = prefix + "setat"
    assert licensed.feature_create(group, "d", {"age": "int64"}) is True
    stamp = int(time.time() * 1000) - 5000
    assert licensed.feature_setat(group, "u1", stamp, {"age": 41}) is True
    values, updated_at_ms, freshness_ms = licensed.feature_get(group, "u1")
    assert values[b"age"] == 41
    assert updated_at_ms == stamp
    assert 4000 <= freshness_ms <= 60_000
    assert licensed.feature_set(group, "u2", {"age": 7}) is True
    _, now_stamp, _ = licensed.feature_get(group, "u2")
    assert now_stamp > stamp
    licensed.feature_drop(group)


def test_vsim_snapshots(licensed, prefix):
    name = "".join(ch for ch in prefix if ch.isalnum()) + "snap"
    blob = bytes(range(256)) + b"\x00blob"
    stored = licensed.vsim_save(name, blob)
    assert stored >= len(blob)  # the server counts its framing too
    assert licensed.vsim_load(name) == blob
    assert name.encode() in licensed.vsim_list()
    assert licensed.vsim_drop(name) is True
    assert licensed.vsim_load(name) is None
    assert licensed.vsim_drop(name) is False


# -- the management families, through execute() ---------------------------------------------------------------


def test_tenant_lifecycle_through_execute(licensed, prefix):
    tid = "t" + "".join(ch for ch in prefix if ch.isalnum())[:20]
    assert licensed.execute("TENANT.CREATE", tid, "SDK test tenant", "MAX_KEYS", 1000) in (b"OK", "OK")
    try:
        assert tid.encode() in licensed.execute("TENANT.LIST")
        info = licensed.execute("TENANT.INFO", tid)
        assert info is not None and len(info) >= 2
        assert licensed.execute("TENANT.SUSPEND", tid) in (b"OK", "OK")
        assert licensed.execute("TENANT.RESUME", tid) in (b"OK", "OK")
        assert licensed.execute("TENANT.STATS", tid) is not None
    finally:
        assert licensed.execute("TENANT.DROP", tid) in (b"OK", "OK", 1)
    assert tid.encode() not in licensed.execute("TENANT.LIST")


def test_iam_roles_through_execute(licensed, prefix):
    role = "r" + "".join(ch for ch in prefix if ch.isalnum())[:20]
    assert licensed.execute("IAM.ROLE.CREATE", role, "read") in (b"OK", "OK", 1)
    try:
        assert role.encode() in licensed.execute("IAM.ROLE.LIST")
        assert licensed.execute("IAM.ROLE.INFO", role) is not None
        assert licensed.execute("IAM.SESSION.LIST") is not None
    finally:
        licensed.execute("IAM.ROLE.DELETE", role)
    assert role.encode() not in licensed.execute("IAM.ROLE.LIST")


def test_audit_encryption_and_replication_reads_through_execute(licensed):
    def fields(reply):
        return reply if isinstance(reply, dict) else dict(zip(reply[::2], reply[1::2]))

    audit = fields(licensed.execute("AUDIT.INFO"))
    assert audit[b"chain_valid"] in (1, True) or audit["chain_valid"] in (1, True)
    assert "chain verified" in str(licensed.execute("AUDIT.VERIFY"))
    enc = fields(licensed.execute("ENCRYPTION.INFO"))
    assert b"AES-256-GCM" in enc.values() or "AES-256-GCM" in enc.values()
    repl = fields(licensed.execute("REPLICATION.INFO"))
    assert b"primary" in repl.values() or "primary" in repl.values()
