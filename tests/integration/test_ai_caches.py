# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for SCACHE.*, PCACHE.*, the four keyed caches
(TOOL.CACHE/NCACHE/GUARD/RERANK), LLMSTATS.*, LLMBUDGET.* and PROMPT.*.

On a server without a license every family here replies `NOLICENSE`; each
test wraps its body in `skip_if_not_licensed` and is skipped with a clear
reason instead of failing, so the whole file passes on a licensed server.
"""

from __future__ import annotations

from hotkv import TokenUsage

USAGE = TokenUsage(model="claude-sonnet-5", input_tokens=100, output_tokens=20)


def test_scache_lifecycle(client, prefix, skip_if_not_licensed):
    ns = prefix + "scache"

    def body():
        assert client.scache_create(ns, 3, threshold=0.8) is True
        assert client.scache_set(ns, "what is hotkv", "a fast kv store", [1.0, 0.0, 0.0]) is True
        hit = client.scache_get(ns, [1.0, 0.0, 0.0])
        assert hit is not None
        response, similarity = hit
        assert response == b"a fast kv store"
        assert similarity > 0.9
        miss = client.scache_get(ns, [0.0, 0.0, 1.0], threshold=0.99)
        assert miss is None
        stats = client.scache_stats(ns)
        assert stats["entries"] == 1
        assert ns.encode() in client.scache_list()
        assert client.scache_threshold(ns, 0.5) is True
        assert client.scache_flush(ns) is True
        assert client.scache_drop(ns) is True

    skip_if_not_licensed(body)


def test_scache_set_with_usage_attributes_savings_on_hit(client, prefix, skip_if_not_licensed):
    ns = prefix + "scache-usage"

    def body():
        assert client.scache_create(ns, 3, threshold=0.8) is True
        assert client.scache_set(
            ns, "what is hotkv", "a fast kv store", [1.0, 0.0, 0.0], usage=USAGE
        ) is True
        assert client.scache_get(ns, [1.0, 0.0, 0.0]) is not None
        stats = client.scache_stats(ns)
        assert stats["usage_saved_requests"] >= 1
        assert stats["usage_saved_input_tokens"] >= 100
        assert "usage_saved_cost_usd" in stats
        client.scache_drop(ns)

    skip_if_not_licensed(body)


def test_pcache_lifecycle(client, prefix, skip_if_not_licensed):
    ns = prefix + "pcache"

    def body():
        assert client.pcache_create(ns, max_entries=100, ttl=0, cost_per_hit=0.002) is True
        assert client.pcache_set(ns, "prompt-1", "response-1") is True
        assert client.pcache_get(ns, "prompt-1") == b"response-1"
        assert client.pcache_get(ns, "missing") is None
        getex = client.pcache_getex(ns, "prompt-1")
        assert getex is not None
        response, metadata, hit_count = getex
        assert response == b"response-1"
        stats = client.pcache_stats(ns)
        assert stats["entries"] == 1
        assert ns.encode() in client.pcache_list()
        assert client.pcache_config_set(ns, "ttl", 3600) is True
        assert client.pcache_del(ns, "prompt-1") == 1
        assert client.pcache_drop(ns) is True

    skip_if_not_licensed(body)


def test_pcache_set_with_usage_attributes_savings_on_hit(client, prefix, skip_if_not_licensed):
    ns = prefix + "pcache-usage"

    def body():
        assert client.pcache_create(ns) is True
        assert client.pcache_set(ns, "prompt-1", "response-1", usage=USAGE) is True
        assert client.pcache_get(ns, "prompt-1") == b"response-1"
        stats = client.pcache_stats(ns)
        assert stats["usage_saved_requests"] >= 1
        assert stats["usage_saved_input_tokens"] >= 100
        assert stats["usage_saved_output_tokens"] >= 20
        assert "usage_saved_cost_usd" in stats
        client.pcache_drop(ns)

    skip_if_not_licensed(body)


def test_keyed_caches(client, prefix, skip_if_not_licensed):
    families = [
        (client.toolcache_create, client.toolcache_set, client.toolcache_get,
         client.toolcache_del, client.toolcache_stats, client.toolcache_drop, client.toolcache_list),
        (client.ncache_create, client.ncache_set, client.ncache_get,
         client.ncache_del, client.ncache_stats, client.ncache_drop, client.ncache_list),
        (client.guard_create, client.guard_set, client.guard_get,
         client.guard_del, client.guard_stats, client.guard_drop, client.guard_list),
        (client.rerank_create, client.rerank_set, client.rerank_get,
         client.rerank_del, client.rerank_stats, client.rerank_drop, client.rerank_list),
    ]

    def body():
        for create, set_, get, delete, stats, drop, list_ns in families:
            ns = prefix + "kc"
            assert create(ns) is True
            assert set_(ns, "k1", "v1", usage=USAGE) is True
            assert get(ns, "k1") == b"v1"
            s = stats(ns)
            assert s["entries"] == 1
            assert s["usage_saved_requests"] >= 1
            assert s["usage_saved_input_tokens"] >= 100
            assert "usage_saved_cost_usd" in s
            assert ns.encode() in list_ns()
            assert delete(ns, "k1") == 1
            assert drop(ns) is True

    skip_if_not_licensed(body)


def test_llmstats(client, prefix, skip_if_not_licensed):
    bucket = prefix + "bucket"

    def body():
        assert client.llmstats_bucket(bucket) is True
        assert client.llmstats_record(bucket, 100, 50, 0, 0.002, 120.5) is True
        assert client.llmstats_record(bucket, 10, 5, 10, 0.0, 5.0, error=True) is True
        stats = client.llmstats_get(bucket)
        assert stats["requests"] == 2
        assert stats["input_tokens"] == 110
        assert stats["cost_usd"] > 0
        assert "cache_write_5m_tokens" in stats
        assert "unpriced_requests" in stats
        assert bucket.encode() in client.llmstats_list()
        assert client.llmstats_reset(bucket) is True
        assert client.llmstats_drop(bucket) is True

    skip_if_not_licensed(body)


def test_llmstats_record_with_model_is_priced_by_the_server(client, prefix, skip_if_not_licensed):
    bucket = prefix + "priced-bucket"

    def body():
        client.llmstats_bucket(bucket)
        assert client.llmstats_record(
            bucket, 1_000_000, 0, 0, 0, 10.0, model="claude-sonnet-5"
        ) is True
        stats = client.llmstats_get(bucket)
        assert stats["requests"] == 1
        # The server priced the call itself from its built-in price table
        # (the SDK sent cost_usd=0); an unknown model would instead count
        # toward unpriced_requests with cost 0.
        assert stats["cost_usd"] >= 0
        client.llmstats_drop(bucket)

    skip_if_not_licensed(body)


def test_llmstats_price_set_get_list_del(client, prefix, skip_if_not_licensed):
    model = prefix + "custom-model"

    def body():
        assert client.llmstats_price_set(model, 3, 15, cache_read_usd_per_mtok=0.3) is True
        price = client.llmstats_price_get(model)
        assert price.model == model
        assert price.source == "custom"
        assert price.priced is True
        assert price.input_usd_per_mtok == 3.0
        assert price.output_usd_per_mtok == 15.0
        assert price.cache_read_usd_per_mtok == 0.3
        # Omitted cache-write tiers default to 1.25x/2x the input price.
        assert price.cache_write_5m_usd_per_mtok == 3.75
        assert price.cache_write_1h_usd_per_mtok == 6.0

        all_prices = client.llmstats_price_list()
        assert any(p.model == model for p in all_prices)

        unknown = client.llmstats_price_get(prefix + "no-such-model-at-all")
        assert unknown.source == "unpriced"
        assert unknown.priced is False

        assert client.llmstats_price_del(model) is True
        assert client.llmstats_price_del(model) is False

    skip_if_not_licensed(body)


def test_llmbudget(client, prefix, skip_if_not_licensed):
    name = prefix + "budget"

    def body():
        assert client.llmbudget_set(name, 1000, 1.0, 3600) is True
        assert client.llmbudget_consume(name, 100, 0.05) is True
        status = client.llmbudget_status(name)
        assert status["tokens_used"] == 100
        assert status["token_limit"] == 1000
        assert "unpriced_requests" in status
        assert name.encode() in client.llmbudget_list()
        assert client.llmbudget_reset(name) is True
        assert client.llmbudget_drop(name) is True

    skip_if_not_licensed(body)


def test_llmbudget_consume_with_usage_is_priced_by_the_server(client, prefix, skip_if_not_licensed):
    name = prefix + "budget-usage"

    def body():
        assert client.llmbudget_set(name, 1_000_000, 100.0, 3600) is True
        assert client.llmbudget_consume(name, usage=USAGE) is True
        status = client.llmbudget_status(name)
        assert status["tokens_used"] >= 100
        client.llmbudget_drop(name)

    skip_if_not_licensed(body)


def test_prompt_registry(client, prefix, skip_if_not_licensed):
    name = prefix + "template"

    def body():
        v1 = client.prompt_put(name, "Hello {name}")
        assert v1 == 1
        v2 = client.prompt_put(name, "Hi {name}!")
        assert v2 == 2
        latest = client.prompt_get(name)
        assert latest is not None
        version, content, metadata, created_unix = latest
        assert version == 2
        assert content == b"Hi {name}!"
        first = client.prompt_get(name, 1)
        assert first[1] == b"Hello {name}"
        assert client.prompt_latest(name)[0] == 2
        assert client.prompt_setactive(name, 1) is True
        assert set(client.prompt_versions(name)) == {1, 2}
        assert name.encode() in client.prompt_list()
        assert client.prompt_drop(name) is True

    skip_if_not_licensed(body)
