# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for SCACHE.*, PCACHE.*, the keyed caches, LLMSTATS.* and
LLMBUDGET.*: exact wire argument lists for every option (and combinations of them), and name-based parsing of
canned RESP2 flat arrays and RESP3 maps. No server needed.
"""

from __future__ import annotations

from hotkv import TokenUsage
from hotkv.commands.ai_caches import LLMPrice

USAGE_FULL = TokenUsage(
    model="claude-sonnet-5",
    input_tokens=100,
    output_tokens=20,
    cache_read_tokens=5,
    cache_write_5m_tokens=7,
    cache_write_1h_tokens=9,
)
USAGE_MODEL_ONLY = TokenUsage(model="gpt-6-sol")


# -- SCACHE.SET ----------------------------------------------------------------


def test_scache_set_without_usage(recording_client):
    recording_client.scache_set("ns", "q", "r", [1.0, 0.0])
    assert recording_client.last_call == ("SCACHE.SET", "ns", "q", "r", "1", "0")


def test_scache_set_with_ttl_and_usage(recording_client):
    recording_client.scache_set("ns", "q", "r", [1.0, 0.0], ttl=60, usage=USAGE_FULL)
    assert recording_client.last_call == (
        "SCACHE.SET", "ns", "q", "r", "1", "0", "TTL", 60,
        "MODEL", "claude-sonnet-5", "INPUT", 100, "OUTPUT", 20,
        "CACHEREAD", 5, "CACHEWRITE5M", 7, "CACHEWRITE1H", 9,
    )


def test_scache_set_with_usage_only(recording_client):
    recording_client.scache_set("ns", "q", "r", [1.0], usage=USAGE_MODEL_ONLY)
    assert recording_client.last_call == ("SCACHE.SET", "ns", "q", "r", "1", "MODEL", "gpt-6-sol")


# -- PCACHE.SET ----------------------------------------------------------------


def test_pcache_set_bare(recording_client):
    recording_client.pcache_set("ns", "prompt", "resp")
    assert recording_client.last_call == ("PCACHE.SET", "ns", "prompt", "resp")


def test_pcache_set_ttl_meta_and_usage_together(recording_client):
    recording_client.pcache_set(
        "ns", "prompt", "resp", ttl=3600, meta='{"k": 1}', usage=USAGE_MODEL_ONLY
    )
    assert recording_client.last_call == (
        "PCACHE.SET", "ns", "prompt", "resp",
        "TTL", 3600, "META", '{"k": 1}', "MODEL", "gpt-6-sol",
    )


# -- keyed caches (TOOL.CACHE / NCACHE / GUARD / RERANK) ------------------------


def test_keyed_cache_set_every_family_sends_correct_prefix_and_usage(recording_client):
    setters = [
        ("TOOL.CACHE.SET", recording_client.toolcache_set),
        ("NCACHE.SET", recording_client.ncache_set),
        ("GUARD.SET", recording_client.guard_set),
        ("RERANK.SET", recording_client.rerank_set),
    ]
    for expected_name, setter in setters:
        setter("ns", "k", "v", ttl=60, meta="m", usage=USAGE_MODEL_ONLY)
        assert recording_client.last_call == (
            expected_name, "ns", "k", "v", "TTL", 60, "META", "m", "MODEL", "gpt-6-sol"
        )


def test_keyed_cache_set_without_any_options(recording_client):
    recording_client.guard_set("ns", "k", "v")
    assert recording_client.last_call == ("GUARD.SET", "ns", "k", "v")


# -- LLMSTATS.RECORD -------------------------------------------------------------


def test_llmstats_record_bare(recording_client):
    recording_client.llmstats_record("bucket", 100, 50, 0, 0.002, 120.0)
    assert recording_client.last_call == ("LLMSTATS.RECORD", "bucket", 100, 50, 0, "0.002", 120)


def test_llmstats_record_sends_whole_milliseconds(recording_client):
    # The server parses the latency as an unsigned integer and refuses "120.5".
    recording_client.llmstats_record("bucket", 1, 1, 0, 0.0, 120.6)
    assert recording_client.last_call[-1] == 121
    recording_client.llmstats_record("bucket", 1, 1, 0, 0.0, 0.2)
    assert recording_client.last_call[-1] == 0
    recording_client.llmstats_record("bucket", 1, 1, 0, 0.0, -3.0)
    assert recording_client.last_call[-1] == 0


def test_llmstats_record_every_new_option_in_order(recording_client):
    recording_client.llmstats_record(
        "bucket", 100, 50, 10, 0.0, 5.0,
        error=True, model="claude-sonnet-5", cache_write_5m_tokens=7, cache_write_1h_tokens=9,
    )
    assert recording_client.last_call == (
        "LLMSTATS.RECORD", "bucket", 100, 50, 10, "0", 5,
        "ERROR", 1, "MODEL", "claude-sonnet-5", "CACHEWRITE5M", 7, "CACHEWRITE1H", 9,
    )


def test_llmstats_record_error_false_still_sent_explicitly(recording_client):
    recording_client.llmstats_record("bucket", 1, 1, 0, 0.0, 1.0, error=False)
    assert recording_client.last_call == (
        "LLMSTATS.RECORD", "bucket", 1, 1, 0, "0", 1, "ERROR", 0
    )


# -- LLMSTATS.PRICE --------------------------------------------------------------


def test_llmstats_price_set_required_only(recording_client):
    recording_client.llmstats_price_set("gpt-6-sol", 3, 15)
    assert recording_client.last_call == (
        "LLMSTATS.PRICE SET", "gpt-6-sol", "INPUT", "3", "OUTPUT", "15"
    )


def test_llmstats_price_set_all_cache_tiers_and_decimal_string(recording_client):
    recording_client.llmstats_price_set(
        "claude-sonnet-5",
        "3.00",
        15,
        cache_read_usd_per_mtok=0.30,
        cache_write_5m_usd_per_mtok=3.75,
        cache_write_1h_usd_per_mtok=6,
    )
    assert recording_client.last_call == (
        "LLMSTATS.PRICE SET", "claude-sonnet-5",
        "INPUT", "3", "OUTPUT", "15",
        "CACHEREAD", "0.3", "CACHEWRITE5M", "3.75", "CACHEWRITE1H", "6",
    )


def test_llmstats_price_set_formats_without_exponent_or_extra_decimals(recording_client):
    recording_client.llmstats_price_set("m", 0.0000001, 1234567.891234)
    _, _, _, input_price, _, output_price = recording_client.last_call
    assert "e" not in input_price.lower()
    assert "e" not in output_price.lower()
    assert input_price == "0"
    assert output_price == "1234567.891234"


def test_llmstats_price_get_parses_flat_array(recording_client):
    flat = [
        "model", "claude-sonnet-5",
        "source", "default",
        "priced", 1,
        "retired", 0,
        "input_usd_per_mtok", "3.000000",
        "output_usd_per_mtok", "15.000000",
        "cache_read_usd_per_mtok", "0.300000",
        "cache_write_5m_usd_per_mtok", "3.750000",
        "cache_write_1h_usd_per_mtok", "6.000000",
    ]
    recording_client.when("LLMSTATS.PRICE GET", flat)
    price = recording_client.llmstats_price_get("claude-sonnet-5")
    assert recording_client.last_call == ("LLMSTATS.PRICE GET", "claude-sonnet-5")
    assert isinstance(price, LLMPrice)
    assert price == LLMPrice(
        model="claude-sonnet-5",
        source="default",
        priced=True,
        retired=False,
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cache_read_usd_per_mtok=0.3,
        cache_write_5m_usd_per_mtok=3.75,
        cache_write_1h_usd_per_mtok=6.0,
    )


def test_llmstats_price_get_unknown_model_from_resp3_map(recording_client):
    reply = {
        "model": "no-such-model",
        "source": "unpriced",
        "priced": 0,
        "retired": 0,
        "input_usd_per_mtok": "0.000000",
        "output_usd_per_mtok": "0.000000",
        "cache_read_usd_per_mtok": "0.000000",
        "cache_write_5m_usd_per_mtok": "0.000000",
        "cache_write_1h_usd_per_mtok": "0.000000",
    }
    recording_client.when("LLMSTATS.PRICE GET", reply)
    price = recording_client.llmstats_price_get("no-such-model")
    assert price.source == "unpriced"
    assert price.priced is False
    assert price.input_usd_per_mtok == 0.0


def test_llmstats_price_list_parses_array_of_arrays(recording_client):
    one = [
        "model", "a", "source", "custom", "priced", 1, "retired", 1,
        "input_usd_per_mtok", "1.000000", "output_usd_per_mtok", "2.000000",
        "cache_read_usd_per_mtok", "0.100000", "cache_write_5m_usd_per_mtok", "1.250000",
        "cache_write_1h_usd_per_mtok", "2.000000",
    ]
    two = [
        "model", "b", "source", "default", "priced", 1, "retired", 0,
        "input_usd_per_mtok", "3.000000", "output_usd_per_mtok", "4.000000",
        "cache_read_usd_per_mtok", "0.300000", "cache_write_5m_usd_per_mtok", "3.750000",
        "cache_write_1h_usd_per_mtok", "6.000000",
    ]
    recording_client.when("LLMSTATS.PRICE LIST", [one, two])
    prices = recording_client.llmstats_price_list()
    assert recording_client.last_call == ("LLMSTATS.PRICE LIST",)
    assert [p.model for p in prices] == ["a", "b"]
    assert prices[0].retired is True
    assert prices[1].retired is False


def test_llmstats_price_del(recording_client):
    recording_client.when("LLMSTATS.PRICE DEL", 1)
    assert recording_client.llmstats_price_del("custom-model") is True
    assert recording_client.last_call == ("LLMSTATS.PRICE DEL", "custom-model")

    recording_client.when("LLMSTATS.PRICE DEL", 0)
    assert recording_client.llmstats_price_del("custom-model") is False


# -- LLMBUDGET.CONSUME -----------------------------------------------------------


def test_llmbudget_consume_plain_tokens_and_cost(recording_client):
    recording_client.llmbudget_consume("name", 100, 0.05)
    assert recording_client.last_call == ("LLMBUDGET.CONSUME", "name", 100, "0.05")


def test_llmbudget_consume_with_usage_sends_zero_tokens_and_cost(recording_client):
    recording_client.llmbudget_consume("name", usage=USAGE_FULL)
    assert recording_client.last_call == (
        "LLMBUDGET.CONSUME", "name", 0, "0",
        "MODEL", "claude-sonnet-5", "INPUT", 100, "OUTPUT", 20,
        "CACHEREAD", 5, "CACHEWRITE5M", 7, "CACHEWRITE1H", 9,
    )


# -- STATS parsing: RESP2 flat arrays and RESP3 maps, by field name ------------


def test_scache_stats_parses_11_field_flat_array(recording_client):
    flat = [
        "entries", 3, "hits", 7, "misses", 2, "hit_rate", "0.777778", "p50_us", 150,
        "usage_saved_requests", 7, "usage_saved_input_tokens", 700,
        "usage_saved_output_tokens", 140, "usage_saved_cache_read_tokens", 0,
        "usage_saved_cache_write_tokens", 0, "usage_saved_cost_usd", "0.000160",
    ]
    recording_client.when("SCACHE.STATS", flat)
    stats = recording_client.scache_stats("ns")
    assert stats["entries"] == 3
    assert stats["hit_rate"] == 0.777778
    assert stats["usage_saved_requests"] == 7
    assert stats["usage_saved_cost_usd"] == 0.00016


def test_pcache_stats_parses_16_field_resp3_map(recording_client):
    reply = {
        "entries": 1, "hits": 4, "misses": 1, "evictions": 0, "memory_bytes": 512,
        "hit_rate": "0.8", "p50_ns": 900, "p95_ns": 1200, "p99_ns": 1500,
        "cost_saved_usd": "0.004000",
        "usage_saved_requests": 4, "usage_saved_input_tokens": 400,
        "usage_saved_output_tokens": 80, "usage_saved_cache_read_tokens": 0,
        "usage_saved_cache_write_tokens": 0, "usage_saved_cost_usd": "0.000600",
    }
    recording_client.when("PCACHE.STATS", reply)
    stats = recording_client.pcache_stats("ns")
    assert stats["cost_saved_usd"] == 0.004
    assert stats["usage_saved_cost_usd"] == 0.0006
    assert stats["usage_saved_output_tokens"] == 80


def test_keyed_cache_stats_parses_11_field_flat_array(recording_client):
    flat = [
        "entries", 2, "hits", 1, "misses", 1, "evictions", 0, "memory_bytes", 64,
        "usage_saved_requests", 1, "usage_saved_input_tokens", 50,
        "usage_saved_output_tokens", 10, "usage_saved_cache_read_tokens", 0,
        "usage_saved_cache_write_tokens", 0, "usage_saved_cost_usd", "0.000010",
    ]
    recording_client.when("TOOL.CACHE.STATS", flat)
    stats = recording_client.toolcache_stats("ns")
    assert stats["entries"] == 2
    assert stats["usage_saved_cost_usd"] == 0.00001


def test_llmstats_get_parses_13_field_reply_including_unpriced(recording_client):
    flat = [
        "requests", 10, "errors", 1, "input_tokens", 1000, "output_tokens", 200,
        "cached_tokens", 50, "cost_usd", "0.005000", "avg_latency_ms", "120.500000",
        "requests_1m", 2, "tokens_1m", 300, "cost_usd_1m", "0.001000",
        "cache_write_5m_tokens", 7, "cache_write_1h_tokens", 9, "unpriced_requests", 1,
    ]
    recording_client.when("LLMSTATS.GET", flat)
    stats = recording_client.llmstats_get("bucket")
    assert stats["cost_usd"] == 0.005
    assert stats["cache_write_5m_tokens"] == 7
    assert stats["unpriced_requests"] == 1


def test_llmbudget_status_parses_9_field_reply(recording_client):
    flat = [
        "tokens_used", 100, "token_limit", 1000, "cost_used_usd", "0.050000",
        "cost_limit_usd", "1.000000", "window_seconds", 3600, "window_remaining", 3000,
        "requests_allowed", 5, "requests_rejected", 0, "unpriced_requests", 2,
    ]
    recording_client.when("LLMBUDGET.STATUS", flat)
    status = recording_client.llmbudget_status("name")
    assert status["cost_used_usd"] == 0.05
    assert status["unpriced_requests"] == 2


def test_stats_reply_tolerates_missing_and_unknown_fields(recording_client):
    recording_client.when("SCACHE.STATS", ["entries", 1, "totally_new_field", "x"])
    stats = recording_client.scache_stats("ns")
    assert stats["entries"] == 1
    assert "hit_rate" not in stats  # missing fields are simply absent, not KeyError
    assert stats["totally_new_field"] == "x"
