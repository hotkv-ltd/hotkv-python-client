# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for `hotkv.ai`'s usage passthrough and `stats()` accessors.
Uses `RecordingClient`/`AsyncRecordingClient`
(see `tests/unit/conftest.py`) instead of a live server."""

from __future__ import annotations

from hotkv import TokenUsage
from hotkv.ai import AsyncPromptCache, AsyncSemanticCache, PromptCache, SemanticCache

USAGE = TokenUsage(model="claude-sonnet-5", input_tokens=42, output_tokens=8)


# -- PromptCache (sync) ---------------------------------------------------------


def test_prompt_cache_set_passes_usage_through(recording_client):
    cache = PromptCache(recording_client, "ns", create=False)
    cache.set("key", "response", usage=USAGE)
    assert recording_client.last_call == (
        "PCACHE.SET", "ns", "key", "response",
        "MODEL", "claude-sonnet-5", "INPUT", 42, "OUTPUT", 8,
    )


def test_prompt_cache_get_or_call_stores_usage_returned_by_fn(recording_client):
    cache = PromptCache(recording_client, "ns", create=False)

    def llm_call():
        return "the answer", USAGE

    response, hit = cache.get_or_call("openai", "gpt-6-sol", [{"role": "user", "content": "hi"}], llm_call)
    assert hit is False
    assert response == "the answer"  # unwrapped, not the (response, usage) pair
    set_call = recording_client.last_call
    assert set_call[0] == "PCACHE.SET"
    assert set_call[1] == "ns"
    assert set_call[3] == "the answer"
    assert set_call[4:] == ("MODEL", "claude-sonnet-5", "INPUT", 42, "OUTPUT", 8)


def test_prompt_cache_get_or_call_without_usage_is_unaffected(recording_client):
    cache = PromptCache(recording_client, "ns", create=False)

    def llm_call():
        return "plain response"

    response, hit = cache.get_or_call("openai", "gpt-6-sol", [{"role": "user", "content": "hi"}], llm_call)
    assert response == "plain response"
    assert "MODEL" not in recording_client.last_call


def test_prompt_cache_stats_delegates_to_pcache_stats(recording_client):
    recording_client.when("PCACHE.STATS", ["entries", 5, "usage_saved_cost_usd", "0.001200"])
    cache = PromptCache(recording_client, "ns", create=False)
    stats = cache.stats()
    assert recording_client.last_call == ("PCACHE.STATS", "ns")
    assert stats["entries"] == 5
    assert stats["usage_saved_cost_usd"] == 0.0012


# -- SemanticCache (sync) ---------------------------------------------------------


def test_semantic_cache_set_passes_usage_through(recording_client):
    cache = SemanticCache(recording_client, "ns", 2, create=False)
    cache.set("query", "response", [1.0, 0.0], usage=USAGE)
    assert recording_client.last_call == (
        "SCACHE.SET", "ns", "query", "response", "1", "0",
        "MODEL", "claude-sonnet-5", "INPUT", 42, "OUTPUT", 8,
    )


def test_semantic_cache_get_or_call_stores_usage_returned_by_fn(recording_client):
    cache = SemanticCache(recording_client, "ns", 2, create=False)

    def llm_call():
        return "semantic answer", USAGE

    response, hit = cache.get_or_call([1.0, 0.0], llm_call, query_text="q")
    assert hit is False
    assert response == "semantic answer"
    assert "MODEL" in recording_client.last_call
    assert "claude-sonnet-5" in recording_client.last_call


def test_semantic_cache_stats_delegates_to_scache_stats(recording_client):
    recording_client.when("SCACHE.STATS", ["entries", 2])
    cache = SemanticCache(recording_client, "ns", 2, create=False)
    stats = cache.stats()
    assert recording_client.last_call == ("SCACHE.STATS", "ns")
    assert stats["entries"] == 2


# -- Async equivalents ------------------------------------------------------------


async def test_async_prompt_cache_stores_usage_returned_by_fn(async_recording_client):
    cache = AsyncPromptCache(async_recording_client, "ns")

    def llm_call():
        return "async answer", USAGE

    response, hit = await cache.get_or_call(
        "openai", "gpt-6-sol", [{"role": "user", "content": "hi"}], llm_call
    )
    assert hit is False
    assert response == "async answer"
    assert "MODEL" in async_recording_client.last_call


async def test_async_semantic_cache_stats_delegates(async_recording_client):
    async_recording_client.when("SCACHE.STATS", ["entries", 9])
    cache = AsyncSemanticCache(async_recording_client, "ns", 2)
    stats = await cache.stats()
    assert stats["entries"] == 9
