# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for the `hotkv.ai` helpers: `PromptCache` and
`SemanticCache`. On a server without a license every test here is wrapped in
`skip_if_not_licensed` and skipped."""

from __future__ import annotations

from hotkv import TokenUsage
from hotkv.ai import PromptCache, SemanticCache


def test_semantic_cache_get_or_call(client, prefix, skip_if_not_licensed):
    calls = []

    def llm_call():
        calls.append(1)
        return "the answer is 42"

    def body():
        # Constructing `SemanticCache` itself issues `SCACHE.CREATE`, so it
        # must run inside `body()` to be caught and skipped like every
        # other call to an unlicensed feature.
        cache = SemanticCache(client, prefix + "semcache", 3, threshold=0.8)
        response1, hit1 = cache.get_or_call(
            [1.0, 0.0, 0.0], llm_call, query_text="what is the answer"
        )
        assert hit1 is False
        assert response1 == "the answer is 42"
        assert len(calls) == 1

        # A near-identical embedding should hit the cache and skip the call.
        response2, hit2 = cache.get_or_call([0.99, 0.01, 0.0], llm_call)
        assert hit2 is True
        assert response2 == b"the answer is 42"
        assert len(calls) == 1  # the LLM was not called again

        client.scache_drop(prefix + "semcache")

    skip_if_not_licensed(body)


def test_semantic_cache_direct_get_set(client, prefix, skip_if_not_licensed):
    ns = prefix + "semdirect"

    def body():
        cache = SemanticCache(client, ns, 2, threshold=0.9)
        cache.set("hello", "world", [1.0, 0.0])
        hit = cache.get([1.0, 0.0])
        assert hit is not None
        response, similarity = hit
        assert response == b"world"
        assert similarity >= 0.9
        client.scache_drop(ns)

    skip_if_not_licensed(body)


def test_semantic_cache_get_or_call_with_usage_reports_savings(client, prefix, skip_if_not_licensed):
    ns = prefix + "semusage"

    def llm_call():
        return "cached answer", TokenUsage(model="claude-sonnet-5", input_tokens=100, output_tokens=20)

    def body():
        cache = SemanticCache(client, ns, 2, threshold=0.8)
        cache.get_or_call([1.0, 0.0], llm_call, query_text="q")
        cache.get_or_call([1.0, 0.0], llm_call, query_text="q")  # a hit
        stats = cache.stats()
        assert stats["usage_saved_requests"] >= 1
        client.scache_drop(ns)

    skip_if_not_licensed(body)


def test_prompt_cache_get_or_call(client, prefix, skip_if_not_licensed):
    def body():
        cache = PromptCache(client, prefix + "promptcache")
        calls = []

        def llm_call():
            calls.append(1)
            return "hello there"

        messages = [{"role": "user", "content": "hi"}]
        response1, hit1 = cache.get_or_call("openai", "gpt-6-sol", messages, llm_call)
        assert hit1 is False
        assert len(calls) == 1

        response2, hit2 = cache.get_or_call("openai", "gpt-6-sol", messages, llm_call)
        assert hit2 is True
        assert len(calls) == 1
        assert response2 == response1

        client.pcache_drop(prefix + "promptcache")

    skip_if_not_licensed(body)


def test_prompt_cache_get_or_call_with_usage_reports_savings(client, prefix, skip_if_not_licensed):
    ns = prefix + "promptusage"

    def llm_call():
        return "hello there", TokenUsage(model="claude-sonnet-5", input_tokens=100, output_tokens=20)

    def body():
        cache = PromptCache(client, ns)
        messages = [{"role": "user", "content": "hi"}]
        cache.get_or_call("openai", "gpt-6-sol", messages, llm_call)
        cache.get_or_call("openai", "gpt-6-sol", messages, llm_call)  # a hit
        stats = cache.stats()
        assert stats["usage_saved_requests"] >= 1
        assert stats["usage_saved_input_tokens"] >= 100
        client.pcache_drop(ns)

    skip_if_not_licensed(body)


def test_prompt_cache_build_key_is_deterministic(client, prefix, skip_if_not_licensed):
    def body():
        cache = PromptCache(client, prefix + "keycache")
        messages = [{"role": "user", "content": "hi"}]
        k1 = cache.build_key("openai", "gpt-6-sol", messages, temperature=0.2)
        k2 = cache.build_key("openai", "gpt-6-sol", messages, temperature=0.2)
        assert k1 == k2
        client.pcache_drop(prefix + "keycache")

    skip_if_not_licensed(body)
