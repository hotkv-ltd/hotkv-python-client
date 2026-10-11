# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for `hotkv.integrations.litellm`. Skips the whole
module if `litellm` is not installed (it is an optional extra)."""

from __future__ import annotations

import pytest

litellm = pytest.importorskip("litellm")

from hotkv.integrations.litellm import HotKVCache, HotKVSemanticCache  # noqa: E402


def test_hotkv_cache_get_set_and_ttl(client, prefix):
    cache = HotKVCache(client, namespace=prefix + "exact", default_ttl=30)
    assert cache.get_cache("missing-key") is None

    cache.set_cache("k1", {"choices": [{"text": "hello"}]})
    assert cache.get_cache("k1") == {"choices": [{"text": "hello"}]}

    cache.set_cache("k2", {"choices": [{"text": "custom-ttl"}]}, ttl=5)
    assert cache.get_cache("k2") == {"choices": [{"text": "custom-ttl"}]}


async def test_hotkv_cache_async_and_pipeline(client, prefix):
    cache = HotKVCache(client, namespace=prefix + "async")
    assert await cache.async_get_cache("missing") is None

    await cache.async_set_cache("a1", {"v": 1})
    assert await cache.async_get_cache("a1") == {"v": 1}

    await cache.async_set_cache_pipeline([("a2", {"v": 2}), ("a3", {"v": 3})])
    assert await cache.async_get_cache("a2") == {"v": 2}
    assert await cache.async_get_cache("a3") == {"v": 3}


def test_hotkv_cache_via_completion_with_mock_response(client, prefix):
    """End to end through `litellm.completion()`: the first call runs the
    (mocked) model, an identical second call is served from HotKV."""
    import litellm as litellm_module

    cache_wrapper = litellm_module.Cache()
    cache_wrapper.cache = HotKVCache(client, namespace=prefix + "completion")
    litellm_module.cache = cache_wrapper
    try:
        messages = [{"role": "user", "content": "say hi"}]
        first = litellm_module.completion(
            model="gpt-6-sol",
            messages=messages,
            mock_response="first-response",
            caching=True,
        )
        second = litellm_module.completion(
            model="gpt-6-sol",
            messages=messages,
            mock_response="a-different-mock-that-should-not-be-used",
            caching=True,
        )
        assert first.choices[0].message.content == "first-response"
        assert second.choices[0].message.content == "first-response"
    finally:
        litellm_module.cache = None


def _fake_embed_factory():
    vectors = {
        "capital of France?": [1.0, 0.0, 0.0],
        "what's the capital of France": [0.99, 0.01, 0.0],
        "capital of Germany?": [0.0, 1.0, 0.0],
    }

    def embed(text: str):
        return vectors.get(text, [0.0, 0.0, 1.0])

    return embed


def test_hotkv_semantic_cache_similarity_and_isolation(client, prefix, skip_if_not_licensed):
    def body():
        # `HotKVSemanticCache.get_cache()`/`set_cache()` treat `NOLICENSE`
        # as "always a miss" rather than raising, so a licensing probe
        # comes first: on an unlicensed server this raises
        # `NotLicensedError`, which `skip_if_not_licensed()` turns into a
        # skip instead of the assertions below failing.
        probe_ns = prefix + "sem-license-probe"
        client.scache_create(probe_ns, 1)
        client.scache_drop(probe_ns)

        embed = _fake_embed_factory()
        cache = HotKVSemanticCache(
            client,
            embedding_model="fake-model",
            namespace=prefix + "sem",
            threshold=0.9,
            embed_fn=embed,
        )

        cache.set_cache(
            "ignored-key",
            {"response": "Paris"},
            messages=[{"role": "user", "content": "capital of France?"}],
        )

        hit = cache.get_cache(
            "ignored-key-2", messages=[{"role": "user", "content": "what's the capital of France"}]
        )
        assert hit == {"response": "Paris"}

        miss = cache.get_cache(
            "ignored-key-3", messages=[{"role": "user", "content": "capital of Germany?"}]
        )
        assert miss is None

        # A cache built for a different embedding model is a separate namespace.
        other = HotKVSemanticCache(
            client,
            embedding_model="a-different-fake-model",
            namespace=prefix + "sem",
            threshold=0.9,
            embed_fn=embed,
        )
        assert other.get_cache(
            "k", messages=[{"role": "user", "content": "what's the capital of France"}]
        ) is None

    skip_if_not_licensed(body)


async def test_hotkv_semantic_cache_async(client, prefix, async_skip_if_not_licensed):
    vectors = {"hello there": [1.0, 0.0, 0.0]}

    async def aembed(text: str):
        return vectors.get(text, [0.0, 1.0, 0.0])

    async def body():
        # See `test_hotkv_semantic_cache_similarity_and_isolation()`: a
        # licensing probe first, since the cache itself swallows
        # `NotLicensedError` as a permanent miss instead of raising.
        probe_ns = prefix + "semasync-license-probe"
        client.scache_create(probe_ns, 1)
        client.scache_drop(probe_ns)

        cache = HotKVSemanticCache(
            client,
            embedding_model="fake-async-model",
            namespace=prefix + "semasync",
            threshold=0.9,
            embed_fn=lambda t: vectors.get(t, [0.0, 1.0, 0.0]),
            aembed_fn=aembed,
        )
        await cache.async_set_cache(
            "ignored", {"response": "hi"}, messages=[{"role": "user", "content": "hello there"}]
        )
        hit = await cache.async_get_cache(
            "ignored2", messages=[{"role": "user", "content": "hello there"}]
        )
        assert hit == {"response": "hi"}

    await async_skip_if_not_licensed(body())
