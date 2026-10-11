# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Asyncio client integration tests, covering a representative slice of the
same ground as the sync tests: basic commands, hashes, sets, HotKV
exclusive commands, pipelines/transactions, pub/sub, SCAN iteration and the
`hotkv.ai` helpers, all through `hotkv.asyncio.HotKV`.
"""

from __future__ import annotations

import pytest

import hotkv
import hotkv.asyncio
from hotkv.ai import AsyncSemanticCache


async def test_strings_and_hashes(aclient, prefix):
    key = prefix + "astr"
    assert await aclient.set(key, "hello") is True
    assert await aclient.get(key) == b"hello"
    assert await aclient.incr(prefix + "acounter") == 1

    hkey = prefix + "ahash"
    assert await aclient.hset(hkey, {"a": "1", "b": "2"}) == 2
    assert await aclient.hgetall(hkey) == {b"a": b"1", b"b": b"2"}
    await aclient.delete(key, prefix + "acounter", hkey)


async def test_sets_and_zsets(aclient, prefix):
    skey = prefix + "aset"
    assert await aclient.sadd(skey, "x", "y") == 2
    assert await aclient.smembers(skey) == {b"x", b"y"}

    zkey = prefix + "azset"
    await aclient.zadd(zkey, {"m1": 1, "m2": 2})
    assert await aclient.zrange(zkey, 0, -1, withscores=True) == [(b"m1", 1.0), (b"m2", 2.0)]
    await aclient.delete(skey, zkey)


async def test_hotkv_exclusive_commands(aclient, prefix):
    key = prefix + "atag"
    await aclient.set(key, "v")
    assert await aclient.tag_set(key, "one", "two") == 2
    assert set(await aclient.tag_get(key)) == {b"one", b"two"}
    assert await aclient.nttl(key) == -1
    await aclient.delete(key)

    gkey = prefix + "agcra"
    result = await aclient.gcra_limit(gkey, 60_000, 5)
    assert result.allowed is True


async def test_pipeline_batches(aclient, prefix):
    k1, k2 = prefix + "ap1", prefix + "ap2"
    async with aclient.pipeline(transaction=False) as pipe:
        await pipe.set(k1, "1")
        await pipe.incr(k1)
        await pipe.set(k2, "x")
        results = await pipe.execute()
    assert results == [True, 2, True]
    await aclient.delete(k1, k2)


async def test_transaction_with_watch(aclient, prefix):
    key = prefix + "abalance"
    await aclient.set(key, 50)

    async def txn(pipe):
        current = await pipe.get(key)
        pipe.multi()
        await pipe.set(key, int(current) + 25)

    await aclient.transaction(txn, key)
    assert await aclient.get(key) == b"75"
    await aclient.delete(key)


async def test_pubsub(aclient, test_url, prefix):
    channel = prefix + "achan"
    pubsub = aclient.pubsub()
    await pubsub.subscribe(channel)
    publisher = hotkv.asyncio.from_url(test_url)
    try:
        await publisher.execute("PUBLISH", channel, "async-hello")
        message = None
        for _ in range(50):
            message = await pubsub.get_message(timeout=0.2, ignore_subscribe_messages=True)
            if message is not None:
                break
        assert message is not None
        assert message.data == b"async-hello"
    finally:
        await pubsub.close()
        await publisher.close()


async def test_scan_iter(aclient, prefix):
    keys = {prefix + f"ascan:{i}" for i in range(4)}
    for k in keys:
        await aclient.set(k, "1")
    found = set()
    async for k in aclient.scan_iter(match=prefix + "ascan:*"):
        found.add(k)
    assert found == {k.encode() for k in keys}
    await aclient.delete(*keys)


async def test_semantic_cache_async(aclient, prefix, async_skip_if_not_licensed):
    calls = []

    async def llm_call():
        calls.append(1)
        return "async answer"

    async def body():
        # Constructing `AsyncSemanticCache` does nothing on the wire (it
        # creates its namespace lazily on first use), so the first real
        # call is what raises `NotLicensedError` on an unlicensed server.
        cache = AsyncSemanticCache(aclient, prefix + "asemcache", 3, threshold=0.8)
        response1, hit1 = await cache.get_or_call([1.0, 0.0, 0.0], llm_call, query_text="q")
        assert hit1 is False
        assert response1 == "async answer"

        response2, hit2 = await cache.get_or_call([0.99, 0.01, 0.0], llm_call)
        assert hit2 is True
        assert len(calls) == 1

        await aclient.scache_drop(prefix + "asemcache")

    await async_skip_if_not_licensed(body())


async def test_eval_auto_async(aclient):
    result = await aclient.eval_auto("return 99")
    assert result == 99
