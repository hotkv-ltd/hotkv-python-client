# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for RATELIMIT.*, TS.* and FEATURE.*.

On a server without a license every family here replies `NOLICENSE`; each
test wraps its body in `skip_if_not_licensed` and is skipped with a clear
reason instead of failing, so the whole file passes on a licensed server.
"""

from __future__ import annotations


def test_ratelimit_lifecycle(client, prefix, skip_if_not_licensed):
    name = prefix + "rl"

    def body():
        assert client.ratelimit_create(name, 2, 60_000, algo="sliding_window") is True
        result1 = client.ratelimit_check(name, "user1")
        assert result1.allowed is True
        assert result1.limit == 2
        result2 = client.ratelimit_check(name, "user1")
        result3 = client.ratelimit_check(name, "user1")
        assert result3.allowed is False
        peek = client.ratelimit_peek(name, "user1")
        assert peek.allowed is False  # non-mutating, still over limit
        stats = client.ratelimit_stats(name)
        assert stats["name"] == name
        assert stats["algo"] == "sliding_window"
        assert name.encode() in client.ratelimit_list()
        assert client.ratelimit_reset(name, "user1") is True
        assert client.ratelimit_drop(name) is True

    skip_if_not_licensed(body)


def test_ts_lifecycle(client, prefix, skip_if_not_licensed):
    import time

    key = prefix + "ts"

    def body():
        # RETENTION filters out samples older than the window relative to
        # real wall clock time, so use realistic "now"-based timestamps
        # rather than small ones.
        now_ms = int(time.time() * 1000)
        assert client.ts_create(key, retention_ms=3_600_000, labels={"region": "eu"}) is True
        ts1 = client.ts_add(key, 1.0, timestamp=now_ms)
        assert ts1 == now_ms
        client.ts_add(key, 2.0, timestamp=now_ms + 1000)
        client.ts_add(key, 3.0, timestamp=now_ms + 2000)
        samples = client.ts_range(key, now_ms - 5000, now_ms + 5000)
        assert samples == [(now_ms, 1.0), (now_ms + 1000, 2.0), (now_ms + 2000, 3.0)]
        agg = client.ts_agg(key, now_ms - 5000, now_ms + 5000, 5000, "avg")
        assert len(agg) >= 1
        info = client.ts_info(key)
        assert info["totalSamples"] == 3
        assert info["retentionMs"] == 3_600_000
        assert ("region", b"eu") in info["labels"] or (b"region", b"eu") in info["labels"]
        assert client.ts_retention(key, 7_200_000) is True
        dest = prefix + "ts_dest"
        client.ts_create(dest)
        assert client.ts_downsample(key, dest, "avg", 1000) is True
        assert key.encode() in client.ts_list()
        assert key.encode() in client.ts_list(label=("region", "eu"))
        assert key.encode() not in client.ts_list(label=("region", "us"))
        assert client.ts_delete(key) is True
        client.ts_delete(dest)

    skip_if_not_licensed(body)


def test_feature_store_lifecycle(client, prefix, skip_if_not_licensed):
    group = prefix + "features"

    def body():
        assert client.feature_create(
            group, "user features", {"age": "int64", "score": "double"}
        ) is True
        assert client.feature_set(group, "user1", {"age": 30, "score": 4.5}) is True
        result = client.feature_get(group, "user1")
        assert result is not None
        values, updated_at_ms, freshness_ms = result
        # int64-typed features come back as a genuine RESP integer, not text.
        assert values[b"age"] == 30
        batch = client.feature_batch(group, ["user1", "missing-user"])
        assert batch[0] is not None
        assert batch[1] is None
        schema = client.feature_schema(group)
        assert ("age", "int64") in schema
        assert group.encode() in client.feature_list()
        stats = client.feature_stats(group)
        assert stats["entities"] == 1
        assert client.feature_drop(group) is True

    skip_if_not_licensed(body)
