# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for AMEM.* (agent memory), AMEM.SUMMARY.*,
EMBED.DEDUP.* and RAG.*.

On a server without a license every family here replies `NOLICENSE`; each
test wraps its body in `skip_if_not_licensed` and is skipped with a clear
reason instead of failing, so the whole file passes on a licensed server.
"""

from __future__ import annotations


def test_amem_conv(client, prefix, skip_if_not_licensed):
    session = prefix + "sess"

    def body():
        assert client.amem_conv_create(session, window=10) is True
        assert client.amem_conv_append(session, "user", "hello") is True
        assert client.amem_conv_append(session, "assistant", "hi there") is True
        turns = client.amem_conv_get(session)
        assert turns == [
            {"role": b"user", "content": b"hello"},
            {"role": b"assistant", "content": b"hi there"},
        ]
        window = client.amem_conv_window(session, 1)
        assert window == [{"role": b"assistant", "content": b"hi there"}]
        assert client.amem_conv_clear(session) is True
        assert client.amem_conv_get(session) == []
        assert client.amem_conv_drop(session) is True

    skip_if_not_licensed(body)


def test_amem_sem(client, prefix, skip_if_not_licensed):
    agent = prefix + "agent"

    def body():
        assert client.amem_sem_create(agent, 3) is True
        assert client.amem_sem_store(agent, "fact1", "the sky is blue", [1.0, 0.0, 0.0]) is True
        results = client.amem_sem_recall(agent, 1, [1.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0]["key"] == b"fact1"
        assert results[0]["content"] == b"the sky is blue"
        assert isinstance(results[0]["similarity"], float)
        assert client.amem_sem_list(agent) == [b"fact1"]
        assert client.amem_sem_forget(agent, "fact1") is True

    skip_if_not_licensed(body)


def test_amem_work(client, prefix, skip_if_not_licensed):
    session = prefix + "work"

    def body():
        assert client.amem_work_set(session, "scratch", "value1") is True
        assert client.amem_work_get(session, "scratch") == b"value1"
        assert client.amem_work_list(session) == [b"scratch"]
        assert client.amem_work_del(session, "scratch") == 1
        assert client.amem_work_clear(session) in (0, 1)

    skip_if_not_licensed(body)


def test_amem_epis(client, prefix, skip_if_not_licensed):
    agent = prefix + "epis"

    def body():
        assert client.amem_epis_log(agent, "tool_call", "did something") is True
        events = client.amem_epis_query(agent)
        assert len(events) == 1
        assert events[0]["type"] == b"tool_call"
        assert isinstance(events[0]["timestamp"], int)
        stats = client.amem_epis_stats(agent)
        assert stats["total_events"] == 1
        assert client.amem_epis_clear(agent) is True

    skip_if_not_licensed(body)


def test_amem_search(client, prefix, skip_if_not_licensed):
    agent = prefix + "cross"

    def body():
        client.amem_sem_create(agent, 2)
        client.amem_sem_store(agent, "k1", "semantic fact", [1.0, 0.0])
        results = client.amem_search(agent, 3, [1.0, 0.0])
        assert isinstance(results, list)

    skip_if_not_licensed(body)


def test_amem_summary(client, prefix, skip_if_not_licensed):
    agent = prefix + "summary"

    def body():
        v1 = client.amem_summary_set(agent, "Initial summary.")
        assert v1 == 1
        v2 = client.amem_summary_append(agent, " More context.")
        assert v2 == 2
        result = client.amem_summary_get(agent)
        summary, version, updated_unix = result
        assert version == 2
        assert client.amem_summary_list()
        assert client.amem_summary_drop(agent) is True

    skip_if_not_licensed(body)


def test_embed_dedup(client, prefix, skip_if_not_licensed):
    ns = prefix + "dedup"

    def body():
        assert client.embed_dedup_put(ns, "hash1", "model-a", "0.1,0.2,0.3") == 1
        assert client.embed_dedup_put(ns, "hash1", "model-a", "0.1,0.2,0.3") == 0
        assert client.embed_dedup_get(ns, "hash1") is not None
        stats = client.embed_dedup_stats(ns)
        assert stats["entries"] == 1
        assert client.embed_dedup_release(ns, "hash1") == 1
        assert client.embed_dedup_purge(ns) in (0, 1)
        assert client.embed_dedup_drop(ns) is True

    skip_if_not_licensed(body)


def test_rag_pipeline(client, prefix, skip_if_not_licensed):
    name = prefix + "rag"

    def body():
        assert client.rag_create(name, 3, chunk_size=256, chunk_overlap=32) is True
        chunks = client.rag_ingest(
            name, "doc1", "hotkv is fast", meta={"source": "readme"}, vectors=[[1.0, 0.0, 0.0]]
        )
        assert chunks >= 1
        results = client.rag_search(name, "hotkv", k=3, vector=[1.0, 0.0, 0.0])
        assert isinstance(results, list)
        if results:
            assert "chunk_id" in results[0]
            assert "score" in results[0]
        stats = client.rag_stats(name)
        assert stats["name"] == name
        assert stats["dimensions"] == 3
        assert name.encode() in client.rag_list()
        assert client.rag_drop(name) is True

    skip_if_not_licensed(body)
