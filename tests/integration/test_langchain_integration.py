# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for `hotkv.integrations.langchain`. Skips the whole
module if `langchain-core` is not installed (it is an optional extra)."""

from __future__ import annotations

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.chat_history import BaseChatMessageHistory  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models import FakeListLLM  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from hotkv.integrations.langchain import (  # noqa: E402
    HotKVCache,
    HotKVChatMessageHistory,
    HotKVSemanticCache,
    HotKVVectorStore,
)


class _ControlledEmbeddings(Embeddings):
    """A deterministic fake embedding model: exact, caller supplied vectors
    for a fixed set of known texts, so semantic similarity tests are
    entirely predictable (no real embedding model, no network, no keys)."""

    def __init__(self, vectors: dict) -> None:
        self._vectors = vectors

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return self._vectors.get(text, [0.0, 0.0, 1.0])


def test_hotkv_cache_exact_match_and_model_isolation(client, prefix):
    from langchain_core.outputs import Generation

    cache = HotKVCache(client, namespace=prefix + "exact")

    # Nothing cached yet: any (prompt, llm_string) pair is a miss.
    assert cache.lookup("hello", "model=a;temperature=0") is None

    cache.update("hello", "model=a;temperature=0", [Generation(text="first")])
    hit = cache.lookup("hello", "model=a;temperature=0")
    assert hit is not None and hit[0].text == "first"

    # A different llm_string (a different model, or the same model with
    # different generation params) never sees this entry, even for the
    # exact same prompt text.
    assert cache.lookup("hello", "model=b;temperature=0") is None
    assert cache.lookup("hello", "model=a;temperature=0.7") is None


def test_hotkv_cache_via_set_llm_cache(client, prefix):
    from langchain_core.globals import set_llm_cache

    cache = HotKVCache(client, namespace=prefix + "global")
    set_llm_cache(cache)
    try:
        llm = FakeListLLM(responses=["first-answer", "second-answer"])
        first = llm.invoke("what is HotKV?")
        second = llm.invoke("what is HotKV?")  # identical call: served from HotKV, not the model
        assert first == "first-answer"
        assert second == "first-answer"

        third = llm.invoke("a different question")  # miss: the model runs again
        assert third == "second-answer"
    finally:
        set_llm_cache(None)


async def test_hotkv_cache_async_methods(client, prefix):
    from langchain_core.outputs import Generation

    cache = HotKVCache(client, namespace=prefix + "async")
    assert await cache.alookup("p", "llm-a") is None
    await cache.aupdate("p", "llm-a", [Generation(text="async-value")])
    hit = await cache.alookup("p", "llm-a")
    assert hit is not None and hit[0].text == "async-value"
    await cache.aclear()
    assert await cache.alookup("p", "llm-a") is None


def test_hotkv_semantic_cache_similarity_and_model_isolation(client, prefix, skip_if_not_licensed):
    def body():
        # `HotKVSemanticCache.lookup()`/`update()` treat `NOLICENSE` as
        # "always a miss" rather than raising (see its docstring), so a
        # licensing probe comes first: on an unlicensed server this raises
        # `NotLicensedError`, which `skip_if_not_licensed()` turns into a
        # skip instead of the assertions below failing.
        probe_ns = prefix + "sem-license-probe"
        client.scache_create(probe_ns, 1)
        client.scache_drop(probe_ns)

        vectors = {
            "capital of France?": [1.0, 0.0, 0.0],
            "what's the capital of France": [0.99, 0.01, 0.0],  # near-duplicate
            "capital of Germany?": [0.0, 1.0, 0.0],  # unrelated
        }
        embedding = _ControlledEmbeddings(vectors)
        cache = HotKVSemanticCache(client, embedding, namespace=prefix + "sem", threshold=0.9)

        from langchain_core.outputs import Generation

        cache.update("capital of France?", "llm-a", [Generation(text="Paris")])

        similar_hit = cache.lookup("what's the capital of France", "llm-a")
        assert similar_hit is not None and similar_hit[0].text == "Paris"

        different_miss = cache.lookup("capital of Germany?", "llm-a")
        assert different_miss is None

        # Same near-duplicate prompt, but a different llm_string: no cross-model leakage.
        isolated_miss = cache.lookup("what's the capital of France", "llm-b")
        assert isolated_miss is None

    skip_if_not_licensed(body)


def test_hotkv_vector_store_add_search_score_order_and_delete(client, prefix):
    class _DocEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [self.embed_query(t) for t in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0] if "hotkv" in text.lower() else [0.0, 1.0, 0.0]

    store = HotKVVectorStore(client, _DocEmbeddings(), index_name=prefix + "vs")
    ids = store.add_texts(
        ["HotKV is a fast in-memory data store", "Cats are small mammals"],
        metadatas=[{"topic": "db"}, {"topic": "animal"}],
    )
    assert len(ids) == 2

    results = store.similarity_search("tell me about hotkv", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert "HotKV" in results[0].page_content
    assert results[0].metadata["topic"] == "db"

    scored = store.similarity_search_with_score("tell me about hotkv", k=2)
    assert [doc.metadata["topic"] for doc, _score in scored] == ["db", "animal"]
    assert scored[0][1] <= scored[1][1]  # nearest first

    assert store.delete(ids=[ids[0]]) is True
    remaining = store.similarity_search("tell me about hotkv", k=2)
    assert all(doc.metadata["topic"] != "db" for doc in remaining)


def test_hotkv_vector_store_from_texts_requires_client(prefix):
    class _E(Embeddings):
        def embed_documents(self, texts):
            return [[0.0] for _ in texts]

        def embed_query(self, text):
            return [0.0]

    with pytest.raises(ValueError):
        HotKVVectorStore.from_texts(["a"], _E())


def test_hotkv_chat_message_history(client, prefix):
    history = HotKVChatMessageHistory(client, session_id=prefix + "session", ttl=60)
    assert isinstance(history, BaseChatMessageHistory)
    assert history.messages == []

    history.add_user_message("hello")
    history.add_ai_message("hi, how can I help?")

    messages = history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage) and messages[0].content == "hello"
    assert isinstance(messages[1], AIMessage) and messages[1].content == "hi, how can I help?"

    history.clear()
    assert history.messages == []
