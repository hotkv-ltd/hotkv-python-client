# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for `hotkv.integrations.llama_index`. Skips the whole
module if `llama-index-core` is not installed (it is an optional extra)."""

from __future__ import annotations

import pytest

llama_index_core = pytest.importorskip("llama_index.core")

from llama_index.core.llms import ChatMessage, MessageRole  # noqa: E402
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode  # noqa: E402
from llama_index.core.vector_stores.types import (  # noqa: E402
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from hotkv.integrations.llama_index import HotKVChatStore, HotKVKVStore, HotKVVectorStore  # noqa: E402


def _node(node_id: str, text: str, embedding, topic: str, ref_doc_id: str) -> TextNode:
    node = TextNode(id_=node_id, text=text, metadata={"topic": topic}, embedding=embedding)
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref_doc_id)
    return node


def test_vector_store_add_query_score_order(client, prefix):
    store = HotKVVectorStore(client, index_name=prefix + "vs", dims=3)
    n1 = _node("n1", "HotKV is a fast in-memory data store", [1.0, 0.0, 0.0], "db", "docA")
    n2 = _node("n2", "Cats are small mammals", [0.0, 1.0, 0.0], "animal", "docB")
    n3 = _node("n3", "HotKV also supports vector search", [0.9, 0.1, 0.0], "db", "docA")

    ids = store.add([n1, n2, n3])
    assert set(ids) == {"n1", "n2", "n3"}

    result = store.query(VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=2))
    assert [n.node_id for n in result.nodes] == ["n1", "n3"]  # nearest first
    assert result.similarities[0] <= result.similarities[1]
    assert result.nodes[0].text == "HotKV is a fast in-memory data store"


def test_vector_store_metadata_filter(client, prefix):
    store = HotKVVectorStore(client, index_name=prefix + "vsfilter", dims=3)
    store.add(
        [
            _node("f1", "HotKV doc", [1.0, 0.0, 0.0], "db", "docA"),
            _node("f2", "animal doc", [0.9, 0.1, 0.0], "animal", "docB"),
        ]
    )
    filters = MetadataFilters(filters=[MetadataFilter(key="topic", value="animal", operator=FilterOperator.EQ)])
    result = store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1, filters=filters)
    )
    assert [n.node_id for n in result.nodes] == ["f2"]


def test_vector_store_delete_by_ref_doc_id(client, prefix):
    store = HotKVVectorStore(client, index_name=prefix + "vsdel", dims=3)
    store.add(
        [
            _node("d1", "doc A part 1", [1.0, 0.0, 0.0], "db", "docA"),
            _node("d2", "doc A part 2", [0.95, 0.05, 0.0], "db", "docA"),
            _node("d3", "doc B", [0.0, 1.0, 0.0], "other", "docB"),
        ]
    )
    store.delete("docA")
    result = store.query(VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=5))
    assert [n.node_id for n in result.nodes] == ["d3"]


def test_kv_store_put_get_all_delete(client, prefix):
    kv = HotKVKVStore(client, key_prefix=prefix + "kv")
    kv.put("doc1", {"a": 1, "b": "x"}, collection="docstore/data")
    kv.put("doc2", {"a": 2}, collection="docstore/data")

    assert kv.get("doc1", collection="docstore/data") == {"a": 1, "b": "x"}
    assert kv.get("missing", collection="docstore/data") is None

    all_items = kv.get_all(collection="docstore/data")
    assert all_items == {"doc1": {"a": 1, "b": "x"}, "doc2": {"a": 2}}

    # A different collection is a separate namespace.
    assert kv.get_all(collection="index_store/data") == {}

    assert kv.delete("doc1", collection="docstore/data") is True
    assert kv.delete("doc1", collection="docstore/data") is False
    assert kv.get("doc1", collection="docstore/data") is None


async def test_kv_store_async(client, prefix):
    kv = HotKVKVStore(client, key_prefix=prefix + "kvasync")
    await kv.aput("k", {"v": 1})
    assert await kv.aget("k") == {"v": 1}
    assert await kv.aget_all() == {"k": {"v": 1}}
    assert await kv.adelete("k") is True


def test_chat_store_round_trip(client, prefix):
    store = HotKVChatStore(client, hash_key=prefix + "chat")
    assert store.get_keys() == []

    store.add_message("sess1", ChatMessage(role=MessageRole.USER, content="hi"))
    store.add_message("sess1", ChatMessage(role=MessageRole.ASSISTANT, content="hello there"))

    messages = store.get_messages("sess1")
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "hi"
    assert messages[1].content == "hello there"

    assert store.get_keys() == ["sess1"]

    last = store.delete_last_message("sess1")
    assert last is not None and last.content == "hello there"
    assert len(store.get_messages("sess1")) == 1

    deleted_all = store.delete_messages("sess1")
    assert deleted_all is not None and len(deleted_all) == 1
    assert store.get_messages("sess1") == []
