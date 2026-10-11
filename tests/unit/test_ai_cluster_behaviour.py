# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests, with a fake client and no server, of the AI helpers' behaviour
that matters on a cluster: a source document's nodes found without the
node-local `TAG.*` index, a namespace that was refused being reported, and a
cached prompt coming back as the text it was stored as."""

from __future__ import annotations

from typing import Any, Dict, List, Set

import pytest

from hotkv.ai import PromptCache
from hotkv.exceptions import ResponseError
from hotkv.integrations._exact_cache import ExactCacheBackend


class FakeClient:
    """Just enough of a client: sets, strings, tags and the calls it was given."""

    def __init__(self) -> None:
        self.sets: Dict[str, Set[str]] = {}
        self.strings: Dict[str, Any] = {}
        self.tags: Dict[str, List[str]] = {}
        self.vectors: Dict[str, Set[str]] = {}
        self.calls: List[tuple] = []
        self.create_error: Any = None
        self.stats_error: Any = None

    def sadd(self, key, *members):
        self.calls.append(("sadd", key, members))
        self.sets.setdefault(key, set()).update(members)
        return len(members)

    def smembers(self, key):
        return {m.encode() for m in self.sets.get(key, set())}

    def set(self, key, value):
        self.strings[key] = value

    def delete(self, *keys):
        self.calls.append(("delete", keys))
        for key in keys:
            self.strings.pop(key, None)
            self.sets.pop(key, None)

    def tag_members(self, tag):
        return [k.encode() for k in self.tags.get(tag, [])]

    def tag_set(self, key, tag):  # the old index: must not be written any more
        raise AssertionError("TAG.SET is node-local state; the store must not use it")

    def vsim_createindex(self, *args, **kwargs):
        return True

    def vsim_set(self, index, key, vector):
        self.vectors.setdefault(index, set()).add(key)

    def vsim_del(self, index, key):
        self.calls.append(("vsim_del", index, key))
        self.vectors.get(index, set()).discard(key)

    def pcache_create(self, namespace, **kwargs):
        if self.create_error is not None:
            raise self.create_error
        return True

    def pcache_stats(self, namespace):
        if self.stats_error is not None:
            raise self.stats_error
        return {}

    def pcache_get(self, namespace, key):
        return self.strings.get(("pc", namespace, key))

    def pcache_set(self, namespace, key, value, **kwargs):
        self.strings[("pc", namespace, key)] = value
        return True


def test_llama_index_vector_store_keeps_a_source_documents_nodes_in_a_set():
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    from hotkv.integrations.llama_index import HotKVVectorStore

    client = FakeClient()
    store = HotKVVectorStore(client, index_name="idx", dims=2)
    nodes = []
    for node_id, ref in (("a", "doc1"), ("b", "doc1"), ("c", "doc2")):
        node = TextNode(id_=node_id, text=node_id, embedding=[1.0, 0.0])
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref)
        nodes.append(node)
    store.add(nodes)
    assert client.sets["idx:ref:doc1"] == {"a", "b"}
    assert client.sets["idx:ref:doc2"] == {"c"}

    store.delete("doc1")
    assert client.vectors["idx"] == {"c"}
    assert "idx:node:a" not in client.strings and "idx:node:b" not in client.strings
    assert "idx:node:c" in client.strings
    assert "idx:ref:doc1" not in client.sets


def test_llama_index_vector_store_still_deletes_nodes_stored_with_a_tag():
    pytest.importorskip("llama_index.core")
    from hotkv.integrations.llama_index import HotKVVectorStore

    client = FakeClient()
    store = HotKVVectorStore(client, index_name="idx", dims=2)
    client.strings["idx:node:old"] = "{}"
    client.vectors["idx"] = {"old"}
    client.tags["hotkv-li-ref:docX"] = ["idx:node:old"]
    store.delete("docX")
    assert client.vectors["idx"] == set()
    assert "idx:node:old" not in client.strings


def test_a_refused_namespace_is_not_mistaken_for_an_existing_one():
    client = FakeClient()
    client.create_error = ResponseError("namespace already exists or limit reached")
    client.stats_error = ResponseError("namespace not found")
    backend = ExactCacheBackend(client, "ns")
    with pytest.raises(ResponseError, match="already exists or limit reached"):
        backend.get("k")


def test_an_existing_namespace_is_used():
    client = FakeClient()
    client.create_error = ResponseError("namespace already exists or limit reached")
    backend = ExactCacheBackend(client, "ns")
    backend.set("k", "v")
    assert backend.get("k") == "v"


def test_prompt_cache_returns_a_hit_as_the_text_the_call_returned():
    client = FakeClient()
    cache = PromptCache(client, "ns")
    messages = [{"role": "user", "content": "hi"}]
    first, hit1 = cache.get_or_call("p", "m", messages, lambda: "hello")
    client.strings[("pc", "ns", cache.build_key("p", "m", messages))] = b"hello"
    second, hit2 = cache.get_or_call("p", "m", messages, lambda: "never called")
    assert (first, hit1) == ("hello", False)
    assert (second, hit2) == ("hello", True)


def test_prompt_cache_keeps_bytes_that_are_not_text():
    client = FakeClient()
    cache = PromptCache(client, "ns")
    messages = [{"role": "user", "content": "hi"}]
    client.strings[("pc", "ns", cache.build_key("p", "m", messages))] = b"\xff\xfe"
    value, hit = cache.get_or_call("p", "m", messages, lambda: "never called")
    assert hit is True and value == b"\xff\xfe"
