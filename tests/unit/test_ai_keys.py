# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for `PromptCache`'s deterministic cache key construction. No
server needed."""

from __future__ import annotations

from hotkv.ai._keys import build_cache_key


def test_same_call_shape_yields_same_key():
    messages = [{"role": "user", "content": "hello"}]
    k1 = build_cache_key("openai", "gpt-6-sol", messages, temperature=0.2)
    k2 = build_cache_key("openai", "gpt-6-sol", messages, temperature=0.2)
    assert k1 == k2


def test_key_independent_of_kwarg_order():
    messages = [{"role": "user", "content": "hello"}]
    k1 = build_cache_key("openai", "gpt-6-sol", messages, temperature=0.2, max_tokens=100)
    k2 = build_cache_key("openai", "gpt-6-sol", messages, max_tokens=100, temperature=0.2)
    assert k1 == k2


def test_key_independent_of_message_dict_key_order():
    m1 = [{"role": "user", "content": "hello"}]
    m2 = [{"content": "hello", "role": "user"}]
    assert build_cache_key("openai", "gpt-6-sol", m1) == build_cache_key("openai", "gpt-6-sol", m2)


def test_different_provider_yields_different_key():
    messages = [{"role": "user", "content": "hello"}]
    k1 = build_cache_key("openai", "gpt-6-sol", messages)
    k2 = build_cache_key("anthropic", "claude-sonnet-5", messages)
    assert k1 != k2


def test_different_messages_yield_different_key():
    k1 = build_cache_key("openai", "gpt-6-sol", [{"role": "user", "content": "hello"}])
    k2 = build_cache_key("openai", "gpt-6-sol", [{"role": "user", "content": "goodbye"}])
    assert k1 != k2


def test_message_order_matters():
    a = {"role": "user", "content": "a"}
    b = {"role": "assistant", "content": "b"}
    k1 = build_cache_key("openai", "gpt-6-sol", [a, b])
    k2 = build_cache_key("openai", "gpt-6-sol", [b, a])
    assert k1 != k2


def test_system_prompt_is_part_of_the_key():
    messages = [{"role": "user", "content": "hello"}]
    k1 = build_cache_key("openai", "gpt-6-sol", messages, system="Be terse.")
    k2 = build_cache_key("openai", "gpt-6-sol", messages, system="Be verbose.")
    k3 = build_cache_key("openai", "gpt-6-sol", messages)
    assert len({k1, k2, k3}) == 3


def test_key_is_hex_sha256():
    key = build_cache_key("openai", "gpt-6-sol", [{"role": "user", "content": "hi"}])
    assert len(key) == 64
    int(key, 16)  # raises ValueError if not valid hex
