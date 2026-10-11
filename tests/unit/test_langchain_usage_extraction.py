# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for `hotkv.integrations.langchain._generations.extract_usage()`:
the best-effort `TokenUsage` extraction `HotKVCache`/`HotKVSemanticCache`
use so a cache entry can attribute a real model call's savings. No server
needed. `langchain-core` is a `dev` extra on Python 3.10 and later (it does not
install on 3.9, where this module skips)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, Generation  # noqa: E402

from hotkv.integrations.langchain._generations import extract_usage


def _chat_generation(*, model=None, usage_metadata=None):
    message = AIMessage(
        content="hi",
        response_metadata={"model_name": model} if model else {},
        usage_metadata=usage_metadata,
    )
    return ChatGeneration(message=message)


def test_extracts_model_and_token_counts_when_both_present():
    generation = _chat_generation(
        model="claude-sonnet-5",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_token_details": {"cache_read": 5},
        },
    )
    usage = extract_usage([generation])
    assert usage is not None
    assert usage.model == "claude-sonnet-5"
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cache_read_tokens == 5
    # No per-tier breakdown exists in LangChain's schema; never guessed.
    assert usage.cache_write_5m_tokens == 0
    assert usage.cache_write_1h_tokens == 0


def test_falls_back_to_plain_model_metadata_key():
    generation = ChatGeneration(
        message=AIMessage(
            content="hi",
            response_metadata={"model": "gpt-6-sol"},
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    )
    assert extract_usage([generation]).model == "gpt-6-sol"


def test_none_when_usage_metadata_missing():
    generation = _chat_generation(model="claude-sonnet-5", usage_metadata=None)
    assert extract_usage([generation]) is None


def test_none_when_model_name_missing():
    generation = _chat_generation(
        model=None, usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    )
    assert extract_usage([generation]) is None


def test_none_for_plain_text_generation():
    assert extract_usage([Generation(text="plain completion, no message at all")]) is None


def test_none_for_empty_generations():
    assert extract_usage([]) is None


def test_finds_usage_on_a_later_generation_in_the_list():
    plain = Generation(text="no usage here")
    with_usage = _chat_generation(
        model="claude-sonnet-5",
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    usage = extract_usage([plain, with_usage])
    assert usage is not None
    assert usage.input_tokens == 7
