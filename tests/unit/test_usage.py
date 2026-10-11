# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for `TokenUsage`'s wire encoding. No server needed."""

from __future__ import annotations

from hotkv import TokenUsage


def test_model_only_sends_just_model():
    assert TokenUsage(model="gpt-6-sol").wire_args() == ["MODEL", "gpt-6-sol"]


def test_zero_counts_are_omitted():
    usage = TokenUsage(model="claude-sonnet-5", input_tokens=0, output_tokens=0)
    assert usage.wire_args() == ["MODEL", "claude-sonnet-5"]


def test_every_count_present():
    usage = TokenUsage(
        model="claude-sonnet-5",
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=5,
        cache_write_5m_tokens=7,
        cache_write_1h_tokens=9,
    )
    assert usage.wire_args() == [
        "MODEL", "claude-sonnet-5",
        "INPUT", 100,
        "OUTPUT", 20,
        "CACHEREAD", 5,
        "CACHEWRITE5M", 7,
        "CACHEWRITE1H", 9,
    ]


def test_partial_counts_keep_only_nonzero_keywords():
    usage = TokenUsage(model="gpt-6-sol", input_tokens=50, cache_write_1h_tokens=3)
    assert usage.wire_args() == ["MODEL", "gpt-6-sol", "INPUT", 50, "CACHEWRITE1H", 3]


def test_model_is_required_positionally():
    # Constructing without `model` is a TypeError, not a silently empty value.
    import pytest

    with pytest.raises(TypeError):
        TokenUsage()  # type: ignore[call-arg]
