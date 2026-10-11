# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for `hotkv.ai._call_result.split_usage()`, the detection
`PromptCache`/`SemanticCache.get_or_call()` use to let the caller's model
function optionally report its own `TokenUsage`. No server needed."""

from __future__ import annotations

from hotkv import TokenUsage
from hotkv.ai._call_result import split_usage


def test_plain_response_has_no_usage():
    assert split_usage("just a response") == ("just a response", None)


def test_response_usage_pair_is_split():
    usage = TokenUsage(model="gpt-6-sol", input_tokens=10)
    assert split_usage(("the response", usage)) == ("the response", usage)


def test_a_plain_two_tuple_response_is_not_mistaken_for_usage():
    # A response that happens to itself be a 2-tuple (e.g. multiple
    # generations) is never confused with a (response, usage) pair: only
    # the second element's *type* (a real TokenUsage) triggers the split.
    assert split_usage(("gen1", "gen2")) == (("gen1", "gen2"), None)


def test_none_response_has_no_usage():
    assert split_usage(None) == (None, None)
