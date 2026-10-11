# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`TokenUsage`: the token-usage group shared by `PCACHE.SET`, `SCACHE.SET`,
every keyed cache `SET` (`TOOL.CACHE`/`NCACHE`/`GUARD`/`RERANK`) and
`LLMBUDGET.CONSUME`.

Recording a `TokenUsage` with a cache entry lets a later cache hit
attribute its tokens, and (once the model is priced) its exact cost, to
the namespace's `usage_saved_*` counters that `*_stats()` reports.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple

__all__ = ["TokenUsage"]


class TokenUsage(NamedTuple):
    """One model call's token usage. `model` is required; every count
    defaults to `0` and is omitted from the wire when zero (`MODEL` itself
    is always sent when a `TokenUsage` is given).

    - `input_tokens`: uncached prompt tokens.
    - `output_tokens`: output tokens.
    - `cache_read_tokens`: provider prompt-cache hit tokens.
    - `cache_write_5m_tokens` / `cache_write_1h_tokens`: Anthropic
      cache-write tokens, by TTL tier.
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    def wire_args(self) -> List[Any]:
        """The `MODEL <model> [INPUT n] ...` `KEY VALUE` pairs this usage
        adds to a command's trailing options."""
        args: List[Any] = ["MODEL", self.model]
        if self.input_tokens:
            args += ["INPUT", self.input_tokens]
        if self.output_tokens:
            args += ["OUTPUT", self.output_tokens]
        if self.cache_read_tokens:
            args += ["CACHEREAD", self.cache_read_tokens]
        if self.cache_write_5m_tokens:
            args += ["CACHEWRITE5M", self.cache_write_5m_tokens]
        if self.cache_write_1h_tokens:
            args += ["CACHEWRITE1H", self.cache_write_1h_tokens]
        return args
