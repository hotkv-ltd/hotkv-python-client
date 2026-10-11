# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Shared by `PromptCache`/`SemanticCache` (and their asyncio equivalents):
`get_or_call()`'s `fn` may return either just the response, or a
`(response, usage)` pair, so a real model call can report its actual token
usage only once it has run. Detected by the second element's type alone
(never just "is a 2-tuple"), so it stays unambiguous no matter what
`response` itself looks like.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

from ..usage import TokenUsage

__all__ = ["CallResult", "split_usage"]

CallResult = Union[Any, Tuple[Any, TokenUsage]]


def split_usage(result: CallResult) -> Tuple[Any, Optional[TokenUsage]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], TokenUsage):
        return result[0], result[1]
    return result, None
