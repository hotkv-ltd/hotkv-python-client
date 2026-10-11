# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""JSON round trip for the `Generation`/`ChatGeneration` objects a LangChain
`BaseCache` stores, shared by `HotKVCache` and `HotKVSemanticCache`.

Deliberately hand rolled rather than `langchain_core.load.dumps()`/`loads()`:
those handle arbitrary LangChain objects (and are themselves a beta API),
while a cache only ever needs to round trip the two concrete shapes a
`RETURN_VAL_TYPE` can take, so this stays small, dependency free of the
`load` module, and produces no deprecation warnings.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_core.outputs import ChatGeneration, Generation

from ...usage import TokenUsage

__all__ = ["dump_generations", "load_generations", "extract_usage"]


def dump_generations(generations: Sequence[Generation]) -> str:
    """Serialize a `lookup()`/`update()` `RETURN_VAL_TYPE` to a JSON string."""
    items: List[Any] = []
    for generation in generations:
        if isinstance(generation, ChatGeneration):
            items.append(
                {
                    "generation_info": generation.generation_info,
                    "message": messages_to_dict([generation.message])[0],
                }
            )
        else:
            items.append({"text": generation.text, "generation_info": generation.generation_info})
    return json.dumps(items)


def extract_usage(generations: Sequence[Generation]) -> Optional[TokenUsage]:
    """Best-effort `TokenUsage` from the first `ChatGeneration` in
    `generations` whose message carries both a model name
    (`response_metadata["model_name"]`/`["model"]`) and `usage_metadata`
    (LangChain's standard token-usage field, populated by most chat model
    integrations). `None` when either is missing, for example a plain
    (non-chat) `Generation`, or a model integration that does not populate
    them, so callers that get `None` simply store without usage, exactly as
    before this existed.

    `usage_metadata` has no per-tier breakdown for cache-write tokens (only
    a single combined `input_token_details["cache_creation"]`, without
    Anthropic's 5-minute/1-hour split), so only `input_tokens`,
    `output_tokens` and `input_token_details["cache_read"]` are mapped;
    cache-write tokens are left at their `TokenUsage` default of `0` rather
    than guessed into the wrong tier.
    """
    for generation in generations:
        message = getattr(generation, "message", None)
        usage = getattr(message, "usage_metadata", None) if message is not None else None
        if not usage:
            continue
        metadata = getattr(message, "response_metadata", None) or {}
        model = metadata.get("model_name") or metadata.get("model")
        if not model:
            continue
        details = usage.get("input_token_details") or {}
        return TokenUsage(
            model=model,
            input_tokens=usage.get("input_tokens", 0) or 0,
            output_tokens=usage.get("output_tokens", 0) or 0,
            cache_read_tokens=details.get("cache_read", 0) or 0,
        )
    return None


def load_generations(payload: str) -> List[Generation]:
    """Inverse of `dump_generations()`."""
    items = json.loads(payload)
    result: List[Generation] = []
    for item in items:
        if "message" in item:
            message = messages_from_dict([item["message"]])[0]
            result.append(ChatGeneration(message=message, generation_info=item.get("generation_info")))
        else:
            result.append(Generation(text=item["text"], generation_info=item.get("generation_info")))
    return result
