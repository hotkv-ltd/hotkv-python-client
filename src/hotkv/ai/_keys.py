# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Deterministic cache key construction, shared by the sync and asyncio
`PromptCache`."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

__all__ = ["build_cache_key"]


def build_cache_key(
    provider: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    system: Optional[str] = None,
    **params: Any,
) -> str:
    """Build a deterministic cache key for one LLM call shape.

    Hashes `provider`, `model`, `system`, `messages` and any generation
    `params` (temperature, max_tokens, ...) as sorted-key JSON with
    SHA-256, so the same logical call always produces the same key no
    matter the dict insertion order used to build it.
    """
    payload = {
        "provider": provider,
        "model": model,
        "system": system,
        "messages": list(messages),
        "params": dict(params),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
