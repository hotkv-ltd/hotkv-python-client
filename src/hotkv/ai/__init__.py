# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Idiomatic, provider neutral AI caching helpers built on the raw
`PCACHE.*`/`SCACHE.*` commands: `PromptCache` (exact-match, deterministic
key) and `SemanticCache` (embedding similarity). Each has an asyncio
counterpart (`AsyncPromptCache`, `AsyncSemanticCache`) that takes a
`hotkv.asyncio.HotKV` client instead.

Neither class imports or depends on any specific LLM or embedding
provider's SDK: you bring the model call and, for `SemanticCache`, the
embedding vector.
"""

from __future__ import annotations

from .prompt_cache import AsyncPromptCache, PromptCache
from .semantic_cache import AsyncSemanticCache, SemanticCache

__all__ = ["PromptCache", "AsyncPromptCache", "SemanticCache", "AsyncSemanticCache"]
