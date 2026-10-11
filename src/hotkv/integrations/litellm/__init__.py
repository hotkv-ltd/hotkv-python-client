# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""LiteLLM integration (extra `litellm`, requires `litellm`):

- `HotKVCache`: exact-key `litellm.caching.base_cache.BaseCache` on `PCACHE.*`
  (core `GET`/`SET` fallback when unlicensed).
- `HotKVSemanticCache`: similarity based `BaseCache` on `SCACHE.*`.

LiteLLM's built-in `type="redis"` cache (`litellm.Cache(type="redis", host=...,
port=...)`) also works against HotKV as-is for exact caching: HotKV speaks
the RESP wire protocol, so no HotKV specific code is needed for that case.
Reach for the classes here when you also want semantic caching, or want
caching to keep working on a server with no `PromptCache` license.

None of this is imported by plain `import hotkv`; `litellm` is only
required once something here is imported.
"""

from __future__ import annotations

from .._require import require

require("litellm", "litellm", "litellm")
del require

from .cache import HotKVCache  # noqa: E402
from .semantic_cache import HotKVSemanticCache  # noqa: E402

__all__ = ["HotKVCache", "HotKVSemanticCache"]
