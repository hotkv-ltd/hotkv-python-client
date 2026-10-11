# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""LangChain integration (extra `langchain`, requires `langchain-core`):

- `HotKVCache`: exact-match `BaseCache` on `PCACHE.*` (core `GET`/`SET`
  fallback when unlicensed).
- `HotKVSemanticCache`: embedding-similarity `BaseCache` on `SCACHE.*`.
- `HotKVVectorStore`: `VectorStore` on `VSIM.*`.
- `HotKVChatMessageHistory`: `BaseChatMessageHistory` on a HotKV list.

None of this is imported by plain `import hotkv`; `langchain-core` is only
required once something here is imported.
"""

from __future__ import annotations

from .._require import require

require("langchain_core", "langchain-core", "langchain")
del require

from .cache import HotKVCache  # noqa: E402
from .chat_history import HotKVChatMessageHistory  # noqa: E402
from .semantic_cache import HotKVSemanticCache  # noqa: E402
from .vectorstore import HotKVVectorStore  # noqa: E402

__all__ = ["HotKVCache", "HotKVSemanticCache", "HotKVVectorStore", "HotKVChatMessageHistory"]
