# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""LlamaIndex integration (extra `llamaindex`, requires `llama-index-core`):

- `HotKVVectorStore`: `BasePydanticVectorStore` on `VSIM.*`.
- `HotKVKVStore`: `BaseKVStore` (docstore/index store) on HotKV hashes.
- `HotKVChatStore`: `BaseChatStore` on a HotKV hash.

None of this is imported by plain `import hotkv`; `llama-index-core` is
only required once something here is imported.
"""

from __future__ import annotations

from .._require import require

require("llama_index.core", "llama-index-core", "llamaindex")
del require

from .chat_store import HotKVChatStore  # noqa: E402
from .kv_store import HotKVKVStore  # noqa: E402
from .vector_store import HotKVVectorStore  # noqa: E402

__all__ = ["HotKVVectorStore", "HotKVKVStore", "HotKVChatStore"]
