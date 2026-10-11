# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVCache`: an exact-match `langchain_core.caches.BaseCache` backed by
`PCACHE.*`, falling back to core `GET`/`SET` when the server has no
prompt-cache license (see `hotkv.integrations._exact_cache`)."""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache

from .._exact_cache import ExactCacheBackend
from ._generations import dump_generations, extract_usage, load_generations

__all__ = ["HotKVCache"]


def _key(prompt: str, llm_string: str) -> str:
    # BaseCache's own contract: a key is derived from the (prompt,
    # llm_string) pair; llm_string already encodes model, params and stop
    # sequences, so two different generation settings never collide.
    digest = hashlib.sha256()
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(llm_string.encode("utf-8"))
    return digest.hexdigest()


class HotKVCache(BaseCache):
    """Exact-match LLM response cache for LangChain, on top of `PCACHE.*`.

    ```python
    from langchain_core.globals import set_llm_cache
    import hotkv
    from hotkv.integrations.langchain import HotKVCache

    client = hotkv.HotKV(host="localhost", port=6379)
    set_llm_cache(HotKVCache(client, namespace="chat-responses", ttl=3600))
    ```

    Once installed, every LangChain LLM/chat model call goes through this
    cache automatically: an identical `(prompt, llm_string)` pair (the
    same model, the same generation parameters, the same rendered prompt)
    is served from HotKV instead of calling the model again. Falls back to
    core `GET`/`SET` (a SHA-256 key, honoring `ttl`) the first time the
    server answers `PCACHE.*` with `NOLICENSE`, so this works unlicensed
    (Starter plan) too, just without `PCACHE.STATS`/eviction accounting.

    The async methods (`alookup`/`aupdate`/`aclear`) are inherited from
    `BaseCache`, which runs the sync implementation below in a thread
    executor; that is enough to make `ainvoke()`/`agenerate()` etc. work
    correctly, since `client` here is a blocking `hotkv.HotKV`.

    When the cached `AIMessage` carries a model name and LangChain's
    standard `usage_metadata`, `update()` stores it with the entry (see
    `hotkv.TokenUsage`) so a later hit attributes its tokens and cost to
    `PCACHE.STATS`'s `usage_saved_*` counters; when it does not (a plain
    text `Generation`, or a model integration that leaves them unset),
    entries are stored exactly as before.
    """

    def __init__(self, client: Any, *, namespace: str = "langchain-cache", ttl: Optional[int] = None) -> None:
        self._backend = ExactCacheBackend(client, namespace, ttl=ttl)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        raw = self._backend.get(_key(prompt, llm_string))
        if raw is None:
            return None
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        return load_generations(text)

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Any]) -> None:
        # When the model's response carries a model name and LangChain's
        # standard `usage_metadata`, it is stored with the entry so a later
        # hit attributes it to `PCACHE.STATS`'s `usage_saved_*` counters.
        self._backend.set(
            _key(prompt, llm_string), dump_generations(return_val), usage=extract_usage(return_val)
        )

    def clear(self, **kwargs: Any) -> None:
        self._backend.flush()
