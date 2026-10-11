# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVChatStore`: a `llama_index.core.storage.chat_store.base.BaseChatStore`
on top of a single HotKV hash, one field per chat key."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore

__all__ = ["HotKVChatStore"]


class HotKVChatStore(BaseChatStore):
    """LlamaIndex chat store (one message list per agent/session key) on HotKV.

    ```python
    from hotkv.integrations.llama_index import HotKVChatStore
    from llama_index.core.memory import ChatMemoryBuffer

    chat_store = HotKVChatStore(client, hash_key="my-app-chats")
    memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, chat_store_key="session-42")
    ```

    All keys managed by one `HotKVChatStore` live as fields of a single
    HotKV hash (`hash_key`), each field's value a JSON array of
    `ChatMessage` dicts; `get_keys()` is one `HKEYS` call. Mutating a
    single message (`add_message()`, `delete_message()`,
    `delete_last_message()`) reads the field, edits the list in memory, and
    writes it back, matching LlamaIndex's own in-memory `SimpleChatStore`
    semantics. That makes one chat key a single-writer structure: two
    processes appending to the same key at the same moment can overwrite each
    other's message (give each conversation one writer, as chat sessions
    have). All the keys of one store live in one hash, hence in one slot and
    on one node of a cluster; use several stores (several `hash_key`s) to
    spread the conversations over the nodes.
    """

    hash_key: str = "hotkv:llamaindex:chat"

    _client: Any = PrivateAttr()

    def __init__(self, client: Any, *, hash_key: str = "hotkv:llamaindex:chat", **kwargs: Any) -> None:
        super().__init__(hash_key=hash_key, **kwargs)
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "HotKVChatStore"

    def _load(self, key: str) -> List[ChatMessage]:
        raw = self._client.hget(self.hash_key, key)
        if raw is None:
            return []
        return [ChatMessage.model_validate(item) for item in json.loads(raw)]

    def _save(self, key: str, messages: List[ChatMessage]) -> None:
        payload = json.dumps([m.model_dump(mode="json") for m in messages])
        self._client.hset(self.hash_key, {key: payload})

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        self._save(key, messages)

    def get_messages(self, key: str) -> List[ChatMessage]:
        return self._load(key)

    def add_message(self, key: str, message: ChatMessage) -> None:
        messages = self._load(key)
        messages.append(message)
        self._save(key, messages)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        messages = self._load(key)
        if not messages:
            return None
        self._client.hdel(self.hash_key, key)
        return messages

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        messages = self._load(key)
        if idx >= len(messages) or idx < -len(messages):
            return None
        message = messages.pop(idx)
        self._save(key, messages)
        return message

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        return self.delete_message(key, -1)

    def get_keys(self) -> List[str]:
        raw = self._client.hkeys(self.hash_key)
        return [k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else k for k in raw]
