# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVChatMessageHistory`: a `langchain_core.chat_history.BaseChatMessageHistory`
storing one session's turns in a HotKV list, one JSON blob per message."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

__all__ = ["HotKVChatMessageHistory"]


class HotKVChatMessageHistory(BaseChatMessageHistory):
    """Persists one chat session's message history in HotKV.

    ```python
    from hotkv.integrations.langchain import HotKVChatMessageHistory

    history = HotKVChatMessageHistory(client, session_id="user-42", ttl=86400)
    history.add_user_message("hello")
    history.add_ai_message("hi, how can I help?")
    history.messages  # [HumanMessage(...), AIMessage(...)]
    ```

    Each message is appended (`RPUSH`) as one JSON blob to a HotKV list
    keyed by `session_id`, so `add_messages()` is a single round trip
    regardless of how many messages are added, and reading back the
    session (`messages`) preserves insertion order. `ttl`, if given, is
    refreshed (`EXPIRE`) after every append, so an idle session expires
    `ttl` seconds after its last message rather than after its first.
    """

    def __init__(
        self,
        client: Any,
        session_id: str,
        *,
        key_prefix: str = "hotkv:langchain:chat:",
        ttl: Optional[int] = None,
    ) -> None:
        self._client = client
        self.session_id = session_id
        self._key = f"{key_prefix}{session_id}"
        self._ttl = ttl

    @property
    def messages(self) -> List[BaseMessage]:
        raw = self._client.lrange(self._key, 0, -1)
        decoded = [item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else item for item in raw]
        return messages_from_dict([json.loads(item) for item in decoded])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        messages = list(messages)
        if not messages:
            return
        payloads = [json.dumps(d) for d in messages_to_dict(messages)]
        self._client.rpush(self._key, *payloads)
        if self._ttl is not None:
            self._client.expire(self._key, self._ttl)

    def clear(self) -> None:
        self._client.delete(self._key)
