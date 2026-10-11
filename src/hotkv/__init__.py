# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Official Python client for HotKV.

Quick start::

    import hotkv

    client = hotkv.HotKV(host="localhost", port=6379)
    client.set("greeting", "hello")
    print(client.get("greeting"))

    # or from a URL
    client = hotkv.from_url("hotkv://user:pass@localhost:6379/0")

For asyncio, see `hotkv.asyncio`; for the AI caching helpers, see `hotkv.ai`.
"""

from __future__ import annotations

from typing import Any

from ._numbers import format_float, format_usd
from .client import HotKV
from .cluster import ClusterHotKV
from .commands.ai_caches import LLMPrice
from .exceptions import (
    AskError,
    AuthenticationError,
    BusyError,
    ClusterDownError,
    ClusterError,
    ConnectionError,
    CrossSlotError,
    DataError,
    ExecAbortError,
    HotKVError,
    MovedError,
    NoAuthError,
    NoScriptError,
    NotLicensedError,
    ProtocolError,
    PubSubError,
    ResponseError,
    TimeoutError,
    TryAgainError,
    WatchError,
    WrongPassError,
    WrongTypeError,
)
from .pipeline import Pipeline
from .pubsub import Message, PubSub
from .usage import TokenUsage

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "HotKV",
    "ClusterHotKV",
    "Pipeline",
    "PubSub",
    "Message",
    "from_url",
    "TokenUsage",
    "LLMPrice",
    "format_float",
    "format_usd",
    "HotKVError",
    "ConnectionError",
    "TimeoutError",
    "AuthenticationError",
    "ProtocolError",
    "DataError",
    "PubSubError",
    "WatchError",
    "ClusterError",
    "ResponseError",
    "NotLicensedError",
    "WrongTypeError",
    "NoAuthError",
    "WrongPassError",
    "NoScriptError",
    "BusyError",
    "ExecAbortError",
    "ClusterDownError",
    "CrossSlotError",
    "TryAgainError",
    "AskError",
    "MovedError",
]


def from_url(url: str, **kwargs: Any) -> HotKV:
    """Build a synchronous `HotKV` client from a connection URL.

    Accepts `hotkv://`, `hotkvs://` (TLS), `redis://`, `rediss://` and
    `unix://`. Keyword arguments override anything the URL also specifies.
    See `hotkv.asyncio.from_url()` for the asyncio equivalent.
    """
    return HotKV.from_url(url, **kwargs)
