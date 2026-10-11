# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Asyncio client for HotKV.

Mirrors `hotkv` (the sync package) option for option and method for
method, built on `asyncio` streams instead of blocking sockets::

    import asyncio
    import hotkv.asyncio

    async def main():
        client = hotkv.asyncio.HotKV(host="localhost", port=6379)
        await client.set("greeting", "hello")
        print(await client.get("greeting"))
        await client.close()

    asyncio.run(main())

Exceptions are shared with the sync client: catch `hotkv.ResponseError`,
`hotkv.NotLicensedError`, and so on, regardless of which client raised them.
"""

from __future__ import annotations

from typing import Any

from .client import HotKV
from .cluster import ClusterHotKV
from .pipeline import AsyncPipeline
from .pubsub import AsyncPubSub, Message

Pipeline = AsyncPipeline
PubSub = AsyncPubSub

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "HotKV",
    "ClusterHotKV",
    "Pipeline",
    "AsyncPipeline",
    "PubSub",
    "AsyncPubSub",
    "Message",
    "from_url",
]


def from_url(url: str, **kwargs: Any) -> HotKV:
    """Build an asyncio `HotKV` client from a connection URL.

    Accepts `hotkv://`, `hotkvs://` (TLS), `redis://`, `rediss://` and
    `unix://`. Keyword arguments override anything the URL also specifies.
    """
    return HotKV.from_url(url, **kwargs)
