# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Fixtures shared by unit tests. None of these need a server."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from hotkv import commands


class RecordingClient(commands.CoreCommands):
    """A fake client for unit-testing typed command wrappers with no server:
    every call records its raw wire arguments (`command_name, *args`), and
    when a canned reply was queued for that command name (`when()`), the
    call runs it through the exact same response callback the real sync and
    asyncio clients use, so parsing is exercised too.
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []
        self._replies: Dict[str, Any] = {}

    def when(self, command_name: str, reply: Any) -> None:
        """Queue the raw (server-shaped) reply the next call to
        `command_name` should return, decoded through its real callback."""
        self._replies[command_name] = reply

    def execute_command(self, *args: Any, **options: Any) -> Any:
        self.calls.append(args)
        command_name = args[0]
        callback = commands.RESPONSE_CALLBACKS.get(command_name)
        raw = self._replies.get(command_name)
        return callback(raw, **options) if callback is not None else raw

    @property
    def last_call(self) -> Tuple[Any, ...]:
        return self.calls[-1]


@pytest.fixture()
def recording_client() -> RecordingClient:
    return RecordingClient()


class AsyncRecordingClient(commands.CoreCommands):
    """Asyncio equivalent of `RecordingClient`, for unit-testing the
    `Async*` AI helpers (`AsyncPromptCache`/`AsyncSemanticCache`) with no
    server and no real event loop I/O."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []
        self._replies: Dict[str, Any] = {}

    def when(self, command_name: str, reply: Any) -> None:
        self._replies[command_name] = reply

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        self.calls.append(args)
        command_name = args[0]
        callback = commands.RESPONSE_CALLBACKS.get(command_name)
        raw = self._replies.get(command_name)
        return callback(raw, **options) if callback is not None else raw

    @property
    def last_call(self) -> Tuple[Any, ...]:
        return self.calls[-1]


@pytest.fixture()
def async_recording_client() -> AsyncRecordingClient:
    return AsyncRecordingClient()
