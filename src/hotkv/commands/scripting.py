# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Lua scripting commands.

`eval()`/`evalsha()`/`script_load()` live here and are shared between the
sync and asyncio clients like every other mixin. The EVALSHA-with-NOSCRIPT-
fallback convenience needs real `try`/`except` control flow around an
`await` on the asyncio side, so it cannot use the same trick; it is defined
once on each concrete client instead, as `eval_auto()`.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from .._typing import ValueT
from ._shared import CallbackMap, ReadonlySet, cb_ok

__all__ = ["ScriptingCommands", "CALLBACKS", "READONLY"]

CALLBACKS: CallbackMap = {
    "SCRIPT EXISTS": lambda reply, **_: [bool(v) for v in reply],
    "SCRIPT FLUSH": cb_ok,
}

READONLY: ReadonlySet = frozenset({"SCRIPT EXISTS"})


class ScriptingCommands:
    """Typed wrappers for EVAL/EVALSHA and the SCRIPT admin commands."""

    def eval(self, script: str, keys: Sequence[ValueT] = (), args: Sequence[ValueT] = ()) -> Any:
        """Run a Lua `script` against `keys` (available as `KEYS`) and `args`
        (available as `ARGV`)."""
        keys = list(keys)
        return self.execute_command("EVAL", script, len(keys), *keys, *args)

    def evalsha(
        self, sha1: str, keys: Sequence[ValueT] = (), args: Sequence[ValueT] = ()
    ) -> Any:
        """Like `eval()`, running a script already cached by `sha1` (see
        `script_load()`). Raises `NoScriptError` if the server does not have it."""
        keys = list(keys)
        return self.execute_command("EVALSHA", sha1, len(keys), *keys, *args)

    def script_load(self, script: str) -> bytes:
        """Cache `script` on the server, returning its SHA1 for `evalsha()`."""
        return self.execute_command("SCRIPT LOAD", script)

    def script_exists(self, *sha1s: str) -> List[bool]:
        """Check whether each of `sha1s` is cached on the server."""
        return self.execute_command("SCRIPT EXISTS", *sha1s)

    def script_flush(self, *, asynchronous: bool = False) -> bool:
        """Clear the server's script cache."""
        args: List[Any] = ["ASYNC"] if asynchronous else []
        return self.execute_command("SCRIPT FLUSH", *args)
