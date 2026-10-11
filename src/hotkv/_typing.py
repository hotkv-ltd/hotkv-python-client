# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Shared type aliases used across the public command surface."""

from __future__ import annotations

from typing import Iterable, Union

KeyT = Union[str, bytes]
"""A HotKV key: `str` is UTF-8 encoded on the wire, `bytes` sent as-is."""

FieldT = Union[str, bytes]
"""A hash field, tag, or similar short identifier."""

ValueT = Union[str, bytes, int, float]
"""A value accepted by write commands; numbers are formatted as text."""

KeysT = Iterable[KeyT]
ValuesT = Iterable[ValueT]
