# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Server administration and introspection commands."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .._typing import ValueT
from ._shared import CallbackMap, ReadonlySet, cb_ok, force_text, pairs_to_dict

__all__ = ["ServerCommands", "CALLBACKS", "READONLY"]


def _coerce_info_scalar(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _cb_info(reply: Any, **_options: Any) -> Dict[str, Any]:
    text = reply.decode("utf-8", "replace") if isinstance(reply, (bytes, bytearray)) else reply
    info: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        info[key] = _coerce_info_scalar(value)
    return info


def _cb_config_get(reply: Any, **_options: Any) -> Dict[str, Any]:
    d = pairs_to_dict(reply)
    return {force_text(k): force_text(v) for k, v in d.items()}


def _cb_time(reply: Any, **_options: Any) -> Tuple[int, int]:
    seconds, microseconds = reply
    return (int(seconds), int(microseconds))


def _cb_pubsub_numsub(reply: Any, **_options: Any) -> Dict[Any, int]:
    """`[channel, count, ...]` (or a RESP3 map) -> `{channel: count}`."""
    return {channel: int(count) for channel, count in pairs_to_dict(reply).items()}


CALLBACKS: CallbackMap = {
    "INFO": _cb_info,
    "PUBSUB NUMSUB": _cb_pubsub_numsub,
    "PUBSUB SHARDNUMSUB": _cb_pubsub_numsub,
    "CONFIG GET": _cb_config_get,
    "CONFIG SET": cb_ok,
    "TIME": _cb_time,
    "FLUSHDB": cb_ok,
    "FLUSHALL": cb_ok,
    "CLIENT SETNAME": cb_ok,
}

READONLY: ReadonlySet = frozenset(
    {
        "PING",
        "ECHO",
        "INFO",
        "DBSIZE",
        "TIME",
        "CONFIG GET",
        "CLIENT GETNAME",
        "CLIENT ID",
        "PUBSUB CHANNELS",
        "PUBSUB NUMSUB",
        "PUBSUB NUMPAT",
        "PUBSUB SHARDCHANNELS",
        "PUBSUB SHARDNUMSUB",
    }
)


class ServerCommands:
    """Typed wrappers for server administration and introspection commands."""

    def ping(self, message: Optional[ValueT] = None) -> Any:
        """Ping the server. Without `message`, returns "PONG"; with one, echoes it."""
        args = [] if message is None else [message]
        return self.execute_command("PING", *args)

    def echo(self, message: ValueT) -> bytes:
        """Echo `message` back."""
        return self.execute_command("ECHO", message)

    def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Return server information and statistics as a `{field: value}` dict."""
        args = [] if section is None else [section]
        return self.execute_command("INFO", *args)

    def dbsize(self) -> int:
        """Number of keys in the currently selected database."""
        return self.execute_command("DBSIZE")

    def flushdb(self, *, asynchronous: bool = False) -> bool:
        """Remove all keys from the currently selected database."""
        args = ["ASYNC"] if asynchronous else []
        return self.execute_command("FLUSHDB", *args)

    def flushall(self, *, asynchronous: bool = False) -> bool:
        """Remove all keys from every database."""
        args = ["ASYNC"] if asynchronous else []
        return self.execute_command("FLUSHALL", *args)

    def time(self) -> Tuple[int, int]:
        """Return the server's current time as `(unix_seconds, microseconds)`."""
        return self.execute_command("TIME")

    def config_get(self, pattern: str = "*") -> Dict[str, Any]:
        """Return configuration parameters matching `pattern`."""
        return self.execute_command("CONFIG GET", pattern)

    def config_set(self, mapping: Dict[str, ValueT]) -> bool:
        """Set one or more configuration parameters."""
        args: List[Any] = []
        for key, value in mapping.items():
            args += [key, value]
        return self.execute_command("CONFIG SET", *args)

    def client_setname(self, name: str) -> bool:
        """Set the display name of the current connection."""
        return self.execute_command("CLIENT SETNAME", name)

    def client_getname(self) -> Optional[bytes]:
        """Return the display name of the current connection, if set."""
        return self.execute_command("CLIENT GETNAME")

    def client_id(self) -> int:
        """Return the current connection's server-assigned ID."""
        return self.execute_command("CLIENT ID")

    # -- publish / pub-sub introspection (subscribing is `client.pubsub()`) --------

    def publish(self, channel: ValueT, message: ValueT) -> int:
        """Publish `message` to `channel`, returning the number of subscribers
        it was delivered to (see `pubsub()` to subscribe)."""
        return self.execute_command("PUBLISH", channel, message)

    def spublish(self, shard_channel: ValueT, message: ValueT) -> int:
        """Publish `message` to a shard channel, returning the number of subscribers."""
        return self.execute_command("SPUBLISH", shard_channel, message)

    def pubsub_channels(self, pattern: Optional[ValueT] = None) -> List[bytes]:
        """Active channels that have at least one subscriber (not counting
        pattern subscriptions), optionally only those matching `pattern`."""
        args = [] if pattern is None else [pattern]
        return self.execute_command("PUBSUB CHANNELS", *args)

    def pubsub_numsub(self, *channels: ValueT) -> Dict[Any, int]:
        """`{channel: subscriber_count}` for each of `channels`."""
        return self.execute_command("PUBSUB NUMSUB", *channels)

    def pubsub_numpat(self) -> int:
        """The number of active pattern subscriptions."""
        return self.execute_command("PUBSUB NUMPAT")

    def pubsub_shardchannels(self, pattern: Optional[ValueT] = None) -> List[bytes]:
        """Active shard channels, optionally only those matching `pattern`."""
        args = [] if pattern is None else [pattern]
        return self.execute_command("PUBSUB SHARDCHANNELS", *args)

    def pubsub_shardnumsub(self, *shard_channels: ValueT) -> Dict[Any, int]:
        """`{shard_channel: subscriber_count}` for each of `shard_channels`."""
        return self.execute_command("PUBSUB SHARDNUMSUB", *shard_channels)
