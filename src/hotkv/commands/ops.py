# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Operational command families: RATELIMIT.* (request rate limiting),
TS.* (time series) and FEATURE.* (feature store).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from .._numbers import format_float
from .._typing import KeyT, ValueT
from ._shared import CallbackMap, ReadonlySet, as_float, cb_ok, force_text, named_fields, pairs_to_dict

__all__ = ["OpsCommands", "RateLimitResult", "CALLBACKS", "READONLY"]


class RateLimitResult(NamedTuple):
    """Reply of `ratelimit_check()`/`ratelimit_peek()`."""

    allowed: bool
    remaining: int
    retry_after_ms: int
    limit: int
    current: int


def _cb_ratelimit(reply: Any, **_options: Any) -> RateLimitResult:
    allowed, remaining, retry_after_ms, limit, current = reply
    return RateLimitResult(bool(allowed), int(remaining), int(retry_after_ms), int(limit), int(current))


def _cb_ts_range(reply: Any, **_options: Any) -> List[Tuple[int, float]]:
    return [(int(ts), as_float(value)) for ts, value in reply]


def _cb_ts_info(reply: Any, **_options: Any) -> Dict[str, Any]:
    info = named_fields(reply, texts={"key"}, floats={"lastValue"})
    labels = info.get("labels")
    if isinstance(labels, list):
        info["labels"] = [(force_text(label), value) for label, value in labels]
    return info


def _cb_feature_entry(item: Any) -> Any:
    if item is None:
        return None
    flat_pairs, updated_at_ms, freshness_ms = item
    return (pairs_to_dict(flat_pairs), int(updated_at_ms), int(freshness_ms))


def _cb_feature_get(reply: Any, **_options: Any) -> Any:
    return _cb_feature_entry(reply)


def _cb_feature_batch(reply: Any, **_options: Any) -> List[Any]:
    return [_cb_feature_entry(item) for item in reply]


def _cb_feature_schema(reply: Any, **_options: Any) -> List[Tuple[str, str]]:
    return [(force_text(name), force_text(ftype)) for name, ftype in reply]


CALLBACKS: CallbackMap = {
    "RATELIMIT.CREATE": cb_ok,
    "RATELIMIT.CHECK": _cb_ratelimit,
    "RATELIMIT.PEEK": _cb_ratelimit,
    "RATELIMIT.RESET": cb_ok,
    "RATELIMIT.STATS": lambda reply, **_: named_fields(reply, texts={"name", "algo"}),
    "RATELIMIT.DROP": cb_ok,
    "TS.CREATE": cb_ok,
    "TS.RANGE": _cb_ts_range,
    "TS.AGG": _cb_ts_range,
    "TS.INFO": _cb_ts_info,
    "TS.DELETE": cb_ok,
    "TS.RETENTION": cb_ok,
    "TS.DOWNSAMPLE": cb_ok,
    "FEATURE.CREATE": cb_ok,
    "FEATURE.SET": cb_ok,
    "FEATURE.SETAT": cb_ok,
    "FEATURE.GET": _cb_feature_get,
    "FEATURE.BATCH": _cb_feature_batch,
    "FEATURE.SCHEMA": _cb_feature_schema,
    "FEATURE.DROP": cb_ok,
    "FEATURE.STATS": lambda reply, **_: named_fields(reply),
}

READONLY: ReadonlySet = frozenset(
    {
        "RATELIMIT.PEEK",
        "RATELIMIT.STATS",
        "RATELIMIT.LIST",
        "TS.RANGE",
        "TS.AGG",
        "TS.INFO",
        "TS.LIST",
        "FEATURE.GET",
        "FEATURE.BATCH",
        "FEATURE.SCHEMA",
        "FEATURE.LIST",
        "FEATURE.STATS",
    }
)


class OpsCommands:
    """Typed wrappers for RATELIMIT.*, TS.* and FEATURE.*."""

    # -- RATELIMIT.* ------------------------------------------------------------

    def ratelimit_create(
        self,
        name: ValueT,
        limit: int,
        window_ms: int,
        *,
        algo: str = "sliding_window",
        refill: Optional[float] = None,
        drain: Optional[float] = None,
    ) -> bool:
        """Create rate limiter `name`."""
        args: List[Any] = [name, limit, window_ms, "ALGO", algo]
        if refill is not None:
            args += ["REFILL", format_float(refill)]
        if drain is not None:
            args += ["DRAIN", format_float(drain)]
        return self.execute_command("RATELIMIT.CREATE", *args)

    def ratelimit_check(self, name: ValueT, key: KeyT, *, cost: Optional[int] = None) -> RateLimitResult:
        """Check and consume from limiter `name` for `key` (mutating)."""
        args: List[Any] = [name, key]
        if cost is not None:
            args += ["COST", cost]
        return self.execute_command("RATELIMIT.CHECK", *args)

    def ratelimit_peek(self, name: ValueT, key: KeyT) -> RateLimitResult:
        """Like `ratelimit_check()`, without consuming anything."""
        return self.execute_command("RATELIMIT.PEEK", name, key)

    def ratelimit_reset(self, name: ValueT, key: KeyT) -> bool:
        """Reset limiter `name`'s state for `key`."""
        return self.execute_command("RATELIMIT.RESET", name, key)

    def ratelimit_stats(self, name: ValueT) -> Dict[str, Any]:
        """Return configuration and totals for limiter `name`."""
        return self.execute_command("RATELIMIT.STATS", name)

    def ratelimit_drop(self, name: ValueT) -> bool:
        """Delete limiter `name`."""
        return self.execute_command("RATELIMIT.DROP", name)

    def ratelimit_list(self) -> List[bytes]:
        """List limiter names."""
        return self.execute_command("RATELIMIT.LIST")

    # -- TS.* (time series) -----------------------------------------------------

    def ts_create(
        self,
        key: KeyT,
        *,
        retention_ms: Optional[int] = None,
        labels: Optional[Mapping[str, ValueT]] = None,
    ) -> bool:
        """Create time series `key`. Pass `retention_ms` when using `labels`
        too: LABELS values are read greedily to the end of the command."""
        args: List[Any] = [key]
        if retention_ms is not None:
            args += ["RETENTION", retention_ms]
        if labels:
            args.append("LABELS")
            for label, value in labels.items():
                args += [label, value]
        return self.execute_command("TS.CREATE", *args)

    def ts_add(self, key: KeyT, value: float, timestamp: Union[int, str] = "*") -> int:
        """Append `value` at `timestamp` ("*" = server time), returning the ms used."""
        return self.execute_command("TS.ADD", key, timestamp, format_float(value))

    def ts_range(self, key: KeyT, from_ms: Union[int, str], to_ms: Union[int, str], *, count: Optional[int] = None) -> List[Tuple[int, float]]:
        """Return `(timestamp_ms, value)` samples between `from_ms` and `to_ms`
        ("-"/"+"" are accepted for the open ends)."""
        args: List[Any] = [key, from_ms, to_ms]
        if count is not None:
            args += ["COUNT", count]
        return self.execute_command("TS.RANGE", *args)

    def ts_agg(
        self,
        key: KeyT,
        from_ms: Union[int, str],
        to_ms: Union[int, str],
        bucket_ms: int,
        agg_type: str,
    ) -> List[Tuple[int, float]]:
        """Return bucketed `(timestamp_ms, value)` samples ("avg"/"min"/"max"/"sum"/"count")."""
        return self.execute_command("TS.AGG", key, from_ms, to_ms, bucket_ms, agg_type)

    def ts_info(self, key: KeyT) -> Dict[str, Any]:
        """Return metadata, retention, memory use and labels for `key`."""
        return self.execute_command("TS.INFO", key)

    def ts_delete(self, key: KeyT) -> bool:
        """Delete time series `key`."""
        return self.execute_command("TS.DELETE", key)

    def ts_retention(self, key: KeyT, retention_ms: int) -> bool:
        """Update the retention window of `key`."""
        return self.execute_command("TS.RETENTION", key, retention_ms)

    def ts_downsample(self, src_key: KeyT, dest_key: KeyT, agg_type: str, bucket_ms: int) -> bool:
        """Add a live downsample rule from `src_key` into `dest_key` (both must exist)."""
        return self.execute_command("TS.DOWNSAMPLE", src_key, dest_key, agg_type, bucket_ms)

    def ts_list(self, *, label: Optional[Tuple[str, ValueT]] = None) -> List[bytes]:
        """List series keys, optionally filtered to a `(label, value)` pair."""
        args: List[Any] = []
        if label is not None:
            args += ["LABEL", label[0], label[1]]
        return self.execute_command("TS.LIST", *args)

    # -- FEATURE.* (feature store) ------------------------------------------------

    def feature_create(self, group: ValueT, description: ValueT, schema: Mapping[str, str]) -> bool:
        """Create feature group `group` with `{feature_name: type}` (types:
        "int64"/"double"/"string"/"int_array"/"double_array")."""
        args: List[Any] = [group, description, "SCHEMA"]
        for name, ftype in schema.items():
            args += [name, ftype]
        return self.execute_command("FEATURE.CREATE", *args)

    def feature_set(self, group: ValueT, entity: ValueT, values: Mapping[str, ValueT]) -> bool:
        """Set one or more feature values for `entity` in `group`."""
        args: List[Any] = [group, entity]
        for name, value in values.items():
            args += [name, value]
        return self.execute_command("FEATURE.SET", *args)

    def feature_setat(
        self, group: ValueT, entity: ValueT, unix_ms: int, values: Mapping[str, ValueT]
    ) -> bool:
        """Like `feature_set()`, stamping the entity with `unix_ms` (a Unix time
        in milliseconds) instead of the server's clock as its update time."""
        args: List[Any] = [group, entity, unix_ms]
        for name, value in values.items():
            args += [name, value]
        return self.execute_command("FEATURE.SETAT", *args)

    def feature_get(
        self, group: ValueT, entity: ValueT, features: Optional[Sequence[str]] = None
    ) -> Optional[Tuple[Dict[str, Any], int, int]]:
        """Return `({feature: value}, updated_at_ms, freshness_ms)` for `entity`,
        or `None`. Omit `features` for all stored features."""
        args: List[Any] = [group, entity, *(features or [])]
        return self.execute_command("FEATURE.GET", *args)

    def feature_batch(
        self, group: ValueT, entities: Sequence[ValueT], *, features: Optional[Sequence[str]] = None
    ) -> List[Optional[Tuple[Dict[str, Any], int, int]]]:
        """Like `feature_get()`, for several entities at once, same order as `entities`."""
        args: List[Any] = [group, *entities]
        if features:
            args += ["FEATURES", *features]
        return self.execute_command("FEATURE.BATCH", *args)

    def feature_schema(self, group: ValueT) -> List[Tuple[str, str]]:
        """Return `(feature_name, type)` pairs for `group`."""
        return self.execute_command("FEATURE.SCHEMA", group)

    def feature_list(self) -> List[bytes]:
        """List feature group names."""
        return self.execute_command("FEATURE.LIST")

    def feature_drop(self, group: ValueT) -> bool:
        """Delete feature group `group`."""
        return self.execute_command("FEATURE.DROP", group)

    def feature_stats(self, group: ValueT) -> Dict[str, int]:
        """Return `entities`/`totalReads`/`totalWrites`/`memoryBytes` for `group`."""
        return self.execute_command("FEATURE.STATS", group)
