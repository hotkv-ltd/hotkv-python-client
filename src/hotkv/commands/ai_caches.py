# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""AI caching command families: SCACHE.* (semantic cache), PCACHE.* (prompt
cache), the four identically-shaped keyed caches (TOOL.CACHE/NCACHE/GUARD/
RERANK), LLMSTATS.*, LLMBUDGET.* and PROMPT.* (the versioned prompt
registry).
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .._numbers import format_cost, format_float, format_usd, format_vector
from .._typing import KeyT, ValueT
from ..usage import TokenUsage
from ._shared import CallbackMap, ReadonlySet, as_bool, as_float, cb_bool, cb_ok, expiry_args, named_fields

__all__ = ["AICacheCommands", "LLMPrice", "CALLBACKS", "READONLY"]

# The 6 `usage_saved_*` fields appended to SCACHE.STATS, PCACHE.STATS and
# every keyed cache's STATS reply; all integers
# except the trailing decimal-USD field.
_USAGE_SAVED_FLOATS = frozenset({"usage_saved_cost_usd"})


def _usage_args(usage: Optional[TokenUsage]) -> List[Any]:
    return usage.wire_args() if usage is not None else []


# -- SCACHE.* / PCACHE.* / PROMPT.* reply shaping ------------------------------


def _cb_scache_get(reply: Any, **_options: Any) -> Optional[Tuple[Any, float]]:
    if reply is None:
        return None
    response, similarity = reply
    return (response, as_float(similarity))


def _cb_pcache_getex(reply: Any, **_options: Any) -> Optional[Tuple[Any, Any, int]]:
    if reply is None:
        return None
    response, metadata, hit_count = reply
    return (response, metadata, int(hit_count))


def _cb_prompt_get(reply: Any, **_options: Any) -> Optional[Tuple[int, Any, Any, int]]:
    if reply is None:
        return None
    version, content, metadata, created_unix = reply
    return (int(version), content, metadata, int(created_unix))


class LLMPrice(NamedTuple):
    """Reply of `llmstats_price_get()`/one element of `llmstats_price_list()`.

    Prices are USD per 1,000,000 tokens. `source` is `"default"` (built in),
    `"custom"` (set with `llmstats_price_set()`) or `"unpriced"` (an unknown
    model: every price is `0.0`, not an error).
    """

    model: str = ""
    source: str = ""
    priced: bool = False
    retired: bool = False
    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0
    cache_read_usd_per_mtok: float = 0.0
    cache_write_5m_usd_per_mtok: float = 0.0
    cache_write_1h_usd_per_mtok: float = 0.0


_PRICE_FLOAT_FIELDS = frozenset(
    {
        "input_usd_per_mtok",
        "output_usd_per_mtok",
        "cache_read_usd_per_mtok",
        "cache_write_5m_usd_per_mtok",
        "cache_write_1h_usd_per_mtok",
    }
)
_PRICE_TEXT_FIELDS = frozenset({"model", "source"})


def _price_from_reply(reply: Any) -> LLMPrice:
    fields = named_fields(reply, floats=_PRICE_FLOAT_FIELDS, texts=_PRICE_TEXT_FIELDS)
    fields["priced"] = as_bool(fields.get("priced", 0))
    fields["retired"] = as_bool(fields.get("retired", 0))
    known = LLMPrice._fields
    return LLMPrice(**{key: value for key, value in fields.items() if key in known})


def _format_price(value: Any) -> str:
    """Format a price as the plain decimal string `LLMSTATS.PRICE SET`
    requires (no exponent, at most 6 fractional digits, no trailing zeros),
    rounding half up. A number, `Decimal` or numeric `str` is accepted; a
    negative, NaN, infinite or out of range value raises `ValueError`.
    """
    return format_usd(value)


_BASE_CALLBACKS: CallbackMap = {
    "SCACHE.CREATE": cb_ok,
    "SCACHE.SET": cb_ok,
    "SCACHE.GET": _cb_scache_get,
    "SCACHE.STATS": lambda reply, **_: named_fields(
        reply, floats={"hit_rate"} | _USAGE_SAVED_FLOATS
    ),
    "SCACHE.FLUSH": cb_bool,
    "SCACHE.DROP": cb_bool,
    "SCACHE.THRESHOLD": cb_ok,
    "PCACHE.CREATE": cb_ok,
    "PCACHE.SET": cb_ok,
    "PCACHE.GETEX": _cb_pcache_getex,
    "PCACHE.DEL": cb_bool,
    "PCACHE.STATS": lambda reply, **_: named_fields(
        reply, floats={"hit_rate", "cost_saved_usd"} | _USAGE_SAVED_FLOATS
    ),
    "PCACHE.DROP": cb_bool,
    "PCACHE.CONFIG": cb_ok,
    "LLMSTATS.BUCKET": cb_ok,
    "LLMSTATS.RECORD": cb_ok,
    "LLMSTATS.GET": lambda reply, **_: named_fields(
        reply, floats={"cost_usd", "avg_latency_ms", "cost_usd_1m"}
    ),
    "LLMSTATS.RESET": cb_bool,
    "LLMSTATS.DROP": cb_bool,
    "LLMSTATS.PRICE SET": cb_ok,
    "LLMSTATS.PRICE GET": lambda reply, **_: _price_from_reply(reply),
    "LLMSTATS.PRICE LIST": lambda reply, **_: [_price_from_reply(item) for item in (reply or [])],
    "LLMSTATS.PRICE DEL": lambda reply, **_: bool(int(reply)),
    "LLMBUDGET.SET": cb_ok,
    "LLMBUDGET.CONSUME": cb_bool,
    "LLMBUDGET.STATUS": lambda reply, **_: named_fields(
        reply, floats={"cost_used_usd", "cost_limit_usd"}
    ),
    "LLMBUDGET.RESET": cb_bool,
    "LLMBUDGET.DROP": cb_bool,
    "PROMPT.GET": _cb_prompt_get,
    "PROMPT.LATEST": _cb_prompt_get,
    "PROMPT.SETACTIVE": cb_ok,
    "PROMPT.DROP": cb_bool,
}

_BASE_READONLY: ReadonlySet = frozenset(
    {
        "SCACHE.GET",
        "SCACHE.STATS",
        "SCACHE.LIST",
        "PCACHE.GET",
        "PCACHE.GETEX",
        "PCACHE.STATS",
        "PCACHE.LIST",
        "LLMSTATS.GET",
        "LLMSTATS.LIST",
        "LLMSTATS.PRICE GET",
        "LLMSTATS.PRICE LIST",
        "LLMBUDGET.STATUS",
        "LLMBUDGET.LIST",
        "PROMPT.GET",
        "PROMPT.LATEST",
        "PROMPT.VERSIONS",
        "PROMPT.LIST",
    }
)


# -- the four identically-shaped keyed caches ---------------------------------

_KEYED_CACHE_FAMILIES = (
    ("TOOL.CACHE", "toolcache"),
    ("NCACHE", "ncache"),
    ("GUARD", "guard"),
    ("RERANK", "rerank"),
)


def _keyed_cache_methods(prefix: str, python_name: str) -> Dict[str, Any]:
    """Build the 8 methods shared by one `<PREFIX>.*` keyed-cache family."""

    def create(
        self, ns: ValueT, *, max_entries: Optional[int] = None, ttl: Optional[int] = None
    ) -> bool:
        args: List[Any] = [ns]
        if max_entries is not None:
            args += ["MAX_ENTRIES", max_entries]
        if ttl is not None:
            args += ["TTL", ttl]
        return self.execute_command(f"{prefix}.CREATE", *args)

    def cache_set(
        self,
        ns: ValueT,
        key: KeyT,
        value: ValueT,
        *,
        ttl: Optional[int] = None,
        pxat: Optional[int] = None,
        meta: Optional[str] = None,
        usage: Optional[TokenUsage] = None,
    ) -> bool:
        args: List[Any] = [ns, key, value]
        args += expiry_args(ttl, pxat)
        if meta is not None:
            args += ["META", meta]
        args += _usage_args(usage)
        return self.execute_command(f"{prefix}.SET", *args)

    def get(self, ns: ValueT, key: KeyT) -> Optional[bytes]:
        return self.execute_command(f"{prefix}.GET", ns, key)

    def delete(self, ns: ValueT, key: KeyT) -> bool:
        return self.execute_command(f"{prefix}.DEL", ns, key)

    def stats(self, ns: ValueT) -> Dict[str, Any]:
        return self.execute_command(f"{prefix}.STATS", ns)

    def flush(self, ns: ValueT) -> int:
        return self.execute_command(f"{prefix}.FLUSH", ns)

    def drop(self, ns: ValueT) -> bool:
        return self.execute_command(f"{prefix}.DROP", ns)

    def list_namespaces(self) -> List[bytes]:
        return self.execute_command(f"{prefix}.LIST")

    create.__doc__ = f"Create a {python_name} namespace `ns`."
    cache_set.__doc__ = (
        f"Store `value` under `key` in {python_name} namespace `ns`. `ttl` is seconds "
        "from now, `pxat` the Unix time in milliseconds the entry expires at (not both)."
    )
    get.__doc__ = f"Look up `key` in {python_name} namespace `ns`; `None` on a miss."
    delete.__doc__ = f"Delete `key` from {python_name} namespace `ns`."
    stats.__doc__ = (
        f"Return `entries`/`hits`/`misses`/`evictions`/`memory_bytes` and the "
        f"`usage_saved_*` savings counters for {python_name} `ns`."
    )
    flush.__doc__ = f"Remove every entry in {python_name} namespace `ns`, returning the count."
    drop.__doc__ = f"Delete {python_name} namespace `ns` entirely."
    list_namespaces.__doc__ = f"List {python_name} namespace names."

    return {
        f"{python_name}_create": create,
        f"{python_name}_set": cache_set,
        f"{python_name}_get": get,
        f"{python_name}_del": delete,
        f"{python_name}_stats": stats,
        f"{python_name}_flush": flush,
        f"{python_name}_drop": drop,
        f"{python_name}_list": list_namespaces,
    }


def _keyed_cache_callbacks(prefix: str) -> CallbackMap:
    return {
        f"{prefix}.CREATE": cb_ok,
        f"{prefix}.SET": cb_ok,
        f"{prefix}.DEL": cb_bool,
        f"{prefix}.STATS": lambda reply, **_: named_fields(reply, floats=_USAGE_SAVED_FLOATS),
        f"{prefix}.DROP": cb_bool,
    }


def _keyed_cache_readonly(prefix: str) -> ReadonlySet:
    return frozenset({f"{prefix}.GET", f"{prefix}.STATS", f"{prefix}.LIST"})


CALLBACKS: CallbackMap = dict(_BASE_CALLBACKS)
READONLY: ReadonlySet = _BASE_READONLY
for _prefix, _python_name in _KEYED_CACHE_FAMILIES:
    CALLBACKS.update(_keyed_cache_callbacks(_prefix))
    READONLY = READONLY | _keyed_cache_readonly(_prefix)


class AICacheCommands:
    """Typed wrappers for SCACHE.*, PCACHE.*, the keyed caches, LLMSTATS.*,
    LLMBUDGET.* and PROMPT.*. See `hotkv.ai` for a higher-level, provider
    neutral `PromptCache`/`SemanticCache` layer built on top of these."""

    # -- SCACHE.* (semantic cache) --------------------------------------------

    def scache_create(
        self,
        ns: ValueT,
        dims: int,
        *,
        metric: str = "cosine",
        threshold: float = 0.85,
        ttl: Optional[int] = None,
    ) -> bool:
        """Create semantic cache namespace `ns` for `dims`-dimensional vectors."""
        args: List[Any] = [ns, dims, "METRIC", metric, "THRESHOLD", format_float(threshold)]
        if ttl is not None:
            args += ["TTL", ttl]
        return self.execute_command("SCACHE.CREATE", *args)

    def scache_set(
        self,
        ns: ValueT,
        query_text: ValueT,
        response: ValueT,
        vector: Sequence[float],
        *,
        ttl: Optional[int] = None,
        pxat: Optional[int] = None,
        usage: Optional[TokenUsage] = None,
    ) -> bool:
        """Cache `response` for `query_text`, indexed by `vector`. `ttl` is
        seconds from now, `pxat` the Unix time in milliseconds the entry
        expires at (not both).

        `usage`, if given, records the original (uncached) call's token
        usage so a later hit attributes it (and its cost, once the model is
        priced) to `ns`'s `usage_saved_*` stats.
        """
        args: List[Any] = [ns, query_text, response, *format_vector(vector)]
        args += expiry_args(ttl, pxat)
        args += _usage_args(usage)
        return self.execute_command("SCACHE.SET", *args)

    def scache_get(
        self, ns: ValueT, vector: Sequence[float], *, threshold: Optional[float] = None
    ) -> Optional[Tuple[Any, float]]:
        """Look up the nearest cached response to `vector`.

        Returns `(response, similarity)` on a hit above the threshold, else `None`.
        """
        args: List[Any] = [ns, *format_vector(vector)]
        if threshold is not None:
            args += ["THRESHOLD", format_float(threshold)]
        return self.execute_command("SCACHE.GET", *args)

    def scache_stats(self, ns: ValueT) -> Dict[str, Any]:
        """Return `entries`/`hits`/`misses`/`hit_rate`/`p50_us` and the
        `usage_saved_*` savings counters for `ns`."""
        return self.execute_command("SCACHE.STATS", ns)

    def scache_flush(self, ns: ValueT) -> bool:
        """Remove every entry in `ns`."""
        return self.execute_command("SCACHE.FLUSH", ns)

    def scache_drop(self, ns: ValueT) -> bool:
        """Delete semantic cache namespace `ns`."""
        return self.execute_command("SCACHE.DROP", ns)

    def scache_list(self) -> List[bytes]:
        """List semantic cache namespace names."""
        return self.execute_command("SCACHE.LIST")

    def scache_threshold(self, ns: ValueT, value: float) -> bool:
        """Update the similarity threshold for `ns`."""
        return self.execute_command("SCACHE.THRESHOLD", ns, format_float(value))

    # -- PCACHE.* (prompt/LLM response cache) -------------------------------------

    def pcache_create(
        self,
        ns: ValueT,
        *,
        max_entries: Optional[int] = None,
        ttl: Optional[int] = None,
        cost_per_hit: Optional[float] = None,
    ) -> bool:
        """Create prompt cache namespace `ns`."""
        args: List[Any] = [ns]
        if max_entries is not None:
            args += ["MAX_ENTRIES", max_entries]
        if ttl is not None:
            args += ["TTL", ttl]
        if cost_per_hit is not None:
            args += ["COST_PER_HIT", format_float(cost_per_hit)]
        return self.execute_command("PCACHE.CREATE", *args)

    def pcache_set(
        self,
        ns: ValueT,
        prompt: ValueT,
        response: ValueT,
        *,
        ttl: Optional[int] = None,
        pxat: Optional[int] = None,
        meta: Optional[str] = None,
        usage: Optional[TokenUsage] = None,
    ) -> bool:
        """Cache `response` for the exact `prompt` text. `ttl` is seconds from
        now, `pxat` the Unix time in milliseconds the entry expires at (not both).

        `usage`, if given, records the original (uncached) call's token
        usage so a later hit attributes it (and its cost, once the model is
        priced) to `ns`'s `usage_saved_*` stats.
        """
        args: List[Any] = [ns, prompt, response]
        args += expiry_args(ttl, pxat)
        if meta is not None:
            args += ["META", meta]
        args += _usage_args(usage)
        return self.execute_command("PCACHE.SET", *args)

    def pcache_get(self, ns: ValueT, prompt: ValueT) -> Optional[bytes]:
        """Return the cached response for `prompt`, or `None` on a miss."""
        return self.execute_command("PCACHE.GET", ns, prompt)

    def pcache_getex(self, ns: ValueT, prompt: ValueT) -> Optional[Tuple[Any, Any, int]]:
        """Like `pcache_get()`, also returning `(response, metadata, hit_count)`."""
        return self.execute_command("PCACHE.GETEX", ns, prompt)

    def pcache_del(self, ns: ValueT, prompt: ValueT) -> bool:
        """Delete the cached entry for `prompt`."""
        return self.execute_command("PCACHE.DEL", ns, prompt)

    def pcache_stats(self, ns: ValueT) -> Dict[str, Any]:
        """Return entry/hit/miss/eviction/latency/cost statistics for `ns`,
        including the `usage_saved_*` savings counters."""
        return self.execute_command("PCACHE.STATS", ns)

    def pcache_flush(self, ns: ValueT) -> int:
        """Remove every entry in `ns`, returning the number flushed."""
        return self.execute_command("PCACHE.FLUSH", ns)

    def pcache_drop(self, ns: ValueT) -> bool:
        """Delete prompt cache namespace `ns`."""
        return self.execute_command("PCACHE.DROP", ns)

    def pcache_list(self) -> List[bytes]:
        """List prompt cache namespace names."""
        return self.execute_command("PCACHE.LIST")

    def pcache_config_set(self, ns: ValueT, parameter: str, value: ValueT) -> bool:
        """Update one config parameter ("MAX_ENTRIES"/"TTL"/"COST_PER_HIT") on `ns`."""
        return self.execute_command("PCACHE.CONFIG", ns, "SET", parameter.upper(), value)

    # -- LLMSTATS.* -----------------------------------------------------------

    def llmstats_bucket(self, name: ValueT) -> bool:
        """Idempotently create LLM stats bucket `name`."""
        return self.execute_command("LLMSTATS.BUCKET", name)

    def llmstats_record(
        self,
        bucket: ValueT,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        cost_usd: float,
        latency_ms: float,
        *,
        error: Optional[bool] = None,
        model: Optional[str] = None,
        cache_write_5m_tokens: Optional[int] = None,
        cache_write_1h_tokens: Optional[int] = None,
    ) -> bool:
        """Record one LLM call's token usage, cost and latency into `bucket`.

        `cached_tokens` is cache-READ tokens. With `model`, the server
        prices the call itself from its `LLMSTATS.PRICE` table and ignores
        `cost_usd` (pass `0.0`); `latency_ms` is sent as whole milliseconds
        (rounded); an unpriced model still records tokens,
        costs `0` and counts toward `unpriced_requests`. For OpenAI, pass
        `prompt_tokens - cached_tokens` as `input_tokens` (OpenAI reports
        cache hits as part of `prompt_tokens`, unlike Anthropic).
        """
        # The server takes the latency as whole milliseconds.
        latency = max(0, int(round(latency_ms)))
        args: List[Any] = [bucket, input_tokens, output_tokens, cached_tokens, format_cost(cost_usd), latency]
        if error is not None:
            args += ["ERROR", int(error)]
        if model is not None:
            args += ["MODEL", model]
        if cache_write_5m_tokens is not None:
            args += ["CACHEWRITE5M", cache_write_5m_tokens]
        if cache_write_1h_tokens is not None:
            args += ["CACHEWRITE1H", cache_write_1h_tokens]
        return self.execute_command("LLMSTATS.RECORD", *args)

    def llmstats_get(self, bucket: ValueT) -> Dict[str, Any]:
        """Return request/token/cost/latency stats for `bucket`, including a
        trailing 1-minute rolling window and `unpriced_requests`."""
        return self.execute_command("LLMSTATS.GET", bucket)

    def llmstats_reset(self, bucket: ValueT) -> bool:
        """Reset counters for `bucket`."""
        return self.execute_command("LLMSTATS.RESET", bucket)

    def llmstats_drop(self, bucket: ValueT) -> bool:
        """Delete stats bucket `bucket`."""
        return self.execute_command("LLMSTATS.DROP", bucket)

    def llmstats_list(self) -> List[bytes]:
        """List stats bucket names."""
        return self.execute_command("LLMSTATS.LIST")

    # -- LLMSTATS.PRICE (per model price table) -------------------------------

    def llmstats_price_set(
        self,
        model: ValueT,
        input_usd_per_mtok: Any,
        output_usd_per_mtok: Any,
        *,
        cache_read_usd_per_mtok: Optional[Any] = None,
        cache_write_5m_usd_per_mtok: Optional[Any] = None,
        cache_write_1h_usd_per_mtok: Optional[Any] = None,
    ) -> bool:
        """Set a custom USD-per-1,000,000-token price for `model`, used by
        `llmstats_record(..., model=...)` and `llmbudget_consume(...,
        usage=...)`. Prices may be a number or a decimal string; omitted
        cache prices default to 0.1x/1.25x/2x `input_usd_per_mtok`.
        """
        args: List[Any] = [
            model,
            "INPUT",
            _format_price(input_usd_per_mtok),
            "OUTPUT",
            _format_price(output_usd_per_mtok),
        ]
        if cache_read_usd_per_mtok is not None:
            args += ["CACHEREAD", _format_price(cache_read_usd_per_mtok)]
        if cache_write_5m_usd_per_mtok is not None:
            args += ["CACHEWRITE5M", _format_price(cache_write_5m_usd_per_mtok)]
        if cache_write_1h_usd_per_mtok is not None:
            args += ["CACHEWRITE1H", _format_price(cache_write_1h_usd_per_mtok)]
        return self.execute_command("LLMSTATS.PRICE SET", *args)

    def llmstats_price_get(self, model: ValueT) -> LLMPrice:
        """Return `model`'s price table entry. An unknown model comes back
        with `source="unpriced"` and every price `0.0`, not an error."""
        return self.execute_command("LLMSTATS.PRICE GET", model)

    def llmstats_price_list(self) -> List[LLMPrice]:
        """List every known model's price table entry (built in and custom)."""
        return self.execute_command("LLMSTATS.PRICE LIST")

    def llmstats_price_del(self, model: ValueT) -> bool:
        """Remove `model`'s custom price override, if any (`False` if there
        was none; the model then falls back to its built-in price, if any)."""
        return self.execute_command("LLMSTATS.PRICE DEL", model)

    # -- LLMBUDGET.* ----------------------------------------------------------

    def llmbudget_set(
        self, name: ValueT, token_limit: int, cost_limit_usd: float, window_seconds: int
    ) -> bool:
        """Create or update budget `name`. `cost_limit_usd` is written as a
        fixed-point decimal of at most 6 fractional digits (rounded half up);
        `0` means no cost limit. A negative, NaN or infinite limit raises
        `ValueError`."""
        return self.execute_command(
            "LLMBUDGET.SET", name, token_limit, format_usd(cost_limit_usd), window_seconds
        )

    def llmbudget_consume(
        self,
        name: ValueT,
        tokens: int = 0,
        cost_usd: float = 0.0,
        *,
        usage: Optional[TokenUsage] = None,
    ) -> bool:
        """Try to consume `tokens`/`cost_usd` from budget `name`.

        Returns `True` if allowed, `False` if it would exceed the cap. Pass
        `usage` instead to have the server compute both from its
        `LLMSTATS.PRICE` table (the server then ignores `tokens`/`cost_usd`,
        so leave them at their defaults). `cost_usd` must be finite and
        not negative, else `ValueError`.
        """
        args: List[Any] = [name, tokens, format_cost(cost_usd)]
        args += _usage_args(usage)
        return self.execute_command("LLMBUDGET.CONSUME", *args)

    def llmbudget_status(self, name: ValueT) -> Dict[str, Any]:
        """Return usage, limits, the current window's remaining time and
        `unpriced_requests` for `name`."""
        return self.execute_command("LLMBUDGET.STATUS", name)

    def llmbudget_reset(self, name: ValueT) -> bool:
        """Reset budget `name`'s usage counters."""
        return self.execute_command("LLMBUDGET.RESET", name)

    def llmbudget_drop(self, name: ValueT) -> bool:
        """Delete budget `name`."""
        return self.execute_command("LLMBUDGET.DROP", name)

    def llmbudget_list(self) -> List[bytes]:
        """List budget names."""
        return self.execute_command("LLMBUDGET.LIST")

    # -- PROMPT.* (versioned prompt/template registry) --------------------------

    def prompt_put(self, name: ValueT, content: ValueT, *, meta: Optional[str] = None) -> int:
        """Store a new version of prompt template `name`, returning its version number."""
        args: List[Any] = [name, content]
        if meta is not None:
            args += ["META", meta]
        return self.execute_command("PROMPT.PUT", *args)

    def prompt_get(
        self, name: ValueT, version: Optional[int] = None
    ) -> Optional[Tuple[int, Any, Any, int]]:
        """Return `(version, content, metadata, created_unix)`; omit `version`
        (or pass `0`) for the latest."""
        args: List[Any] = [name] if version is None else [name, version]
        return self.execute_command("PROMPT.GET", *args)

    def prompt_latest(self, name: ValueT) -> Optional[Tuple[int, Any, Any, int]]:
        """Alias for `prompt_get(name)`."""
        return self.execute_command("PROMPT.LATEST", name)

    def prompt_setactive(self, name: ValueT, version: int) -> bool:
        """Mark `version` as the active version of `name`."""
        return self.execute_command("PROMPT.SETACTIVE", name, version)

    def prompt_versions(self, name: ValueT) -> List[int]:
        """List every stored version number of `name`."""
        return self.execute_command("PROMPT.VERSIONS", name)

    def prompt_drop(self, name: ValueT) -> bool:
        """Delete prompt template `name` and all its versions."""
        return self.execute_command("PROMPT.DROP", name)

    def prompt_list(self) -> List[bytes]:
        """List prompt template names."""
        return self.execute_command("PROMPT.LIST")


for _prefix, _python_name in _KEYED_CACHE_FAMILIES:
    for _method_name, _method in _keyed_cache_methods(_prefix, _python_name).items():
        setattr(AICacheCommands, _method_name, _method)
del _prefix, _python_name, _method_name, _method
