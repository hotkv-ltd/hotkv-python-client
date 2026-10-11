# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Sorted set commands."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from .._typing import KeyT, KeysT, ValueT
from ._shared import CallbackMap, ReadonlySet, as_float, as_int, flat_to_pairs

__all__ = ["SortedSetCommands", "CALLBACKS", "READONLY"]


def _cb_score(reply: Any, **_options: Any) -> Optional[float]:
    return as_float(reply)


def _cb_scores(reply: Any, **_options: Any) -> List[Optional[float]]:
    return [as_float(v) for v in (reply or [])]


def _cb_withscores(reply: Any, withscores: bool = False, **_options: Any) -> Any:
    if withscores and isinstance(reply, list):
        return [(member, as_float(score)) for member, score in flat_to_pairs(reply)]
    return reply


def _cb_pop(reply: Any, **_options: Any) -> List[Tuple[Any, float]]:
    return [(member, as_float(score)) for member, score in flat_to_pairs(reply)]


def _cb_bzpop(reply: Any, **_options: Any) -> Optional[Tuple[Any, Any, float]]:
    if reply is None:
        return None
    key, member, score = reply
    return (key, member, as_float(score))


def _cb_zadd(reply: Any, incr: bool = False, **_options: Any) -> Union[Optional[int], Optional[float]]:
    # Plain ZADD replies with the count of newly added (or, with CH,
    # added-plus-changed) members. With INCR it instead replies with the
    # member's new score (a native RESP3 double, or a bulk string under
    # RESP2), or nil if NX/XX aborted the update.
    return as_float(reply) if incr else as_int(reply)


def _cb_rank_withscore(
    reply: Any, withscore: bool = False, **_options: Any
) -> Union[Optional[int], Optional[Tuple[int, float]]]:
    if not withscore:
        return as_int(reply)
    if reply is None:
        return None
    rank, score = reply
    return (as_int(rank), as_float(score))


CALLBACKS: CallbackMap = {
    "ZADD": _cb_zadd,
    "ZSCORE": _cb_score,
    "ZINCRBY": _cb_score,
    "ZMSCORE": _cb_scores,
    "ZRANK": _cb_rank_withscore,
    "ZREVRANK": _cb_rank_withscore,
    "ZRANGE": _cb_withscores,
    "ZRANGEBYSCORE": _cb_withscores,
    "ZREVRANGE": _cb_withscores,
    "ZREVRANGEBYSCORE": _cb_withscores,
    "ZUNION": _cb_withscores,
    "ZINTER": _cb_withscores,
    "ZDIFF": _cb_withscores,
    "ZRANDMEMBER": _cb_withscores,
    "ZPOPMIN": _cb_pop,
    "ZPOPMAX": _cb_pop,
    "BZPOPMIN": _cb_bzpop,
    "BZPOPMAX": _cb_bzpop,
}

READONLY: ReadonlySet = frozenset(
    {
        "ZSCORE",
        "ZMSCORE",
        "ZCARD",
        "ZCOUNT",
        "ZRANK",
        "ZREVRANK",
        "ZRANGE",
        "ZRANGEBYSCORE",
        "ZREVRANGE",
        "ZREVRANGEBYSCORE",
        "ZRANGEBYLEX",
        "ZREVRANGEBYLEX",
        "ZLEXCOUNT",
        "ZRANDMEMBER",
        "ZUNION",
        "ZINTER",
        "ZDIFF",
        "ZINTERCARD",
    }
)


class SortedSetCommands:
    """Typed wrappers for sorted set commands."""

    def zadd(
        self,
        key: KeyT,
        mapping: Mapping[ValueT, float],
        *,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
        ch: bool = False,
        incr: bool = False,
    ) -> Union[Optional[int], Optional[float]]:
        """Add or update `{member: score}` pairs.

        Returns the number of newly added members (or, with `ch=True`,
        newly added plus changed members). `nx`/`xx` only add a new
        member / only update an existing one; `gt`/`lt` only update when
        the new score is greater/less than the current one. With
        `incr=True` (and exactly one `mapping` entry), increments that
        member's score instead of setting it, returning the new score, or
        `None` if `nx`/`xx`/`gt`/`lt` aborted the update. `nx` is mutually
        exclusive with `xx`, `gt` and `lt`; `gt` and `lt` are mutually
        exclusive with each other (the server rejects an invalid
        combination with a normal `ResponseError`).
        """
        args: List[Any] = [key]
        if nx:
            args.append("NX")
        elif xx:
            args.append("XX")
        if gt:
            args.append("GT")
        elif lt:
            args.append("LT")
        if ch:
            args.append("CH")
        if incr:
            args.append("INCR")
        for member, score in mapping.items():
            args += [score, member]
        return self.execute_command("ZADD", *args, incr=incr)

    def zrem(self, key: KeyT, *members: ValueT) -> int:
        """Remove `members` from the sorted set at `key`."""
        return self.execute_command("ZREM", key, *members)

    def zscore(self, key: KeyT, member: ValueT) -> Optional[float]:
        """Return the score of `member`, or `None`."""
        return self.execute_command("ZSCORE", key, member)

    def zmscore(self, key: KeyT, members: Sequence[ValueT]) -> List[Optional[float]]:
        """Return the scores of several members, `None` for any that are unset."""
        return self.execute_command("ZMSCORE", key, *members)

    def zcard(self, key: KeyT) -> int:
        """Number of members in the sorted set at `key`."""
        return self.execute_command("ZCARD", key)

    def zcount(self, key: KeyT, min_score: ValueT, max_score: ValueT) -> int:
        """Count members with a score between `min_score` and `max_score`."""
        return self.execute_command("ZCOUNT", key, min_score, max_score)

    def zincrby(self, key: KeyT, amount: float, member: ValueT) -> float:
        """Increment `member`'s score by `amount`, returning the new score."""
        return self.execute_command("ZINCRBY", key, amount, member)

    def zrank(
        self, key: KeyT, member: ValueT, withscore: bool = False
    ) -> Union[Optional[int], Optional[Tuple[int, float]]]:
        """Rank of `member` (ascending, 0-based), or `None`.

        With `withscore=True`, returns `(rank, score)` instead, or `None`
        if `member` is not in the set.
        """
        args: List[Any] = [key, member]
        if withscore:
            args.append("WITHSCORE")
        return self.execute_command("ZRANK", *args, withscore=withscore)

    def zrevrank(
        self, key: KeyT, member: ValueT, withscore: bool = False
    ) -> Union[Optional[int], Optional[Tuple[int, float]]]:
        """Rank of `member` (descending, 0-based), or `None`; see `zrank()`
        for `withscore`."""
        args: List[Any] = [key, member]
        if withscore:
            args.append("WITHSCORE")
        return self.execute_command("ZREVRANK", *args, withscore=withscore)

    def zrange(
        self,
        key: KeyT,
        start: ValueT,
        stop: ValueT,
        *,
        byscore: bool = False,
        bylex: bool = False,
        rev: bool = False,
        offset: Optional[int] = None,
        count: Optional[int] = None,
        withscores: bool = False,
    ) -> Any:
        """Members between `start` and `stop`.

        By default `start`/`stop` are 0-based rank indexes (may be
        negative). `byscore=True` treats them as score bounds (a bound may
        be prefixed with `"("` for exclusive, or be `"-inf"`/`"+inf"`);
        `bylex=True` treats them as lexicographic bounds (`"-"`, `"+"`,
        `"[member"`, `"(member"`), meaningful only when every member
        shares one score. `rev=True` reverses the ordering; with
        `byscore`, pass `start` as the higher bound and `stop` as the
        lower one, as with `zrevrangebyscore()`. `offset`/`count` send a
        `LIMIT` clause and require `byscore` or `bylex`.
        """
        args: List[Any] = [key, start, stop]
        if byscore:
            args.append("BYSCORE")
        if bylex:
            args.append("BYLEX")
        if rev:
            args.append("REV")
        if offset is not None or count is not None:
            args += ["LIMIT", offset or 0, count if count is not None else -1]
        if withscores:
            args.append("WITHSCORES")
        return self.execute_command("ZRANGE", *args, withscores=withscores)

    def zrangestore(
        self,
        destination: KeyT,
        source: KeyT,
        start: ValueT,
        stop: ValueT,
        *,
        byscore: bool = False,
        bylex: bool = False,
        rev: bool = False,
        offset: Optional[int] = None,
        count: Optional[int] = None,
    ) -> int:
        """Like `zrange()`, storing the result into `destination` instead
        of returning it. Returns the size of the stored result."""
        args: List[Any] = [destination, source, start, stop]
        if byscore:
            args.append("BYSCORE")
        if bylex:
            args.append("BYLEX")
        if rev:
            args.append("REV")
        if offset is not None or count is not None:
            args += ["LIMIT", offset or 0, count if count is not None else -1]
        return self.execute_command("ZRANGESTORE", *args)

    def zrangebyscore(
        self,
        key: KeyT,
        min_score: ValueT,
        max_score: ValueT,
        *,
        offset: Optional[int] = None,
        count: Optional[int] = None,
        withscores: bool = False,
    ) -> Any:
        """Members with scores between `min_score` and `max_score` (ascending)."""
        args: List[Any] = [key, min_score, max_score]
        if withscores:
            args.append("WITHSCORES")
        if offset is not None or count is not None:
            args += ["LIMIT", offset or 0, count if count is not None else -1]
        return self.execute_command("ZRANGEBYSCORE", *args, withscores=withscores)

    def zrevrangebyscore(
        self,
        key: KeyT,
        max_score: ValueT,
        min_score: ValueT,
        *,
        offset: Optional[int] = None,
        count: Optional[int] = None,
        withscores: bool = False,
    ) -> Any:
        """Members with scores between `min_score` and `max_score` (descending)."""
        args: List[Any] = [key, max_score, min_score]
        if withscores:
            args.append("WITHSCORES")
        if offset is not None or count is not None:
            args += ["LIMIT", offset or 0, count if count is not None else -1]
        return self.execute_command("ZREVRANGEBYSCORE", *args, withscores=withscores)

    def zrevrange(
        self, key: KeyT, start: int, stop: int, *, withscores: bool = False
    ) -> Any:
        """Members ranked `start`..`stop` (descending by score)."""
        args: List[Any] = [key, start, stop]
        if withscores:
            args.append("WITHSCORES")
        return self.execute_command("ZREVRANGE", *args, withscores=withscores)

    def zrangebylex(
        self,
        key: KeyT,
        min_lex: ValueT,
        max_lex: ValueT,
        *,
        offset: Optional[int] = None,
        count: Optional[int] = None,
    ) -> List[bytes]:
        """Members between `min_lex` and `max_lex` in lexicographic order
        (all members must share the same score for this to be meaningful)."""
        args: List[Any] = [key, min_lex, max_lex]
        if offset is not None or count is not None:
            args += ["LIMIT", offset or 0, count if count is not None else -1]
        return self.execute_command("ZRANGEBYLEX", *args)

    def zrevrangebylex(
        self,
        key: KeyT,
        max_lex: ValueT,
        min_lex: ValueT,
        *,
        offset: Optional[int] = None,
        count: Optional[int] = None,
    ) -> List[bytes]:
        """Like `zrangebylex()`, in reverse order."""
        args: List[Any] = [key, max_lex, min_lex]
        if offset is not None or count is not None:
            args += ["LIMIT", offset or 0, count if count is not None else -1]
        return self.execute_command("ZREVRANGEBYLEX", *args)

    def zlexcount(self, key: KeyT, min_lex: ValueT, max_lex: ValueT) -> int:
        """Count members between `min_lex` and `max_lex`."""
        return self.execute_command("ZLEXCOUNT", key, min_lex, max_lex)

    def zpopmin(self, key: KeyT, count: Optional[int] = None) -> List[Tuple[Any, float]]:
        """Remove and return up to `count` (default 1) lowest-scoring members."""
        args: List[Any] = [key] if count is None else [key, count]
        return self.execute_command("ZPOPMIN", *args)

    def zpopmax(self, key: KeyT, count: Optional[int] = None) -> List[Tuple[Any, float]]:
        """Remove and return up to `count` (default 1) highest-scoring members."""
        args: List[Any] = [key] if count is None else [key, count]
        return self.execute_command("ZPOPMAX", *args)

    def bzpopmin(self, keys: Any, timeout: float) -> Optional[Tuple[Any, Any, float]]:
        """Blocking `zpopmin()` across several keys: `(key, member, score)`."""
        if isinstance(keys, (str, bytes)):
            keys = [keys]
        return self.execute_command("BZPOPMIN", *keys, timeout)

    def bzpopmax(self, keys: Any, timeout: float) -> Optional[Tuple[Any, Any, float]]:
        """Blocking `zpopmax()` across several keys: `(key, member, score)`."""
        if isinstance(keys, (str, bytes)):
            keys = [keys]
        return self.execute_command("BZPOPMAX", *keys, timeout)

    def zrandmember(
        self, key: KeyT, count: Optional[int] = None, *, withscores: bool = False
    ) -> Any:
        """Return one random member, or `count` members (negative allows repeats)."""
        args: List[Any] = [key]
        if count is not None:
            args.append(count)
            if withscores:
                args.append("WITHSCORES")
        return self.execute_command("ZRANDMEMBER", *args, withscores=withscores)

    def zremrangebyscore(self, key: KeyT, min_score: ValueT, max_score: ValueT) -> int:
        """Remove members with a score between `min_score` and `max_score`."""
        return self.execute_command("ZREMRANGEBYSCORE", key, min_score, max_score)

    def zremrangebyrank(self, key: KeyT, start: int, stop: int) -> int:
        """Remove members ranked between `start` and `stop`."""
        return self.execute_command("ZREMRANGEBYRANK", key, start, stop)

    def zremrangebylex(self, key: KeyT, min_lex: ValueT, max_lex: ValueT) -> int:
        """Remove members between `min_lex` and `max_lex`."""
        return self.execute_command("ZREMRANGEBYLEX", key, min_lex, max_lex)

    def zunionstore(
        self,
        destination: KeyT,
        keys: KeysT,
        *,
        weights: Optional[Sequence[float]] = None,
        aggregate: Optional[str] = None,
    ) -> int:
        """Store the union of `keys` into `destination`, returning its size.

        `weights` multiplies each key's scores before combining them (one
        weight per key, default `1` each); `aggregate` is `"SUM"`
        (default), `"MIN"` or `"MAX"`.
        """
        keys = list(keys)
        args: List[Any] = [destination, len(keys), *keys]
        if weights:
            args += ["WEIGHTS", *weights]
        if aggregate:
            args += ["AGGREGATE", aggregate.upper()]
        return self.execute_command("ZUNIONSTORE", *args)

    def zinterstore(
        self,
        destination: KeyT,
        keys: KeysT,
        *,
        weights: Optional[Sequence[float]] = None,
        aggregate: Optional[str] = None,
    ) -> int:
        """Store the intersection of `keys` into `destination`; see
        `zunionstore()` for `weights`/`aggregate`."""
        keys = list(keys)
        args: List[Any] = [destination, len(keys), *keys]
        if weights:
            args += ["WEIGHTS", *weights]
        if aggregate:
            args += ["AGGREGATE", aggregate.upper()]
        return self.execute_command("ZINTERSTORE", *args)

    def zdiffstore(self, destination: KeyT, keys: KeysT) -> int:
        """Store the difference of `keys` into `destination`."""
        keys = list(keys)
        return self.execute_command("ZDIFFSTORE", destination, len(keys), *keys)

    def zunion(self, keys: KeysT, *, withscores: bool = False) -> Any:
        """Union of `keys` without storing it."""
        keys = list(keys)
        args: List[Any] = [len(keys), *keys]
        if withscores:
            args.append("WITHSCORES")
        return self.execute_command("ZUNION", *args, withscores=withscores)

    def zinter(self, keys: KeysT, *, withscores: bool = False) -> Any:
        """Intersection of `keys` without storing it."""
        keys = list(keys)
        args: List[Any] = [len(keys), *keys]
        if withscores:
            args.append("WITHSCORES")
        return self.execute_command("ZINTER", *args, withscores=withscores)

    def zdiff(self, keys: KeysT, *, withscores: bool = False) -> Any:
        """Difference of `keys` without storing it."""
        keys = list(keys)
        args: List[Any] = [len(keys), *keys]
        if withscores:
            args.append("WITHSCORES")
        return self.execute_command("ZDIFF", *args, withscores=withscores)

    def zintercard(self, keys: KeysT, *, limit: Optional[int] = None) -> int:
        """Cardinality of the intersection of `keys`, without materializing it."""
        keys = list(keys)
        args: List[Any] = [len(keys), *keys]
        if limit is not None:
            args += ["LIMIT", limit]
        return self.execute_command("ZINTERCARD", *args)
