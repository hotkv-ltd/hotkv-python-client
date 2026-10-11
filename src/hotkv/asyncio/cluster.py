# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Asyncio cluster client; see `hotkv.cluster.ClusterHotKV` for the routing
rules, the failover and reshard handling, the fan-out of per-node commands
and the scope (single commands only, no cross-node pipelines/transactions or
pub/sub), which this mirrors with `await` added at each network touch point.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple

from .. import commands
from .._cluster_common import (
    Address,
    command_name,
    fanout_kind,
    is_permanent_failure,
    merge_replies,
    parse_slots,
    routing_key,
)
from .._crc16 import TOTAL_SLOTS, key_slot
from .._protocol import encode_arg
from ..exceptions import (
    AskError,
    ClusterDownError,
    ClusterError,
    ConnectionError,
    MovedError,
    NoScriptError,
    ProtocolError,
    ResponseError,
    TimeoutError,
    TryAgainError,
)
from ._pool import AsyncConnectionPool

__all__ = ["ClusterHotKV"]

_RETRY_EXCEPTIONS = (ConnectionError, TimeoutError, ProtocolError)
_REFRESH_RETRY_EXCEPTIONS = _RETRY_EXCEPTIONS + (ResponseError,)
_TRANSIENT_CODES = frozenset({"CLUSTERDOWN", "TRYAGAIN", "LOADING", "MASTERDOWN", "READONLY"})
_SUSPECT_SECONDS = 30.0


class _StalePrimaries(Exception):
    """The set of primaries changed under a fanned-out command."""


class ClusterHotKV(commands.CoreCommands):
    """Slot-aware asyncio HotKV cluster client.

    `retry_timeout` and `retry_writes` are as for the sync `ClusterHotKV`,
    including its default of delivering a write at least once: see there for
    the commands a repeated write changes.
    """

    RESPONSE_CALLBACKS = commands.RESPONSE_CALLBACKS
    READONLY_COMMANDS = commands.READONLY_COMMANDS

    def __init__(
        self,
        startup_nodes: Iterable[Address],
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_name: Optional[str] = None,
        connect_timeout: Optional[float] = 5.0,
        command_timeout: Optional[float] = None,
        tls: bool = False,
        tls_ca_file: Optional[str] = None,
        tls_certfile: Optional[str] = None,
        tls_keyfile: Optional[str] = None,
        tls_server_hostname: Optional[str] = None,
        tls_insecure_skip_verify: bool = False,
        resp3: bool = True,
        decode_responses: bool = False,
        encoding: str = "utf-8",
        encoding_errors: str = "strict",
        max_connections_per_node: int = 16,
        read_from_replicas: bool = False,
        max_redirects: int = 16,
        retry_timeout: float = 10.0,
        retry_writes: bool = True,
    ) -> None:
        self._startup_nodes: List[Address] = [(h, int(p)) for h, p in startup_nodes]
        if not self._startup_nodes:
            raise ValueError("ClusterHotKV needs at least one startup node")
        self._node_kwargs: Dict[str, Any] = dict(
            username=username,
            password=password,
            client_name=client_name,
            connect_timeout=connect_timeout,
            command_timeout=command_timeout,
            tls=tls,
            tls_ca_file=tls_ca_file,
            tls_certfile=tls_certfile,
            tls_keyfile=tls_keyfile,
            tls_server_hostname=tls_server_hostname,
            tls_insecure_skip_verify=tls_insecure_skip_verify,
            resp3=resp3,
            decode_responses=decode_responses,
            encoding=encoding,
            encoding_errors=encoding_errors,
        )
        self._max_connections_per_node = max_connections_per_node
        self.read_from_replicas = read_from_replicas
        self.max_redirects = max_redirects
        self.retry_timeout = retry_timeout
        self.retry_writes = retry_writes

        self._pools: Dict[Tuple[Address, bool], AsyncConnectionPool] = {}
        self._slot_masters: List[Optional[Address]] = [None] * TOTAL_SLOTS
        self._slot_replicas: List[List[Address]] = [[] for _ in range(TOTAL_SLOTS)]
        # When each node last failed to answer; see the sync client.
        self._suspect: Dict[Address, float] = {}
        self._ready = False

    async def _ensure_ready(self) -> None:
        if not self._ready:
            await self.refresh_slots()
            self._ready = True

    # -- slot map --------------------------------------------------------------

    async def refresh_slots(self) -> None:
        """Rebuild the slot map from `CLUSTER SLOTS`, trying each known node."""
        last_error: Optional[BaseException] = None
        for address in self._known_addresses():
            pool = self._pool_for(address)
            try:
                async with pool.connection() as conn:
                    await conn.send_command("CLUSTER", "SLOTS")
                    reply = await conn.read_response()
                if isinstance(reply, ResponseError):
                    raise reply
                self._apply_slots(reply, address[0])
                return
            except _REFRESH_RETRY_EXCEPTIONS as exc:
                last_error = exc
                self._suspect[address] = time.monotonic()
                continue
        raise ClusterError(f"could not load the cluster slot map from any node: {last_error}")

    def _known_addresses(self) -> List[Address]:
        seen = set()
        ordered: List[Address] = []
        candidates = list(self._startup_nodes) + [a for (a, _ro) in self._pools.keys()]
        candidates += [a for a in self._slot_masters if a is not None]
        for address in candidates:
            if address not in seen:
                seen.add(address)
                ordered.append(address)
        now = time.monotonic()
        ordered.sort(key=lambda a: now - self._suspect.get(a, -1e9) < _SUSPECT_SECONDS)
        return ordered

    def _apply_slots(self, reply: Any, asked_host: str = "") -> None:
        self._slot_masters, self._slot_replicas = parse_slots(reply, asked_host)

    def _pool_for(self, address: Address, readonly: bool = False) -> AsyncConnectionPool:
        pool = self._pools.get((address, readonly))
        if pool is None:
            kwargs = dict(self._node_kwargs, host=address[0], port=address[1], readonly=readonly)
            pool = AsyncConnectionPool(kwargs, max_connections=self._max_connections_per_node)
            self._pools[(address, readonly)] = pool
        return pool

    async def _pick(
        self, args: Tuple[Any, ...], *, readonly: bool, avoid_replicas: bool = False
    ) -> Tuple[Address, bool]:
        key = routing_key(command_name(args[0]), args)
        if key is None:
            return self._any_address(), False
        slot = key_slot(key if isinstance(key, (str, bytes)) else encode_arg(key))
        master = self._slot_masters[slot]
        replicas = list(self._slot_replicas[slot])
        if master is None:
            await self.refresh_slots()
            master = self._slot_masters[slot]
        if master is None:
            raise ClusterError(f"no node is known to own slot {slot}")
        if readonly and self.read_from_replicas and replicas and not avoid_replicas:
            return random.choice(replicas), True
        return master, False

    async def _target_address(self, args: Tuple[Any, ...], *, readonly: bool) -> Address:
        address, _replica = await self._pick(args, readonly=readonly)
        return address

    def _any_address(self) -> Address:
        for address in self._slot_masters:
            if address is not None:
                return address
        return self._startup_nodes[0]

    # -- command execution --------------------------------------------------------

    async def _round_trip(
        self, address: Address, args: Tuple[Any, ...], *, asking: bool, replica: bool
    ) -> Any:
        """One command on one node; see the sync client."""
        pool = self._pool_for(address, readonly=replica)
        conn = await pool.get_connection()
        sent = False
        try:
            await conn.connect()
            if asking:
                await conn.send_command("ASKING")
                ack = await conn.read_response()
                if isinstance(ack, ResponseError):
                    raise ack
            sent = True
            await conn.send_command(*args)
            reply = await conn.read_response()
        except _RETRY_EXCEPTIONS as exc:
            await pool.discard(conn)
            self._suspect[address] = time.monotonic()
            exc.sent = sent  # type: ignore[attr-defined]
            raise
        except BaseException:
            await pool.release(conn)
            raise
        await pool.release(conn)
        return reply

    def _may_retry(self, exc: BaseException, readonly: bool) -> bool:
        # A refused password or certificate stays refused: retrying would only
        # wait out `retry_timeout` and hide the cause behind a ClusterError.
        if is_permanent_failure(exc):
            return False
        if readonly or self.retry_writes:
            return True
        return not getattr(exc, "sent", True)

    @staticmethod
    async def _backoff(attempt: int, deadline: float) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.05 * (2 ** min(attempt, 4)), 0.5, remaining))
        return True

    async def _refresh_quietly(self) -> None:
        try:
            await self.refresh_slots()
        except ClusterError:
            pass

    async def _send(self, args: Tuple[Any, ...], *, readonly: bool) -> Any:
        name = command_name(args[0])
        deadline = time.monotonic() + self.retry_timeout
        redirects = 0
        attempt = 0
        ask_target: Optional[Address] = None
        avoid_replicas = False
        last_error: Optional[BaseException] = None
        while True:
            replica = False
            if ask_target is not None:
                address, asking, ask_target = ask_target, True, None
            else:
                asking = False
                try:
                    address, replica = await self._pick(
                        args, readonly=readonly, avoid_replicas=avoid_replicas
                    )
                except ClusterError as exc:
                    last_error = exc
                    if not await self._backoff(attempt, deadline):
                        raise
                    attempt += 1
                    await self._refresh_quietly()
                    continue
            try:
                reply = await self._round_trip(address, args, asking=asking, replica=replica)
            except _RETRY_EXCEPTIONS as exc:
                last_error = exc
                if not self._may_retry(exc, readonly):
                    raise
                if not await self._backoff(attempt, deadline):
                    raise ClusterError(
                        f"giving up on {name!r} after {self.retry_timeout}s of retries: {exc}"
                    ) from exc
                attempt += 1
                avoid_replicas = avoid_replicas or replica
                await self._refresh_quietly()
                continue
            if isinstance(reply, MovedError):
                redirects += 1
                if redirects > self.max_redirects:
                    break
                last_error = reply
                await self._refresh_quietly()
                self._note_moved(args, (reply.host or address[0], reply.port))
                avoid_replicas = True
                continue
            if isinstance(reply, AskError):
                redirects += 1
                if redirects > self.max_redirects:
                    break
                last_error = reply
                ask_target = (reply.host or address[0], reply.port)
                continue
            if self._is_transient(reply):
                last_error = reply
                if not await self._backoff(attempt, deadline):
                    raise reply
                attempt += 1
                avoid_replicas = avoid_replicas or replica
                await self._refresh_quietly()
                continue
            return reply
        raise ClusterError(
            f"too many redirects ({self.max_redirects}) executing {name!r}: {last_error}"
        )

    @staticmethod
    def _is_transient(reply: Any) -> bool:
        return isinstance(reply, ResponseError) and (
            isinstance(reply, (ClusterDownError, TryAgainError)) or reply.code in _TRANSIENT_CODES
        )

    def _note_moved(self, args: Tuple[Any, ...], target: Address) -> None:
        key = routing_key(command_name(args[0]), args)
        if key is None:
            return
        slot = key_slot(key if isinstance(key, (str, bytes)) else encode_arg(key))
        self._slot_masters[slot] = target

    async def _send_to_primaries(self, args: Tuple[Any, ...]) -> List[Any]:
        deadline = time.monotonic() + self.retry_timeout
        attempt = 0
        while True:
            try:
                replies = []
                for address in await self._primaries():
                    reply = await self._round_trip(address, args, asking=False, replica=False)
                    if isinstance(reply, MovedError) or self._is_transient(reply):
                        raise _StalePrimaries(str(reply))
                    replies.append(reply)
                return replies
            except (_StalePrimaries,) + _RETRY_EXCEPTIONS as exc:
                if isinstance(exc, _RETRY_EXCEPTIONS) and is_permanent_failure(exc):
                    raise
                if (
                    not self.retry_writes
                    and isinstance(exc, _RETRY_EXCEPTIONS)
                    and getattr(exc, "sent", True)
                ):
                    raise
                if not await self._backoff(attempt, deadline):
                    raise ClusterError(f"giving up on {command_name(args[0])!r}: {exc}") from exc
                attempt += 1
                await self._refresh_quietly()

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        await self._ensure_ready()
        name = command_name(args[0])
        kind = fanout_kind(name, args)
        if kind is not None:
            replies = await self._send_to_primaries(args)
            for reply in replies:
                if isinstance(reply, ResponseError):
                    raise reply
            reply = merge_replies(kind, replies)
        else:
            reply = await self._send(args, readonly=name in self.READONLY_COMMANDS)
            if isinstance(reply, ResponseError):
                raise reply
        callback = self.RESPONSE_CALLBACKS.get(args[0])
        return callback(reply, **options) if callback is not None else reply

    async def _primaries(self) -> List[Address]:
        masters = self.masters
        if not masters:
            await self.refresh_slots()
            masters = self.masters
        if not masters:
            raise ClusterError("the cluster has no primary with slots")
        return masters

    async def execute(self, *args: Any, **options: Any) -> Any:
        """Escape hatch: send any command, typed wrapper or not."""
        return await self.execute_command(*args, **options)

    command = execute

    async def execute_on_node(self, address: Address, *args: Any, **options: Any) -> Any:
        """Send one command to one specific node, with no routing and no
        redirect following; see the sync client."""
        pool = self._pool_for((address[0], int(address[1])))
        async with pool.connection() as conn:
            await conn.send_command(*args)
            reply = await conn.read_response()
        if isinstance(reply, ResponseError):
            raise reply
        callback = self.RESPONSE_CALLBACKS.get(args[0])
        return callback(reply, **options) if callback is not None else reply

    async def execute_on_primaries(self, *args: Any, **options: Any) -> Dict[Address, Any]:
        """Send one command to every primary that serves slots; the reply of
        each, by address."""
        return {a: await self.execute_on_node(a, *args, **options) for a in await self._primaries()}

    # -- SCAN family iteration ----------------------------------------------------

    async def scan_iter(
        self, match: Optional[str] = None, count: Optional[int] = None, type: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Iterate over every key of the cluster: `SCAN` pages through each
        primary in turn."""
        await self._ensure_ready()
        for address in await self._primaries():
            cursor = 0
            while True:
                args: List[Any] = [cursor]
                if match is not None:
                    args += ["MATCH", match]
                if count is not None:
                    args += ["COUNT", count]
                if type is not None:
                    args += ["TYPE", type]
                cursor, keys = await self.execute_on_node(address, "SCAN", *args)
                for key in keys:
                    yield key
                if cursor == 0:
                    break

    async def sscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> AsyncIterator[bytes]:
        """Iterate over every member of the set at `key`."""
        cursor = 0
        while True:
            cursor, members = await self.sscan(key, cursor, match=match, count=count)
            for member in members:
                yield member
            if cursor == 0:
                return

    async def hscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> AsyncIterator[Tuple[Any, Any]]:
        """Iterate over every `(field, value)` pair of the hash at `key`."""
        cursor = 0
        while True:
            cursor, page = await self.hscan(key, cursor, match=match, count=count)
            for item in page.items():
                yield item
            if cursor == 0:
                return

    async def zscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> AsyncIterator[Tuple[Any, float]]:
        """Iterate over every `(member, score)` pair of the sorted set at `key`."""
        cursor = 0
        while True:
            cursor, page = await self.zscan(key, cursor, match=match, count=count)
            for item in page:
                yield item
            if cursor == 0:
                return

    # -- scripting convenience ----------------------------------------------------

    async def eval_auto(self, script: str, keys: Any = (), args: Any = ()) -> Any:
        """Asyncio equivalent of `hotkv.ClusterHotKV.eval_auto()`."""
        sha1 = hashlib.sha1(script.encode("utf-8")).hexdigest()
        try:
            return await self.evalsha(sha1, keys, args)
        except NoScriptError:
            return await self.eval(script, keys, args)

    # -- lifecycle --------------------------------------------------------------

    async def close(self) -> None:
        """Close every per-node connection pool."""
        pools = list(self._pools.values())
        self._pools.clear()
        for pool in pools:
            await pool.close()

    async def __aenter__(self) -> "ClusterHotKV":
        await self._ensure_ready()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    @property
    def masters(self) -> List[Address]:
        """The distinct set of currently known master node addresses."""
        return sorted(set(a for a in self._slot_masters if a is not None))
