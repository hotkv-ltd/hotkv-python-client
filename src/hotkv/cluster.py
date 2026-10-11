# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Cluster client: slot-aware routing across a HotKV cluster.

Builds its slot map from `CLUSTER SLOTS`, hashes the key of each command with
the same CRC16 + `{hash tag}` scheme as `CLUSTER KEYSLOT`, follows `MOVED`
(refreshing the map) and `ASK` (sending `ASKING` first, without remapping),
and bounds the number of redirects it will follow.

Beyond the redirects, the client rides out the events of a live cluster:

* a primary that goes away (failover) or a reshard in flight: connection
  errors, `CLUSTERDOWN`, `TRYAGAIN` and `LOADING` replies make it refresh the
  slot map and try again with a short backoff, for up to `retry_timeout`
  seconds;
* a command that is answered from a node's own state rather than a slot (the
  `*.LIST` commands of the AI engines, `TAG.MEMBERS`/`TAG.INTER`, `KEYS`,
  `DBSIZE`, `FLUSHALL`) is sent to every primary and the answers merged, so
  `pcache_list()` lists the namespaces of the whole cluster, not of one node;
* `scan_iter()` walks the keyspace of every primary.

The key of a command is its first key (`EVAL`'s and `XREAD`'s too, where it is
not the first argument); a command that names several keys needs them in one
slot, as on any cluster: give them a common hash tag.

Scope: this routes single commands (the full typed surface from
`CoreCommands`, plus the `execute()` escape hatch). Pipelines, transactions
and pub/sub are intentionally out of scope here, since they need every key
involved to share one slot; use the plain `HotKV` client pointed at a
specific node for those.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from . import commands
from ._cluster_common import (
    Address,
    command_name,
    fanout_kind,
    is_permanent_failure,
    merge_replies,
    parse_slots,
    routing_key,
)
from ._crc16 import TOTAL_SLOTS, key_slot
from ._protocol import encode_arg
from ._pool import ConnectionPool
from .exceptions import (
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

__all__ = ["ClusterHotKV"]

_RETRY_EXCEPTIONS = (ConnectionError, TimeoutError, ProtocolError)
_REFRESH_RETRY_EXCEPTIONS = _RETRY_EXCEPTIONS + (ResponseError,)
# Replies that say the command did not run and may be asked again shortly.
_TRANSIENT_CODES = frozenset({"CLUSTERDOWN", "TRYAGAIN", "LOADING", "MASTERDOWN", "READONLY"})
_SUSPECT_SECONDS = 30.0


class _StalePrimaries(Exception):
    """The set of primaries changed under a fanned-out command."""


def _to_str(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


class ClusterHotKV(commands.CoreCommands):
    """Slot-aware HotKV cluster client.

    `retry_timeout` bounds how long one command keeps trying through a
    failover or a reshard before it gives up with `ClusterError`.
    `retry_writes` (default `True`) lets a
    command that may have reached a node whose connection then broke be sent
    again; with `False`, only reads and commands that never left the client
    (the connection could not be opened) are retried, and a write that fails
    in flight raises.

    The default delivers a write at least once. A write that did reach the
    node before the connection broke (or before `command_timeout` ran out) is
    applied twice when the node had applied the first copy, which changes the
    result of a command that is not idempotent: `INCR`, `RPUSH`, `XADD`,
    `TS.ADD`, `LLMBUDGET.CONSUME`, `RATELIMIT.CHECK`, `LLMSTATS.RECORD`,
    `AMEM.CONV.APPEND` and the other counters and appends. Pass
    `retry_writes=False` where that matters: the failure then reaches the
    caller, who knows whether the command can be repeated.
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

        self._lock = threading.RLock()
        self._pools: Dict[Tuple[Address, bool], ConnectionPool] = {}
        self._slot_masters: List[Optional[Address]] = [None] * TOTAL_SLOTS
        self._slot_replicas: List[List[Address]] = [[] for _ in range(TOTAL_SLOTS)]
        # When each node last failed to answer: a node that did is asked last
        # for the slot map, so one that hangs is not waited for again and again.
        self._suspect: Dict[Address, float] = {}
        self.refresh_slots()

    # -- slot map --------------------------------------------------------------

    def refresh_slots(self) -> None:
        """Rebuild the slot map from `CLUSTER SLOTS`, trying each known node."""
        last_error: Optional[BaseException] = None
        for address in self._known_addresses():
            pool = self._pool_for(address)
            try:
                with pool.connection() as conn:
                    conn.send_command("CLUSTER", "SLOTS")
                    reply = conn.read_response()
                if isinstance(reply, ResponseError):
                    raise reply
                self._apply_slots(reply, address[0])
                return
            except _REFRESH_RETRY_EXCEPTIONS as exc:
                last_error = exc
                self._suspect[address] = time.monotonic()
                continue
        raise ClusterError(f"could not load the cluster slot map from any node: {last_error}")

    def _known_addresses(self) -> Iterable[Address]:
        seen = set()
        with self._lock:
            candidates = list(self._startup_nodes) + [a for (a, _ro) in self._pools.keys()]
            candidates += [a for a in self._slot_masters if a is not None]
        ordered = []
        for address in candidates:
            if address not in seen:
                seen.add(address)
                ordered.append(address)
        now = time.monotonic()
        ordered.sort(key=lambda a: now - self._suspect.get(a, -1e9) < _SUSPECT_SECONDS)
        yield from ordered

    def _apply_slots(self, reply: Any, asked_host: str = "") -> None:
        masters, replicas = parse_slots(reply, asked_host)
        with self._lock:
            self._slot_masters = masters
            self._slot_replicas = replicas

    def _pool_for(self, address: Address, readonly: bool = False) -> ConnectionPool:
        with self._lock:
            pool = self._pools.get((address, readonly))
            if pool is None:
                kwargs = dict(self._node_kwargs, host=address[0], port=address[1], readonly=readonly)
                pool = ConnectionPool(kwargs, max_connections=self._max_connections_per_node)
                self._pools[(address, readonly)] = pool
            return pool

    def _target_address(self, args: Tuple[Any, ...], *, readonly: bool) -> Address:
        """The node a command goes to first: the master of its key's slot (a
        replica of it for a read with `read_from_replicas`), else any master."""
        address, _replica = self._pick(args, readonly=readonly)
        return address

    def _pick(self, args: Tuple[Any, ...], *, readonly: bool, avoid_replicas: bool = False) -> Tuple[Address, bool]:
        key = routing_key(command_name(args[0]), args)
        if key is None:
            return self._any_address(), False
        slot = key_slot(key if isinstance(key, (str, bytes)) else encode_arg(key))
        with self._lock:
            master = self._slot_masters[slot]
            replicas = list(self._slot_replicas[slot])
        if master is None:
            self.refresh_slots()
            with self._lock:
                master = self._slot_masters[slot]
        if master is None:
            raise ClusterError(f"no node is known to own slot {slot}")
        if readonly and self.read_from_replicas and replicas and not avoid_replicas:
            return random.choice(replicas), True
        return master, False

    def _any_address(self) -> Address:
        with self._lock:
            for address in self._slot_masters:
                if address is not None:
                    return address
        return self._startup_nodes[0]

    # -- command execution --------------------------------------------------------

    def _round_trip(self, address: Address, args: Tuple[Any, ...], *, asking: bool, replica: bool) -> Any:
        """One command on one node. Returns the reply (an error reply is returned,
        not raised). A transport failure is raised with `.sent` telling whether
        the command had been written."""
        pool = self._pool_for(address, readonly=replica)
        conn = pool.get_connection()
        sent = False
        try:
            conn.connect()
            if asking:
                conn.send_command("ASKING")
                ack = conn.read_response()
                if isinstance(ack, ResponseError):
                    raise ack
            sent = True
            conn.send_command(*args)
            reply = conn.read_response()
        except _RETRY_EXCEPTIONS as exc:
            pool.discard(conn)
            self._suspect[address] = time.monotonic()
            exc.sent = sent  # type: ignore[attr-defined]
            raise
        except BaseException:
            pool.release(conn)
            raise
        pool.release(conn)
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
    def _backoff(attempt: int, deadline: float) -> bool:
        """Sleep before the next try; False once the deadline has passed."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05 * (2 ** min(attempt, 4)), 0.5, remaining))
        return True

    def _refresh_quietly(self) -> None:
        try:
            self.refresh_slots()
        except ClusterError:
            pass

    def _send(self, args: Tuple[Any, ...], *, readonly: bool) -> Any:
        """Send `args` to the node that owns its key, following redirects and
        riding out a failover. Returns the raw reply (an error reply is
        returned, not raised, except when the retries run out)."""
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
                    address, replica = self._pick(args, readonly=readonly, avoid_replicas=avoid_replicas)
                except ClusterError as exc:
                    # The slot has no owner in the map (the cluster is mid-change).
                    last_error = exc
                    if not self._backoff(attempt, deadline):
                        raise
                    attempt += 1
                    self._refresh_quietly()
                    continue
            try:
                reply = self._round_trip(address, args, asking=asking, replica=replica)
            except _RETRY_EXCEPTIONS as exc:
                last_error = exc
                if not self._may_retry(exc, readonly):
                    raise
                if not self._backoff(attempt, deadline):
                    raise ClusterError(
                        f"giving up on {name!r} after {self.retry_timeout}s of retries: {exc}"
                    ) from exc
                attempt += 1
                avoid_replicas = avoid_replicas or replica
                self._refresh_quietly()
                continue
            if isinstance(reply, MovedError):
                redirects += 1
                if redirects > self.max_redirects:
                    break
                last_error = reply
                self._refresh_quietly()
                # The node that answered MOVED knows where the slot went before
                # CLUSTER SLOTS may show it: follow it.
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
                if not self._backoff(attempt, deadline):
                    raise reply
                attempt += 1
                avoid_replicas = avoid_replicas or replica
                self._refresh_quietly()
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
        with self._lock:
            self._slot_masters[slot] = target

    def _send_to_primaries(self, args: Tuple[Any, ...]) -> List[Any]:
        """Send `args` to every primary and return the raw replies. A primary
        that is gone, or no longer one, makes the round start again from a
        fresh slot map, until `retry_timeout`."""
        deadline = time.monotonic() + self.retry_timeout
        attempt = 0
        while True:
            try:
                replies = []
                for address in self._primaries():
                    reply = self._round_trip(address, args, asking=False, replica=False)
                    if isinstance(reply, MovedError) or self._is_transient(reply):
                        raise _StalePrimaries(str(reply))
                    replies.append(reply)
                return replies
            except (_StalePrimaries,) + _RETRY_EXCEPTIONS as exc:
                if isinstance(exc, _RETRY_EXCEPTIONS) and is_permanent_failure(exc):
                    raise
                if not self.retry_writes and isinstance(exc, _RETRY_EXCEPTIONS) and getattr(exc, "sent", True):
                    raise
                if not self._backoff(attempt, deadline):
                    raise ClusterError(f"giving up on {command_name(args[0])!r}: {exc}") from exc
                attempt += 1
                self._refresh_quietly()

    def execute_command(self, *args: Any, **options: Any) -> Any:
        name = command_name(args[0])
        kind = fanout_kind(name, args)
        if kind is not None:
            replies = self._send_to_primaries(args)
            for reply in replies:
                if isinstance(reply, ResponseError):
                    raise reply
            reply = merge_replies(kind, replies)
        else:
            reply = self._send(args, readonly=name in self.READONLY_COMMANDS)
            if isinstance(reply, ResponseError):
                raise reply
        callback = self.RESPONSE_CALLBACKS.get(args[0])
        return callback(reply, **options) if callback is not None else reply

    def _primaries(self) -> List[Address]:
        masters = self.masters
        if not masters:
            self.refresh_slots()
            masters = self.masters
        if not masters:
            raise ClusterError("the cluster has no primary with slots")
        return masters

    def execute(self, *args: Any, **options: Any) -> Any:
        """Escape hatch: send any command, typed wrapper or not."""
        return self.execute_command(*args, **options)

    command = execute

    def execute_on_node(self, address: Address, *args: Any, **options: Any) -> Any:
        """Send one command to one specific node (a primary or a replica),
        with no routing and no redirect following: for per-node administration
        and for scans."""
        pool = self._pool_for((address[0], int(address[1])))
        with pool.connection() as conn:
            conn.send_command(*args)
            reply = conn.read_response()
        if isinstance(reply, ResponseError):
            raise reply
        callback = self.RESPONSE_CALLBACKS.get(args[0])
        return callback(reply, **options) if callback is not None else reply

    def execute_on_primaries(self, *args: Any, **options: Any) -> Dict[Address, Any]:
        """Send one command to every primary that serves slots and return the
        reply of each, by address."""
        return {a: self.execute_on_node(a, *args, **options) for a in self._primaries()}

    # -- SCAN family iteration ----------------------------------------------------

    def scan_iter(
        self, match: Optional[str] = None, count: Optional[int] = None, type: Optional[str] = None
    ) -> Iterator[bytes]:
        """Iterate over every key of the cluster: `SCAN` pages through each
        primary in turn (a key is returned once unless it moves between
        nodes during the scan, as with any `SCAN` on a changing keyspace)."""
        for address in self._primaries():
            cursor = 0
            while True:
                cursor, keys = self.execute_on_node(
                    address, "SCAN", *self._scan_args(cursor, match, count, type)
                )
                yield from keys
                if cursor == 0:
                    break

    @staticmethod
    def _scan_args(cursor: int, match: Optional[str], count: Optional[int], type: Optional[str]) -> List[Any]:
        args: List[Any] = [cursor]
        if match is not None:
            args += ["MATCH", match]
        if count is not None:
            args += ["COUNT", count]
        if type is not None:
            args += ["TYPE", type]
        return args

    def sscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> Iterator[bytes]:
        """Iterate over every member of the set at `key`."""
        cursor = 0
        while True:
            cursor, members = self.sscan(key, cursor, match=match, count=count)
            yield from members
            if cursor == 0:
                return

    def hscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> Iterator[Tuple[Any, Any]]:
        """Iterate over every `(field, value)` pair of the hash at `key`."""
        cursor = 0
        while True:
            cursor, page = self.hscan(key, cursor, match=match, count=count)
            yield from page.items()
            if cursor == 0:
                return

    def zscan_iter(
        self, key: Any, match: Optional[str] = None, count: Optional[int] = None
    ) -> Iterator[Tuple[Any, float]]:
        """Iterate over every `(member, score)` pair of the sorted set at `key`."""
        cursor = 0
        while True:
            cursor, page = self.zscan(key, cursor, match=match, count=count)
            yield from page
            if cursor == 0:
                return

    # -- scripting convenience ----------------------------------------------------

    def eval_auto(self, script: str, keys: Any = (), args: Any = ()) -> Any:
        """Run `script` with `EVALSHA` (the digest is computed locally) and fall
        back to `EVAL`, which also caches it on that node, on `NOSCRIPT`."""
        sha1 = hashlib.sha1(script.encode("utf-8")).hexdigest()
        try:
            return self.evalsha(sha1, keys, args)
        except NoScriptError:
            return self.eval(script, keys, args)

    # -- lifecycle --------------------------------------------------------------

    def close(self) -> None:
        """Close every per-node connection pool."""
        with self._lock:
            pools = list(self._pools.values())
            self._pools.clear()
        for pool in pools:
            pool.close()

    def __enter__(self) -> "ClusterHotKV":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def masters(self) -> List[Address]:
        """The distinct set of currently known master node addresses."""
        with self._lock:
            return sorted(set(a for a in self._slot_masters if a is not None))
