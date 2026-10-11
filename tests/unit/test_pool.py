# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for `ConnectionPool` against a fake connection, so no server
or real socket is needed. Covers bounded size, idle health checks, and the
discard-on-transport-error-but-release-otherwise rule."""

from __future__ import annotations

import threading
import time

import pytest

from hotkv._pool import ConnectionPool
from hotkv.exceptions import ConnectionError as HotKVConnectionError
from hotkv.exceptions import ProtocolError, ResponseError


class FakeConnection:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._connected = True
        self.ping_calls = 0
        self.ping_should_fail = False

    @property
    def connected(self):
        return self._connected

    def disconnect(self):
        self._connected = False

    def ping(self):
        self.ping_calls += 1
        if self.ping_should_fail:
            raise HotKVConnectionError("ping failed")
        return True


def make_pool(**kwargs):
    return ConnectionPool({}, connection_class=FakeConnection, **kwargs)


def test_creates_up_to_max_connections():
    pool = make_pool(max_connections=2, idle_timeout=None)
    c1 = pool.get_connection()
    c2 = pool.get_connection()
    assert c1 is not c2
    assert pool.size == 2
    with pytest.raises(HotKVConnectionError):
        pool.get_connection(timeout=0.05)


def test_release_makes_connection_reusable():
    pool = make_pool(max_connections=1, idle_timeout=None)
    c1 = pool.get_connection()
    pool.release(c1)
    c2 = pool.get_connection(timeout=0.5)
    assert c1 is c2
    assert pool.size == 1


def test_discard_allows_a_fresh_connection():
    pool = make_pool(max_connections=1, idle_timeout=None)
    c1 = pool.get_connection()
    pool.discard(c1)
    assert c1.connected is False
    assert pool.size == 0
    c2 = pool.get_connection(timeout=0.5)
    assert c2 is not c1


def test_context_manager_releases_on_success():
    pool = make_pool(max_connections=1, idle_timeout=None)
    with pool.connection() as conn:
        assert conn.connected
    assert pool.size == 1
    # the same (still connected) instance should be reusable
    with pool.connection() as conn2:
        assert conn2.connected


def test_context_manager_releases_on_response_error():
    # A ResponseError (a normal per-command failure) must NOT poison the
    # connection: it should be released, not discarded.
    pool = make_pool(max_connections=1, idle_timeout=None)
    with pytest.raises(ResponseError):
        with pool.connection() as conn:
            raise ResponseError("bad command", "ERR")
    assert conn.connected is True
    assert pool.size == 1


def test_context_manager_discards_on_transport_error():
    pool = make_pool(max_connections=1, idle_timeout=None)
    with pytest.raises(HotKVConnectionError):
        with pool.connection() as conn:
            raise HotKVConnectionError("socket died")
    assert conn.connected is False
    assert pool.size == 0


def test_context_manager_discards_on_protocol_error():
    pool = make_pool(max_connections=1, idle_timeout=None)
    with pytest.raises(ProtocolError):
        with pool.connection() as conn:
            raise ProtocolError("desynced")
    assert conn.connected is False
    assert pool.size == 0


def test_context_manager_releases_on_unrelated_exception():
    pool = make_pool(max_connections=1, idle_timeout=None)
    with pytest.raises(ValueError):
        with pool.connection() as conn:
            raise ValueError("caller's own bug")
    assert conn.connected is True
    assert pool.size == 1


def test_idle_connection_is_health_checked_and_replaced_on_failure():
    pool = make_pool(max_connections=1, idle_timeout=0.01)
    c1 = pool.get_connection()
    c1.ping_should_fail = True
    pool.release(c1)
    time.sleep(0.05)
    c2 = pool.get_connection(timeout=0.5)
    assert c2 is not c1
    assert c1.ping_calls == 1
    assert c1.connected is False


def test_idle_connection_passes_health_check():
    pool = make_pool(max_connections=1, idle_timeout=0.01)
    c1 = pool.get_connection()
    pool.release(c1)
    time.sleep(0.05)
    c2 = pool.get_connection(timeout=0.5)
    assert c2 is c1
    assert c1.ping_calls == 1


def test_close_disconnects_everything():
    pool = make_pool(max_connections=2, idle_timeout=None)
    c1 = pool.get_connection()
    c2 = pool.get_connection()
    pool.release(c1)
    pool.close()
    assert c1.connected is False
    assert c2.connected is False
    assert pool.size == 0
    with pytest.raises(HotKVConnectionError):
        pool.get_connection(timeout=0.05)


def test_thread_safety_under_concurrent_borrow_and_release():
    pool = make_pool(max_connections=4, idle_timeout=None)
    errors = []
    seen_sizes = []

    def worker():
        try:
            for _ in range(50):
                conn = pool.get_connection(timeout=5)
                seen_sizes.append(pool.size)
                time.sleep(0.001)
                pool.release(conn)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert pool.size <= 4
    assert max(seen_sizes) <= 4
