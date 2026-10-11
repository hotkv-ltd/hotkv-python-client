# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Fixtures for integration tests. Every test in this directory needs a
live HotKV server reachable at `HOTKV_TEST_URL`; when that is not set, all
of them are skipped (not failed) with a clear reason.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

import hotkv
import hotkv.asyncio

HOTKV_TEST_URL = os.environ.get("HOTKV_TEST_URL")


@pytest.fixture(autouse=True)
def _require_server(request):
    if request.node.module.__name__.endswith("test_remote"):
        return  # these tests carry their own gate, HOTKV_REMOTE_HOST
    if not HOTKV_TEST_URL:
        pytest.skip("HOTKV_TEST_URL is not set; skipping integration tests")


@pytest.fixture()
def test_url() -> str:
    return HOTKV_TEST_URL


@pytest.fixture()
def client(request):
    c = hotkv.from_url(request.getfixturevalue("test_url"))
    yield c
    c.close()


@pytest_asyncio.fixture()
async def aclient(request):
    c = hotkv.asyncio.from_url(request.getfixturevalue("test_url"))
    yield c
    await c.close()


@pytest.fixture()
def prefix() -> str:
    """A short prefix unique to this test, so tests never collide with each
    other's keys on the shared test server."""
    token = f"pytest:{uuid.uuid4().hex[:12]}"
    return token + ":"


@pytest.fixture()
def skip_if_not_licensed():
    """Call `skip_if_not_licensed(callable_, *args, **kwargs)`; if the
    server replies `NOLICENSE`, the test is skipped with a clear reason
    instead of failing."""

    def _run(callable_, *args, **kwargs):
        try:
            return callable_(*args, **kwargs)
        except hotkv.NotLicensedError as exc:
            pytest.skip(f"the server is not licensed for this feature: {exc}")

    return _run


@pytest.fixture()
def async_skip_if_not_licensed():
    """Async equivalent of `skip_if_not_licensed()`, given a coroutine."""

    async def _run(coro):
        try:
            return await coro
        except hotkv.NotLicensedError as exc:
            pytest.skip(f"the server is not licensed for this feature: {exc}")

    return _run
