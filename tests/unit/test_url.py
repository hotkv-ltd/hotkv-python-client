# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for connection URL parsing. No server needed."""

from __future__ import annotations

import pytest

from hotkv._url import parse_url


def test_basic_hotkv_url():
    opts = parse_url("hotkv://localhost:6399")
    assert opts["host"] == "localhost"
    assert opts["port"] == 6399
    assert opts["tls"] is False


def test_default_port():
    opts = parse_url("hotkv://localhost")
    assert opts["port"] == 6379


def test_hotkvs_enables_tls():
    opts = parse_url("hotkvs://localhost:6400")
    assert opts["tls"] is True


def test_redis_scheme_accepted_for_migration():
    opts = parse_url("redis://localhost:6379")
    assert opts["tls"] is False
    opts_tls = parse_url("rediss://localhost:6379")
    assert opts_tls["tls"] is True


def test_username_and_password():
    opts = parse_url("hotkv://alice:s3cret@localhost:6379")
    assert opts["username"] == "alice"
    assert opts["password"] == "s3cret"


def test_password_only_userinfo():
    opts = parse_url("hotkv://:s3cret@localhost:6379")
    assert opts.get("username") is None
    assert opts["password"] == "s3cret"


def test_db_from_path():
    opts = parse_url("hotkv://localhost:6379/3")
    assert opts["db"] == 3


def test_db_from_query_overrides_path():
    opts = parse_url("hotkv://localhost:6379/3?db=7")
    assert opts["db"] == 7


def test_unix_socket_url():
    opts = parse_url("unix:///tmp/hotkv.sock")
    assert opts["unix_socket_path"] == "/tmp/hotkv.sock"
    assert "host" not in opts


def test_unix_socket_url_with_query_db():
    opts = parse_url("unix:///tmp/hotkv.sock?db=2&password=pw")
    assert opts["unix_socket_path"] == "/tmp/hotkv.sock"
    assert opts["db"] == 2
    assert opts["password"] == "pw"


def test_query_parameters():
    opts = parse_url(
        "hotkv://localhost:6379"
        "?client_name=myapp&connect_timeout=2.5&command_timeout=10"
        "&tls_ca_file=/etc/ca.pem&tls_server_hostname=hotkv.internal"
        "&tls_insecure_skip_verify=true"
    )
    assert opts["client_name"] == "myapp"
    assert opts["connect_timeout"] == 2.5
    assert opts["command_timeout"] == 10.0
    assert opts["tls_ca_file"] == "/etc/ca.pem"
    assert opts["tls"] is True  # inferred from tls_ca_file
    assert opts["tls_server_hostname"] == "hotkv.internal"
    assert opts["tls_insecure_skip_verify"] is True


def test_unsupported_scheme_raises():
    with pytest.raises(ValueError):
        parse_url("http://localhost:6379")


def test_invalid_db_path_raises():
    with pytest.raises(ValueError):
        parse_url("hotkv://localhost:6379/not-a-number")


def test_percent_encoded_credentials_are_decoded():
    opts = parse_url("hotkv://us%40er:p%40ss@localhost:6379")
    assert opts["username"] == "us@er"
    assert opts["password"] == "p@ss"
