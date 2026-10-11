# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for the typed error hierarchy and `parse_error()`. No server needed."""

from __future__ import annotations

from hotkv.exceptions import (
    AskError,
    ClusterDownError,
    MovedError,
    NoAuthError,
    NotLicensedError,
    ResponseError,
    WrongTypeError,
    parse_error,
)


def test_generic_err_reply():
    err = parse_error("ERR wrong number of arguments for 'get' command")
    assert isinstance(err, ResponseError)
    assert err.code == "ERR"
    assert "wrong number of arguments" in err.message


def test_wrongtype_reply():
    err = parse_error("WRONGTYPE Operation against a key holding the wrong kind of value")
    assert isinstance(err, WrongTypeError)
    assert err.code == "WRONGTYPE"


def test_notlicensed_reply_matches_prefix_not_text():
    err = parse_error("NOLICENSE enterprise feature 'prompt-cache' requires a valid license")
    assert isinstance(err, NotLicensedError)
    assert err.code == "NOLICENSE"
    # Any message text under the same prefix is still a NotLicensedError.
    err2 = parse_error("NOLICENSE enterprise feature 'time-series' requires a valid license")
    assert isinstance(err2, NotLicensedError)


def test_noauth_reply():
    err = parse_error("NOAUTH Authentication required.")
    assert isinstance(err, NoAuthError)


def test_clusterdown_reply():
    err = parse_error("CLUSTERDOWN The cluster is down")
    assert isinstance(err, ClusterDownError)


def test_moved_reply_parses_slot_host_port():
    err = parse_error("MOVED 1234 127.0.0.1:6381")
    assert isinstance(err, MovedError)
    assert err.slot == 1234
    assert err.host == "127.0.0.1"
    assert err.port == 6381


def test_ask_reply_parses_slot_host_port():
    err = parse_error("ASK 5061 10.0.0.2:6400")
    assert isinstance(err, AskError)
    assert err.slot == 5061
    assert err.host == "10.0.0.2"
    assert err.port == 6400


def test_unrecognized_leading_token_falls_back_to_generic():
    err = parse_error("this is not a code just free text")
    assert type(err) is ResponseError
    assert err.code == "ERR"


def test_response_error_str_is_message_without_code():
    err = parse_error("WRONGTYPE bad type")
    assert str(err) == "bad type"


def test_all_typed_errors_are_response_errors():
    for cls in (NotLicensedError, WrongTypeError, NoAuthError, MovedError, AskError, ClusterDownError):
        assert issubclass(cls, ResponseError)
