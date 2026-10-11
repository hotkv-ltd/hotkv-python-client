# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for the RESP2/RESP3 encoder and incremental parser. No server
needed."""

from __future__ import annotations

import pytest

from hotkv._protocol import (
    NEED_MORE,
    PushMessage,
    Reader,
    VerbatimString,
    build_args,
    decode_value,
    encode_command,
)
from hotkv.exceptions import DataError, ProtocolError, ResponseError


def parse_all(data: bytes):
    """Feed `data` one byte at a time and return every parsed reply, to
    exercise the streaming/partial-read path on every type."""
    reader = Reader()
    values = []
    for i in range(len(data)):
        reader.feed(data[i : i + 1])
        while True:
            value = reader.gets()
            if value is NEED_MORE:
                break
            values.append(value)
    return values


def test_simple_string_and_error():
    values = parse_all(b"+OK\r\n-ERR bad thing\r\n")
    assert values[0] == "OK"
    assert isinstance(values[1], ResponseError)
    assert values[1].code == "ERR"
    assert values[1].message == "bad thing"


def test_integer():
    assert parse_all(b":1000\r\n") == [1000]
    assert parse_all(b":-7\r\n") == [-7]


def test_bulk_string_and_null():
    values = parse_all(b"$5\r\nhello\r\n$-1\r\n$0\r\n\r\n")
    assert values == [b"hello", None, b""]


def test_binary_safe_bulk_string():
    payload = bytes(range(256))
    data = b"$" + str(len(payload)).encode() + b"\r\n" + payload + b"\r\n"
    assert parse_all(data) == [payload]


def test_array_and_null_array():
    values = parse_all(b"*3\r\n:1\r\n:2\r\n:3\r\n*-1\r\n*0\r\n")
    assert values == [[1, 2, 3], None, []]


def test_nested_array():
    values = parse_all(b"*2\r\n*2\r\n:1\r\n:2\r\n*1\r\n+ok\r\n")
    assert values == [[[1, 2], ["ok"]]]


def test_resp3_null():
    assert parse_all(b"_\r\n") == [None]


def test_resp3_double():
    values = parse_all(b",3.14\r\n,inf\r\n,-inf\r\n")
    assert values[0] == pytest.approx(3.14)
    assert values[1] == float("inf")
    assert values[2] == float("-inf")


def test_resp3_boolean():
    assert parse_all(b"#t\r\n#f\r\n") == [True, False]


def test_resp3_blob_error():
    values = parse_all(b"!21\r\nSYNTAX invalid syntax\r\n")
    assert isinstance(values[0], ResponseError)
    assert values[0].code == "SYNTAX"


def test_resp3_verbatim_string():
    values = parse_all(b"=9\r\ntxt:hello\r\n")
    assert isinstance(values[0], VerbatimString)
    assert values[0] == b"hello"
    assert values[0].format == "txt"


def test_resp3_big_number():
    values = parse_all(b"(3492890328409238509324850943850943825024385\r\n")
    assert values[0] == 3492890328409238509324850943850943825024385


def test_resp3_map():
    values = parse_all(b"%2\r\n+a\r\n:1\r\n+b\r\n:2\r\n")
    assert values[0] == {"a": 1, "b": 2}


def test_resp3_set():
    values = parse_all(b"~3\r\n:1\r\n:2\r\n:1\r\n")
    assert values[0] == {1, 2}


def test_resp3_attribute_is_transparent():
    # An attribute precedes the real reply; the parser should surface only
    # the real reply, still correctly positioned in the stream.
    values = parse_all(b"|1\r\n+key\r\n+val\r\n:5\r\n+next\r\n")
    assert values == [5, "next"]


def test_resp3_push():
    values = parse_all(b">2\r\n+message\r\n:9\r\n")
    assert isinstance(values[0], PushMessage)
    assert list(values[0]) == ["message", 9]


def test_unknown_type_byte_raises_protocol_error():
    reader = Reader()
    reader.feed(b"?garbage\r\n")
    with pytest.raises(ProtocolError):
        reader.gets()


def test_gets_returns_need_more_without_consuming():
    reader = Reader()
    reader.feed(b"$5\r\nhel")
    assert reader.gets() is NEED_MORE
    reader.feed(b"lo\r\n")
    assert reader.gets() == b"hello"


def test_multiple_replies_in_one_feed():
    reader = Reader()
    reader.feed(b"+OK\r\n+OK\r\n+OK\r\n")
    assert [reader.gets(), reader.gets(), reader.gets()] == ["OK", "OK", "OK"]
    assert reader.gets() is NEED_MORE


def test_build_args_splits_multi_word_command():
    assert build_args("GET", "key") == ["GET", "key"]
    assert build_args("CONFIG GET", "maxmemory") == ["CONFIG", "GET", "maxmemory"]


def test_encode_command_roundtrip():
    packed = encode_command(build_args("SET", "k", "v"))
    assert packed == b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n"


def test_encode_command_binary_safe():
    packed = encode_command(build_args("SET", "k", b"\x00\x01\xff"))
    assert packed == b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$3\r\n\x00\x01\xff\r\n"


def test_encode_command_numbers():
    packed = encode_command(["INCRBY", "k", 42])
    assert b"$2\r\n42\r\n" in packed
    packed_float = encode_command(["INCRBYFLOAT", "k", 3.5])
    assert b"$3\r\n3.5\r\n" in packed_float


def test_encode_command_rejects_bool():
    with pytest.raises(DataError):
        encode_command(["SET", "k", True])


def test_encode_command_rejects_unsupported_type():
    with pytest.raises(DataError):
        encode_command(["SET", "k", object()])


def test_decode_value_recursive():
    decoded = decode_value([b"a", {b"x": b"y"}, {b"z"}], "utf-8", "strict")
    assert decoded == ["a", {"x": "y"}, {"z"}]


def test_decode_value_preserves_push_type():
    push = PushMessage([b"message", b"chan", b"payload"])
    decoded = decode_value(push, "utf-8", "strict")
    assert isinstance(decoded, PushMessage)
    assert decoded == ["message", "chan", "payload"]
