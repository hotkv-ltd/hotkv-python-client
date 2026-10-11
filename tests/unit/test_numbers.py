# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Unit tests for how numbers are written on the wire: plain decimal, never an
exponent, shortest round-trip digits, NaN refused, USD as fixed point. No
server needed. The live counterparts are in tests/integration/test_number_wire.py."""

from __future__ import annotations

import math
import random
import re
import struct
import sys
from decimal import Decimal

import pytest

import hotkv
from hotkv import format_float, format_usd
from hotkv._numbers import format_component, format_cost, format_vector
from hotkv._protocol import encode_arg, encode_command
from hotkv.exceptions import DataError

PLAIN = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$")


@pytest.mark.parametrize(
    "value, text",
    [
        (0.0, "0"),
        (-0.0, "0"),
        (3.0, "3"),
        (-3.0, "-3"),
        (0.5, "0.5"),
        (0.1, "0.1"),
        (0.30000000000000004, "0.30000000000000004"),
        (1e-05, "0.00001"),
        (1e-07, "0.0000001"),
        (123456789.125, "123456789.125"),
        (1e21, "1000000000000000000000"),
        (1e22, "10000000000000000000000"),
        (-2.5e-10, "-0.00000000025"),
        (5, "5"),
        (-7, "-7"),
        (Decimal("1.50"), "1.5"),
        ("2.50", "2.5"),
    ],
)
def test_format_float_is_plain_decimal(value, text):
    assert format_float(value) == text


def test_tiny_and_huge_doubles_round_trip():
    for value in (1e-300, 5e-324, 2.2250738585072014e-308, 1e-05, 1e21, 1.7976931348623157e308, -1.7976931348623157e308):
        text = format_float(value)
        assert PLAIN.match(text), text
        assert "e" not in text.lower() and "+" not in text
        assert float(text) == value
    assert format_float(1e-300) == "0." + "0" * 299 + "1"
    assert len(format_float(1.7976931348623157e308)) == 309


def test_round_trip_over_random_doubles():
    rng = random.Random(20261010)
    for _ in range(20000):
        bits = rng.getrandbits(64)
        value = struct.unpack("<d", bits.to_bytes(8, "little"))[0]
        if math.isnan(value) or math.isinf(value):
            continue
        text = format_float(value)
        assert PLAIN.match(text), (value, text)
        assert float(text) == value, (value, text)
    for _ in range(5000):
        value = rng.uniform(-1000, 1000) * 10 ** rng.randint(-12, 12)
        assert float(format_float(value)) == value


def test_negative_zero_is_zero():
    assert format_float(-0.0) == "0"
    assert encode_arg(-0.0) == b"0"


def test_nan_is_refused_everywhere():
    with pytest.raises(DataError):
        format_float(float("nan"))
    with pytest.raises(DataError):
        format_float(float("nan"), allow_inf=True)
    with pytest.raises(DataError):
        encode_arg(float("nan"))
    with pytest.raises(ValueError):
        format_usd(float("nan"))
    with pytest.raises(ValueError):
        format_cost(float("nan"))
    with pytest.raises(ValueError):
        format_component(float("nan"))
    with pytest.raises(DataError):
        format_float("nan")


def test_infinity_only_where_allowed():
    with pytest.raises(DataError):
        format_float(float("inf"))
    with pytest.raises(DataError):
        format_float(float("-inf"))
    assert format_float(float("inf"), allow_inf=True) == "+inf"
    assert format_float(float("-inf"), allow_inf=True) == "-inf"
    # the generic argument encoder leaves it to the command (a sorted set takes one)
    assert encode_arg(float("inf")) == b"+inf"
    assert encode_arg(float("-inf")) == b"-inf"


def test_bool_and_junk_are_refused():
    for bad in (True, False, None, object(), "abc", "1_0", "", [1.0]):
        with pytest.raises(DataError):
            format_float(bad)


def test_underscores_do_not_sneak_in_as_digits():
    with pytest.raises(DataError):
        format_float("1_000")


def test_the_wire_encoder_never_writes_an_exponent():
    assert encode_arg(1e-05) == b"0.00001"
    assert encode_arg(1e21) == b"1000000000000000000000"
    assert encode_arg(3.0) == b"3"
    assert encode_arg(7) == b"7"
    assert encode_command(["ZADD", "k", 1e-05, "m"]) == (
        b"*4\r\n$4\r\nZADD\r\n$1\r\nk\r\n$7\r\n0.00001\r\n$1\r\nm\r\n"
    )


def test_numpy_style_scalars_are_encoded():
    import numbers

    class Real32(float):
        pass

    assert isinstance(Real32(1.5), numbers.Real)
    assert encode_arg(Real32(1.5)) == b"1.5"


# -- USD ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, text",
    [
        (0, "0"),
        (0.0, "0"),
        (-0.0, "0"),
        (50.0, "50"),
        (1e-05, "0.00001"),
        (0.000001, "0.000001"),
        (0.0000005, "0.000001"),  # half up
        (0.0000004, "0"),
        (0.1 + 0.2, "0.3"),
        (1234567.891234, "1234567.891234"),
        (1234567.8912344, "1234567.891234"),
        (1e-07, "0"),
        (2, "2"),
        (Decimal("0.0000015"), "0.000002"),
        ("3.00", "3"),
        ("1e-5", "0.00001"),
    ],
)
def test_format_usd(value, text):
    assert format_usd(value) == text


def test_usd_output_matches_the_servers_grammar():
    # parse_usd_to_micros: digits, optional '.', a digit before it, at most 6 fractional digits.
    grammar = re.compile(r"^[0-9]+(\.[0-9]{1,6})?$")
    rng = random.Random(7)
    for _ in range(5000):
        value = rng.uniform(0, 1000) * 10 ** rng.randint(-9, 3)
        text = format_usd(value)
        assert grammar.match(text), (value, text)
        assert abs(float(text) - value) <= 5e-7 + abs(value) * 1e-15


def test_usd_refuses_negative_nan_inf_and_overflow():
    for bad in (-1, -0.01, -1e-09, float("-inf"), float("inf"), float("nan"), "abc", "-5", 1e30, 18446744073710.0, True):
        with pytest.raises(ValueError):
            format_usd(bad)
    # the largest amount that fits in an unsigned 64-bit count of micro-dollars
    assert format_usd(Decimal("18446744073709.551615")) == "18446744073709.551615"
    with pytest.raises(ValueError):
        format_usd(Decimal("18446744073709.551616"))


def test_cost_is_a_plain_finite_non_negative_decimal():
    assert format_cost(0.0) == "0"
    assert format_cost(0.05) == "0.05"
    assert format_cost(1e-05) == "0.00001"
    assert format_cost(1e-300) == "0." + "0" * 299 + "1"
    for bad in (-0.5, float("inf"), float("nan"), 1e30):
        with pytest.raises(ValueError):
            format_cost(bad)


# -- vector components ----------------------------------------------------------------


def test_vector_components_are_plain_and_fit_a_float32():
    assert format_vector([0.25, -1.5, 1e-05, 3.0, 0]) == ["0.25", "-1.5", "0.00001", "3", "0"]
    assert format_component(1e-300) == "0"  # below a float32 subnormal: out of range for the server
    assert format_component(-1e-50) == "0"
    assert format_component(1.0e-45) == "0.000000000000000000000000000000000000000000001"
    assert format_component(3.4028234663852886e38) == "340282346638528860000000000000000000000"
    for bad in (1e39, 1.7976931348623157e308, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            format_component(bad)


# -- the typed commands send the plain forms ---------------------------------------------


def test_llmbudget_set_sends_a_fixed_point_cost_limit(recording_client):
    recording_client.llmbudget_set("b", 1000, 1e-05, 60)
    assert recording_client.last_call == ("LLMBUDGET.SET", "b", 1000, "0.00001", 60)
    recording_client.llmbudget_set("b", 1000, 50, 60)
    assert recording_client.last_call == ("LLMBUDGET.SET", "b", 1000, "50", 60)
    recording_client.llmbudget_set("b", 1000, 0.0000005, 60)
    assert recording_client.last_call[3] == "0.000001"
    with pytest.raises(ValueError):
        recording_client.llmbudget_set("b", 1000, -1.0, 60)


def test_llmbudget_consume_and_llmstats_record_send_plain_costs(recording_client):
    recording_client.llmbudget_consume("b", 10, 1e-05)
    assert recording_client.last_call == ("LLMBUDGET.CONSUME", "b", 10, "0.00001")
    recording_client.llmstats_record("s", 1, 2, 3, 2.5e-06, 10.0)
    assert recording_client.last_call == ("LLMSTATS.RECORD", "s", 1, 2, 3, "0.0000025", 10)
    with pytest.raises(ValueError):
        recording_client.llmstats_record("s", 1, 2, 3, -1.0, 10.0)


def test_llmstats_price_set_sends_fixed_point_prices(recording_client):
    recording_client.llmstats_price_set("m", 1e-05, 2.5e1, cache_read_usd_per_mtok=1e-06)
    assert recording_client.last_call == (
        "LLMSTATS.PRICE SET", "m", "INPUT", "0.00001", "OUTPUT", "25", "CACHEREAD", "0.000001"
    )


def test_cache_thresholds_and_vectors_are_plain(recording_client):
    recording_client.scache_create("ns", 2, threshold=1e-05)
    assert recording_client.last_call == ("SCACHE.CREATE", "ns", 2, "METRIC", "cosine", "THRESHOLD", "0.00001")
    recording_client.scache_threshold("ns", 0.5)
    assert recording_client.last_call == ("SCACHE.THRESHOLD", "ns", "0.5")
    recording_client.scache_get("ns", [1e-05, 1.0], threshold=2e-05)
    assert recording_client.last_call == ("SCACHE.GET", "ns", "0.00001", "1", "THRESHOLD", "0.00002")
    recording_client.pcache_create("ns", cost_per_hit=1e-06)
    assert recording_client.last_call == ("PCACHE.CREATE", "ns", "COST_PER_HIT", "0.000001")
    with pytest.raises(ValueError):
        recording_client.scache_threshold("ns", float("inf"))
    with pytest.raises(ValueError):
        recording_client.scache_set("ns", "q", "r", [float("nan")])


def test_vsim_amem_and_rag_vectors_are_plain(recording_client):
    for searching in ("VSIM.SEARCH", "AMEM.SEM.RECALL", "AMEM.SEARCH", "RAG.SEARCH"):
        recording_client.when(searching, [])
    recording_client.when("RAG.INGEST", 1)
    recording_client.vsim_set("i", "k", [1e-05, -2e-30, 3.0])
    assert recording_client.last_call == ("VSIM.SET", "i", "k", "0.00001", "-0.000000000000000000000000000002", "3")
    recording_client.vsim_search("i", 3, [1e-05, 1.0])
    assert recording_client.last_call == ("VSIM.SEARCH", "i", 3, "0.00001", "1")
    recording_client.amem_sem_store("a", "k", "c", [1e-05, 1.0])
    assert recording_client.last_call == ("AMEM.SEM.STORE", "a", "k", "c", "0.00001", "1")
    recording_client.amem_sem_recall("a", 2, [1e-05, 1.0])
    assert recording_client.last_call == ("AMEM.SEM.RECALL", "a", 2, "0.00001", "1")
    recording_client.amem_search("a", 2, [1e-05, 1.0])
    assert recording_client.last_call == ("AMEM.SEARCH", "a", 2, "0.00001", "1")
    recording_client.rag_ingest("p", "d", "text", vectors=[[1e-05, 1.0], [2.0, 3e-05]])
    assert recording_client.last_call == ("RAG.INGEST", "p", "d", "text", "VECTORS", "0.00001,1", "2,0.00003")
    recording_client.rag_search("p", "q", k=2, alpha=1e-05, vector=[1e-05, 1.0])
    assert recording_client.last_call == (
        "RAG.SEARCH", "p", "q", "K", 2, "ALPHA", "0.00001", "VECTOR", "0.00001,1"
    )


def test_ratelimit_and_time_series_floats_are_plain(recording_client):
    recording_client.ratelimit_create("rl", 10, 1000, algo="token_bucket", refill=1e-05, drain=2.5)
    assert recording_client.last_call == (
        "RATELIMIT.CREATE", "rl", 10, 1000, "ALGO", "token_bucket", "REFILL", "0.00001", "DRAIN", "2.5"
    )
    recording_client.ts_add("ts", 1e-300, 1000)
    assert recording_client.last_call == ("TS.ADD", "ts", 1000, "0." + "0" * 299 + "1")
    recording_client.ts_add("ts", 3.0)
    assert recording_client.last_call == ("TS.ADD", "ts", "*", "3")
    with pytest.raises(ValueError):
        recording_client.ts_add("ts", float("nan"))


def test_numbers_are_exported():
    assert hotkv.format_float is format_float
    assert hotkv.format_usd is format_usd
    assert issubclass(hotkv.DataError, ValueError)
    assert sys.version_info >= (3, 9)
