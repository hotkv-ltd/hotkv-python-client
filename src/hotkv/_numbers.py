# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""How the client writes a number on the wire.

Python's `repr(1e-05)` is `'1e-05'`. Several HotKV parsers refuse an exponent
(`LLMBUDGET.SET`'s cost limit and the `LLMSTATS.PRICE SET` prices are exact
fixed-point decimals, with at most 6 fractional digits), and others refuse a
leading `+` or a non-finite value. Every float argument the client sends is
therefore written by the one helper here, `format_float()`, as a plain decimal:

* digits, an optional `-` and an optional `.`; never an exponent, never a `+`,
  never a thousands separator or a locale's decimal comma;
* the shortest digits that read back as the same double
  (`float(format_float(x)) == x`), so `0.1` is `0.1`, not `0.1000000000000000055`;
* no trailing zeros and no trailing `.`: `3.0` is `3`; `-0.0` is `0`;
* NaN is refused everywhere; an infinity only where the caller says the command
  takes one (a sorted set's scores and score bounds), as `+inf` / `-inf`.

`format_usd()` is the fixed-point variant for the amounts the server stores as
whole micro-dollars. `format_component()` writes one component of a vector,
which the server reads as a 32-bit float.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Any, List, Sequence

from .exceptions import DataError

__all__ = [
    "format_float",
    "format_usd",
    "format_cost",
    "format_component",
    "format_vector",
]

_U64_MAX = (1 << 64) - 1
_MICRO = Decimal("0.000001")
# The smallest magnitude (a half of the smallest float32 subnormal, 2**-150)
# that does not round to zero as a 32-bit float; above this the server's
# float32 parser keeps the value, below it the value is zero.
_FLOAT32_ZERO_BELOW = 7.006492321624085e-46
# The first double that rounds to infinity as a 32-bit float; the server
# refuses it.
_FLOAT32_OVERFLOW_AT = 3.4028235677973366e38


def _plain(number: Decimal) -> str:
    """`number` as plain decimal text: no exponent, no trailing zeros, no `-0`."""
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-", "-0"):
        return "0"
    return text


def _refuse_bool(value: Any) -> None:
    if isinstance(value, bool):
        raise DataError(f"refusing to encode bool {value!r} as a number; pass 0 or 1")


def _decimal_of(value: Any) -> Decimal:
    """The exact decimal a caller's number stands for: a float by its shortest
    round-trip digits (what a person means by `0.1`), an int, `Decimal` or
    numeric string exactly. Raises for NaN and for anything that is not a number."""
    _refuse_bool(value)
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, (str, bytes, bytearray)):
        text = bytes(value).decode("utf-8", "replace") if not isinstance(value, str) else value
        if "_" in text:
            raise DataError(f"{text!r} is not a number")
        try:
            return Decimal(text.strip())
        except InvalidOperation:
            raise DataError(f"{text!r} is not a number") from None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise DataError(f"unsupported number type: {type(value).__name__}") from None
    return Decimal(repr(number))  # "nan", "inf" and "-inf" included


def format_float(value: Any, *, allow_inf: bool = False) -> str:
    """`value` as a plain decimal string (see the module notes).

    Raises `DataError` (also a `ValueError`) for NaN, and for an infinity unless
    `allow_inf` is true, in which case it is written `+inf` or `-inf`.
    """
    number = _decimal_of(value)
    if number.is_nan():
        raise DataError("NaN cannot be sent as a command argument")
    if number.is_infinite():
        if not allow_inf:
            raise DataError("an infinite value is not accepted by this command")
        return "-inf" if number < 0 else "+inf"
    return _plain(number)


def format_cost(value: Any) -> str:
    """A cost in dollars for `LLMSTATS.RECORD` / `LLMBUDGET.CONSUME`: a finite,
    non-negative plain decimal that fits the server's micro-dollar counters
    (those two commands take any decimal double, exponent or not)."""
    number = _decimal_of(value)
    if not number.is_finite():
        raise DataError("a cost must be a finite number")
    if number < 0:
        raise DataError("a cost cannot be negative")
    if number * 1_000_000 > _U64_MAX:
        raise DataError("a cost that large does not fit in micro-dollars")
    return _plain(number)


def format_usd(value: Any) -> str:
    """An amount in dollars as the fixed-point decimal the server stores as whole
    micro-dollars: plain digits, at most 6 fractional digits, rounded half up
    (`0.0000005` is `0.000001`), no sign, no exponent.

    Accepts a float, int, `Decimal` or numeric string. Raises `DataError` (also
    a `ValueError`) for a negative amount, NaN, an infinity and an amount that
    does not fit in an unsigned 64-bit count of micro-dollars.
    """
    number = _decimal_of(value)
    if not number.is_finite():
        raise DataError("an amount of money must be a finite number")
    if number < 0:
        raise DataError("an amount of money cannot be negative")
    if number > Decimal(_U64_MAX) / 1_000_000 + 1:
        raise DataError("that amount does not fit in an unsigned 64-bit count of micro-dollars")
    with localcontext() as ctx:
        ctx.prec = 60
        micros = number.quantize(_MICRO, rounding=ROUND_HALF_UP)
        if micros * 1_000_000 > _U64_MAX:
            raise DataError("that amount does not fit in an unsigned 64-bit count of micro-dollars")
    return _plain(micros)


def format_component(value: Any) -> str:
    """One component of a vector. The server reads it as a 32-bit float, which
    refuses a magnitude below the smallest subnormal as out of range, so such a
    component is written `0`; one too large for a float32 is an error."""
    number = _decimal_of(value)
    if not number.is_finite():
        raise DataError("a vector component must be a finite number")
    magnitude = abs(float(number))
    if magnitude >= _FLOAT32_OVERFLOW_AT:
        raise DataError(f"vector component {value!r} is too large for a 32-bit float")
    if magnitude < _FLOAT32_ZERO_BELOW:
        return "0"
    return _plain(number)


def format_vector(vector: Sequence[Any]) -> List[str]:
    """Each component of `vector`, written by `format_component()`."""
    return [format_component(component) for component in vector]
