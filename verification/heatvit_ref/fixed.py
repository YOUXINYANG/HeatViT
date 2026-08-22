"""Bit-exact integer fixed-point reference helpers.

Only explicit integers are accepted; Python floats are rejected so the
golden model cannot silently drift from the RTL fixed-point contract.
"""

import math


def _check_int(name, value):
    if isinstance(value, float):
        raise TypeError(f"{name} must be an integer, got float")


def round_shift_away(value: int, shift: int) -> int:
    """Arithmetic shift with round-to-nearest, ties away from zero."""
    _check_int("value", value)
    _check_int("shift", shift)
    if shift < 0:
        return value << (-shift)
    if shift == 0:
        return value
    magnitude = abs(value)
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    return -rounded if value < 0 else rounded


def sat_signed(value: int, bits: int) -> int:
    """Saturate to the signed two's-complement range of ``bits`` bits."""
    _check_int("value", value)
    _check_int("bits", bits)
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    return min(upper, max(lower, value))


def requant(value: int, src_exp: int, dst_exp: int, bits: int) -> int:
    """Rescale by 2**(dst_exp-src_exp), rounding away, then saturate."""
    _check_int("value", value)
    _check_int("src_exp", src_exp)
    _check_int("dst_exp", dst_exp)
    _check_int("bits", bits)
    shift = dst_exp - src_exp
    shifted = round_shift_away(value, shift) if shift >= 0 else value << (-shift)
    return sat_signed(shifted, bits)


def udiv(numerator: int, denominator: int):
    """Unsigned floor division returning ``(quotient, remainder)``."""
    _check_int("numerator", numerator)
    _check_int("denominator", denominator)
    if denominator == 0:
        raise ValueError("division by zero")
    return divmod(numerator, denominator)


def round_div(numerator: int, denominator: int) -> int:
    """Nearest-integer division; ties round away from zero."""
    _check_int("numerator", numerator)
    _check_int("denominator", denominator)
    if denominator == 0:
        raise ValueError("division by zero")
    sign = -1 if numerator < 0 else 1
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    quotient, remainder = divmod(abs(numerator), denominator)
    if 2 * remainder >= denominator:
        quotient += 1
    return sign * quotient


def isqrt(radicand: int):
    """Floor integer square root returning ``(root, remainder)``."""
    _check_int("radicand", radicand)
    if radicand < 0:
        raise ValueError("negative radicand")
    radicand = int(radicand)
    root = math.isqrt(radicand)
    return root, radicand - root * root
