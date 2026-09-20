"""Bit-exact FP8 conversion and sequential FP32 accumulation."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import bisect
import math
from enum import Enum
from functools import lru_cache
from typing import Iterable

import numpy as np


class FP8Format(str, Enum):
    """FP8 encodings reserved by the HASLAB architecture."""

    E4M3FN = "e4m3fn"
    E5M2 = "e5m2"


def _coerce_format(fmt: FP8Format | str) -> FP8Format:
    try:
        return FP8Format(fmt)
    except ValueError as exc:
        raise ValueError(f"unsupported FP8 format: {fmt!r}") from exc


def _format_fields(fmt: FP8Format) -> tuple[int, int, int]:
    if fmt is FP8Format.E4M3FN:
        return 4, 3, 7
    return 5, 2, 15


def _decode_scalar(raw: int, fmt: FP8Format) -> float:
    if not 0 <= raw <= 0xFF:
        raise ValueError("FP8 byte must be in the range 0..255")

    exponent_bits, fraction_bits, bias = _format_fields(fmt)
    sign = -1.0 if raw & 0x80 else 1.0
    exponent_mask = (1 << exponent_bits) - 1
    fraction_mask = (1 << fraction_bits) - 1
    exponent = (raw >> fraction_bits) & exponent_mask
    fraction = raw & fraction_mask

    if exponent == 0:
        if fraction == 0:
            return math.copysign(0.0, sign)
        value = math.ldexp(fraction / (1 << fraction_bits), 1 - bias)
        return math.copysign(value, sign)

    if fmt is FP8Format.E4M3FN:
        if exponent == exponent_mask and fraction == fraction_mask:
            return math.nan
    elif exponent == exponent_mask:
        if fraction == 0:
            return math.copysign(math.inf, sign)
        return math.nan

    significand = 1.0 + fraction / (1 << fraction_bits)
    return math.copysign(math.ldexp(significand, exponent - bias), sign)


@lru_cache(maxsize=2)
def _positive_finite_table(fmt: FP8Format) -> tuple[tuple[float, int], ...]:
    values: list[tuple[float, int]] = []
    for raw in range(0x80):
        value = _decode_scalar(raw, fmt)
        if math.isfinite(value):
            values.append((value, raw))
    values.sort(key=lambda item: item[0])
    return tuple(values)


def _encode_scalar(value: float, fmt: FP8Format) -> int:
    value = float(value)
    if math.isnan(value):
        return 0x7F

    negative = math.copysign(1.0, value) < 0.0
    sign_bit = 0x80 if negative else 0
    magnitude = abs(value)
    table = _positive_finite_table(fmt)

    if magnitude == 0.0:
        return sign_bit

    max_value, max_raw = table[-1]
    if math.isinf(magnitude) or magnitude >= max_value:
        return sign_bit | max_raw

    numeric_values = [item[0] for item in table]
    upper_index = bisect.bisect_left(numeric_values, magnitude)
    if numeric_values[upper_index] == magnitude:
        chosen_raw = table[upper_index][1]
    else:
        lower_value, lower_raw = table[upper_index - 1]
        upper_value, upper_raw = table[upper_index]
        lower_distance = magnitude - lower_value
        upper_distance = upper_value - magnitude
        if lower_distance < upper_distance:
            chosen_raw = lower_raw
        elif upper_distance < lower_distance:
            chosen_raw = upper_raw
        else:
            # Round-to-nearest, ties-to-even uses the stored significand LSB.
            chosen_raw = lower_raw if (lower_raw & 1) == 0 else upper_raw
    return sign_bit | chosen_raw


def decode_fp8(values: int | Iterable[int] | np.ndarray, fmt: FP8Format | str) -> np.ndarray:
    """Decode FP8 bytes to float32, preserving signed zero and special values."""

    resolved = _coerce_format(fmt)
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("FP8 storage must be an integer byte array")
    if np.any(array < 0) or np.any(array > 255):
        raise ValueError("FP8 storage values must be in the range 0..255")
    output = np.empty(array.shape, dtype=np.float32)
    for index in np.ndindex(array.shape):
        output[index] = np.float32(_decode_scalar(int(array[index]), resolved))
    return output


def encode_fp8(values: float | Iterable[float] | np.ndarray, fmt: FP8Format | str) -> np.ndarray:
    """Encode to FP8 using RNE, finite saturation, and canonical positive NaN."""

    resolved = _coerce_format(fmt)
    array = np.asarray(values, dtype=np.float64)
    output = np.empty(array.shape, dtype=np.uint8)
    for index in np.ndindex(array.shape):
        output[index] = _encode_scalar(float(array[index]), resolved)
    return output


def _binary32_scale(scale: float | np.ndarray) -> np.ndarray:
    result = np.asarray(scale, dtype=np.float32)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("FP8 scale must be finite and strictly positive")
    return result


def quantize_fp8(
    values: float | Iterable[float] | np.ndarray,
    scale: float | np.ndarray,
    fmt: FP8Format | str,
) -> np.ndarray:
    """Scale real values and encode FP8 bytes."""

    scale32 = _binary32_scale(scale)
    scaled = np.asarray(values, dtype=np.float64) / scale32.astype(np.float64)
    return encode_fp8(scaled, fmt)


def dequantize_fp8(
    values: int | Iterable[int] | np.ndarray,
    scale: float | np.ndarray,
    fmt: FP8Format | str,
) -> np.ndarray:
    """Decode FP8 and apply binary32 scale, rounding results to float32."""

    scale32 = _binary32_scale(scale)
    decoded = decode_fp8(values, fmt)
    return np.asarray(decoded * scale32, dtype=np.float32)


def matmul_fp8(
    left: np.ndarray,
    right: np.ndarray,
    fmt: FP8Format | str,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Decode FP8 matrices and accumulate K terms sequentially in float32."""

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("FP8 matmul inputs must be rank-2")
    if left_array.shape[1] != right_array.shape[0]:
        raise ValueError("FP8 matmul inner dimensions must match")
    a = decode_fp8(left_array, fmt)
    b = decode_fp8(right_array, fmt)
    output = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)

    bias32: np.ndarray | None = None
    if bias is not None:
        bias32 = np.asarray(bias, dtype=np.float32)
        if bias32.shape != (b.shape[1],):
            raise ValueError("FP8 matmul bias must have one value per output column")

    with np.errstate(over="ignore", invalid="ignore"):
        for row in range(a.shape[0]):
            for column in range(b.shape[1]):
                accumulator = np.float32(0.0)
                for reduction in range(a.shape[1]):
                    product = np.float32(a[row, reduction] * b[reduction, column])
                    accumulator = np.float32(accumulator + product)
                if bias32 is not None:
                    accumulator = np.float32(accumulator + bias32[column])
                output[row, column] = accumulator
    return output
