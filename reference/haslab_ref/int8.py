"""Exact HASLAB v0 integer quantization and accumulation semantics."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

from typing import Iterable

import numpy as np

from .errors import ArithmeticOverflowError

INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
MULTIPLIER_MAX = (1 << 31) - 1
SHIFT_MAX = 62


def _require_integer_array(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    return array


def _require_int8_values(values: object, name: str) -> np.ndarray:
    array = _require_integer_array(values, name)
    if np.any(array < INT8_MIN) or np.any(array > INT8_MAX):
        raise ValueError(f"{name} values must fit signed INT8")
    return array.astype(np.int8, copy=False)


def _require_int32_values(values: object, name: str) -> np.ndarray:
    array = _require_integer_array(values, name)
    if np.any(array < INT32_MIN) or np.any(array > INT32_MAX):
        raise ValueError(f"{name} values must fit signed INT32")
    return array.astype(np.int32, copy=False)


def _binary32_scale(scale: float | np.ndarray) -> np.ndarray:
    result = np.asarray(scale, dtype=np.float32)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("scale must be finite and strictly positive")
    return result


def clamp(values: object, minimum: int | float, maximum: int | float) -> np.ndarray:
    """Clamp values inclusively; reject an inverted interval."""

    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum")
    return np.clip(np.asarray(values), minimum, maximum)


def clamp_int8(values: object) -> np.ndarray:
    """Saturate integer values to signed INT8."""

    array = _require_integer_array(values, "values")
    return np.clip(array, INT8_MIN, INT8_MAX).astype(np.int8)


def quantize_int8(values: object, scale: float | np.ndarray) -> np.ndarray:
    """Symmetric INT8 quantization using binary32 scale and RNE."""

    scale32 = _binary32_scale(scale)
    real = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(real)):
        raise ValueError("INT8 quantization input must be finite")
    rounded = np.rint(real / scale32.astype(np.float64))
    return np.clip(rounded, INT8_MIN, INT8_MAX).astype(np.int8)


def dequantize_int8(values: object, scale: float | np.ndarray) -> np.ndarray:
    """Return float32 values using symmetric zero-point-zero dequantization."""

    array = _require_int8_values(values, "values")
    scale32 = _binary32_scale(scale)
    return np.asarray(array.astype(np.float32) * scale32, dtype=np.float32)


def round_shift_rne(value: int, shift: int) -> int:
    """Return RNE(value / 2**shift), symmetrically for signed integers."""

    if not isinstance(value, (int, np.integer)):
        raise TypeError("value must be an integer")
    if not isinstance(shift, (int, np.integer)):
        raise TypeError("shift must be an integer")
    shift = int(shift)
    if not 0 <= shift <= SHIFT_MAX:
        raise ValueError(f"shift must be in the range 0..{SHIFT_MAX}")
    value = int(value)
    if shift == 0:
        return value

    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    denominator = 1 << shift
    quotient, remainder = divmod(magnitude, denominator)
    doubled = remainder << 1
    if doubled > denominator or (doubled == denominator and (quotient & 1)):
        quotient += 1
    return sign * quotient


def _channel_parameters(
    parameter: int | Iterable[int] | np.ndarray,
    channels: int,
    name: str,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    array = _require_integer_array(parameter, name).astype(np.int64, copy=False)
    if array.ndim == 0:
        array = np.full((channels,), int(array), dtype=np.int64)
    elif array.shape != (channels,):
        raise ValueError(f"{name} must be scalar or have one value per final-axis channel")
    if np.any(array < minimum) or np.any(array > maximum):
        raise ValueError(f"{name} must be in the range {minimum}..{maximum}")
    return array


def requantize_int32(
    values: object,
    multipliers: int | Iterable[int] | np.ndarray,
    shifts: int | Iterable[int] | np.ndarray,
) -> np.ndarray:
    """Apply exact M/2**R fixed-point scaling and signed INT8 saturation."""

    source = _require_int32_values(values, "values")
    if source.ndim == 0:
        source = source.reshape((1,))
        scalar = True
    else:
        scalar = False
    channels = source.shape[-1]
    multiplier_values = _channel_parameters(
        multipliers, channels, "multipliers", 0, MULTIPLIER_MAX
    )
    shift_values = _channel_parameters(shifts, channels, "shifts", 0, SHIFT_MAX)
    output = np.empty(source.shape, dtype=np.int8)
    for index in np.ndindex(source.shape):
        channel = index[-1]
        product = int(source[index]) * int(multiplier_values[channel])
        rounded = round_shift_rne(product, int(shift_values[channel]))
        output[index] = min(INT8_MAX, max(INT8_MIN, rounded))
    return output.reshape(()) if scalar else output


def map_i8(
    values: object,
    multipliers: int | Iterable[int] | np.ndarray,
    shifts: int | Iterable[int] | np.ndarray,
) -> np.ndarray:
    """Scale an INT8 tensor once and saturate to INT8."""

    source = _require_int8_values(values, "values").astype(np.int32)
    return requantize_int32(source, multipliers, shifts)


def scale_i8(
    values: object,
    multipliers: int | Iterable[int] | np.ndarray,
    shifts: int | Iterable[int] | np.ndarray,
) -> np.ndarray:
    """Named alias for the v0 MAP_I8 numerical operation."""

    return map_i8(values, multipliers, shifts)


def add_i8(
    left: object,
    right: object,
    left_multipliers: int | Iterable[int] | np.ndarray,
    right_multipliers: int | Iterable[int] | np.ndarray,
    shifts: int | Iterable[int] | np.ndarray,
) -> np.ndarray:
    """Scaled equal-shape residual add with one final RNE and saturation."""

    a = _require_int8_values(left, "left")
    b = _require_int8_values(right, "right")
    if a.shape != b.shape:
        raise ValueError("ADD_I8 inputs must have identical shapes")
    if a.ndim == 0:
        a = a.reshape((1,))
        b = b.reshape((1,))
        scalar = True
    else:
        scalar = False
    channels = a.shape[-1]
    ma = _channel_parameters(
        left_multipliers, channels, "left_multipliers", 0, MULTIPLIER_MAX
    )
    mb = _channel_parameters(
        right_multipliers, channels, "right_multipliers", 0, MULTIPLIER_MAX
    )
    shift_values = _channel_parameters(shifts, channels, "shifts", 0, SHIFT_MAX)
    output = np.empty(a.shape, dtype=np.int8)
    for index in np.ndindex(a.shape):
        channel = index[-1]
        numerator = int(a[index]) * int(ma[channel]) + int(b[index]) * int(mb[channel])
        rounded = round_shift_rne(numerator, int(shift_values[channel]))
        output[index] = min(INT8_MAX, max(INT8_MIN, rounded))
    return output.reshape(()) if scalar else output


def multiply_i8_to_i32(left: object, right: object) -> np.ndarray:
    """Elementwise INT8 multiply, widened exactly to INT32 without broadcasting."""

    a = _require_int8_values(left, "left")
    b = _require_int8_values(right, "right")
    if a.shape != b.shape:
        raise ValueError("elementwise multiply inputs must have identical shapes")
    return a.astype(np.int32) * b.astype(np.int32)


def add_i32_exact(left: object, right: object) -> np.ndarray:
    """Elementwise INT32 add with explicit overflow rejection and no broadcasting."""

    a = _require_int32_values(left, "left")
    b = _require_int32_values(right, "right")
    if a.shape != b.shape:
        raise ValueError("elementwise add inputs must have identical shapes")
    wide = a.astype(np.int64) + b.astype(np.int64)
    if np.any(wide < INT32_MIN) or np.any(wide > INT32_MAX):
        raise ArithmeticOverflowError("signed INT32 elementwise addition overflow")
    return wide.astype(np.int32)


def accumulate_i8(left: object, right: object, initial: int = 0) -> np.int32:
    """Multiply equal-length INT8 vectors and add each product in order."""

    a = _require_int8_values(left, "left").reshape(-1)
    b = _require_int8_values(right, "right").reshape(-1)
    if a.shape != b.shape:
        raise ValueError("accumulation vectors must have identical lengths")
    if not INT32_MIN <= int(initial) <= INT32_MAX:
        raise ValueError("initial accumulator must fit signed INT32")
    accumulator = int(initial)
    for index in range(a.size):
        accumulator += int(a[index]) * int(b[index])
        if not INT32_MIN <= accumulator <= INT32_MAX:
            raise ArithmeticOverflowError(
                f"signed INT32 accumulation overflow at reduction index {index}"
            )
    return np.int32(accumulator)


def matmul_i8(left: object, right: object, bias: object | None = None) -> np.ndarray:
    """Rank-2 INT8 matrix multiplication with ordered INT32 accumulation."""

    a = _require_int8_values(left, "left")
    b = _require_int8_values(right, "right")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("INT8 matmul inputs must be rank-2")
    if a.shape[1] != b.shape[0]:
        raise ValueError("INT8 matmul inner dimensions must match")

    bias_values: np.ndarray | None = None
    if bias is not None:
        bias_values = _require_int32_values(bias, "bias")
        if bias_values.shape != (b.shape[1],):
            raise ValueError("INT8 matmul bias must have one value per output column")

    output = np.empty((a.shape[0], b.shape[1]), dtype=np.int32)
    for row in range(a.shape[0]):
        for column in range(b.shape[1]):
            accumulator = int(accumulate_i8(a[row, :], b[:, column]))
            if bias_values is not None:
                accumulator += int(bias_values[column])
                if not INT32_MIN <= accumulator <= INT32_MAX:
                    raise ArithmeticOverflowError("signed INT32 bias addition overflow")
            output[row, column] = accumulator
    return output
