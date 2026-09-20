"""HASLAB v0 logical tensor operations and activation references."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

from typing import Iterable

import numpy as np

from .errors import ArithmeticOverflowError, UnsupportedOperationError
from .int8 import (
    INT32_MAX,
    INT32_MIN,
    MULTIPLIER_MAX,
    SHIFT_MAX,
    _channel_parameters,
    _require_int8_values,
    _require_int32_values,
    quantize_int8,
    round_shift_rne,
)


def sigmoid(values: object) -> np.ndarray:
    """Stable binary64 logistic sigmoid used for host/reference calculations."""

    source = np.asarray(values, dtype=np.float64)
    result = np.empty(source.shape, dtype=np.float64)
    positive = source >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-source[positive]))
    exp_x = np.exp(source[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def silu(values: object) -> np.ndarray:
    """Binary64 SiLU reference: x * sigmoid(x)."""

    source = np.asarray(values, dtype=np.float64)
    return source * sigmoid(source)


def build_silu_lut(delta: float, output_scale: float) -> np.ndarray:
    """Build the 1,024-entry v0 SiLU table for indices -512..511."""

    delta32 = np.float32(delta)
    if not np.isfinite(delta32) or delta32 <= 0:
        raise ValueError("SiLU LUT delta must be finite and strictly positive")
    grid = np.arange(-512, 512, dtype=np.float64) * float(delta32)
    return quantize_int8(silu(grid), output_scale)


def silu_lut_i32(
    accumulators: object,
    multipliers: int | Iterable[int] | np.ndarray,
    shifts: int | Iterable[int] | np.ndarray,
    lut: object,
) -> np.ndarray:
    """Map INT32 accumulators through the v0 channel-scaled SiLU LUT."""

    source = _require_int32_values(accumulators, "accumulators")
    if source.ndim == 0:
        source = source.reshape((1,))
        scalar = True
    else:
        scalar = False
    table = _require_int8_values(lut, "lut").reshape(-1)
    if table.size != 1024:
        raise ValueError("SiLU LUT must contain exactly 1,024 INT8 entries")
    channels = source.shape[-1]
    multiplier_values = _channel_parameters(
        multipliers, channels, "multipliers", 0, MULTIPLIER_MAX
    )
    shift_values = _channel_parameters(shifts, channels, "shifts", 0, SHIFT_MAX)
    output = np.empty(source.shape, dtype=np.int8)
    for index in np.ndindex(source.shape):
        channel = index[-1]
        grid_value = round_shift_rne(
            int(source[index]) * int(multiplier_values[channel]),
            int(shift_values[channel]),
        )
        grid_value = min(511, max(-512, grid_value))
        output[index] = table[grid_value + 512]
    return output.reshape(()) if scalar else output


def conv2d_i8(
    activations: object,
    weights: object,
    bias: object | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Logical NCHW/OIHW dense convolution with exact INT32 accumulation.

    Reduction order is kernel row, kernel column, then input channel, matching
    the v0 contract. Inputs outside the logical image are exact INT8 zero.
    """

    source = _require_int8_values(activations, "activations")
    kernel = _require_int8_values(weights, "weights")
    if source.ndim != 4 or source.shape[0] != 1:
        raise ValueError("v0 convolution requires NCHW activations with N=1")
    if kernel.ndim != 4:
        raise ValueError("v0 convolution requires OIHW weights")
    if source.shape[1] != kernel.shape[1]:
        raise ValueError("convolution input-channel counts must match")
    if kernel.shape[2] != kernel.shape[3] or kernel.shape[2] not in (1, 3):
        raise ValueError("v0 convolution kernel must be square 1x1 or 3x3")
    if stride not in (1, 2):
        raise ValueError("v0 convolution stride must be 1 or 2")
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("convolution padding must be a nonnegative integer")

    bias_values: np.ndarray | None = None
    if bias is not None:
        bias_values = _require_int32_values(bias, "bias")
        if bias_values.shape != (kernel.shape[0],):
            raise ValueError("convolution bias must have one value per output channel")

    _, input_channels, input_h, input_w = source.shape
    output_channels, _, kernel_h, kernel_w = kernel.shape
    numerator_h = input_h + 2 * padding - kernel_h
    numerator_w = input_w + 2 * padding - kernel_w
    if numerator_h < 0 or numerator_w < 0:
        raise ValueError("kernel does not fit padded input")
    output_h = numerator_h // stride + 1
    output_w = numerator_w // stride + 1
    output = np.empty((1, output_channels, output_h, output_w), dtype=np.int32)

    for output_channel in range(output_channels):
        for output_y in range(output_h):
            for output_x in range(output_w):
                accumulator = 0
                for kernel_y in range(kernel_h):
                    input_y = output_y * stride + kernel_y - padding
                    for kernel_x in range(kernel_w):
                        input_x = output_x * stride + kernel_x - padding
                        for input_channel in range(input_channels):
                            activation = 0
                            if 0 <= input_y < input_h and 0 <= input_x < input_w:
                                activation = int(source[0, input_channel, input_y, input_x])
                            accumulator += activation * int(
                                kernel[output_channel, input_channel, kernel_y, kernel_x]
                            )
                            if not INT32_MIN <= accumulator <= INT32_MAX:
                                raise ArithmeticOverflowError(
                                    "signed INT32 convolution accumulation overflow"
                                )
                if bias_values is not None:
                    accumulator += int(bias_values[output_channel])
                    if not INT32_MIN <= accumulator <= INT32_MAX:
                        raise ArithmeticOverflowError("signed INT32 convolution bias overflow")
                output[0, output_channel, output_y, output_x] = accumulator
    return output


def maxpool5_i8(padded_hwc8: object, logical_channels: int | None = None) -> np.ndarray:
    """5x5 stride-one pool over an already padded HWC8 input."""

    source = _require_int8_values(padded_hwc8, "padded_hwc8")
    if source.ndim != 4 or source.shape[-1] != 8:
        raise ValueError("pool input must have shape [H+4,W+4,Cgroups,8]")
    if source.shape[0] < 5 or source.shape[1] < 5:
        raise ValueError("pool input must contain at least one 5x5 window")
    physical_channels = source.shape[2] * 8
    if logical_channels is None:
        logical_channels = physical_channels
    if not 0 < logical_channels <= physical_channels:
        raise ValueError("logical_channels exceeds HWC8 storage")
    output_h = source.shape[0] - 4
    output_w = source.shape[1] - 4
    output = np.empty((output_h, output_w, source.shape[2], 8), dtype=np.int8)
    for y in range(output_h):
        for x in range(output_w):
            output[y, x] = np.max(source[y : y + 5, x : x + 5], axis=(0, 1))
    for channel in range(logical_channels, physical_channels):
        output[:, :, channel // 8, channel % 8] = 0
    return output


def upsample2_nearest_i8(values: object, logical_channels: int | None = None) -> np.ndarray:
    """Replicate each HWC8 spatial value into an exact 2x2 block."""

    source = _require_int8_values(values, "values")
    if source.ndim != 4 or source.shape[-1] != 8:
        raise ValueError("upsample input must have shape [H,W,Cgroups,8]")
    physical_channels = source.shape[2] * 8
    if logical_channels is None:
        logical_channels = physical_channels
    if not 0 < logical_channels <= physical_channels:
        raise ValueError("logical_channels exceeds HWC8 storage")
    output = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
    for channel in range(logical_channels, physical_channels):
        output[:, :, channel // 8, channel % 8] = 0
    return output


def argmax(values: object, axis: int | None = None) -> np.ndarray | np.int64:
    """Return the lowest-index maximum; reject empty or NaN-bearing inputs."""

    source = np.asarray(values)
    if source.size == 0:
        raise ValueError("argmax input must not be empty")
    if np.issubdtype(source.dtype, np.floating) and np.any(np.isnan(source)):
        raise ValueError("argmax input must not contain NaN")
    return np.argmax(source, axis=axis)


def random_sample(*_args: object, **_kwargs: object) -> None:
    """Reject random sampling because v0 defines no RNG or distribution contract."""

    raise UnsupportedOperationError("random sampling is not part of HASLAB v0")
