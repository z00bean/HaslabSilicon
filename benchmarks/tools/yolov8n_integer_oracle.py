"""Vectorized, independent INT8 oracle for pinned multi-input graph checks.

The convolution uses FLOAT64 matrix products only as an exact integer carrier:
every INT8 product and bounded reduction fits exactly below 2**53. Fixed-point
rounding and LUT addressing remain integer operations. This is not a runtime
or target implementation.
"""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import numpy as np

from haslab_ref import hwc8_to_nchw, nchw_to_hwc8, quantize_int8


def conv2d_exact(source: np.ndarray, weights: np.ndarray, bias: np.ndarray,
                 *, stride: int, padding: int) -> np.ndarray:
    if source.ndim != 4 or source.shape[0] != 1 or source.dtype != np.int8:
        raise ValueError("expected batch-one NCHW INT8 source")
    if weights.ndim != 4 or weights.dtype != np.int8 or weights.shape[1] != source.shape[1]:
        raise ValueError("expected matching OIHW INT8 weights")
    if bias.shape != (weights.shape[0],) or bias.dtype != np.int32:
        raise ValueError("expected one INT32 bias per output channel")
    kernel = weights.shape[2]
    if kernel not in (1, 3) or weights.shape[3] != kernel or stride not in (1, 2):
        raise ValueError("unsupported pinned convolution")
    # Bound every partial integer sum as well as the final sum. This makes
    # the vectorized reduction equivalent to sequential INT32 accumulation.
    bound = source.shape[1] * kernel * kernel * 128 * 128 + int(np.max(np.abs(bias.astype(np.int64))))
    if bound >= (1 << 31):
        raise ValueError("integer reduction may overflow a sequential INT32 accumulator")
    padded = np.pad(source, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]
    _, _, out_height, out_width, _, _ = windows.shape
    matrix = windows.transpose(0, 2, 3, 1, 4, 5).reshape(-1, source.shape[1] * kernel * kernel)
    sums = matrix.astype(np.float64) @ weights.reshape(weights.shape[0], -1).astype(np.float64).T
    result = sums.astype(np.int64).reshape(1, out_height, out_width, -1).transpose(0, 3, 1, 2)
    result += bias.astype(np.int64).reshape(1, -1, 1, 1)
    if np.any(result < -(1 << 31)) or np.any(result > (1 << 31) - 1):
        raise OverflowError("INT32 accumulator overflow")
    return result.astype(np.int32)


def silu_exact(accumulators: np.ndarray, multipliers: np.ndarray,
               shifts: np.ndarray, lut: np.ndarray) -> np.ndarray:
    if accumulators.dtype != np.int32 or accumulators.shape[-1] != 8:
        raise ValueError("expected INT32 HWC8 accumulators")
    channels = accumulators.shape[2] * 8
    if multipliers.shape != (channels,) or shifts.shape != (channels,):
        raise ValueError("fixed-point parameters do not match channels")
    if lut.shape != (1024,) or lut.dtype != np.int8:
        raise ValueError("expected pinned 1024-entry INT8 LUT")
    values = accumulators.reshape(*accumulators.shape[:2], channels).astype(np.int64)
    product = values * multipliers.astype(np.int64).reshape(1, 1, channels)
    denominator = np.left_shift(np.int64(1), shifts.astype(np.int64)).reshape(1, 1, channels)
    magnitude = np.abs(product)
    quotient, remainder = np.divmod(magnitude, denominator)
    round_up = (remainder > denominator // 2) | ((remainder == denominator // 2) & (quotient & 1 != 0))
    grid = np.where(product < 0, -(quotient + round_up), quotient + round_up)
    grid = np.clip(grid, -512, 511).astype(np.intp)
    return lut[grid + 512].reshape(accumulators.shape)


def conv_silu(source: np.ndarray, layer: object) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(layer.weights, dtype=np.float32)
    scales = np.asarray(layer.weight_scales, dtype=np.float32)
    weight_i8 = quantize_int8(weights, scales.reshape(-1, 1, 1, 1))
    bias_i32 = np.rint(np.asarray(layer.bias, dtype=np.float64) /
                       (np.float64(np.float32(layer.input_scale)) * scales.astype(np.float64))).astype(np.int32)
    raw = conv2d_exact(source, weight_i8, bias_i32,
                       stride=int(getattr(layer, "stride", 2)),
                       padding=int(getattr(layer, "padding", 1)))
    physical = silu_exact(nchw_to_hwc8(raw), np.asarray(layer.multipliers, dtype=np.int64),
                          np.asarray(layer.shifts, dtype=np.int64),
                          np.asarray(layer.lut, dtype=np.int8))
    return physical, hwc8_to_nchw(physical, weights.shape[0])


def conv_raw(source: np.ndarray, layer: object) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(layer.weights, dtype=np.float32)
    scales = np.asarray(layer.weight_scales, dtype=np.float32)
    weight_i8 = quantize_int8(weights, scales.reshape(-1, 1, 1, 1))
    bias_i32 = np.rint(np.asarray(layer.bias, dtype=np.float64) /
                       (np.float64(np.float32(layer.input_scale)) * scales.astype(np.float64))).astype(np.int32)
    raw = conv2d_exact(source, weight_i8, bias_i32,
                       stride=layer.stride, padding=layer.padding)
    return nchw_to_hwc8(raw), raw
