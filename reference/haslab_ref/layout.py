"""Conversions between logical tensors and HASLAB blocked layouts."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import numpy as np


def _require_numeric(array: np.ndarray, name: str) -> None:
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise TypeError(f"{name} must have a numeric dtype")


def nchw_to_hwc8(values: object, pad_value: int | float = 0) -> np.ndarray:
    """Convert logical NCHW with N=1 to physical [H,W,Cgroups,8]."""

    source = np.asarray(values)
    _require_numeric(source, "values")
    if source.ndim != 4 or source.shape[0] != 1:
        raise ValueError("HWC8 conversion requires rank-4 NCHW with N=1")
    _, channels, height, width = source.shape
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("tensor dimensions must be positive")
    groups = (channels + 7) // 8
    output = np.full((height, width, groups, 8), pad_value, dtype=source.dtype)
    for channel in range(channels):
        output[:, :, channel // 8, channel % 8] = source[0, channel, :, :]
    return output


def hwc8_to_nchw(values: object, logical_channels: int) -> np.ndarray:
    """Convert physical [H,W,Cgroups,8] to logical NCHW with N=1."""

    source = np.asarray(values)
    _require_numeric(source, "values")
    if source.ndim != 4 or source.shape[-1] != 8:
        raise ValueError("HWC8 storage must have shape [H,W,Cgroups,8]")
    if not isinstance(logical_channels, (int, np.integer)) or logical_channels <= 0:
        raise ValueError("logical_channels must be a positive integer")
    if logical_channels > source.shape[2] * 8:
        raise ValueError("logical channel count exceeds physical HWC8 storage")
    height, width = source.shape[:2]
    output = np.empty((1, logical_channels, height, width), dtype=source.dtype)
    for channel in range(logical_channels):
        output[0, channel, :, :] = source[:, :, channel // 8, channel % 8]
    return output


def oihw_to_khwci8(values: object, pad_value: int | float = 0) -> np.ndarray:
    """Pack logical [Cout,Cin,Kh,Kw] weights as [Kh,Kw,CinPad,CoutGroups,8]."""

    source = np.asarray(values)
    _require_numeric(source, "values")
    if source.ndim != 4:
        raise ValueError("weight conversion requires rank-4 OIHW")
    output_channels, input_channels, kernel_h, kernel_w = source.shape
    if min(source.shape) <= 0:
        raise ValueError("weight dimensions must be positive")
    input_padded = ((input_channels + 7) // 8) * 8
    output_groups = (output_channels + 7) // 8
    packed = np.full(
        (kernel_h, kernel_w, input_padded, output_groups, 8),
        pad_value,
        dtype=source.dtype,
    )
    for output_channel in range(output_channels):
        group, lane = divmod(output_channel, 8)
        for input_channel in range(input_channels):
            packed[:, :, input_channel, group, lane] = source[
                output_channel, input_channel, :, :
            ]
    return packed


def khwci8_to_oihw(
    values: object,
    logical_output_channels: int,
    logical_input_channels: int,
) -> np.ndarray:
    """Unpack [Kh,Kw,CinPad,CoutGroups,8] weights to logical OIHW."""

    source = np.asarray(values)
    _require_numeric(source, "values")
    if source.ndim != 5 or source.shape[-1] != 8:
        raise ValueError("KHWCI8 storage must have shape [Kh,Kw,CinPad,CoutGroups,8]")
    if logical_output_channels <= 0 or logical_input_channels <= 0:
        raise ValueError("logical channel counts must be positive")
    if logical_output_channels > source.shape[3] * 8:
        raise ValueError("logical output channels exceed packed storage")
    if logical_input_channels > source.shape[2]:
        raise ValueError("logical input channels exceed packed storage")
    kernel_h, kernel_w = source.shape[:2]
    unpacked = np.empty(
        (logical_output_channels, logical_input_channels, kernel_h, kernel_w),
        dtype=source.dtype,
    )
    for output_channel in range(logical_output_channels):
        group, lane = divmod(output_channel, 8)
        for input_channel in range(logical_input_channels):
            unpacked[output_channel, input_channel, :, :] = source[
                :, :, input_channel, group, lane
            ]
    return unpacked
