"""HASLAB golden numerical reference model.

The package defines behavior for verification. It is not an optimized inference
runtime and does not imply that a corresponding FPGA or ASIC operation exists.
"""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from .errors import ArithmeticOverflowError, UnsupportedOperationError
from .fp8 import (
    FP8Format,
    decode_fp8,
    dequantize_fp8,
    encode_fp8,
    matmul_fp8,
    quantize_fp8,
)
from .int8 import (
    INT32_MAX,
    INT32_MIN,
    accumulate_i8,
    add_i8,
    add_i32_exact,
    clamp,
    clamp_int8,
    dequantize_int8,
    map_i8,
    matmul_i8,
    multiply_i8_to_i32,
    quantize_int8,
    requantize_int32,
    round_shift_rne,
    scale_i8,
)
from .layout import (
    hwc8_to_nchw,
    khwci8_to_oihw,
    nchw_to_hwc8,
    oihw_to_khwci8,
)
from .ops import (
    argmax,
    build_silu_lut,
    conv2d_i8,
    maxpool5_i8,
    random_sample,
    sigmoid,
    silu,
    silu_lut_i32,
    upsample2_nearest_i8,
)

__all__ = [
    "ArithmeticOverflowError",
    "FP8Format",
    "INT32_MAX",
    "INT32_MIN",
    "UnsupportedOperationError",
    "accumulate_i8",
    "add_i8",
    "add_i32_exact",
    "argmax",
    "build_silu_lut",
    "clamp",
    "clamp_int8",
    "conv2d_i8",
    "decode_fp8",
    "dequantize_fp8",
    "dequantize_int8",
    "encode_fp8",
    "hwc8_to_nchw",
    "khwci8_to_oihw",
    "map_i8",
    "matmul_fp8",
    "matmul_i8",
    "maxpool5_i8",
    "multiply_i8_to_i32",
    "nchw_to_hwc8",
    "oihw_to_khwci8",
    "quantize_fp8",
    "quantize_int8",
    "random_sample",
    "requantize_int32",
    "round_shift_rne",
    "scale_i8",
    "sigmoid",
    "silu",
    "silu_lut_i32",
    "upsample2_nearest_i8",
]
