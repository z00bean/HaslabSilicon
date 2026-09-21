# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

import numpy as np

from haslab_compiler import compile_conv_silu_slice
from haslab_ref import (
    build_silu_lut,
    conv2d_i8,
    nchw_to_hwc8,
    quantize_int8,
    silu_lut_i32,
)
from haslab_runtime import (
    HxbFormatError,
    RuntimeErrorCode,
    RuntimeFailure,
    RuntimeStatus,
    SimulatorRuntime,
    load_hxb,
)


def fixture() -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.zeros((16, 3, 3, 3), dtype=np.float32)
    for output in range(8):
        weights[output, output % 3, :, :] = (output % 3) - 1
        weights[output, (output + 1) % 3, 1, 1] = 2
    bias = np.arange(-8, 8, dtype=np.float32)
    scales = np.ones(16, dtype=np.float32)
    multipliers = np.ones(16, dtype=np.int64)
    shifts = np.zeros(16, dtype=np.int64)
    lut = build_silu_lut(1.0, 1.0)
    package = compile_conv_silu_slice(
        weights=weights,
        bias=bias,
        input_scale=1.0,
        weight_scales=scales,
        output_scale=1.0,
        multipliers=multipliers,
        shifts=shifts,
        lut=lut,
        source_model_sha256="00" * 32,
    )
    values = (np.arange(1 * 3 * 17 * 17, dtype=np.int32) % 15 - 7).astype(np.int8)
    patch = values.reshape(1, 3, 17, 17)
    return package, patch, weights, bias, lut


class VerticalSliceRuntimeTests(unittest.TestCase):
    def test_compiler_package_executes_bit_exactly_against_golden_model(self) -> None:
        package, patch, weights, bias, lut = fixture()
        self.assertEqual(package, fixture()[0])
        runtime = SimulatorRuntime()
        loaded = runtime.load_model(package)
        runtime.bind_input(nchw_to_hwc8(patch).tobytes())
        token = runtime.submit()
        completion = runtime.wait(token)
        self.assertEqual(completion.status, RuntimeStatus.SUCCESS)
        self.assertIsNone(completion.error)
        self.assertEqual(loaded.manifest["commands"]["count"], 8)

        quantized_weights = quantize_int8(weights[:8], np.ones((8, 1, 1, 1), dtype=np.float32))
        quantized_bias = np.rint(bias[:8]).astype(np.int32)
        accumulators = conv2d_i8(
            patch,
            quantized_weights,
            quantized_bias,
            stride=2,
            padding=0,
        )
        expected = silu_lut_i32(
            nchw_to_hwc8(accumulators),
            np.ones(8, dtype=np.int64),
            np.zeros(8, dtype=np.int64),
            lut,
        )
        self.assertEqual(completion.output, expected.tobytes())

    def test_loader_rejects_corruption_before_execution(self) -> None:
        package = bytearray(fixture()[0])
        package[64] ^= 1
        with self.assertRaises(HxbFormatError):
            load_hxb(package)

    def test_binding_size_is_exact_and_reset_invalidates_token(self) -> None:
        package, patch, _, _, _ = fixture()
        runtime = SimulatorRuntime()
        runtime.load_model(package)
        with self.assertRaises(RuntimeFailure) as caught:
            runtime.bind_input(b"short")
        self.assertEqual(caught.exception.code, RuntimeErrorCode.BAD_BINDING)
        runtime.bind_input(nchw_to_hwc8(patch).tobytes())
        token = runtime.submit()
        runtime.reset()
        with self.assertRaises(RuntimeFailure) as caught:
            runtime.wait(token)
        self.assertEqual(caught.exception.code, RuntimeErrorCode.STALE_TOKEN)

    def test_arithmetic_fault_is_returned_as_structured_completion(self) -> None:
        weights = np.full((16, 3, 3, 3), 127.0, dtype=np.float32)
        bias = np.zeros(16, dtype=np.float32)
        bias[0] = np.float32(2_147_483_520)
        package = compile_conv_silu_slice(
            weights=weights,
            bias=bias,
            input_scale=1.0,
            weight_scales=np.ones(16, dtype=np.float32),
            output_scale=1.0,
            multipliers=np.ones(16, dtype=np.int64),
            shifts=np.zeros(16, dtype=np.int64),
            lut=build_silu_lut(1.0, 1.0),
            source_model_sha256="00" * 32,
        )
        patch = np.full((1, 3, 17, 17), 127, dtype=np.int8)
        runtime = SimulatorRuntime()
        runtime.load_model(package)
        runtime.bind_input(nchw_to_hwc8(patch).tobytes())
        completion = runtime.wait(runtime.submit())
        self.assertEqual(completion.status, RuntimeStatus.FAILURE)
        self.assertEqual(completion.error["runtime_code"], RuntimeErrorCode.DEVICE_FAULT.value)
        self.assertEqual(completion.error["sequence"], 6)


if __name__ == "__main__":
    unittest.main()
