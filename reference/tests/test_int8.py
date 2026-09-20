# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import math
import unittest

import numpy as np

from haslab_ref import (
    INT32_MAX,
    INT32_MIN,
    ArithmeticOverflowError,
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


class Int8ConversionTests(unittest.TestCase):
    def test_quantization_rne_and_saturation(self) -> None:
        values = np.array([-200.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 200.0])
        expected = np.array([-128, -2, -2, 0, 0, 2, 2, 127], dtype=np.int8)
        np.testing.assert_array_equal(quantize_int8(values, 1.0), expected)

    def test_dequantization_is_float32(self) -> None:
        result = dequantize_int8(np.array([-2, 3], dtype=np.int8), 0.25)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.array([-0.5, 0.75], dtype=np.float32))

    def test_nonfinite_input_and_bad_scales_rejected(self) -> None:
        for value in [math.inf, -math.inf, math.nan]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    quantize_int8([value], 1.0)
        for scale in [0.0, -1.0, math.inf, math.nan]:
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    quantize_int8([1.0], scale)

    def test_clamping(self) -> None:
        np.testing.assert_array_equal(
            clamp_int8(np.array([-1000, -128, 0, 127, 1000])),
            np.array([-128, -128, 0, 127, 127], dtype=np.int8),
        )
        np.testing.assert_array_equal(clamp([0, 5, 10], 2, 8), [2, 5, 8])
        with self.assertRaises(ValueError):
            clamp([1], 2, 1)


class ExactIntegerArithmeticTests(unittest.TestCase):
    def test_signed_rne_examples_and_ties(self) -> None:
        cases = {
            (5, 1): 2,
            (7, 1): 4,
            (-5, 1): -2,
            (-7, 1): -4,
            (1, 1): 0,
            (-1, 1): 0,
            (6, 1): 3,
            (123, 0): 123,
        }
        for arguments, expected in cases.items():
            with self.subTest(arguments=arguments):
                self.assertEqual(round_shift_rne(*arguments), expected)

    def test_requantize_per_channel_and_saturate(self) -> None:
        values = np.array([[5, 7, 1_000, -1_000]], dtype=np.int32)
        result = requantize_int32(values, [1, 1, 1, 1], [1, 1, 0, 0])
        np.testing.assert_array_equal(result, np.array([[2, 4, 127, -128]], dtype=np.int8))

    def test_map_and_scale_are_identical(self) -> None:
        values = np.array([[-3, 5]], dtype=np.int8)
        np.testing.assert_array_equal(
            map_i8(values, [2, 3], [1, 1]), np.array([[-3, 8]], dtype=np.int8)
        )
        np.testing.assert_array_equal(
            scale_i8(values, [2, 3], [1, 1]), map_i8(values, [2, 3], [1, 1])
        )

    def test_scaled_add_rounds_only_once(self) -> None:
        result = add_i8(
            np.array([[1, -1]], dtype=np.int8),
            np.array([[1, -1]], dtype=np.int8),
            [1, 1],
            [2, 2],
            [1, 1],
        )
        np.testing.assert_array_equal(result, np.array([[2, -2]], dtype=np.int8))

    def test_elementwise_multiply_widens(self) -> None:
        result = multiply_i8_to_i32(
            np.array([-128, 127], dtype=np.int8), np.array([-128, 127], dtype=np.int8)
        )
        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(result, [16384, 16129])

    def test_exact_int32_add_and_overflow(self) -> None:
        np.testing.assert_array_equal(add_i32_exact([1, -2], [3, 4]), [4, 2])
        with self.assertRaises(ArithmeticOverflowError):
            add_i32_exact([INT32_MAX], [1])
        with self.assertRaises(ArithmeticOverflowError):
            add_i32_exact([INT32_MIN], [-1])

    def test_accumulation_order_and_overflow(self) -> None:
        self.assertEqual(int(accumulate_i8([2, -3, 4], [5, 6, -7], 10)), -26)
        with self.assertRaises(ArithmeticOverflowError):
            accumulate_i8([1], [1], INT32_MAX)

    def test_matmul_and_bias(self) -> None:
        left = np.array([[1, 2, 3], [-1, 0, 2]], dtype=np.int8)
        right = np.array([[2, 1], [0, -1], [3, 2]], dtype=np.int8)
        result = matmul_i8(left, right, bias=np.array([10, -5], dtype=np.int32))
        np.testing.assert_array_equal(result, np.array([[21, 0], [14, -2]], dtype=np.int32))

    def test_matmul_bias_overflow(self) -> None:
        with self.assertRaises(ArithmeticOverflowError):
            matmul_i8(
                np.array([[1]], dtype=np.int8),
                np.array([[1]], dtype=np.int8),
                bias=np.array([INT32_MAX], dtype=np.int32),
            )

    def test_shape_and_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            add_i8([1, 2], [1], 1, 1, 0)
        with self.assertRaises(ValueError):
            requantize_int32([[1, 2]], [1, 2, 3], 0)
        with self.assertRaises(ValueError):
            round_shift_rne(1, 63)
        with self.assertRaises(ValueError):
            matmul_i8(np.ones((2, 3), dtype=np.int8), np.ones((2, 2), dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
