# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import math
import unittest

import numpy as np

from haslab_ref import (
    FP8Format,
    decode_fp8,
    dequantize_fp8,
    encode_fp8,
    matmul_fp8,
    quantize_fp8,
)


class FP8ConversionTests(unittest.TestCase):
    def test_e4m3fn_known_encodings(self) -> None:
        raw = np.array([0x00, 0x80, 0x01, 0x38, 0x7E, 0x7F], dtype=np.uint8)
        values = decode_fp8(raw, FP8Format.E4M3FN)
        self.assertEqual(values[0], 0.0)
        self.assertFalse(np.signbit(values[0]))
        self.assertEqual(values[1], 0.0)
        self.assertTrue(np.signbit(values[1]))
        self.assertEqual(values[2], np.float32(2.0**-9))
        self.assertEqual(values[3], 1.0)
        self.assertEqual(values[4], 448.0)
        self.assertTrue(math.isnan(float(values[5])))

    def test_e5m2_known_encodings(self) -> None:
        raw = np.array([0x00, 0x80, 0x01, 0x3C, 0x7B, 0x7C, 0x7D], dtype=np.uint8)
        values = decode_fp8(raw, FP8Format.E5M2)
        self.assertFalse(np.signbit(values[0]))
        self.assertTrue(np.signbit(values[1]))
        self.assertEqual(values[2], np.float32(2.0**-16))
        self.assertEqual(values[3], 1.0)
        self.assertEqual(values[4], 57344.0)
        self.assertTrue(math.isinf(float(values[5])))
        self.assertTrue(math.isnan(float(values[6])))

    def test_all_finite_encodings_round_trip(self) -> None:
        for fmt in FP8Format:
            for raw in range(256):
                decoded = float(decode_fp8(np.uint8(raw), fmt))
                if math.isfinite(decoded):
                    encoded = int(encode_fp8(decoded, fmt))
                    self.assertEqual(encoded, raw, (fmt, raw, decoded))

    def test_special_values_have_explicit_encoding_policy(self) -> None:
        values = np.array([math.inf, -math.inf, math.nan, 0.0, -0.0])
        np.testing.assert_array_equal(
            encode_fp8(values, FP8Format.E4M3FN),
            np.array([0x7E, 0xFE, 0x7F, 0x00, 0x80], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            encode_fp8(values, FP8Format.E5M2),
            np.array([0x7B, 0xFB, 0x7F, 0x00, 0x80], dtype=np.uint8),
        )

    def test_round_to_nearest_ties_to_even(self) -> None:
        values = np.array([1.0625, 1.1875, -1.0625, -1.1875])
        np.testing.assert_array_equal(
            encode_fp8(values, FP8Format.E4M3FN),
            np.array([0x38, 0x3A, 0xB8, 0xBA], dtype=np.uint8),
        )

    def test_subnormal_ties_to_even(self) -> None:
        half_first = 2.0**-10
        halfway_first_second = 3.0 * 2.0**-10
        np.testing.assert_array_equal(
            encode_fp8(
                np.array([half_first, halfway_first_second]), FP8Format.E4M3FN
            ),
            np.array([0x00, 0x02], dtype=np.uint8),
        )

    def test_quantize_and_dequantize_scale(self) -> None:
        encoded = quantize_fp8(
            np.array([2.0, -4.0], dtype=np.float32), 2.0, FP8Format.E4M3FN
        )
        np.testing.assert_array_equal(encoded, np.array([0x38, 0xC0], dtype=np.uint8))
        decoded = dequantize_fp8(encoded, 2.0, FP8Format.E4M3FN)
        np.testing.assert_array_equal(decoded, np.array([2.0, -4.0], dtype=np.float32))

    def test_invalid_scale_and_storage_rejected(self) -> None:
        for scale in [0.0, -1.0, math.inf, math.nan]:
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    quantize_fp8([1.0], scale, FP8Format.E4M3FN)
        with self.assertRaises(TypeError):
            decode_fp8(np.array([1.0]), FP8Format.E4M3FN)
        with self.assertRaises(ValueError):
            decode_fp8(np.array([256]), FP8Format.E4M3FN)

    def test_fp8_matmul_uses_sequential_float32_accumulation(self) -> None:
        left = encode_fp8(np.array([[1.0, 2.0], [-1.0, 0.5]]), FP8Format.E4M3FN)
        right = encode_fp8(np.array([[3.0, -2.0], [4.0, 1.0]]), FP8Format.E4M3FN)
        result = matmul_fp8(left, right, FP8Format.E4M3FN, bias=np.array([1.0, -1.0]))
        np.testing.assert_array_equal(
            result, np.array([[12.0, -1.0], [0.0, 1.5]], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
