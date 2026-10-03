# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "benchmarks/tools"))

from haslab_ref import conv2d_i8, silu_lut_i32  # noqa: E402
from yolov8n_integer_oracle import conv2d_exact, silu_exact  # noqa: E402


class VectorizedOracleTests(unittest.TestCase):
    def test_convolution_matches_scalar_reference_with_border_and_stride(self):
        rng = np.random.default_rng(73)
        source = rng.integers(-128, 128, (1, 3, 5, 6), dtype=np.int8)
        weights = rng.integers(-128, 128, (8, 3, 3, 3), dtype=np.int8)
        bias = rng.integers(-1000, 1000, 8, dtype=np.int32)
        for stride in (1, 2):
            expected = conv2d_i8(source, weights, bias, stride=stride, padding=1)
            actual = conv2d_exact(source, weights, bias, stride=stride, padding=1)
            np.testing.assert_array_equal(actual, expected)

    def test_silu_integer_rounding_matches_scalar_reference(self):
        rng = np.random.default_rng(991)
        values = rng.integers(-100000, 100000, (3, 4, 2, 8), dtype=np.int32)
        multipliers = rng.integers(0, 1 << 25, 16, dtype=np.int64)
        shifts = rng.integers(0, 30, 16, dtype=np.int64)
        lut = rng.integers(-128, 128, 1024, dtype=np.int8)
        expected = np.concatenate(
            [silu_lut_i32(values[:, :, group:group + 1],
                            multipliers[group * 8:(group + 1) * 8],
                            shifts[group * 8:(group + 1) * 8], lut)
             for group in range(2)], axis=2
        )
        np.testing.assert_array_equal(silu_exact(values, multipliers, shifts, lut), expected)


if __name__ == "__main__":
    unittest.main()
