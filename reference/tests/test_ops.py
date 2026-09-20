# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

import numpy as np

from haslab_ref import (
    UnsupportedOperationError,
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


class ActivationTests(unittest.TestCase):
    def test_sigmoid_is_stable_at_large_magnitude(self) -> None:
        result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.5)
        self.assertEqual(result[2], 1.0)

    def test_silu_known_values(self) -> None:
        result = silu(np.array([-1.0, 0.0, 1.0]))
        expected = np.array([-0.2689414213699951, 0.0, 0.7310585786300049])
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-15)

    def test_silu_lut_generation_and_index_clamping(self) -> None:
        lut = build_silu_lut(delta=1.0, output_scale=1.0)
        self.assertEqual(lut.shape, (1024,))
        self.assertEqual(lut.dtype, np.int8)
        self.assertEqual(int(lut[512]), 0)
        accumulators = np.array([[-600, 0, 600]], dtype=np.int32)
        result = silu_lut_i32(accumulators, [1, 1, 1], [0, 0, 0], lut)
        np.testing.assert_array_equal(
            result, np.array([[lut[0], lut[512], lut[1023]]], dtype=np.int8)
        )

    def test_silu_lut_validates_size(self) -> None:
        with self.assertRaises(ValueError):
            silu_lut_i32([0], 1, 0, np.zeros(10, dtype=np.int8))


class TensorOperationTests(unittest.TestCase):
    def test_one_by_one_convolution(self) -> None:
        activations = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.int8)
        weights = np.array([[[[2]], [[-1]]], [[[1]], [[1]]]], dtype=np.int8)
        result = conv2d_i8(activations, weights, bias=np.array([5, -2], dtype=np.int32))
        expected = np.array([[[[4, 5]], [[2, 4]]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_three_by_three_padding_and_reduction_order(self) -> None:
        activations = np.arange(1, 10, dtype=np.int8).reshape(1, 1, 3, 3)
        weights = np.ones((1, 1, 3, 3), dtype=np.int8)
        result = conv2d_i8(activations, weights, stride=1, padding=1)
        expected = np.array([[[[12, 21, 16], [27, 45, 33], [24, 39, 28]]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_stride_two_convolution(self) -> None:
        activations = np.arange(1, 26, dtype=np.int8).reshape(1, 1, 5, 5)
        weights = np.ones((1, 1, 3, 3), dtype=np.int8)
        result = conv2d_i8(activations, weights, stride=2)
        np.testing.assert_array_equal(result, np.array([[[[63, 81], [153, 171]]]], dtype=np.int32))

    def test_convolution_rejects_unsupported_kernel(self) -> None:
        with self.assertRaises(ValueError):
            conv2d_i8(
                np.ones((1, 1, 5, 5), dtype=np.int8),
                np.ones((1, 1, 5, 5), dtype=np.int8),
            )

    def test_pool_uses_compiled_halo_and_zeroes_padding_lanes(self) -> None:
        padded = np.full((6, 6, 1, 8), -128, dtype=np.int8)
        padded[2, 2, 0, 0] = 7
        padded[3, 3, 0, 1] = 9
        padded[:, :, 0, 3:] = 100
        result = maxpool5_i8(padded, logical_channels=3)
        self.assertEqual(result.shape, (2, 2, 1, 8))
        np.testing.assert_array_equal(result[:, :, 0, 0], 7)
        np.testing.assert_array_equal(result[:, :, 0, 1], 9)
        np.testing.assert_array_equal(result[:, :, 0, 3:], 0)

    def test_nearest_upsample_replication_and_padding(self) -> None:
        source = np.zeros((1, 2, 1, 8), dtype=np.int8)
        source[0, 0, 0, 0] = 3
        source[0, 1, 0, 0] = 4
        source[:, :, 0, 4:] = 99
        result = upsample2_nearest_i8(source, logical_channels=4)
        self.assertEqual(result.shape, (2, 4, 1, 8))
        np.testing.assert_array_equal(result[:, :, 0, 0], [[3, 3, 4, 4], [3, 3, 4, 4]])
        np.testing.assert_array_equal(result[:, :, 0, 4:], 0)


class DecisionOperationTests(unittest.TestCase):
    def test_argmax_returns_first_tie(self) -> None:
        values = np.array([[1, 5, 5], [7, 7, 1]], dtype=np.int8)
        np.testing.assert_array_equal(argmax(values, axis=1), np.array([1, 0]))
        self.assertEqual(int(argmax(values)), 3)

    def test_argmax_rejects_nan_and_empty(self) -> None:
        with self.assertRaises(ValueError):
            argmax(np.array([1.0, np.nan]))
        with self.assertRaises(ValueError):
            argmax(np.array([]))

    def test_random_sampling_is_explicitly_unsupported(self) -> None:
        with self.assertRaises(UnsupportedOperationError):
            random_sample([0.5, 0.5], seed=1)


if __name__ == "__main__":
    unittest.main()
