# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

import numpy as np

from haslab_ref import (
    hwc8_to_nchw,
    khwci8_to_oihw,
    nchw_to_hwc8,
    oihw_to_khwci8,
)


class ActivationLayoutTests(unittest.TestCase):
    def test_nchw_hwc8_round_trip_and_padding(self) -> None:
        logical = np.arange(1 * 3 * 2 * 3, dtype=np.int8).reshape(1, 3, 2, 3)
        physical = nchw_to_hwc8(logical)
        self.assertEqual(physical.shape, (2, 3, 1, 8))
        np.testing.assert_array_equal(physical[:, :, 0, 3:], 0)
        np.testing.assert_array_equal(hwc8_to_nchw(physical, 3), logical)

    def test_contract_byte_offset_example(self) -> None:
        logical = np.zeros((1, 3, 1, 2), dtype=np.int8)
        logical[0, 2, 0, 1] = 99
        physical = nchw_to_hwc8(logical)
        self.assertEqual(physical.tobytes()[10], 99)
        self.assertEqual(physical.strides[0], 16)

    def test_more_than_one_channel_group(self) -> None:
        logical = np.arange(9, dtype=np.int8).reshape(1, 9, 1, 1)
        physical = nchw_to_hwc8(logical, pad_value=-7)
        self.assertEqual(physical.shape, (1, 1, 2, 8))
        self.assertEqual(physical[0, 0, 1, 0], 8)
        np.testing.assert_array_equal(physical[0, 0, 1, 1:], -7)

    def test_invalid_activation_shapes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            nchw_to_hwc8(np.zeros((2, 3, 4, 4), dtype=np.int8))
        with self.assertRaises(ValueError):
            hwc8_to_nchw(np.zeros((2, 3, 8), dtype=np.int8), 3)
        with self.assertRaises(ValueError):
            hwc8_to_nchw(np.zeros((1, 1, 1, 8), dtype=np.int8), 9)


class WeightLayoutTests(unittest.TestCase):
    def test_oihw_khwci8_round_trip(self) -> None:
        logical = np.arange(10 * 3 * 3 * 3, dtype=np.int16).reshape(10, 3, 3, 3)
        packed = oihw_to_khwci8(logical)
        self.assertEqual(packed.shape, (3, 3, 8, 2, 8))
        np.testing.assert_array_equal(packed[:, :, 3:, :, :], 0)
        np.testing.assert_array_equal(khwci8_to_oihw(packed, 10, 3), logical)

    def test_weight_address_order(self) -> None:
        logical = np.zeros((8, 1, 1, 1), dtype=np.int8)
        logical[:, 0, 0, 0] = np.arange(8, dtype=np.int8)
        packed = oihw_to_khwci8(logical)
        np.testing.assert_array_equal(packed[0, 0, 0, 0, :], np.arange(8, dtype=np.int8))

    def test_reference_tile_size(self) -> None:
        logical = np.zeros((8, 32, 3, 3), dtype=np.int8)
        packed = oihw_to_khwci8(logical)
        self.assertEqual(packed.nbytes, 2304)


if __name__ == "__main__":
    unittest.main()
