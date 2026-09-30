# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

import numpy as np

from haslab_runtime import (Letterbox, decode_host_tail, map_boxes_to_original,
                            select_detections, unpack_boundaries)
from haslab_runtime.yolov8n_tail import BOUNDARY_SHAPES


class Yolov8nTailTests(unittest.TestCase):
    def test_int32_hwc8_per_channel_unpack(self):
        records, parts = [], []
        offset = 0
        for branch, (_, channels, height, width) in enumerate(BOUNDARY_SHAPES):
            physical = np.zeros((height, width, channels // 8, 8), dtype="<i4")
            physical[0, 0, 0, 0] = 2 + branch
            raw = physical.tobytes()
            parts.append(raw)
            scales = np.full(channels, 0.25, dtype=np.float32)
            scales[0] = 0.125
            records.append(dict(name=f"model.22.boundary{branch}", addend=offset,
                                bytes=len(raw), dtype="int32", layout="HWC8",
                                logical_shape=[1, channels, height, width],
                                physical_shape=list(physical.shape),
                                scales_binary32_per_channel=scales.tolist()))
            offset += len(raw)
        decoded = unpack_boundaries(b"".join(parts), records)
        for branch, values in enumerate(decoded):
            self.assertEqual(values.shape, BOUNDARY_SHAPES[branch])
            self.assertEqual(values[0, 0, 0, 0], (2 + branch) * 0.125)
            self.assertEqual(np.count_nonzero(values), 1)
        records[0]["scales_binary32_per_channel"][0] = 0
        with self.assertRaises(ValueError):
            unpack_boundaries(b"".join(parts), records)

    def test_uniform_dfl_grid_stride_and_class_sigmoid(self):
        boundaries = tuple(np.zeros(shape, dtype=np.float32) for shape in BOUNDARY_SHAPES)
        result, stages = decode_host_tail(boundaries, trace=True)
        self.assertEqual(result.shape, (1, 84, 2100))
        self.assertTrue(np.allclose(stages["dfl_distances"], 7.5))
        self.assertTrue(np.allclose(stages["dfl_softmax"], 1 / 16))
        self.assertTrue(np.array_equal(result[0, :2, 0], [4, 4]))
        self.assertTrue(np.array_equal(result[0, :2, 1600], [8, 8]))
        self.assertTrue(np.array_equal(result[0, :2, 2000], [16, 16]))
        self.assertTrue(np.array_equal(result[0, 4:, :], np.full((80, 2100), 0.5)))
        self.assertEqual(result[0, 2, 0], 120)
        boundaries[0][0, 64, 0, 0] = 100
        boundaries[0][0, 65, 0, 0] = -100
        result = decode_host_tail(boundaries)
        self.assertEqual(result[0, 4, 0], 1)
        self.assertLess(result[0, 5, 0], 1e-40)
        with self.assertRaises(ValueError):
            decode_host_tail(boundaries[:2])

    def test_letterbox_mapping_and_class_aware_nms(self):
        letterbox = Letterbox.centered(480, 640)
        self.assertEqual((letterbox.resized_height, letterbox.resized_width,
                          letterbox.top, letterbox.left), (240, 320, 40, 0))
        mapped = map_boxes_to_original(np.array([[60, 60, 100, 100]], dtype=np.float32), letterbox)
        np.testing.assert_array_equal(mapped[0], [120, 40, 200, 120])
        decoded = np.zeros((1, 84, 2100), dtype=np.float32)
        decoded[0, :4, :3] = np.array([[80, 80, 80], [80, 80, 80],
                                        [40, 40, 40], [40, 40, 40]], dtype=np.float32)
        decoded[0, 4, 0] = 0.9
        decoded[0, 4, 1] = 0.8  # identical same-class box suppressed
        decoded[0, 5, 2] = 0.7  # identical different-class box retained
        result = select_detections(decoded, letterbox, confidence=0.25)
        self.assertEqual(result.shape, (2, 6))
        np.testing.assert_array_equal(result[:, 5], [0, 1])
        np.testing.assert_array_equal(result[:, :4], [[120, 40, 200, 120]] * 2)
        np.testing.assert_array_equal(select_detections(decoded, letterbox, confidence=0.9),
                                      np.empty((0, 6), dtype=np.float32))
        decoded[0, 4, 1] = 0.9
        decoded[0, 0, 1] = 81  # still overlapping, but its mapped box differs
        tied = select_detections(decoded, letterbox, confidence=0.25)
        self.assertEqual(tied.shape, (2, 6))
        np.testing.assert_array_equal(tied[0, :4], [120, 40, 200, 120])
        self.assertEqual(select_detections(decoded, letterbox, max_detections=1).shape, (1, 6))


if __name__ == "__main__":
    unittest.main()
