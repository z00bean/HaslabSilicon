# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in ("reference", "simulation", "runtime", "benchmarks/tools"):
    sys.path.insert(0, str(ROOT / directory))

from verify_yolov8n_host_tail import compare  # noqa: E402


class HostTailComparisonTests(unittest.TestCase):
    def test_both_sides_can_have_no_detections(self):
        empty = np.empty((0, 6), dtype=np.float32)
        result = compare(empty, empty.copy())
        self.assertEqual(result["values"], 0)
        self.assertEqual(result["maximum_absolute_error"], 0.0)
        self.assertTrue(result["allclose_atol_0.0001_rtol_0.00001"])


if __name__ == "__main__":
    unittest.main()
