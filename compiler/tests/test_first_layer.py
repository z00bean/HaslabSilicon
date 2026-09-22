# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

import numpy as np

from haslab_compiler import compile_conv_silu_first_layer
from haslab_ref import build_silu_lut
from haslab_runtime import load_hxb


def build_package() -> bytes:
    return compile_conv_silu_first_layer(
        weights=np.zeros((16, 3, 3, 3), dtype=np.float32),
        bias=np.zeros(16, dtype=np.float32),
        input_scale=1.0,
        weight_scales=np.ones(16, dtype=np.float32),
        output_scale=1.0,
        multipliers=np.ones(16, dtype=np.int64),
        shifts=np.zeros(16, dtype=np.int64),
        lut=build_silu_lut(1.0, 1.0),
        source_model_sha256="00" * 32,
    )


class FirstLayerScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = build_package()
        cls.package = load_hxb(cls.raw)

    def test_schedule_counts_and_extents_are_exact(self) -> None:
        manifest = self.package.manifest
        schedule = manifest["schedule"]
        self.assertEqual(manifest["schema"], "haslab.first-layer.v1")
        self.assertEqual(manifest["commands"]["count"], 8884)
        self.assertEqual(manifest["commands"]["bytes"], 8884 * 128)
        self.assertEqual(len(manifest["relocations"]), 7205)
        self.assertEqual(schedule["spatial_tiles"], 400)
        self.assertEqual(schedule["tile_executions"], 800)
        self.assertEqual(schedule["boundary_fill_commands"], 78)
        self.assertEqual(
            schedule["opcode_counts"],
            {
                "CONV_I8": 800,
                "DMA_COPY2D": 7205,
                "END": 1,
                "EPILOGUE": 800,
                "FILL8": 78,
            },
        )
        self.assertEqual(
            schedule["dma_bytes"],
            {
                "constants": 2432,
                "input": 1838736,
                "output": 409600,
                "total": 2250768,
            },
        )
        self.assertEqual(schedule["macs"], 11059200)
        self.assertEqual(manifest["buffers"]["input"]["bytes"], 819200)
        self.assertEqual(manifest["buffers"]["output"]["bytes"], 409600)
        self.assertEqual(len(self.package.constants), 2432)

    def test_full_layer_package_is_byte_deterministic(self) -> None:
        self.assertEqual(self.raw, build_package())


if __name__ == "__main__":
    unittest.main()
