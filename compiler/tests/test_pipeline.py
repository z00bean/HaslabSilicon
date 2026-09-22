# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

import numpy as np

from haslab_compiler import ConvSiluLayerSpec, CompileError, compile_conv_silu_pipeline
from haslab_ref import (
    build_silu_lut,
    conv2d_i8,
    hwc8_to_nchw,
    nchw_to_hwc8,
    quantize_int8,
    silu_lut_i32,
)
from haslab_runtime import RuntimeStatus, SimulatorRuntime, load_hxb


def layer(name: str, h: int, w: int, inputs: int, outputs: int, scale_in: float, scale_out: float) -> ConvSiluLayerSpec:
    return ConvSiluLayerSpec(
        name=name,
        input_h=h,
        input_w=w,
        input_channels=inputs,
        weights=np.zeros((outputs, inputs, 3, 3), dtype=np.float32),
        bias=np.zeros(outputs, dtype=np.float32),
        input_scale=scale_in,
        weight_scales=np.ones(outputs, dtype=np.float32),
        output_scale=scale_out,
        multipliers=np.ones(outputs, dtype=np.int64),
        shifts=np.zeros(outputs, dtype=np.int64),
        lut=build_silu_lut(1.0, scale_out),
        node_names=(f"{name}.conv", f"{name}.sigmoid", f"{name}.mul"),
    )


def build_package() -> bytes:
    return compile_conv_silu_pipeline(
        layers=[
            layer("model.0", 320, 320, 3, 16, 1.0, 2.0),
            layer("model.1", 160, 160, 16, 32, 2.0, 3.0),
        ],
        source_model_sha256="00" * 32,
    )


class PipelineScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = build_package()
        cls.package = load_hxb(cls.raw)

    def test_two_layer_schedule_counts_are_exact(self) -> None:
        manifest = self.package.manifest
        schedule = manifest["schedule"]
        self.assertEqual(manifest["schema"], "haslab.conv-silu-pipeline.v1")
        self.assertEqual(manifest["commands"]["count"], 16245)
        self.assertEqual(len(manifest["relocations"]), 13386)
        self.assertEqual(
            schedule["opcode_counts"],
            {
                "CONV_I8": 1600,
                "DMA_COPY2D": 13386,
                "END": 1,
                "EPILOGUE": 1200,
                "FILL8": 58,
            },
        )
        self.assertEqual(
            schedule["dma_bytes"],
            {"constants": 8576, "input": 1376344, "output": 614400, "total": 1999320},
        )
        self.assertEqual(schedule["macs"], 40550400)
        first, second = schedule["layers"]
        self.assertEqual(first["command_count"], 8442)
        self.assertEqual(first["input_channel_chunks"], 1)
        self.assertEqual(second["command_count"], 7802)
        self.assertEqual(second["input_channel_chunks"], 2)
        self.assertEqual(second["boundary_fill_commands"], 19)

    def test_both_intermediates_are_retained(self) -> None:
        manifest = self.package.manifest
        self.assertEqual(manifest["buffers"]["output"]["bytes"], 614400)
        self.assertEqual(manifest["buffers"]["constants"]["bytes"], 8576)
        self.assertEqual(
            manifest["output"]["tensors"],
            [
                {
                    "addend": 0,
                    "bytes": 409600,
                    "dtype": "int8",
                    "layout": "HWC8",
                    "logical_shape": [1, 16, 160, 160],
                    "name": "model.0",
                    "physical_shape": [160, 160, 2, 8],
                    "scale_binary32": 2.0,
                },
                {
                    "addend": 409600,
                    "bytes": 204800,
                    "dtype": "int8",
                    "layout": "HWC8",
                    "logical_shape": [1, 32, 80, 80],
                    "name": "model.1",
                    "physical_shape": [80, 80, 4, 8],
                    "scale_binary32": 3.0,
                },
            ],
        )

    def test_package_is_byte_deterministic(self) -> None:
        self.assertEqual(self.raw, build_package())

    def test_disconnected_scale_is_rejected(self) -> None:
        with self.assertRaisesRegex(CompileError, "input scale"):
            compile_conv_silu_pipeline(
                layers=[
                    layer("a", 320, 320, 3, 16, 1.0, 2.0),
                    layer("b", 160, 160, 16, 32, 2.5, 3.0),
                ],
                source_model_sha256="00" * 32,
            )

    def test_small_two_layer_pipeline_executes_exactly(self) -> None:
        first = layer("a", 32, 32, 3, 16, 1.0, 1.0)
        second = layer("b", 16, 16, 16, 32, 1.0, 1.0)
        first_weights = (
            np.arange(16 * 3 * 3 * 3, dtype=np.int32).reshape(16, 3, 3, 3) % 5 - 2
        ).astype(np.float32)
        second_weights = (
            np.arange(32 * 16 * 3 * 3, dtype=np.int32).reshape(32, 16, 3, 3) % 3 - 1
        ).astype(np.float32)
        first = ConvSiluLayerSpec(**{**first.__dict__, "weights": first_weights})
        second = ConvSiluLayerSpec(**{**second.__dict__, "weights": second_weights})
        package = compile_conv_silu_pipeline(
            layers=[first, second], source_model_sha256="11" * 32
        )
        source = (
            np.arange(1 * 3 * 32 * 32, dtype=np.int32).reshape(1, 3, 32, 32) % 17 - 8
        ).astype(np.int8)

        def golden(values: np.ndarray, spec: ConvSiluLayerSpec) -> tuple[np.ndarray, np.ndarray]:
            weights = quantize_int8(spec.weights, 1.0)
            accumulators = conv2d_i8(
                values,
                weights,
                np.zeros(np.asarray(spec.weights).shape[0], dtype=np.int32),
                stride=2,
                padding=1,
            )
            physical_accumulators = nchw_to_hwc8(accumulators)
            physical = np.empty(physical_accumulators.shape, dtype=np.int8)
            for group in range(physical.shape[2]):
                physical[:, :, group : group + 1, :] = silu_lut_i32(
                    physical_accumulators[:, :, group : group + 1, :],
                    np.ones(8, dtype=np.int64),
                    np.zeros(8, dtype=np.int64),
                    spec.lut,
                )
            return physical, hwc8_to_nchw(physical, np.asarray(spec.weights).shape[0])

        expected_first, logical_first = golden(source, first)
        expected_second, _ = golden(logical_first, second)
        runtime = SimulatorRuntime()
        parsed = runtime.load_model(package)
        runtime.bind_input(np.ascontiguousarray(nchw_to_hwc8(source)).tobytes())
        completion = runtime.wait(runtime.submit())
        self.assertIs(completion.status, RuntimeStatus.SUCCESS)
        assert completion.output is not None
        tensors = parsed.manifest["output"]["tensors"]
        actual_first = np.frombuffer(
            completion.output[: tensors[0]["bytes"]], dtype=np.int8
        ).reshape(expected_first.shape)
        start = tensors[1]["addend"]
        actual_second = np.frombuffer(
            completion.output[start : start + tensors[1]["bytes"]], dtype=np.int8
        ).reshape(expected_second.shape)
        np.testing.assert_array_equal(actual_first, expected_first)
        np.testing.assert_array_equal(actual_second, expected_second)


if __name__ == "__main__":
    unittest.main()
