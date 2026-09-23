# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

import numpy as np

from haslab_compiler import (
    C2fBlockSpec,
    C2fConvSpec,
    CompileError,
    ConvSiluLayerSpec,
    compile_first_c2f,
)
from haslab_ref import (
    add_i8,
    build_silu_lut,
    conv2d_i8,
    hwc8_to_nchw,
    map_i8,
    nchw_to_hwc8,
    quantize_int8,
    silu_lut_i32,
)
from haslab_runtime import RuntimeStatus, SimulatorRuntime, load_hxb


def stem_layer(name: str, h: int, w: int, inputs: int, outputs: int) -> ConvSiluLayerSpec:
    weights = (
        np.arange(outputs * inputs * 3 * 3, dtype=np.int32).reshape(outputs, inputs, 3, 3)
        % 5
        - 2
    ).astype(np.float32)
    return ConvSiluLayerSpec(
        name=name,
        input_h=h,
        input_w=w,
        input_channels=inputs,
        weights=weights,
        bias=np.zeros(outputs, dtype=np.float32),
        input_scale=1.0,
        weight_scales=np.ones(outputs, dtype=np.float32),
        output_scale=1.0,
        multipliers=np.ones(outputs, dtype=np.int64),
        shifts=np.zeros(outputs, dtype=np.int64),
        lut=build_silu_lut(1.0, 1.0),
        node_names=(f"{name}.conv", f"{name}.sigmoid", f"{name}.mul"),
    )


def c2f_conv(
    name: str,
    inputs: int,
    outputs: int,
    kernel: int,
    input_scale: float,
    output_scale: float,
) -> C2fConvSpec:
    weights = (
        np.arange(outputs * inputs * kernel * kernel, dtype=np.int32).reshape(
            outputs, inputs, kernel, kernel
        )
        % 3
        - 1
    ).astype(np.float32)
    return C2fConvSpec(
        name=name,
        input_h=8,
        input_w=8,
        input_channels=inputs,
        weights=weights,
        bias=np.zeros(outputs, dtype=np.float32),
        input_scale=input_scale,
        weight_scales=np.ones(outputs, dtype=np.float32),
        output_scale=output_scale,
        multipliers=np.ones(outputs, dtype=np.int64),
        shifts=np.zeros(outputs, dtype=np.int64),
        lut=build_silu_lut(1.0, output_scale),
        node_names=(f"{name}.conv", f"{name}.sigmoid", f"{name}.mul"),
        kernel=kernel,
        stride=1,
        padding=(kernel - 1) // 2,
    )


def fixture() -> tuple[list[ConvSiluLayerSpec], C2fBlockSpec]:
    stem = [stem_layer("model.0", 32, 32, 3, 16), stem_layer("model.1", 16, 16, 16, 32)]
    block = C2fBlockSpec(
        cv1=c2f_conv("model.2.cv1", 32, 32, 1, 1.0, 2.0),
        bottleneck_cv1=c2f_conv("model.2.m.0.cv1", 16, 16, 3, 2.0, 3.0),
        bottleneck_cv2=c2f_conv("model.2.m.0.cv2", 16, 16, 3, 3.0, 4.0),
        cv2=c2f_conv("model.2.cv2", 48, 32, 1, 6.0, 7.0),
        residual_scale=5.0,
        concat_scale=6.0,
        split_nodes=("model.2.constant", "model.2.split"),
        add_node="model.2.add",
        concat_node="model.2.concat",
    )
    return stem, block


def conv_silu(source: np.ndarray, spec: object) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(spec.weights, dtype=np.float32)
    weights_i8 = quantize_int8(weights, np.ones(weights.shape[0], dtype=np.float32).reshape(-1, 1, 1, 1))
    accumulators = conv2d_i8(
        source,
        weights_i8,
        np.zeros(weights.shape[0], dtype=np.int32),
        stride=getattr(spec, "stride", 2),
        padding=getattr(spec, "padding", 1),
    )
    physical_accumulators = nchw_to_hwc8(accumulators)
    physical = np.empty(physical_accumulators.shape, dtype=np.int8)
    for group in range(physical.shape[2]):
        physical[:, :, group : group + 1, :] = silu_lut_i32(
            physical_accumulators[:, :, group : group + 1, :],
            np.ones(8, dtype=np.int64),
            np.zeros(8, dtype=np.int64),
            np.asarray(spec.lut, dtype=np.int8),
        )
    return physical, hwc8_to_nchw(physical, weights.shape[0])


class C2fScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stem, cls.block = fixture()
        cls.raw = compile_first_c2f(
            stem_layers=cls.stem, block=cls.block, source_model_sha256="22" * 32
        )
        cls.package = load_hxb(cls.raw)

    def test_manifest_records_views_and_branch_operations(self) -> None:
        manifest = self.package.manifest
        self.assertEqual(manifest["schema"], "haslab.first-c2f.v1")
        self.assertEqual(
            [view["name"] for view in manifest["output"]["views"]],
            ["model.2.split0", "model.2.split1"],
        )
        self.assertTrue(all(view["byte_allocation"] == 0 for view in manifest["output"]["views"]))
        self.assertEqual(
            [item["kind"] for item in manifest["schedule"]["c2f_operations"]],
            ["conv_silu", "conv_silu", "conv_silu", "residual_add", "channel_concat", "conv_silu"],
        )
        self.assertEqual(manifest["commands"]["count"], len(self.package.commands) // 128)

    def test_package_is_byte_deterministic(self) -> None:
        stem, block = fixture()
        self.assertEqual(
            self.raw,
            compile_first_c2f(stem_layers=stem, block=block, source_model_sha256="22" * 32),
        )

    def test_bad_final_input_scale_is_rejected(self) -> None:
        stem, block = fixture()
        bad_cv2 = C2fConvSpec(**{**block.cv2.__dict__, "input_scale": 6.5})
        with self.assertRaisesRegex(CompileError, "input scale"):
            compile_first_c2f(
                stem_layers=stem,
                block=C2fBlockSpec(**{**block.__dict__, "cv2": bad_cv2}),
                source_model_sha256="22" * 32,
            )

    def test_complete_branch_executes_exactly(self) -> None:
        source = (
            np.arange(1 * 3 * 32 * 32, dtype=np.int32).reshape(1, 3, 32, 32) % 17 - 8
        ).astype(np.int8)
        runtime = SimulatorRuntime()
        parsed = runtime.load_model(self.raw)
        runtime.bind_input(np.ascontiguousarray(nchw_to_hwc8(source)).tobytes())
        completion = runtime.wait(runtime.submit())
        self.assertIs(completion.status, RuntimeStatus.SUCCESS)
        assert completion.output is not None

        first_physical, first = conv_silu(source, self.stem[0])
        second_physical, second = conv_silu(first, self.stem[1])
        cv1_physical, _ = conv_silu(second, self.block.cv1)
        split0 = cv1_physical[:, :, :2, :]
        split1 = cv1_physical[:, :, 2:, :]
        bottleneck1_physical, bottleneck1 = conv_silu(
            hwc8_to_nchw(split1, 16), self.block.bottleneck_cv1
        )
        bottleneck2_physical, _ = conv_silu(bottleneck1, self.block.bottleneck_cv2)

        operations = parsed.manifest["schedule"]["c2f_operations"]
        add_fixed = operations[3]["fixed_point"]
        residual = add_i8(
            split1,
            bottleneck2_physical,
            add_fixed["left_multiplier"],
            add_fixed["right_multiplier"],
            add_fixed["shift"],
        )
        concat_parts = []
        for values, fixed in zip((split0, split1, residual), operations[4]["fixed_point"]):
            concat_parts.append(map_i8(values, fixed["multiplier"], fixed["shift"]))
        concat = np.concatenate(concat_parts, axis=2)
        final_physical, _ = conv_silu(hwc8_to_nchw(concat, 48), self.block.cv2)
        expected = [
            first_physical,
            second_physical,
            cv1_physical,
            bottleneck1_physical,
            bottleneck2_physical,
            residual,
            concat,
            final_physical,
        ]
        for record, golden in zip(parsed.manifest["output"]["tensors"], expected):
            start = int(record["addend"])
            actual = np.frombuffer(
                completion.output[start : start + int(record["bytes"])], dtype=np.int8
            ).reshape(record["physical_shape"])
            np.testing.assert_array_equal(actual, golden, err_msg=record["name"])


if __name__ == "__main__":
    unittest.main()
