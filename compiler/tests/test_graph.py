# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

import numpy as np

from haslab_compiler import (
    C2fBlockSpec,
    C2fConvSpec,
    C2fStageSpec,
    ConvSiluLayerSpec,
    build_two_c2f_graph,
    compile_graph,
    first_stage,
)
from haslab_ref import build_silu_lut, nchw_to_hwc8
from haslab_runtime import RuntimeStatus, SimulatorRuntime, load_hxb


def stem(name: str, spatial: int, inputs: int, outputs: int) -> ConvSiluLayerSpec:
    weights = (
        np.arange(outputs * inputs * 3 * 3, dtype=np.int32).reshape(outputs, inputs, 3, 3)
        % 3
        - 1
    ).astype(np.float32)
    return ConvSiluLayerSpec(
        name=name,
        input_h=spatial,
        input_w=spatial,
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


def conv(
    name: str,
    spatial: int,
    inputs: int,
    outputs: int,
    kernel: int,
    stride: int = 1,
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
        input_h=spatial,
        input_w=spatial,
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
        kernel=kernel,
        stride=stride,
        padding=(kernel - 1) // 2,
    )


def graph_fixture():
    stem_layers = [stem("model.0", 64, 3, 16), stem("model.1", 32, 16, 32)]
    first_block = C2fBlockSpec(
        cv1=conv("model.2.cv1", 16, 32, 32, 1),
        bottleneck_cv1=conv("model.2.m.0.cv1", 16, 16, 16, 3),
        bottleneck_cv2=conv("model.2.m.0.cv2", 16, 16, 16, 3),
        cv2=conv("model.2.cv2", 16, 48, 32, 1),
        residual_scale=1.0,
        concat_scale=1.0,
        split_nodes=("model.2.constant", "model.2.split"),
        add_node="model.2.add",
        concat_node="model.2.concat",
    )
    second = C2fStageSpec(
        cv1=conv("model.4.cv1", 8, 64, 64, 1),
        bottlenecks=(
            (conv("model.4.m.0.cv1", 8, 32, 32, 3), conv("model.4.m.0.cv2", 8, 32, 32, 3)),
            (conv("model.4.m.1.cv1", 8, 32, 32, 3), conv("model.4.m.1.cv2", 8, 32, 32, 3)),
        ),
        cv2=conv("model.4.cv2", 8, 128, 64, 1),
        residual_scales=(1.0, 1.0),
        concat_scale=1.0,
        split_nodes=("model.4.constant", "model.4.split"),
        add_nodes=("model.4.add0", "model.4.add1"),
        concat_node="model.4.concat",
    )
    graph = build_two_c2f_graph(
        first=first_stage(first_block),
        downsample=conv("model.3", 16, 32, 64, 3, stride=2),
        second=second,
    )
    return stem_layers, graph


class GraphScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stem, cls.graph = graph_fixture()
        cls.diagnostic_raw = compile_graph(
            stem_layers=cls.stem,
            graph=cls.graph,
            source_model_sha256="33" * 32,
            allocation_mode="diagnostic",
        )
        cls.release_raw = compile_graph(
            stem_layers=cls.stem,
            graph=cls.graph,
            source_model_sha256="33" * 32,
            allocation_mode="release",
        )
        cls.diagnostic = load_hxb(cls.diagnostic_raw)
        cls.release = load_hxb(cls.release_raw)

    def test_release_allocator_reuses_only_disjoint_lifetimes(self) -> None:
        diagnostic = self.diagnostic.manifest["schedule"]["allocation"]
        release = self.release.manifest["schedule"]["allocation"]
        self.assertEqual(diagnostic["mode"], "diagnostic")
        self.assertEqual(release["mode"], "release")
        self.assertLess(release["selected_peak_output_bytes"], diagnostic["selected_peak_output_bytes"])
        self.assertEqual(
            release["reuse_savings_bytes"],
            diagnostic["selected_peak_output_bytes"] - release["selected_peak_output_bytes"],
        )
        plan = self.release.manifest["output"]["allocation_plan"]
        for index, left in enumerate(plan):
            for right in plan[index + 1 :]:
                memory_overlaps = max(left["addend"], right["addend"]) < min(
                    left["addend"] + left["bytes"], right["addend"] + right["bytes"]
                )
                lifetime_overlaps = max(left["first_operation"], right["first_operation"]) <= min(
                    left["last_operation"], right["last_operation"]
                )
                self.assertFalse(memory_overlaps and lifetime_overlaps, (left, right))

    def test_release_and_diagnostic_final_outputs_match(self) -> None:
        source = (
            np.arange(1 * 3 * 64 * 64, dtype=np.int32).reshape(1, 3, 64, 64) % 17 - 8
        ).astype(np.int8)
        bound = np.ascontiguousarray(nchw_to_hwc8(source)).tobytes()
        outputs = []
        for raw in (self.diagnostic_raw, self.release_raw):
            runtime = SimulatorRuntime()
            package = runtime.load_model(raw)
            runtime.bind_input(bound)
            completion = runtime.wait(runtime.submit())
            self.assertIs(completion.status, RuntimeStatus.SUCCESS)
            assert completion.output is not None
            final = next(item for item in package.manifest["output"]["allocation_plan"] if item["name"] == "model.4")
            outputs.append(completion.output[final["addend"] : final["addend"] + final["bytes"]])
        self.assertEqual(outputs[0], outputs[1])

    def test_graph_is_deterministic_and_streams_oversized_layer_weights(self) -> None:
        stem_layers, graph = graph_fixture()
        self.assertEqual(
            self.release_raw,
            compile_graph(
                stem_layers=stem_layers,
                graph=graph,
                source_model_sha256="33" * 32,
                allocation_mode="release",
            ),
        )
        operation = next(
            item
            for item in self.release.manifest["schedule"]["graph_operations"]
            if item["name"] == "model.3"
        )
        self.assertEqual(
            operation["weight_residency"], "one output group streamed per spatial tile"
        )
        self.assertEqual([item["name"] for item in self.release.manifest["output"]["tensors"]], ["model.4"])
        self.assertEqual(self.release.manifest["output"]["views"], [])


if __name__ == "__main__":
    unittest.main()
