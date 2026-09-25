# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

import numpy as np

from haslab_compiler import (
    AddOp,
    C2fBlockSpec,
    C2fConvSpec,
    C2fStageSpec,
    ConcatOp,
    ConvSiluOp,
    ConvSiluLayerSpec,
    GraphIR,
    GraphTensor,
    MaxPoolOp,
    Upsample2Op,
    build_two_c2f_graph,
    compile_graph,
    first_stage,
)
from haslab_ref import (
    build_silu_lut,
    maxpool5_i8,
    nchw_to_hwc8,
    upsample2_nearest_i8,
)
from haslab_runtime import RuntimeStatus, SimulatorRuntime, load_hxb
from haslab_compiler.c2f import _conv_tile_shape


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

    def test_tile_selection_handles_twenty_by_twenty_wide_convolution(self) -> None:
        self.assertEqual(_conv_tile_shape(20, 20, 8, 3, 2), (5, 5))
        tile_h, tile_w = _conv_tile_shape(20, 20, 8, 3, 2)
        patch_bytes = 8 * ((tile_h - 1) * 2 + 3) * ((tile_w - 1) * 2 + 3) * 8
        self.assertLessEqual(patch_bytes, 16_384)

    def test_wide_concat_streams_parameters_and_executes(self) -> None:
        branch_names = tuple(f"wide.branch{index}" for index in range(4))
        layers = tuple(conv(name, 16, 32, 64, 1) for name in branch_names)
        graph = GraphIR(
            tensors=tuple(
                GraphTensor(name, 16, 16, 64, 1.0) for name in branch_names
            )
            + (GraphTensor("wide.concat", 16, 16, 256, 1.0),),
            operations=tuple(
                ConvSiluOp("model.1", name, layer)
                for name, layer in zip(branch_names, layers)
            )
            + (ConcatOp(branch_names, "wide.concat", "wide.concat.node"),),
            source_nodes=tuple(
                node for layer in layers for node in layer.node_names
            )
            + ("wide.concat.node",),
            final_outputs=("wide.concat",),
        )
        raw = compile_graph(
            stem_layers=self.stem,
            graph=graph,
            source_model_sha256="44" * 32,
            allocation_mode="release",
        )
        package = load_hxb(raw)
        operation = package.manifest["schedule"]["graph_operations"][-1]
        self.assertEqual(operation["parameter_residency"], "one concat source at a time")
        source = np.zeros((1, 3, 64, 64), dtype=np.int8)
        runtime = SimulatorRuntime()
        runtime.load_model(raw)
        runtime.bind_input(np.ascontiguousarray(nchw_to_hwc8(source)).tobytes())
        completion = runtime.wait(runtime.submit())
        self.assertIs(completion.status, RuntimeStatus.SUCCESS)

    def test_single_wide_concat_source_streams_one_group_record(self) -> None:
        layer = conv("verywide.source", 16, 32, 256, 1)
        graph = GraphIR(
            tensors=(
                GraphTensor("verywide.source", 16, 16, 256, 1.0),
                GraphTensor("verywide.concat", 16, 16, 256, 1.0),
            ),
            operations=(
                ConvSiluOp("model.1", "verywide.source", layer),
                ConcatOp(
                    ("verywide.source",),
                    "verywide.concat",
                    "verywide.concat.node",
                ),
            ),
            source_nodes=layer.node_names + ("verywide.concat.node",),
            final_outputs=("verywide.concat",),
        )
        raw = compile_graph(
            stem_layers=self.stem,
            graph=graph,
            source_model_sha256="88" * 32,
            allocation_mode="release",
        )
        package = load_hxb(raw)
        operation = package.manifest["schedule"]["graph_operations"][-1]
        self.assertEqual(
            operation["parameter_residency"],
            "one concat channel-group record streamed per spatial tile",
        )
        source = np.zeros((1, 3, 64, 64), dtype=np.int8)
        runtime = SimulatorRuntime()
        runtime.load_model(raw)
        runtime.bind_input(np.ascontiguousarray(nchw_to_hwc8(source)).tobytes())
        completion = runtime.wait(runtime.submit())
        self.assertIs(completion.status, RuntimeStatus.SUCCESS)

    def test_maxpool_graph_operation_compiles_halo_and_executes_exactly(self) -> None:
        graph = GraphIR(
            tensors=(GraphTensor("pool", 16, 16, 32, 1.0),),
            operations=(MaxPoolOp("model.1", "pool", "pool.node"),),
            source_nodes=("pool.node",),
            final_outputs=("pool",),
        )
        raw = compile_graph(
            stem_layers=self.stem,
            graph=graph,
            source_model_sha256="55" * 32,
            allocation_mode="diagnostic",
        )
        package = load_hxb(raw)
        operation = package.manifest["schedule"]["graph_operations"][0]
        self.assertEqual(operation["kind"], "maxpool5")
        self.assertEqual(operation["tile_shape"], [8, 8])
        self.assertEqual(operation["output_group_tiles"], 16)
        self.assertEqual(operation["opcode_counts"]["MAXPOOL5_I8"], 16)
        self.assertEqual(operation["boundary_fill_commands"], 16)

        source = (
            np.arange(1 * 3 * 64 * 64, dtype=np.int32).reshape(1, 3, 64, 64)
            % 17
            - 8
        ).astype(np.int8)
        runtime = SimulatorRuntime()
        runtime.load_model(raw)
        runtime.bind_input(np.ascontiguousarray(nchw_to_hwc8(source)).tobytes())
        completion = runtime.wait(runtime.submit())
        self.assertIs(completion.status, RuntimeStatus.SUCCESS)
        assert completion.output is not None
        records = {
            item["name"]: item for item in package.manifest["output"]["tensors"]
        }
        model1_record = records["model.1"]
        pool_record = records["pool"]
        model1 = np.frombuffer(
            completion.output[
                model1_record["addend"] : model1_record["addend"]
                + model1_record["bytes"]
            ],
            dtype=np.int8,
        ).reshape(model1_record["physical_shape"])
        actual = np.frombuffer(
            completion.output[
                pool_record["addend"] : pool_record["addend"] + pool_record["bytes"]
            ],
            dtype=np.int8,
        ).reshape(pool_record["physical_shape"])
        padded = np.full((20, 20, 4, 8), -128, dtype=np.int8)
        padded[2:-2, 2:-2] = model1
        np.testing.assert_array_equal(actual, maxpool5_i8(padded))

    def test_wide_conv_and_add_stream_parameter_records(self) -> None:
        wide_layer = conv("wide.conv", 16, 32, 256, 1)
        left_layer = conv("wide.left", 16, 32, 128, 1)
        right_layer = conv("wide.right", 16, 32, 128, 1)
        graph = GraphIR(
            tensors=(
                GraphTensor("wide.conv", 16, 16, 256, 1.0),
                GraphTensor("wide.left", 16, 16, 128, 1.0),
                GraphTensor("wide.right", 16, 16, 128, 1.0),
                GraphTensor("wide.add", 16, 16, 128, 1.0),
            ),
            operations=(
                ConvSiluOp("model.1", "wide.conv", wide_layer),
                ConvSiluOp("model.1", "wide.left", left_layer),
                ConvSiluOp("model.1", "wide.right", right_layer),
                AddOp("wide.left", "wide.right", "wide.add", "wide.add.node"),
            ),
            source_nodes=wide_layer.node_names
            + left_layer.node_names
            + right_layer.node_names
            + ("wide.add.node",),
            final_outputs=("wide.add",),
        )
        package = load_hxb(
            compile_graph(
                stem_layers=self.stem,
                graph=graph,
                source_model_sha256="66" * 32,
                allocation_mode="release",
            )
        )
        operations = package.manifest["schedule"]["graph_operations"]
        self.assertIn("output-group record", operations[0]["parameter_residency"])
        self.assertIn("left/right channel-group pair", operations[3]["parameter_residency"])

    def test_upsample_graph_operation_executes_exactly(self) -> None:
        graph = GraphIR(
            tensors=(GraphTensor("upsample", 32, 32, 32, 1.0),),
            operations=(Upsample2Op("model.1", "upsample", "resize.node"),),
            source_nodes=("resize.constant", "resize.node"),
            final_outputs=("upsample",),
        )
        raw = compile_graph(
            stem_layers=self.stem,
            graph=graph,
            source_model_sha256="77" * 32,
            allocation_mode="diagnostic",
        )
        package = load_hxb(raw)
        operation = package.manifest["schedule"]["graph_operations"][0]
        self.assertEqual(operation["kind"], "upsample2_nearest")
        self.assertEqual(operation["tile_shape"], [8, 8])
        self.assertEqual(operation["input_group_tiles"], 16)
        self.assertEqual(operation["opcode_counts"]["UPSAMPLE2_I8"], 16)

        source = (
            np.arange(1 * 3 * 64 * 64, dtype=np.int32).reshape(1, 3, 64, 64)
            % 17
            - 8
        ).astype(np.int8)
        runtime = SimulatorRuntime()
        runtime.load_model(raw)
        runtime.bind_input(np.ascontiguousarray(nchw_to_hwc8(source)).tobytes())
        completion = runtime.wait(runtime.submit())
        self.assertIs(completion.status, RuntimeStatus.SUCCESS)
        assert completion.output is not None
        records = {
            item["name"]: item for item in package.manifest["output"]["tensors"]
        }
        model1_record = records["model.1"]
        upsample_record = records["upsample"]
        model1 = np.frombuffer(
            completion.output[
                model1_record["addend"] : model1_record["addend"]
                + model1_record["bytes"]
            ],
            dtype=np.int8,
        ).reshape(model1_record["physical_shape"])
        actual = np.frombuffer(
            completion.output[
                upsample_record["addend"] : upsample_record["addend"]
                + upsample_record["bytes"]
            ],
            dtype=np.int8,
        ).reshape(upsample_record["physical_shape"])
        np.testing.assert_array_equal(actual, upsample2_nearest_i8(model1))


if __name__ == "__main__":
    unittest.main()
