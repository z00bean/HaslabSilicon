#!/usr/bin/env python3
"""Compile and validate pinned YOLOv8n nodes 0..217 through the learned detection head."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from haslab_compiler import (
    C2fExtension,
    build_backbone_graph,
    compile_graph,
    extend_with_bottom_up_neck_stage,
    extend_with_detection_head,
    extend_with_top_down_neck_stage,
    first_stage,
)
from haslab_compiler.c2f import load_pinned_through_detection_head
from haslab_ref import (
    add_i8,
    conv2d_i8,
    hwc8_to_nchw,
    map_i8,
    maxpool5_i8,
    nchw_to_hwc8,
    quantize_int8,
    upsample2_nearest_i8,
)
from haslab_runtime import RuntimeStatus, SimulatorRuntime, load_hxb

from compile_yolov8n_first_c2f import (
    exact_result,
    float_result,
    integer_conv_silu,
    synthetic_input,
)

ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "benchmarks/manifests/yolov8n-320-opset13"
DEFAULT_MODEL = ROOT / "models/generated/yolov8n-320-opset13.onnx"
DEFAULT_CALIBRATION = WORKLOAD / "int8-calibration.json"
DEFAULT_LUTS = WORKLOAD / "int8-silu-luts.bin"
DEFAULT_ARTIFACTS = ROOT / "artifacts/m6"
DEFAULT_REPORT = WORKLOAD / "m6-through-detection-head.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def execute(package: bytes, physical_input: bytes):
    runtime = SimulatorRuntime(ext_bytes=128 << 20)
    parsed = runtime.load_model(package)
    runtime.bind_input(physical_input)
    completion = runtime.wait(runtime.submit())
    if completion.status is not RuntimeStatus.SUCCESS or completion.output is None:
        raise SystemExit(f"simulator execution failed: {completion.error}")
    if runtime.last_submission_stats is None:
        raise SystemExit("runtime did not record FIFO submission statistics")
    return runtime, parsed, completion.output


def tensor_from_output(output: bytes, record: dict[str, object]) -> np.ndarray:
    start = int(record["addend"])
    end = start + int(record["bytes"])
    dtype = np.dtype("<i4") if record["dtype"] == "int32" else np.dtype("i1")
    return np.frombuffer(output[start:end], dtype=dtype).copy().reshape(record["physical_shape"])


def integer_conv_raw(source: np.ndarray, layer: object) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(layer.weights, dtype=np.float32)
    scales = np.asarray(layer.weight_scales, dtype=np.float32)
    weights_i8 = quantize_int8(weights, scales.reshape(weights.shape[0], 1, 1, 1))
    bias_i32 = np.rint(
        np.asarray(layer.bias, dtype=np.float64)
        / (np.float64(np.float32(layer.input_scale)) * scales.astype(np.float64))
    ).astype(np.int32)
    logical = conv2d_i8(
        source,
        weights_i8,
        bias_i32,
        stride=layer.stride,
        padding=layer.padding,
    )
    return nchw_to_hwc8(logical), logical


def float_result_per_channel(
    name: str, simulator: np.ndarray, scales: object, reference: np.ndarray
) -> dict[str, object]:
    scale_array = np.asarray(scales, dtype=np.float32).reshape(1, -1, 1, 1)
    difference = simulator.astype(np.float32) * scale_array - reference.astype(np.float32)
    return {
        "compared_values": int(reference.size),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "name": name,
        "root_mean_square_error": float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))),
    }


def apply_concat(
    sources: list[np.ndarray], operation: dict[str, object]
) -> np.ndarray:
    return np.concatenate(
        [
            map_i8(values, fixed["multiplier"], fixed["shift"])
            for values, fixed in zip(sources, operation["fixed_point"])
        ],
        axis=2,
    )


def apply_add(
    left: np.ndarray, right: np.ndarray, operation: dict[str, object]
) -> np.ndarray:
    fixed = operation["fixed_point"]
    return add_i8(
        left,
        right,
        fixed["left_multiplier"],
        fixed["right_multiplier"],
        fixed["shift"],
    )


def apply_pool(values: np.ndarray) -> np.ndarray:
    padded = np.full(
        (values.shape[0] + 4, values.shape[1] + 4, values.shape[2], 8),
        -128,
        dtype=np.int8,
    )
    padded[2:-2, 2:-2] = values
    return maxpool5_i8(padded)


def calculate_golden(
    full_i8: np.ndarray, stem: object, first: object, downsample2: object,
    second: object, downsample3: object, third: object, downsample4: object,
    fourth: object, sppf: object, first_neck: object, second_neck: object,
    first_bottom_up: object, second_bottom_up: object, head: object,
    operations: list[dict[str, object]],
    *, conv_silu=integer_conv_silu, conv_raw=integer_conv_raw,
) -> dict[str, np.ndarray]:
    """Independent integer reference for every materialized/aliased boundary."""
    golden: dict[str, np.ndarray] = {}
    golden["model.0"], logical = conv_silu(full_i8, stem[0])
    golden["model.1"], logical = conv_silu(logical, stem[1])
    golden["model.2.cv1"], _ = conv_silu(logical, first.cv1)
    golden["model.2.split0"] = golden["model.2.cv1"][:, :, :2, :]
    golden["model.2.split1"] = golden["model.2.cv1"][:, :, 2:, :]
    golden["model.2.m.0.cv1"], logical = conv_silu(
        hwc8_to_nchw(golden["model.2.split1"], 16), first.bottleneck_cv1
    )
    golden["model.2.m.0.cv2"], _ = conv_silu(logical, first.bottleneck_cv2)
    golden["model.2.m.0.add"] = apply_add(
        golden["model.2.split1"], golden["model.2.m.0.cv2"], operations[3]
    )
    golden["model.2.concat"] = apply_concat(
        [
            golden["model.2.split0"],
            golden["model.2.split1"],
            golden["model.2.m.0.add"],
        ],
        operations[4],
    )
    golden["model.2"], logical = conv_silu(
        hwc8_to_nchw(golden["model.2.concat"], 48), first.cv2
    )
    golden["model.3"], logical = conv_silu(logical, downsample2)
    golden["model.4.cv1"], _ = conv_silu(logical, second.cv1)
    golden["model.4.split0"] = golden["model.4.cv1"][:, :, :4, :]
    golden["model.4.split1"] = golden["model.4.cv1"][:, :, 4:, :]
    branch = golden["model.4.split1"]
    for index, (conv1, conv2) in enumerate(second.bottlenecks):
        name1 = f"model.4.m.{index}.cv1"
        name2 = f"model.4.m.{index}.cv2"
        add_name = f"model.4.m.{index}.add"
        golden[name1], logical = conv_silu(hwc8_to_nchw(branch, 32), conv1)
        golden[name2], _ = conv_silu(logical, conv2)
        golden[add_name] = apply_add(branch, golden[name2], operations[10 + index * 3])
        branch = golden[add_name]
    golden["model.4.concat"] = apply_concat(
        [
            golden["model.4.split0"],
            golden["model.4.split1"],
            golden["model.4.m.0.add"],
            golden["model.4.m.1.add"],
        ],
        operations[14],
    )
    golden["model.4"], _ = conv_silu(
        hwc8_to_nchw(golden["model.4.concat"], 128), second.cv2
    )
    golden["model.5"], logical = conv_silu(
        hwc8_to_nchw(golden["model.4"], 64), downsample3
    )
    golden["model.6.cv1"], _ = conv_silu(logical, third.cv1)
    golden["model.6.split0"] = golden["model.6.cv1"][:, :, :8, :]
    golden["model.6.split1"] = golden["model.6.cv1"][:, :, 8:, :]
    branch = golden["model.6.split1"]
    for index, (conv1, conv2) in enumerate(third.bottlenecks):
        name1 = f"model.6.m.{index}.cv1"
        name2 = f"model.6.m.{index}.cv2"
        add_name = f"model.6.m.{index}.add"
        golden[name1], logical = conv_silu(hwc8_to_nchw(branch, 64), conv1)
        golden[name2], _ = conv_silu(logical, conv2)
        golden[add_name] = apply_add(branch, golden[name2], operations[20 + index * 3])
        branch = golden[add_name]
    golden["model.6.concat"] = apply_concat(
        [
            golden["model.6.split0"],
            golden["model.6.split1"],
            golden["model.6.m.0.add"],
            golden["model.6.m.1.add"],
        ],
        operations[24],
    )
    golden["model.6"], _ = conv_silu(
        hwc8_to_nchw(golden["model.6.concat"], 256), third.cv2
    )
    golden["model.7"], logical = conv_silu(
        hwc8_to_nchw(golden["model.6"], 128), downsample4
    )
    golden["model.8.cv1"], _ = conv_silu(logical, fourth.cv1)
    golden["model.8.split0"] = golden["model.8.cv1"][:, :, :16, :]
    golden["model.8.split1"] = golden["model.8.cv1"][:, :, 16:, :]
    golden["model.8.m.0.cv1"], logical = conv_silu(
        hwc8_to_nchw(golden["model.8.split1"], 128), fourth.bottlenecks[0][0]
    )
    golden["model.8.m.0.cv2"], _ = conv_silu(
        logical, fourth.bottlenecks[0][1]
    )
    golden["model.8.m.0.add"] = apply_add(
        golden["model.8.split1"], golden["model.8.m.0.cv2"], operations[30]
    )
    golden["model.8.concat"] = apply_concat(
        [
            golden["model.8.split0"],
            golden["model.8.split1"],
            golden["model.8.m.0.add"],
        ],
        operations[31],
    )
    golden["model.8"], logical = conv_silu(
        hwc8_to_nchw(golden["model.8.concat"], 384), fourth.cv2
    )
    golden["model.9.cv1"], _ = conv_silu(logical, sppf.cv1)
    golden["model.9.pool0"] = apply_pool(golden["model.9.cv1"])
    golden["model.9.pool1"] = apply_pool(golden["model.9.pool0"])
    golden["model.9.pool2"] = apply_pool(golden["model.9.pool1"])
    golden["model.9.concat"] = apply_concat(
        [
            golden["model.9.cv1"],
            golden["model.9.pool0"],
            golden["model.9.pool1"],
            golden["model.9.pool2"],
        ],
        operations[37],
    )
    golden["model.9"], _ = conv_silu(
        hwc8_to_nchw(golden["model.9.concat"], 512), sppf.cv2
    )
    golden["model.10"] = upsample2_nearest_i8(golden["model.9"])
    golden["model.11.concat"] = apply_concat(
        [golden["model.10"], golden["model.6"]], operations[40]
    )
    golden["model.12.cv1"], _ = conv_silu(
        hwc8_to_nchw(golden["model.11.concat"], 384), first_neck.stage.cv1
    )
    golden["model.12.split0"] = golden["model.12.cv1"][:, :, :8, :]
    golden["model.12.split1"] = golden["model.12.cv1"][:, :, 8:, :]
    golden["model.12.m.0.cv1"], logical = conv_silu(
        hwc8_to_nchw(golden["model.12.split1"], 64),
        first_neck.stage.bottlenecks[0][0],
    )
    golden["model.12.m.0.cv2"], _ = conv_silu(
        logical, first_neck.stage.bottlenecks[0][1]
    )
    golden["model.12.concat"] = apply_concat(
        [
            golden["model.12.split0"],
            golden["model.12.split1"],
            golden["model.12.m.0.cv2"],
        ],
        operations[44],
    )
    golden["model.12"], _ = conv_silu(
        hwc8_to_nchw(golden["model.12.concat"], 192), first_neck.stage.cv2
    )
    golden["model.13"] = upsample2_nearest_i8(golden["model.12"])
    golden["model.14.concat"] = apply_concat(
        [golden["model.13"], golden["model.4"]], operations[47]
    )
    golden["model.15.cv1"], _ = conv_silu(
        hwc8_to_nchw(golden["model.14.concat"], 192), second_neck.stage.cv1
    )
    golden["model.15.split0"] = golden["model.15.cv1"][:, :, :4, :]
    golden["model.15.split1"] = golden["model.15.cv1"][:, :, 4:, :]
    golden["model.15.m.0.cv1"], logical = conv_silu(
        hwc8_to_nchw(golden["model.15.split1"], 32),
        second_neck.stage.bottlenecks[0][0],
    )
    golden["model.15.m.0.cv2"], _ = conv_silu(
        logical, second_neck.stage.bottlenecks[0][1]
    )
    golden["model.15.concat"] = apply_concat(
        [
            golden["model.15.split0"],
            golden["model.15.split1"],
            golden["model.15.m.0.cv2"],
        ],
        operations[51],
    )
    golden["model.15"], _ = conv_silu(
        hwc8_to_nchw(golden["model.15.concat"], 96), second_neck.stage.cv2
    )
    golden["model.16"], logical = conv_silu(
        hwc8_to_nchw(golden["model.15"], 64), first_bottom_up.downsample
    )
    golden["model.17.concat"] = apply_concat(
        [golden["model.16"], golden["model.12"]], operations[54]
    )
    golden["model.18.cv1"], _ = conv_silu(
        hwc8_to_nchw(golden["model.17.concat"], 192), first_bottom_up.stage.cv1
    )
    golden["model.18.split0"] = golden["model.18.cv1"][:, :, :8, :]
    golden["model.18.split1"] = golden["model.18.cv1"][:, :, 8:, :]
    golden["model.18.m.0.cv1"], logical = conv_silu(
        hwc8_to_nchw(golden["model.18.split1"], 64),
        first_bottom_up.stage.bottlenecks[0][0],
    )
    golden["model.18.m.0.cv2"], _ = conv_silu(
        logical, first_bottom_up.stage.bottlenecks[0][1]
    )
    golden["model.18.concat"] = apply_concat(
        [
            golden["model.18.split0"],
            golden["model.18.split1"],
            golden["model.18.m.0.cv2"],
        ],
        operations[58],
    )
    golden["model.18"], _ = conv_silu(
        hwc8_to_nchw(golden["model.18.concat"], 192), first_bottom_up.stage.cv2
    )
    golden["model.19"], logical = conv_silu(
        hwc8_to_nchw(golden["model.18"], 128), second_bottom_up.downsample
    )
    golden["model.20.concat"] = apply_concat(
        [golden["model.19"], golden["model.9"]], operations[61]
    )
    golden["model.21.cv1"], _ = conv_silu(
        hwc8_to_nchw(golden["model.20.concat"], 384), second_bottom_up.stage.cv1
    )
    golden["model.21.split0"] = golden["model.21.cv1"][:, :, :16, :]
    golden["model.21.split1"] = golden["model.21.cv1"][:, :, 16:, :]
    golden["model.21.m.0.cv1"], logical = conv_silu(
        hwc8_to_nchw(golden["model.21.split1"], 128),
        second_bottom_up.stage.bottlenecks[0][0],
    )
    golden["model.21.m.0.cv2"], _ = conv_silu(
        logical, second_bottom_up.stage.bottlenecks[0][1]
    )
    golden["model.21.concat"] = apply_concat(
        [
            golden["model.21.split0"],
            golden["model.21.split1"],
            golden["model.21.m.0.cv2"],
        ],
        operations[65],
    )
    golden["model.21"], _ = conv_silu(
        hwc8_to_nchw(golden["model.21.concat"], 384), second_bottom_up.stage.cv2
    )

    for scale_index, branch in enumerate(head.branches):
        source = golden[branch.source]
        regression0, regression1, regression2 = branch.regression
        classification0, classification1, classification2 = branch.classification
        golden[regression0.name], logical = conv_silu(
            hwc8_to_nchw(source, regression0.input_channels), regression0
        )
        golden[regression1.name], logical = conv_silu(logical, regression1)
        golden[regression2.name], _ = conv_raw(logical, regression2)
        golden[classification0.name], logical = conv_silu(
            hwc8_to_nchw(source, classification0.input_channels), classification0
        )
        golden[classification1.name], logical = conv_silu(logical, classification1)
        golden[classification2.name], _ = conv_raw(logical, classification2)
        golden[branch.concat_name] = np.concatenate(
            [golden[regression2.name], golden[classification2.name]], axis=2
        )

    return golden


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--luts", type=Path, default=DEFAULT_LUTS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    import onnx
    import onnxruntime
    from onnx import TensorProto, helper

    (
        stem,
        first,
        downsample2,
        second,
        downsample3,
        third,
        downsample4,
        fourth,
        sppf,
        first_neck,
        second_neck,
        first_bottom_up,
        second_bottom_up,
        head,
        model_hash,
    ) = load_pinned_through_detection_head(
        args.model, args.calibration, args.luts
    )
    graph = build_backbone_graph(
        first=first_stage(first),
        extensions=(
            C2fExtension(downsample2, second),
            C2fExtension(downsample3, third),
            C2fExtension(downsample4, fourth),
        ),
        sppf=sppf,
    )
    graph = extend_with_top_down_neck_stage(graph, first_neck)
    graph = extend_with_top_down_neck_stage(graph, second_neck)
    graph = extend_with_bottom_up_neck_stage(graph, first_bottom_up)
    graph = extend_with_bottom_up_neck_stage(graph, second_bottom_up)
    graph = extend_with_detection_head(graph, head)
    packages = {
        mode: compile_graph(
            stem_layers=stem,
            graph=graph,
            source_model_sha256=model_hash,
            allocation_mode=mode,
        )
        for mode in ("diagnostic", "release")
    }
    for mode, package in packages.items():
        repeated = compile_graph(
            stem_layers=stem,
            graph=graph,
            source_model_sha256=model_hash,
            allocation_mode=mode,
        )
        if repeated != package:
            raise SystemExit(f"{mode} graph compilation was not byte-identical")

    full_float = synthetic_input()
    full_i8 = quantize_int8(full_float, stem[0].input_scale)
    physical_input = np.ascontiguousarray(nchw_to_hwc8(full_i8)).tobytes()
    diagnostic_runtime, diagnostic_package, diagnostic_output = execute(
        packages["diagnostic"], physical_input
    )
    release_runtime, release_package, release_output = execute(
        packages["release"], physical_input
    )

    simulator = {
        record["name"]: tensor_from_output(diagnostic_output, record)
        for record in diagnostic_package.manifest["output"]["tensors"]
    }
    simulator["model.2.split0"] = simulator["model.2.cv1"][:, :, :2, :]
    simulator["model.2.split1"] = simulator["model.2.cv1"][:, :, 2:, :]
    simulator["model.4.split0"] = simulator["model.4.cv1"][:, :, :4, :]
    simulator["model.4.split1"] = simulator["model.4.cv1"][:, :, 4:, :]
    simulator["model.6.split0"] = simulator["model.6.cv1"][:, :, :8, :]
    simulator["model.6.split1"] = simulator["model.6.cv1"][:, :, 8:, :]
    simulator["model.8.split0"] = simulator["model.8.cv1"][:, :, :16, :]
    simulator["model.8.split1"] = simulator["model.8.cv1"][:, :, 16:, :]
    simulator["model.12.split0"] = simulator["model.12.cv1"][:, :, :8, :]
    simulator["model.12.split1"] = simulator["model.12.cv1"][:, :, 8:, :]
    simulator["model.15.split0"] = simulator["model.15.cv1"][:, :, :4, :]
    simulator["model.15.split1"] = simulator["model.15.cv1"][:, :, 4:, :]
    simulator["model.18.split0"] = simulator["model.18.cv1"][:, :, :8, :]
    simulator["model.18.split1"] = simulator["model.18.cv1"][:, :, 8:, :]
    simulator["model.21.split0"] = simulator["model.21.cv1"][:, :, :16, :]
    simulator["model.21.split1"] = simulator["model.21.cv1"][:, :, 16:, :]
    operations = diagnostic_package.manifest["schedule"]["graph_operations"]

    golden = calculate_golden(
        full_i8, stem, first, downsample2, second, downsample3, third,
        downsample4, fourth, sppf, first_neck, second_neck, first_bottom_up,
        second_bottom_up, head, operations,
    )

    ordered_names = [
        "model.0", "model.1", "model.2.cv1", "model.2.split0", "model.2.split1",
        "model.2.m.0.cv1", "model.2.m.0.cv2", "model.2.m.0.add", "model.2.concat",
        "model.2", "model.3", "model.4.cv1", "model.4.split0", "model.4.split1",
        "model.4.m.0.cv1", "model.4.m.0.cv2", "model.4.m.0.add",
        "model.4.m.1.cv1", "model.4.m.1.cv2", "model.4.m.1.add",
        "model.4.concat", "model.4",
        "model.5", "model.6.cv1", "model.6.split0", "model.6.split1",
        "model.6.m.0.cv1", "model.6.m.0.cv2", "model.6.m.0.add",
        "model.6.m.1.cv1", "model.6.m.1.cv2", "model.6.m.1.add",
        "model.6.concat", "model.6",
        "model.7", "model.8.cv1", "model.8.split0", "model.8.split1",
        "model.8.m.0.cv1", "model.8.m.0.cv2", "model.8.m.0.add",
        "model.8.concat", "model.8", "model.9.cv1", "model.9.pool0",
        "model.9.pool1", "model.9.pool2", "model.9.concat", "model.9",
        "model.10", "model.11.concat", "model.12.cv1", "model.12.split0",
        "model.12.split1", "model.12.m.0.cv1", "model.12.m.0.cv2",
        "model.12.concat", "model.12",
        "model.13", "model.14.concat", "model.15.cv1", "model.15.split0",
        "model.15.split1", "model.15.m.0.cv1", "model.15.m.0.cv2",
        "model.15.concat", "model.15",
        "model.16", "model.17.concat", "model.18.cv1", "model.18.split0",
        "model.18.split1", "model.18.m.0.cv1", "model.18.m.0.cv2",
        "model.18.concat", "model.18",
        "model.19", "model.20.concat", "model.21.cv1", "model.21.split0",
        "model.21.split1", "model.21.m.0.cv1", "model.21.m.0.cv2",
        "model.21.concat", "model.21",
    ]
    for branch in head.branches:
        ordered_names.extend(
            [
                branch.regression[0].name,
                branch.regression[1].name,
                branch.regression[2].name,
                branch.classification[0].name,
                branch.classification[1].name,
                branch.classification[2].name,
                branch.concat_name,
            ]
        )
    exact = [exact_result(name, simulator[name], golden[name]) for name in ordered_names]
    if not all(item["pass"] for item in exact):
        failure = next(item for item in exact if not item["pass"])
        raise SystemExit(f"{failure['name']} differs at {failure['mismatch_count']} values")

    release_results = []
    for release_record in release_package.manifest["output"]["tensors"]:
        name = release_record["name"]
        release_final = tensor_from_output(release_output, release_record)
        release_results.append(exact_result(name, release_final, golden[name]))
    if not all(item["pass"] for item in release_results):
        raise SystemExit("release learned-head boundary differs from the golden model")

    model = onnx.load(args.model)
    output_specs = [
        (2, [1, 16, 160, 160], "model.0", stem[0].output_scale),
        (5, [1, 32, 80, 80], "model.1", stem[1].output_scale),
        (8, [1, 32, 80, 80], "model.2.cv1", first.cv1.output_scale),
        (13, [1, 16, 80, 80], "model.2.m.0.cv1", first.bottleneck_cv1.output_scale),
        (16, [1, 16, 80, 80], "model.2.m.0.cv2", first.bottleneck_cv2.output_scale),
        (17, [1, 16, 80, 80], "model.2.m.0.add", first.residual_scale),
        (18, [1, 48, 80, 80], "model.2.concat", first.concat_scale),
        (21, [1, 32, 80, 80], "model.2", first.cv2.output_scale),
        (24, [1, 64, 40, 40], "model.3", downsample2.output_scale),
        (27, [1, 64, 40, 40], "model.4.cv1", second.cv1.output_scale),
        (32, [1, 32, 40, 40], "model.4.m.0.cv1", second.bottlenecks[0][0].output_scale),
        (35, [1, 32, 40, 40], "model.4.m.0.cv2", second.bottlenecks[0][1].output_scale),
        (36, [1, 32, 40, 40], "model.4.m.0.add", second.residual_scales[0]),
        (39, [1, 32, 40, 40], "model.4.m.1.cv1", second.bottlenecks[1][0].output_scale),
        (42, [1, 32, 40, 40], "model.4.m.1.cv2", second.bottlenecks[1][1].output_scale),
        (43, [1, 32, 40, 40], "model.4.m.1.add", second.residual_scales[1]),
        (44, [1, 128, 40, 40], "model.4.concat", second.concat_scale),
        (47, [1, 64, 40, 40], "model.4", second.cv2.output_scale),
        (50, [1, 128, 20, 20], "model.5", downsample3.output_scale),
        (53, [1, 128, 20, 20], "model.6.cv1", third.cv1.output_scale),
        (58, [1, 64, 20, 20], "model.6.m.0.cv1", third.bottlenecks[0][0].output_scale),
        (61, [1, 64, 20, 20], "model.6.m.0.cv2", third.bottlenecks[0][1].output_scale),
        (62, [1, 64, 20, 20], "model.6.m.0.add", third.residual_scales[0]),
        (65, [1, 64, 20, 20], "model.6.m.1.cv1", third.bottlenecks[1][0].output_scale),
        (68, [1, 64, 20, 20], "model.6.m.1.cv2", third.bottlenecks[1][1].output_scale),
        (69, [1, 64, 20, 20], "model.6.m.1.add", third.residual_scales[1]),
        (70, [1, 256, 20, 20], "model.6.concat", third.concat_scale),
        (73, [1, 128, 20, 20], "model.6", third.cv2.output_scale),
        (76, [1, 256, 10, 10], "model.7", downsample4.output_scale),
        (79, [1, 256, 10, 10], "model.8.cv1", fourth.cv1.output_scale),
        (84, [1, 128, 10, 10], "model.8.m.0.cv1", fourth.bottlenecks[0][0].output_scale),
        (87, [1, 128, 10, 10], "model.8.m.0.cv2", fourth.bottlenecks[0][1].output_scale),
        (88, [1, 128, 10, 10], "model.8.m.0.add", fourth.residual_scales[0]),
        (89, [1, 384, 10, 10], "model.8.concat", fourth.concat_scale),
        (92, [1, 256, 10, 10], "model.8", fourth.cv2.output_scale),
        (95, [1, 128, 10, 10], "model.9.cv1", sppf.cv1.output_scale),
        (96, [1, 128, 10, 10], "model.9.pool0", sppf.cv1.output_scale),
        (97, [1, 128, 10, 10], "model.9.pool1", sppf.cv1.output_scale),
        (98, [1, 128, 10, 10], "model.9.pool2", sppf.cv1.output_scale),
        (99, [1, 512, 10, 10], "model.9.concat", sppf.concat_scale),
        (102, [1, 256, 10, 10], "model.9", sppf.cv2.output_scale),
        (104, [1, 256, 20, 20], "model.10", sppf.cv2.output_scale),
        (105, [1, 384, 20, 20], "model.11.concat", first_neck.concat_scale),
        (108, [1, 128, 20, 20], "model.12.cv1", first_neck.stage.cv1.output_scale),
        (112, [1, 64, 20, 20], "model.12.m.0.cv1", first_neck.stage.bottlenecks[0][0].output_scale),
        (115, [1, 64, 20, 20], "model.12.m.0.cv2", first_neck.stage.bottlenecks[0][1].output_scale),
        (116, [1, 192, 20, 20], "model.12.concat", first_neck.stage.concat_scale),
        (119, [1, 128, 20, 20], "model.12", first_neck.stage.cv2.output_scale),
        (121, [1, 128, 40, 40], "model.13", first_neck.stage.cv2.output_scale),
        (122, [1, 192, 40, 40], "model.14.concat", second_neck.concat_scale),
        (125, [1, 64, 40, 40], "model.15.cv1", second_neck.stage.cv1.output_scale),
        (129, [1, 32, 40, 40], "model.15.m.0.cv1", second_neck.stage.bottlenecks[0][0].output_scale),
        (132, [1, 32, 40, 40], "model.15.m.0.cv2", second_neck.stage.bottlenecks[0][1].output_scale),
        (133, [1, 96, 40, 40], "model.15.concat", second_neck.stage.concat_scale),
        (136, [1, 64, 40, 40], "model.15", second_neck.stage.cv2.output_scale),
        (139, [1, 64, 20, 20], "model.16", first_bottom_up.downsample.output_scale),
        (140, [1, 192, 20, 20], "model.17.concat", first_bottom_up.concat_scale),
        (143, [1, 128, 20, 20], "model.18.cv1", first_bottom_up.stage.cv1.output_scale),
        (147, [1, 64, 20, 20], "model.18.m.0.cv1", first_bottom_up.stage.bottlenecks[0][0].output_scale),
        (150, [1, 64, 20, 20], "model.18.m.0.cv2", first_bottom_up.stage.bottlenecks[0][1].output_scale),
        (151, [1, 192, 20, 20], "model.18.concat", first_bottom_up.stage.concat_scale),
        (154, [1, 128, 20, 20], "model.18", first_bottom_up.stage.cv2.output_scale),
        (157, [1, 128, 10, 10], "model.19", second_bottom_up.downsample.output_scale),
        (158, [1, 384, 10, 10], "model.20.concat", second_bottom_up.concat_scale),
        (161, [1, 256, 10, 10], "model.21.cv1", second_bottom_up.stage.cv1.output_scale),
        (165, [1, 128, 10, 10], "model.21.m.0.cv1", second_bottom_up.stage.bottlenecks[0][0].output_scale),
        (168, [1, 128, 10, 10], "model.21.m.0.cv2", second_bottom_up.stage.bottlenecks[0][1].output_scale),
        (169, [1, 384, 10, 10], "model.21.concat", second_bottom_up.stage.concat_scale),
        (172, [1, 256, 10, 10], "model.21", second_bottom_up.stage.cv2.output_scale),
    ]
    for scale_index, branch in enumerate(head.branches):
        height = branch.regression[0].input_h
        width = branch.regression[0].input_w
        start = 173 + scale_index * 15
        regression_raw_scales = np.float32(branch.regression[2].input_scale) * np.asarray(
            branch.regression[2].weight_scales, dtype=np.float32
        )
        classification_raw_scales = np.float32(branch.classification[2].input_scale) * np.asarray(
            branch.classification[2].weight_scales, dtype=np.float32
        )
        output_specs.extend(
            [
                (start + 2, [1, 64, height, width], branch.regression[0].name, branch.regression[0].output_scale),
                (start + 5, [1, 64, height, width], branch.regression[1].name, branch.regression[1].output_scale),
                (start + 6, [1, 64, height, width], branch.regression[2].name, regression_raw_scales),
                (start + 9, [1, 80, height, width], branch.classification[0].name, branch.classification[0].output_scale),
                (start + 12, [1, 80, height, width], branch.classification[1].name, branch.classification[1].output_scale),
                (start + 13, [1, 80, height, width], branch.classification[2].name, classification_raw_scales),
                (
                    start + 14,
                    [1, 144, height, width],
                    branch.concat_name,
                    np.concatenate([regression_raw_scales, classification_raw_scales]),
                ),
            ]
        )
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(model.graph.node[index].output[0], TensorProto.FLOAT, shape)
            for index, shape, _, _ in output_specs
        ]
    )
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    float_outputs = session.run(None, {"images": full_float})
    float_comparisons = []
    for (_, _, name, scale), reference in zip(output_specs, float_outputs):
        logical_simulator = hwc8_to_nchw(simulator[name], reference.shape[1])
        if np.asarray(scale).ndim:
            float_comparisons.append(
                float_result_per_channel(name, logical_simulator, scale, reference)
            )
        else:
            float_comparisons.append(float_result(name, logical_simulator, scale, reference))

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for mode, package in packages.items():
        (args.artifact_dir / f"yolov8n-through-detection-head-{mode}.hxb").write_bytes(package)
    (args.artifact_dir / "yolov8n-through-detection-head-input.bin").write_bytes(physical_input)
    (args.artifact_dir / "yolov8n-through-detection-head-diagnostic-output.bin").write_bytes(
        diagnostic_output
    )
    (args.artifact_dir / "yolov8n-through-detection-head-release-output.bin").write_bytes(
        release_output
    )
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))

    def package_record(mode: str, parsed, runtime, output: bytes) -> dict[str, object]:
        allocation = parsed.manifest["schedule"]["allocation"]
        buffers = parsed.manifest["buffers"]
        return {
            "allocation": allocation,
            "bytes": len(packages[mode]),
            "commands_bytes": len(parsed.commands),
            "constant_bytes": len(parsed.constants),
            "external_buffers": buffers,
            "external_bytes": sum(int(record["bytes"]) for record in buffers.values()),
            "fifo": asdict(runtime.last_submission_stats),
            "output_sha256": sha256(output),
            "repeated_compilation_byte_identical": True,
            "schema": parsed.manifest["schema"],
            "sha256": sha256(packages[mode]),
        }

    report = {
        "schema_version": 1,
        "status": "passing_nodes_0_through_217_learned_head_diagnostic_and_release",
        "scope": {
            "node_indices": [0, 217],
            "nodes": diagnostic_package.manifest["source"]["nodes"],
            "logical_input_shape": [1, 3, 320, 320],
            "compared_boundaries": ordered_names,
            "limitation": "This executes the accelerator partition through its three raw INT32 learned-head tensors; the declared host decoding, DFL, sigmoid, and NMS tail remains future work.",
        },
        "source_model": calibration["source_model"],
        "calibration_sha256": sha256(args.calibration.read_bytes()),
        "lut_binary_sha256": sha256(args.luts.read_bytes()),
        "packages": {
            "diagnostic": package_record(
                "diagnostic", diagnostic_package, diagnostic_runtime, diagnostic_output
            ),
            "release": package_record("release", release_package, release_runtime, release_output),
        },
        "allocation_plan": {
            "diagnostic_retained_tensors": diagnostic_package.manifest["output"]["tensors"],
            "diagnostic_views": diagnostic_package.manifest["output"]["views"],
            "release_final_tensors": release_package.manifest["output"]["tensors"],
            "release_lifetimes": release_package.manifest["output"]["allocation_plan"],
        },
        "schedule": diagnostic_package.manifest["schedule"],
        "exact_integer_comparisons": exact,
        "exact_integer_totals": {
            "compared_values": sum(int(item["compared_values"]) for item in exact),
            "mismatch_count": sum(int(item["mismatch_count"]) for item in exact),
            "pass": all(bool(item["pass"]) for item in exact),
        },
        "release_final_comparisons": release_results,
        "onnx_float_comparisons": float_comparisons,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "platform": platform.platform(),
        },
    }
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
