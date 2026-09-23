#!/usr/bin/env python3
"""Compile, execute, and validate nodes 0..21 through the first YOLOv8n C2f."""

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

from haslab_compiler import compile_pinned_first_c2f, load_pinned_first_c2f
from haslab_ref import (
    add_i8,
    conv2d_i8,
    hwc8_to_nchw,
    map_i8,
    nchw_to_hwc8,
    quantize_int8,
    silu_lut_i32,
)
from haslab_runtime import RuntimeStatus, SimulatorRuntime, load_hxb

ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "benchmarks/manifests/yolov8n-320-opset13"
DEFAULT_MODEL = ROOT / "models/generated/yolov8n-320-opset13.onnx"
DEFAULT_CALIBRATION = WORKLOAD / "int8-calibration.json"
DEFAULT_LUTS = WORKLOAD / "int8-silu-luts.bin"
DEFAULT_PACKAGE = ROOT / "artifacts/m6/yolov8n-first-c2f.hxb"
DEFAULT_INPUT = ROOT / "artifacts/m6/yolov8n-first-c2f-input.bin"
DEFAULT_OUTPUT = ROOT / "artifacts/m6/yolov8n-first-c2f-output.bin"
DEFAULT_REPORT = WORKLOAD / "m6-first-c2f.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_input() -> np.ndarray:
    channel = np.arange(3, dtype=np.int32).reshape(1, 3, 1, 1)
    y = np.arange(320, dtype=np.int32).reshape(1, 1, 320, 1)
    x = np.arange(320, dtype=np.int32).reshape(1, 1, 1, 320)
    values = (channel * 37 + y * 3 + x * 5) % 256
    return np.ascontiguousarray(values.astype(np.float32) / np.float32(255.0))


def integer_conv_silu(source: np.ndarray, layer: object) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(layer.weights, dtype=np.float32)
    scales = np.asarray(layer.weight_scales, dtype=np.float32)
    weights_i8 = quantize_int8(weights, scales.reshape(weights.shape[0], 1, 1, 1))
    bias_i32 = np.rint(
        np.asarray(layer.bias, dtype=np.float64)
        / (np.float32(layer.input_scale).astype(np.float64) * scales.astype(np.float64))
    ).astype(np.int32)
    accumulators = conv2d_i8(
        source,
        weights_i8,
        bias_i32,
        stride=int(getattr(layer, "stride", 2)),
        padding=int(getattr(layer, "padding", 1)),
    )
    physical_accumulators = nchw_to_hwc8(accumulators)
    physical = np.empty(physical_accumulators.shape, dtype=np.int8)
    multipliers = np.asarray(layer.multipliers, dtype=np.int64)
    shifts = np.asarray(layer.shifts, dtype=np.int64)
    for group in range(physical.shape[2]):
        start = group * 8
        physical[:, :, group : group + 1, :] = silu_lut_i32(
            physical_accumulators[:, :, group : group + 1, :],
            multipliers[start : start + 8],
            shifts[start : start + 8],
            np.asarray(layer.lut, dtype=np.int8),
        )
    return physical, hwc8_to_nchw(physical, weights.shape[0])


def exact_result(name: str, simulator: np.ndarray, golden: np.ndarray) -> dict[str, object]:
    mismatches = int(np.count_nonzero(simulator != golden))
    return {
        "compared_values": int(golden.size),
        "mismatch_count": mismatches,
        "name": name,
        "output_sha256": sha256(np.ascontiguousarray(simulator).tobytes()),
        "pass": mismatches == 0,
        "reference": "independent haslab_ref graph evaluation",
    }


def float_result(name: str, simulator: np.ndarray, scale: float, reference: np.ndarray) -> dict[str, object]:
    dequantized = simulator.astype(np.float32) * np.float32(scale)
    difference = dequantized.astype(np.float64) - reference.astype(np.float64)
    return {
        "compared_values": int(reference.size),
        "int8_saturation_count": int(np.count_nonzero((simulator == -128) | (simulator == 127))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "name": name,
        "root_mean_square_error": float(np.sqrt(np.mean(difference * difference))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--luts", type=Path, default=DEFAULT_LUTS)
    parser.add_argument("--output-package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-tensor", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    import onnx
    import onnxruntime
    from onnx import TensorProto, helper

    package = compile_pinned_first_c2f(args.model, args.calibration, args.luts)
    if compile_pinned_first_c2f(args.model, args.calibration, args.luts) != package:
        raise SystemExit("repeated first-C2f compilation was not byte-identical")
    parsed = load_hxb(package)
    stem, block, _ = load_pinned_first_c2f(args.model, args.calibration, args.luts)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))

    full_float = synthetic_input()
    full_i8 = quantize_int8(full_float, stem[0].input_scale)
    physical_input = np.ascontiguousarray(nchw_to_hwc8(full_i8)).tobytes()
    runtime = SimulatorRuntime(ext_bytes=4 << 20)
    runtime.load_model(package)
    runtime.bind_input(physical_input)
    completion = runtime.wait(runtime.submit())
    if completion.status is not RuntimeStatus.SUCCESS or completion.output is None:
        raise SystemExit(f"simulator execution failed: {completion.error}")
    if runtime.last_submission_stats is None:
        raise SystemExit("runtime did not record FIFO submission statistics")

    simulator: dict[str, np.ndarray] = {}
    for tensor in parsed.manifest["output"]["tensors"]:
        start = int(tensor["addend"])
        end = start + int(tensor["bytes"])
        simulator[tensor["name"]] = np.frombuffer(completion.output[start:end], dtype=np.int8).copy().reshape(
            tensor["physical_shape"]
        )
    simulator["model.2.split0"] = simulator["model.2.cv1"][:, :, :2, :]
    simulator["model.2.split1"] = simulator["model.2.cv1"][:, :, 2:, :]

    golden: dict[str, np.ndarray] = {}
    golden["model.0"], logical = integer_conv_silu(full_i8, stem[0])
    golden["model.1"], logical = integer_conv_silu(logical, stem[1])
    golden["model.2.cv1"], _ = integer_conv_silu(logical, block.cv1)
    golden["model.2.split0"] = golden["model.2.cv1"][:, :, :2, :]
    golden["model.2.split1"] = golden["model.2.cv1"][:, :, 2:, :]
    golden["model.2.m.0.cv1"], logical = integer_conv_silu(
        hwc8_to_nchw(golden["model.2.split1"], 16), block.bottleneck_cv1
    )
    golden["model.2.m.0.cv2"], _ = integer_conv_silu(logical, block.bottleneck_cv2)
    operations = parsed.manifest["schedule"]["c2f_operations"]
    add_fixed = operations[3]["fixed_point"]
    golden["model.2.m.0.add"] = add_i8(
        golden["model.2.split1"],
        golden["model.2.m.0.cv2"],
        add_fixed["left_multiplier"],
        add_fixed["right_multiplier"],
        add_fixed["shift"],
    )
    concat_parts = []
    for name, fixed in zip(
        ("model.2.split0", "model.2.split1", "model.2.m.0.add"),
        operations[4]["fixed_point"],
    ):
        concat_parts.append(map_i8(golden[name], fixed["multiplier"], fixed["shift"]))
    golden["model.2.concat"] = np.concatenate(concat_parts, axis=2)
    golden["model.2"], _ = integer_conv_silu(
        hwc8_to_nchw(golden["model.2.concat"], 48), block.cv2
    )

    ordered_names = [
        "model.0", "model.1", "model.2.cv1", "model.2.split0", "model.2.split1",
        "model.2.m.0.cv1", "model.2.m.0.cv2", "model.2.m.0.add", "model.2.concat", "model.2",
    ]
    exact = [exact_result(name, simulator[name], golden[name]) for name in ordered_names]
    if not all(item["pass"] for item in exact):
        failure = next(item for item in exact if not item["pass"])
        raise SystemExit(f"{failure['name']} differs at {failure['mismatch_count']} values")

    model = onnx.load(args.model)
    output_specs = [
        (2, [1, 16, 160, 160], "model.0", stem[0].output_scale),
        (5, [1, 32, 80, 80], "model.1", stem[1].output_scale),
        (8, [1, 32, 80, 80], "model.2.cv1", block.cv1.output_scale),
        (13, [1, 16, 80, 80], "model.2.m.0.cv1", block.bottleneck_cv1.output_scale),
        (16, [1, 16, 80, 80], "model.2.m.0.cv2", block.bottleneck_cv2.output_scale),
        (17, [1, 16, 80, 80], "model.2.m.0.add", block.residual_scale),
        (18, [1, 48, 80, 80], "model.2.concat", block.concat_scale),
        (21, [1, 32, 80, 80], "model.2", block.cv2.output_scale),
    ]
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(model.graph.node[index].output[0], TensorProto.FLOAT, shape)
            for index, shape, _, _ in output_specs
        ]
    )
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    float_outputs = session.run(None, {"images": full_float})
    float_comparisons = []
    for (_, _, name, scale), reference in zip(output_specs, float_outputs):
        logical_simulator = hwc8_to_nchw(simulator[name], reference.shape[1])
        float_comparisons.append(float_result(name, logical_simulator, scale, reference))

    args.output_package.parent.mkdir(parents=True, exist_ok=True)
    args.output_package.write_bytes(package)
    args.output_input.write_bytes(physical_input)
    args.output_tensor.write_bytes(completion.output)
    buffers = parsed.manifest["buffers"]
    report = {
        "schema_version": 1,
        "status": "passing_first_c2f_nodes_0_through_21",
        "scope": {
            "node_indices": [0, 21],
            "nodes": parsed.manifest["source"]["nodes"],
            "logical_input_shape": [1, 3, 320, 320],
            "retained_boundaries": ordered_names,
            "limitation": "This executes through the first C2f block, not the remaining accelerator graph or host tail.",
        },
        "source_model": calibration["source_model"],
        "calibration_sha256": sha256(args.calibration.read_bytes()),
        "lut_binary_sha256": sha256(args.luts.read_bytes()),
        "package": {
            "schema": parsed.manifest["schema"],
            "bytes": len(package),
            "sha256": sha256(package),
            "repeated_compilation_byte_identical": True,
            "commands_bytes": len(parsed.commands),
            "constant_bytes": len(parsed.constants),
        },
        "allocation": {
            "external_buffers": buffers,
            "external_bytes": sum(int(record["bytes"]) for record in buffers.values()),
            "retained_tensors": parsed.manifest["output"]["tensors"],
            "zero_allocation_views": parsed.manifest["output"]["views"],
            "local_capacity_bytes": {
                "input": 16384,
                "weight": 16384,
                "accumulator": 8192,
                "output": 8192,
                "parameter": 3072,
            },
        },
        "schedule": parsed.manifest["schedule"],
        "input": {
            "generator": "((channel*37 + y*3 + x*5) mod 256) / FLOAT32(255)",
            "physical_bytes": len(physical_input),
            "physical_sha256": sha256(physical_input),
        },
        "exact_integer_comparisons": exact,
        "exact_integer_totals": {
            "compared_values": sum(int(item["compared_values"]) for item in exact),
            "mismatch_count": sum(int(item["mismatch_count"]) for item in exact),
            "pass": all(bool(item["pass"]) for item in exact),
        },
        "onnx_float_comparisons": float_comparisons,
        "runtime": {
            "backend": "haslab_sim functional command simulator",
            "completion": completion.status.value,
            "final_sequence": runtime.device.last_completed,
            "fifo": asdict(runtime.last_submission_stats),
        },
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
