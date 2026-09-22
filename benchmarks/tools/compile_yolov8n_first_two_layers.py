#!/usr/bin/env python3
"""Compile, execute, and validate the first two pinned YOLOv8n Conv-SiLU blocks."""

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
    compile_pinned_first_two_layers,
    load_pinned_first_two_layers,
)
from haslab_ref import (
    conv2d_i8,
    hwc8_to_nchw,
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
DEFAULT_PACKAGE = ROOT / "artifacts/m6/yolov8n-first-two-conv-silu.hxb"
DEFAULT_INPUT = ROOT / "artifacts/m6/yolov8n-first-two-conv-silu-input.bin"
DEFAULT_OUTPUT = ROOT / "artifacts/m6/yolov8n-first-two-conv-silu-output.bin"
DEFAULT_FIRST = ROOT / "artifacts/m6/yolov8n-first-two-layer0.bin"
DEFAULT_SECOND = ROOT / "artifacts/m6/yolov8n-first-two-layer1.bin"
DEFAULT_REPORT = WORKLOAD / "m6-first-two-conv-silu-layers.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_input() -> np.ndarray:
    channel = np.arange(3, dtype=np.int32).reshape(1, 3, 1, 1)
    y = np.arange(320, dtype=np.int32).reshape(1, 1, 320, 1)
    x = np.arange(320, dtype=np.int32).reshape(1, 1, 1, 320)
    values = (channel * 37 + y * 3 + x * 5) % 256
    return np.ascontiguousarray(values.astype(np.float32) / np.float32(255.0))


def integer_layer(source: np.ndarray, layer: object) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(layer.weights, dtype=np.float32)
    scales = np.asarray(layer.weight_scales, dtype=np.float32)
    weights_i8 = quantize_int8(weights, scales.reshape(weights.shape[0], 1, 1, 1))
    bias_i32 = np.rint(
        np.asarray(layer.bias, dtype=np.float64)
        / (np.float32(layer.input_scale).astype(np.float64) * scales.astype(np.float64))
    ).astype(np.int32)
    accumulators = conv2d_i8(source, weights_i8, bias_i32, stride=2, padding=1)
    physical_accumulators = nchw_to_hwc8(accumulators)
    physical = np.empty(physical_accumulators.shape, dtype=np.int8)
    for group in range(physical.shape[2]):
        start = group * 8
        physical[:, :, group : group + 1, :] = silu_lut_i32(
            physical_accumulators[:, :, group : group + 1, :],
            np.asarray(layer.multipliers, dtype=np.int64)[start : start + 8],
            np.asarray(layer.shifts, dtype=np.int64)[start : start + 8],
            np.asarray(layer.lut, dtype=np.int8),
        )
    return physical, hwc8_to_nchw(physical, weights.shape[0])


def numerical_result(simulator: np.ndarray, golden: np.ndarray) -> dict[str, object]:
    mismatches = int(np.count_nonzero(simulator != golden))
    return {
        "compared_values": int(golden.size),
        "mismatch_count": mismatches,
        "output_sha256": sha256(np.ascontiguousarray(simulator).tobytes()),
        "pass": mismatches == 0,
    }


def float_result(simulator: np.ndarray, scale: float, reference: np.ndarray) -> dict[str, object]:
    dequantized = simulator.astype(np.float32) * np.float32(scale)
    difference = dequantized.astype(np.float64) - reference.astype(np.float64)
    return {
        "compared_values": int(reference.size),
        "int8_saturation_count": int(
            np.count_nonzero((simulator == -128) | (simulator == 127))
        ),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
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
    parser.add_argument("--output-first", type=Path, default=DEFAULT_FIRST)
    parser.add_argument("--output-second", type=Path, default=DEFAULT_SECOND)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    import onnx
    import onnxruntime
    from onnx import TensorProto, helper

    package = compile_pinned_first_two_layers(args.model, args.calibration, args.luts)
    if compile_pinned_first_two_layers(args.model, args.calibration, args.luts) != package:
        raise SystemExit("repeated pipeline compilation was not byte-identical")
    parsed = load_hxb(package)
    layers, _ = load_pinned_first_two_layers(args.model, args.calibration, args.luts)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))

    full_float = synthetic_input()
    full_i8 = quantize_int8(full_float, layers[0].input_scale)
    physical_input = np.ascontiguousarray(nchw_to_hwc8(full_i8)).tobytes()
    runtime = SimulatorRuntime()
    runtime.load_model(package)
    runtime.bind_input(physical_input)
    token = runtime.submit()
    completion = runtime.wait(token)
    if completion.status is not RuntimeStatus.SUCCESS or completion.output is None:
        raise SystemExit(f"simulator execution failed: {completion.error}")
    if runtime.last_submission_stats is None:
        raise SystemExit("runtime did not record FIFO submission statistics")

    tensors = parsed.manifest["output"]["tensors"]
    simulator_tensors = []
    for tensor in tensors:
        start = int(tensor["addend"])
        end = start + int(tensor["bytes"])
        simulator_tensors.append(
            np.frombuffer(completion.output[start:end], dtype=np.int8)
            .copy()
            .reshape(tensor["physical_shape"])
        )
    golden_first_physical, golden_first = integer_layer(full_i8, layers[0])
    golden_second_physical, _ = integer_layer(golden_first, layers[1])
    golden_tensors = [golden_first_physical, golden_second_physical]
    exact = []
    for layer, simulator, golden in zip(layers, simulator_tensors, golden_tensors):
        result = numerical_result(simulator, golden)
        result.update(
            {
                "name": layer.name,
                "reference": "haslab_ref conv2d_i8 + per-group silu_lut_i32",
            }
        )
        exact.append(result)
        if not result["pass"]:
            first = np.argwhere(simulator != golden)[0].tolist()
            raise SystemExit(
                f"{layer.name} differs from integer golden at {result['mismatch_count']} values; first={first}"
            )

    model = onnx.load(args.model)
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                model.graph.node[2].output[0], TensorProto.FLOAT, [1, 16, 160, 160]
            ),
            helper.make_tensor_value_info(
                model.graph.node[5].output[0], TensorProto.FLOAT, [1, 32, 80, 80]
            ),
        ]
    )
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    float_outputs = session.run(None, {"images": full_float})
    float_comparisons = []
    for layer, simulator, reference in zip(layers, simulator_tensors, float_outputs):
        simulator_nchw = hwc8_to_nchw(simulator, np.asarray(layer.weights).shape[0])[0]
        result = float_result(simulator_nchw, layer.output_scale, reference[0])
        result.update(
            {
                "name": layer.name,
                "reference": f"ONNX Runtime output of {layer.node_names[2]} for the same input",
            }
        )
        float_comparisons.append(result)

    args.output_package.parent.mkdir(parents=True, exist_ok=True)
    args.output_package.write_bytes(package)
    args.output_input.write_bytes(physical_input)
    args.output_tensor.write_bytes(completion.output)
    args.output_first.write_bytes(np.ascontiguousarray(simulator_tensors[0]).tobytes())
    args.output_second.write_bytes(np.ascontiguousarray(simulator_tensors[1]).tobytes())
    schedule = parsed.manifest["schedule"]
    buffers = parsed.manifest["buffers"]
    ext_allocation = sum(int(record["bytes"]) for record in buffers.values())
    report = {
        "schema_version": 1,
        "status": "passing_first_two_conv_silu_layers",
        "scope": {
            "nodes": [name for layer in layers for name in layer.node_names],
            "logical_input_shape": [1, 3, 320, 320],
            "retained_logical_shapes": [[1, 16, 160, 160], [1, 32, 80, 80]],
            "limitation": "This executes the first two Conv-SiLU blocks, not the remaining accelerator graph or host tail.",
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
            "external_bytes": ext_allocation,
            "retained_tensors": tensors,
            "local_peak_bytes": {
                "input": 4624,
                "weight": 4608,
                "accumulator": 2048,
                "output": 512,
                "parameter": 1536,
            },
        },
        "schedule": schedule,
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
            "final_sequence": token.final_sequence,
            "reset_generation": token.reset_generation,
            "fifo": asdict(runtime.last_submission_stats),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
