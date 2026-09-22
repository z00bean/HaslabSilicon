#!/usr/bin/env python3
"""Compile, execute, and validate the complete pinned YOLOv8n first layer."""

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

from haslab_compiler import compile_pinned_first_layer
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
DEFAULT_PACKAGE = ROOT / "artifacts/m6/yolov8n-first-conv-silu-layer.hxb"
DEFAULT_INPUT = ROOT / "artifacts/m6/yolov8n-first-conv-silu-layer-input.bin"
DEFAULT_OUTPUT = ROOT / "artifacts/m6/yolov8n-first-conv-silu-layer-output.bin"
DEFAULT_REPORT = WORKLOAD / "m6-first-conv-silu-layer.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_input() -> np.ndarray:
    channel = np.arange(3, dtype=np.int32).reshape(1, 3, 1, 1)
    y = np.arange(320, dtype=np.int32).reshape(1, 1, 320, 1)
    x = np.arange(320, dtype=np.int32).reshape(1, 1, 1, 320)
    values = (channel * 37 + y * 3 + x * 5) % 256
    return np.ascontiguousarray(values.astype(np.float32) / np.float32(255.0))


def scale_record(calibration: dict[str, object], name: str) -> np.ndarray:
    record = next(item for item in calibration["scales"] if item["name"] == name)
    return np.asarray(record["values"], dtype=np.float32)


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
    from onnx import TensorProto, helper, numpy_helper

    package = compile_pinned_first_layer(args.model, args.calibration, args.luts)
    if compile_pinned_first_layer(args.model, args.calibration, args.luts) != package:
        raise SystemExit("repeated first-layer compilation was not byte-identical")
    parsed = load_hxb(package)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    input_scale = float(scale_record(calibration, "images_scale")[0])
    weight_scales = scale_record(calibration, "model.0.conv.weight_scale")
    silu_record = calibration["silu"]["records"][0]
    multipliers = np.asarray(
        silu_record["accumulator_to_grid"]["multipliers"], dtype=np.int64
    )
    shifts = np.asarray(
        silu_record["accumulator_to_grid"]["shifts"], dtype=np.int64
    )
    lut_blob = args.luts.read_bytes()
    lut_start = int(silu_record["table_offset"])
    lut_bytes = int(silu_record["table_bytes"])
    lut = np.frombuffer(lut_blob[lut_start : lut_start + lut_bytes], dtype=np.int8).copy()

    full_float = synthetic_input()
    full_i8 = quantize_int8(full_float, input_scale)
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

    model = onnx.load(args.model)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    conv = model.graph.node[0]
    weights = np.asarray(initializers[conv.input[1]], dtype=np.float32)
    bias = np.asarray(initializers[conv.input[2]], dtype=np.float32)
    weight_i8 = quantize_int8(weights, weight_scales.reshape(16, 1, 1, 1))
    bias_i32 = np.rint(
        bias.astype(np.float64)
        / (np.float32(input_scale).astype(np.float64) * weight_scales.astype(np.float64))
    ).astype(np.int32)
    accumulators = conv2d_i8(full_i8, weight_i8, bias_i32, stride=2, padding=1)
    accumulator_hwc8 = nchw_to_hwc8(accumulators)
    golden = np.empty((160, 160, 2, 8), dtype=np.int8)
    for group in range(2):
        start = group * 8
        golden[:, :, group : group + 1, :] = silu_lut_i32(
            accumulator_hwc8[:, :, group : group + 1, :],
            multipliers[start : start + 8],
            shifts[start : start + 8],
            lut,
        )
    simulator = np.frombuffer(completion.output, dtype=np.int8).reshape(160, 160, 2, 8)
    mismatch_count = int(np.count_nonzero(simulator != golden))
    if mismatch_count:
        mismatch = np.argwhere(simulator != golden)[0].tolist()
        raise SystemExit(
            f"simulator differs from integer golden at {mismatch_count} values; first={mismatch}"
        )

    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(
            model.graph.node[2].output[0], TensorProto.FLOAT, [1, 16, 160, 160]
        )
    )
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    float_output = session.run(None, {"images": full_float})[0][0]
    simulator_nchw = hwc8_to_nchw(simulator, 16)[0].astype(np.float32)
    dequantized = simulator_nchw * np.float32(silu_record["output_scale_binary32"])
    difference = dequantized.astype(np.float64) - float_output.astype(np.float64)

    args.output_package.parent.mkdir(parents=True, exist_ok=True)
    args.output_package.write_bytes(package)
    args.output_input.write_bytes(physical_input)
    args.output_tensor.write_bytes(completion.output)
    schedule = parsed.manifest["schedule"]
    runtime_stats = asdict(runtime.last_submission_stats)
    report = {
        "schema_version": 1,
        "status": "passing_complete_first_layer",
        "scope": {
            "nodes": [model.graph.node[index].name for index in range(3)],
            "logical_input_shape": [1, 3, 320, 320],
            "logical_output_shape": [1, 16, 160, 160],
            "physical_output_shape": [160, 160, 2, 8],
            "limitation": "This executes the complete first Conv-SiLU layer, not the remaining accelerator graph or host tail.",
        },
        "source_model": calibration["source_model"],
        "calibration_sha256": sha256(args.calibration.read_bytes()),
        "lut_binary_sha256": sha256(lut_blob),
        "package": {
            "schema": parsed.manifest["schema"],
            "bytes": len(package),
            "sha256": sha256(package),
            "repeated_compilation_byte_identical": True,
            "commands_bytes": len(parsed.commands),
            "constant_bytes": len(parsed.constants),
        },
        "schedule": schedule,
        "input": {
            "generator": "((channel*37 + y*3 + x*5) mod 256) / FLOAT32(255)",
            "physical_bytes": len(physical_input),
            "physical_sha256": sha256(physical_input),
        },
        "exact_integer_comparison": {
            "reference": "haslab_ref conv2d_i8 + per-group silu_lut_i32",
            "compared_values": int(golden.size),
            "mismatch_count": mismatch_count,
            "output_sha256": sha256(completion.output),
            "pass": mismatch_count == 0,
        },
        "onnx_float_comparison": {
            "reference": "ONNX Runtime output of node 2 for the same deterministic input",
            "compared_values": int(float_output.size),
            "mean_absolute_error": float(np.mean(np.abs(difference))),
            "root_mean_square_error": float(np.sqrt(np.mean(difference * difference))),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
            "int8_saturation_count": int(
                np.count_nonzero((simulator == -128) | (simulator == 127))
            ),
        },
        "runtime": {
            "backend": "haslab_sim functional command simulator",
            "completion": completion.status.value,
            "final_sequence": token.final_sequence,
            "reset_generation": token.reset_generation,
            "fifo": runtime_stats,
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
