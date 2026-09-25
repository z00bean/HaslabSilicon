#!/usr/bin/env python3
"""Compile and validate pinned YOLOv8n nodes 0..119 in two allocation modes."""

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
    extend_with_first_neck_stage,
    first_stage,
)
from haslab_compiler.c2f import load_pinned_through_first_neck
from haslab_ref import (
    add_i8,
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
DEFAULT_REPORT = WORKLOAD / "m6-through-first-neck.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def execute(package: bytes, physical_input: bytes):
    runtime = SimulatorRuntime(ext_bytes=8 << 20)
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
    return np.frombuffer(output[start:end], dtype=np.int8).copy().reshape(record["physical_shape"])


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
        neck,
        model_hash,
    ) = load_pinned_through_first_neck(args.model, args.calibration, args.luts)
    graph = extend_with_first_neck_stage(
        build_backbone_graph(
            first=first_stage(first),
            extensions=(
                C2fExtension(downsample2, second),
                C2fExtension(downsample3, third),
                C2fExtension(downsample4, fourth),
            ),
            sppf=sppf,
        ),
        neck,
    )
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
    operations = diagnostic_package.manifest["schedule"]["graph_operations"]

    golden: dict[str, np.ndarray] = {}
    golden["model.0"], logical = integer_conv_silu(full_i8, stem[0])
    golden["model.1"], logical = integer_conv_silu(logical, stem[1])
    golden["model.2.cv1"], _ = integer_conv_silu(logical, first.cv1)
    golden["model.2.split0"] = golden["model.2.cv1"][:, :, :2, :]
    golden["model.2.split1"] = golden["model.2.cv1"][:, :, 2:, :]
    golden["model.2.m.0.cv1"], logical = integer_conv_silu(
        hwc8_to_nchw(golden["model.2.split1"], 16), first.bottleneck_cv1
    )
    golden["model.2.m.0.cv2"], _ = integer_conv_silu(logical, first.bottleneck_cv2)
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
    golden["model.2"], logical = integer_conv_silu(
        hwc8_to_nchw(golden["model.2.concat"], 48), first.cv2
    )
    golden["model.3"], logical = integer_conv_silu(logical, downsample2)
    golden["model.4.cv1"], _ = integer_conv_silu(logical, second.cv1)
    golden["model.4.split0"] = golden["model.4.cv1"][:, :, :4, :]
    golden["model.4.split1"] = golden["model.4.cv1"][:, :, 4:, :]
    branch = golden["model.4.split1"]
    for index, (conv1, conv2) in enumerate(second.bottlenecks):
        name1 = f"model.4.m.{index}.cv1"
        name2 = f"model.4.m.{index}.cv2"
        add_name = f"model.4.m.{index}.add"
        golden[name1], logical = integer_conv_silu(hwc8_to_nchw(branch, 32), conv1)
        golden[name2], _ = integer_conv_silu(logical, conv2)
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
    golden["model.4"], _ = integer_conv_silu(
        hwc8_to_nchw(golden["model.4.concat"], 128), second.cv2
    )
    golden["model.5"], logical = integer_conv_silu(
        hwc8_to_nchw(golden["model.4"], 64), downsample3
    )
    golden["model.6.cv1"], _ = integer_conv_silu(logical, third.cv1)
    golden["model.6.split0"] = golden["model.6.cv1"][:, :, :8, :]
    golden["model.6.split1"] = golden["model.6.cv1"][:, :, 8:, :]
    branch = golden["model.6.split1"]
    for index, (conv1, conv2) in enumerate(third.bottlenecks):
        name1 = f"model.6.m.{index}.cv1"
        name2 = f"model.6.m.{index}.cv2"
        add_name = f"model.6.m.{index}.add"
        golden[name1], logical = integer_conv_silu(hwc8_to_nchw(branch, 64), conv1)
        golden[name2], _ = integer_conv_silu(logical, conv2)
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
    golden["model.6"], _ = integer_conv_silu(
        hwc8_to_nchw(golden["model.6.concat"], 256), third.cv2
    )
    golden["model.7"], logical = integer_conv_silu(
        hwc8_to_nchw(golden["model.6"], 128), downsample4
    )
    golden["model.8.cv1"], _ = integer_conv_silu(logical, fourth.cv1)
    golden["model.8.split0"] = golden["model.8.cv1"][:, :, :16, :]
    golden["model.8.split1"] = golden["model.8.cv1"][:, :, 16:, :]
    golden["model.8.m.0.cv1"], logical = integer_conv_silu(
        hwc8_to_nchw(golden["model.8.split1"], 128), fourth.bottlenecks[0][0]
    )
    golden["model.8.m.0.cv2"], _ = integer_conv_silu(
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
    golden["model.8"], logical = integer_conv_silu(
        hwc8_to_nchw(golden["model.8.concat"], 384), fourth.cv2
    )
    golden["model.9.cv1"], _ = integer_conv_silu(logical, sppf.cv1)
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
    golden["model.9"], _ = integer_conv_silu(
        hwc8_to_nchw(golden["model.9.concat"], 512), sppf.cv2
    )
    golden["model.10"] = upsample2_nearest_i8(golden["model.9"])
    golden["model.11.concat"] = apply_concat(
        [golden["model.10"], golden["model.6"]], operations[40]
    )
    golden["model.12.cv1"], _ = integer_conv_silu(
        hwc8_to_nchw(golden["model.11.concat"], 384), neck.stage.cv1
    )
    golden["model.12.split0"] = golden["model.12.cv1"][:, :, :8, :]
    golden["model.12.split1"] = golden["model.12.cv1"][:, :, 8:, :]
    golden["model.12.m.0.cv1"], logical = integer_conv_silu(
        hwc8_to_nchw(golden["model.12.split1"], 64),
        neck.stage.bottlenecks[0][0],
    )
    golden["model.12.m.0.cv2"], _ = integer_conv_silu(
        logical, neck.stage.bottlenecks[0][1]
    )
    golden["model.12.concat"] = apply_concat(
        [
            golden["model.12.split0"],
            golden["model.12.split1"],
            golden["model.12.m.0.cv2"],
        ],
        operations[44],
    )
    golden["model.12"], _ = integer_conv_silu(
        hwc8_to_nchw(golden["model.12.concat"], 192), neck.stage.cv2
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
    ]
    exact = [exact_result(name, simulator[name], golden[name]) for name in ordered_names]
    if not all(item["pass"] for item in exact):
        failure = next(item for item in exact if not item["pass"])
        raise SystemExit(f"{failure['name']} differs at {failure['mismatch_count']} values")

    release_record = release_package.manifest["output"]["tensors"][0]
    release_final = tensor_from_output(release_output, release_record)
    release_result = exact_result("model.12", release_final, golden["model.12"])
    if not release_result["pass"]:
        raise SystemExit("release first-neck output differs from the golden model")

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
        (105, [1, 384, 20, 20], "model.11.concat", neck.concat_scale),
        (108, [1, 128, 20, 20], "model.12.cv1", neck.stage.cv1.output_scale),
        (112, [1, 64, 20, 20], "model.12.m.0.cv1", neck.stage.bottlenecks[0][0].output_scale),
        (115, [1, 64, 20, 20], "model.12.m.0.cv2", neck.stage.bottlenecks[0][1].output_scale),
        (116, [1, 192, 20, 20], "model.12.concat", neck.stage.concat_scale),
        (119, [1, 128, 20, 20], "model.12", neck.stage.cv2.output_scale),
    ]
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
        float_comparisons.append(float_result(name, logical_simulator, scale, reference))

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for mode, package in packages.items():
        (args.artifact_dir / f"yolov8n-through-first-neck-{mode}.hxb").write_bytes(package)
    (args.artifact_dir / "yolov8n-through-first-neck-input.bin").write_bytes(physical_input)
    (args.artifact_dir / "yolov8n-through-first-neck-diagnostic-output.bin").write_bytes(
        diagnostic_output
    )
    (args.artifact_dir / "yolov8n-through-first-neck-release-output.bin").write_bytes(
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
        "status": "passing_nodes_0_through_119_first_neck_diagnostic_and_release",
        "scope": {
            "node_indices": [0, 119],
            "nodes": diagnostic_package.manifest["source"]["nodes"],
            "logical_input_shape": [1, 3, 320, 320],
            "compared_boundaries": ordered_names,
            "limitation": "This executes the backbone and first top-down neck C2f, not the remaining neck, learned heads, or host tail.",
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
        "release_final_comparison": release_result,
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
