#!/usr/bin/env python3
"""Differentially verify the explicit host tail on a pinned accelerator output."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from haslab_runtime import (Letterbox, decode_host_tail, load_hxb,
                            select_detections, unpack_boundaries)

ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "benchmarks/manifests/yolov8n-320-opset13"
STAGES = {
    "concat": (224, 0),
    "regression": (226, 0),
    "classes": (226, 1),
    "dfl_softmax": (230, 0),
    "dfl_distances": (233, 0),
    "anchors": (248, 0),
    "xy1": (249, 0),
    "xy2": (251, 0),
    "boxes": (258, 0),
    "strides": (257, 0),
    "scores": (259, 0),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference_tail(model_path: Path, boundaries: tuple[np.ndarray, ...]):
    """Copy only ONNX nodes 218..260 into an independent ORT tail session."""
    import onnx
    import onnxruntime
    from onnx import TensorProto, helper

    model = onnx.load(str(model_path))
    if len(model.graph.node) != 261:
        raise ValueError("pinned model node count changed")
    names = [model.graph.node[i].output[index] for i, index in STAGES.values()]
    inputs = [model.graph.node[i].input[0] for i in (221, 222, 223)]
    outputs = [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 2100])]
    graph = helper.make_graph(
        list(model.graph.node[218:261]), "pinned-yolov8n-host-tail",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, list(arr.shape))
         for name, arr in zip(inputs, boundaries)], outputs,
        initializer=[init for init in model.graph.initializer
                     if init.name == "model.22.dfl.conv.weight"],
    )
    submodel = helper.make_model(graph, opset_imports=list(model.opset_import))
    submodel.ir_version = model.ir_version
    submodel = onnx.shape_inference.infer_shapes(submodel)
    inferred = {value.name: value for value in submodel.graph.value_info}
    final_output = submodel.graph.output[0]
    del submodel.graph.output[:]
    submodel.graph.output.extend([inferred[name] for name in names] + [final_output])
    onnx.checker.check_model(submodel)
    session = onnxruntime.InferenceSession(submodel.SerializeToString(),
                                           providers=["CPUExecutionProvider"])
    values = session.run(None, dict(zip(inputs, boundaries)))
    return dict(zip(STAGES, values[:-1])), values[-1], onnx.__version__, onnxruntime.__version__


def compare(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AssertionError(f"shape/dtype mismatch: {left.shape}/{left.dtype}, {right.shape}/{right.dtype}")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"values": int(delta.size), "maximum_absolute_error": float(delta.max()),
            "mean_absolute_error": float(delta.mean()),
            "allclose_atol_0.0001_rtol_0.00001": bool(np.allclose(left, right, atol=1e-4, rtol=1e-5))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/generated/yolov8n-320-opset13.onnx")
    parser.add_argument("--package", type=Path,
                        default=ROOT / "artifacts/m6/yolov8n-through-detection-head-release.hxb")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/m6/yolov8n-through-detection-head-release-output.bin")
    parser.add_argument("--report", type=Path, default=WORKLOAD / "m6-host-tail.json")
    args = parser.parse_args()

    previous = json.loads((WORKLOAD / "m6-through-detection-head.json").read_text())
    model_data = args.model.read_bytes()
    package_data = args.package.read_bytes()
    output_data = args.output.read_bytes()
    if digest(model_data) != previous["source_model"]["sha256"]:
        raise ValueError("model hash disagrees with accelerator report")
    if digest(package_data) != previous["packages"]["release"]["sha256"]:
        raise ValueError("package hash disagrees with accelerator report")
    if digest(output_data) != previous["packages"]["release"]["output_sha256"]:
        raise ValueError("output hash disagrees with accelerator report")
    package = load_hxb(package_data)
    boundaries = unpack_boundaries(output_data, package.manifest["output"]["tensors"])
    actual, trace = decode_host_tail(boundaries, trace=True)
    expected_trace, expected, onnx_version, ort_version = reference_tail(args.model, boundaries)
    stage_results = {name: compare(trace[name], expected_trace[name]) for name in STAGES}
    final_result = compare(actual, expected)
    if not all(item["allclose_atol_0.0001_rtol_0.00001"] for item in stage_results.values()) \
            or not final_result["allclose_atol_0.0001_rtol_0.00001"]:
        raise AssertionError("explicit host tail differs from pinned ONNX tail")
    letterbox = Letterbox.centered(480, 640)
    detections = select_detections(actual, letterbox)
    expected_detections = select_detections(expected, letterbox)
    same_classes = bool(np.array_equal(detections[:, 5], expected_detections[:, 5]))
    detection_comparison = compare(detections, expected_detections)
    if not same_classes or not detection_comparison["allclose_atol_0.0001_rtol_0.00001"]:
        raise AssertionError("detections differ from the independent ONNX-tail input")
    eval_detections = select_detections(actual, letterbox, confidence=0.001)
    expected_eval_detections = select_detections(expected, letterbox, confidence=0.001)
    eval_same_classes = bool(np.array_equal(eval_detections[:, 5], expected_eval_detections[:, 5]))
    eval_comparison = compare(eval_detections, expected_eval_detections)
    if not eval_same_classes or not eval_comparison["allclose_atol_0.0001_rtol_0.00001"]:
        raise AssertionError("evaluation-profile detections differ from the independent ONNX-tail input")
    report = {
        "schema_version": 1,
        "status": "complete",
        "scope": "one pinned synthetic accelerator output; host nodes 218..260 and deterministic 480x640 postprocessing",
        "source_model_sha256": digest(model_data),
        "release_package_sha256": digest(package_data),
        "release_output_sha256": digest(output_data),
        "decoded_shape": list(actual.shape),
        "boundary_shapes": [list(x.shape) for x in boundaries],
        "stage_comparisons": stage_results,
        "decoded_comparison": final_result,
        "operational_postprocessing": {
            "letterbox_original_shape": [480, 640],
            "confidence": 0.25, "iou": 0.7, "max_detections": 300,
            "detection_count": len(detections),
            "same_class_order": same_classes,
            "comparison": detection_comparison,
        },
        "evaluation_postprocessing": {
            "letterbox_original_shape": [480, 640],
            "confidence": 0.001, "iou": 0.7, "max_detections": 300,
            "detection_count": len(eval_detections),
            "same_class_order": eval_same_classes,
            "comparison": eval_comparison,
        },
        "environment": {"numpy": np.__version__, "onnx": onnx_version,
                        "onnxruntime": ort_version},
        "limitation": "No COCO command-path accuracy, multi-input inference, or hardware result is implied.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decoded": final_result,
                      "operational_detections": report["operational_postprocessing"],
                      "evaluation_detections": report["evaluation_postprocessing"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
