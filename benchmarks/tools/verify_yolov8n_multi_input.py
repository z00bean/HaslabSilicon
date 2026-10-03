#!/usr/bin/env python3
"""Differentially run pinned YOLOv8n commands and host tail on diverse inputs."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from haslab_compiler.c2f import load_pinned_through_detection_head
from haslab_ref import nchw_to_hwc8, quantize_int8
from haslab_runtime import (Letterbox, RuntimeStatus, SimulatorRuntime,
                            decode_host_tail, select_detections, unpack_boundaries)

from compile_yolov8n_detection_head import calculate_golden, tensor_from_output
from quantize_yolov8n_int8 import preprocess
from verify_yolov8n_host_tail import STAGES, compare, reference_tail
from yolov8n_integer_oracle import conv_raw, conv_silu

ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "benchmarks/manifests/yolov8n-320-opset13"
MODEL = ROOT / "models/generated/yolov8n-320-opset13.onnx"
REPORT = WORKLOAD / "m6-multi-input.json"
CASES = ("zero", "max", "min", "seeded", "calibration-0", "calibration-1")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_case(name: str, input_scale: float, selection: dict,
              image_dir: Path) -> tuple[np.ndarray, Letterbox, dict]:
    if name in {"zero", "max", "min"}:
        level = {"zero": 0, "max": 127, "min": -128}[name]
        source = np.full((1, 3, 320, 320), level, dtype=np.int8)
        return source, Letterbox.centered(320, 320), {"kind": "constant_int8", "value": level}
    if name == "seeded":
        source = np.random.default_rng(0x4841534C4142).integers(
            -128, 128, size=(1, 3, 320, 320), dtype=np.int8
        )
        return source, Letterbox.centered(320, 320), {
            "kind": "numpy_PCG64_full_range_int8", "seed_hex": "0x4841534c4142"
        }
    if name.startswith("calibration-"):
        index = int(name.split("-")[1])
        item = selection["selection"]["images"][index]
        image_path = image_dir / item["filename"]
        image_bytes = image_path.read_bytes()
        if len(image_bytes) != item["bytes"] or sha256(image_bytes) != item["sha256"]:
            raise ValueError(f"pinned calibration image changed: {item['filename']}")
        float_input = preprocess(image_path)
        source = quantize_int8(float_input, input_scale)
        return source, Letterbox.centered(item["height"], item["width"]), {
            "kind": "pinned_train2017_calibration_image",
            "rank": index, "filename": item["filename"],
            "image_sha256": item["sha256"],
            "original_shape": [item["height"], item["width"]],
        }
    raise ValueError(f"unknown case: {name}")


def assert_equal(name: str, actual: np.ndarray, expected: np.ndarray,
                 physical_input: bytes, artifact_dir: Path) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape or dtype mismatch")
    mismatch = np.argwhere(actual != expected)
    if mismatch.size:
        first = tuple(int(value) for value in mismatch[0])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "multi-input-failing-input.bin"
        path.write_bytes(physical_input)
        raise AssertionError(
            f"{name}: {len(mismatch)} mismatches; first at {first}: "
            f"simulator={int(actual[first])}, oracle={int(expected[first])}; "
            f"reproducer={path}"
        )
    return {"name": name, "values": int(actual.size),
            "output_sha256": sha256(np.ascontiguousarray(actual).tobytes())}


def postprocess_comparison(actual: np.ndarray, expected: np.ndarray,
                           letterbox: Letterbox, confidence: float) -> dict:
    found = select_detections(actual, letterbox, confidence=confidence)
    reference = select_detections(expected, letterbox, confidence=confidence)
    same_classes = bool(np.array_equal(found[:, 5], reference[:, 5]))
    numerical = compare(found, reference)
    if not same_classes or not numerical["allclose_atol_0.0001_rtol_0.00001"]:
        raise AssertionError(f"host detections differ at confidence {confidence}")
    return {"confidence": confidence, "count": len(found),
            "same_class_order": same_classes, "comparison": numerical}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--allocation-mode", choices=("diagnostic", "release"),
                        default="diagnostic")
    parser.add_argument("--reuse-output", action="store_true",
                        help="reuse an exact cached output after validating its package/input hashes")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--image-dir", type=Path,
                        default=ROOT / "datasets/coco/images/train2017-calibration-512")
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("cases must not be repeated")

    previous = json.loads((WORKLOAD / "m6-through-detection-head.json").read_text())
    package_path = ROOT / f"artifacts/m6/yolov8n-through-detection-head-{args.allocation_mode}.hxb"
    model_data, package_data = MODEL.read_bytes(), package_path.read_bytes()
    if sha256(model_data) != previous["source_model"]["sha256"]:
        raise ValueError("pinned model hash changed")
    if sha256(package_data) != previous["packages"][args.allocation_mode]["sha256"]:
        raise ValueError("package hash changed")
    layers = load_pinned_through_detection_head(
        MODEL, WORKLOAD / "int8-calibration.json", WORKLOAD / "int8-silu-luts.bin"
    )[:-1]
    selection = json.loads((WORKLOAD / "calibration-selection.json").read_text())
    runtime = SimulatorRuntime(ext_bytes=128 << 20)
    package = runtime.load_model(package_data)
    records = package.manifest["output"]["tensors"]
    boundary_records = [record for record in records
                        if record["name"].startswith("model.22.boundary")]
    operations = package.manifest["schedule"]["graph_operations"]
    if len(records) != (90 if args.allocation_mode == "diagnostic" else 3) \
            or len(boundary_records) != 3:
        raise ValueError("package boundaries changed")

    report = {
        "schema_version": 1,
        "status": "in_progress",
        "scope": "pinned full accelerator commands and explicit host tail; batch one",
        "allocation_mode": args.allocation_mode,
        "source_model_sha256": sha256(model_data),
        "package_sha256": sha256(package_data),
        "comparison_policy": f"all {len(records)} stored boundaries bit-exact against independent vectorized integer oracle; host nodes 218..260 against isolated ONNX Runtime tail",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "requested_cases": list(args.cases),
        "cases": {},
        "limitation": "No COCO command-path accuracy or hardware performance result is implied.",
    }
    if args.report.exists():
        old = json.loads(args.report.read_text())
        old_hash = old.get("package_sha256", old.get("diagnostic_package_sha256"))
        if old_hash != report["package_sha256"]:
            raise ValueError("existing report refers to a different package")
        report["cases"] = old["cases"]
        report["requested_cases"] = sorted(set(old.get("requested_cases", old["cases"])) | set(args.cases))
    for name in args.cases:
        if name in report["cases"]:
            print(f"SKIP {name}: recorded already", flush=True)
            continue
        started = time.monotonic()
        source, letterbox, provenance = load_case(
            name, layers[0][0].input_scale, selection, args.image_dir
        )
        physical_input = np.ascontiguousarray(nchw_to_hwc8(source)).tobytes()
        oracle = calculate_golden(
            source, *layers, operations, conv_silu=conv_silu, conv_raw=conv_raw
        )
        output_path = ROOT / "artifacts/m6" / f"multi-input-{args.allocation_mode}-{name}-output.bin"
        receipt_path = output_path.with_suffix(".json")
        if args.reuse_output:
            receipt = json.loads(receipt_path.read_text())
            output_data = output_path.read_bytes()
            if (receipt["package_sha256"] != sha256(package_data)
                    or receipt["input_sha256"] != sha256(physical_input)
                    or receipt["output_sha256"] != sha256(output_data)
                    or len(output_data) != package.manifest["buffers"]["output"]["bytes"]):
                raise ValueError("cached output does not match the exact package/input")
            stats = receipt["simulator_stats"]
        else:
            runtime.bind_input(physical_input)
            completion = runtime.wait(runtime.submit())
            if completion.status is not RuntimeStatus.SUCCESS or completion.output is None:
                raise AssertionError(f"{name}: simulator failed: {completion.error}")
            output_data = completion.output
            stats = vars(runtime.last_submission_stats)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(output_data)
            receipt_path.write_text(json.dumps({
                "package_sha256": sha256(package_data),
                "input_sha256": sha256(physical_input),
                "output_sha256": sha256(output_data),
                "simulator_stats": stats,
            }, indent=2, sort_keys=True) + "\n")
        comparisons = [
            assert_equal(record["name"], tensor_from_output(output_data, record),
                         oracle[record["name"]], physical_input, ROOT / "artifacts/m6")
            for record in records
        ]
        boundaries = unpack_boundaries(output_data, boundary_records)
        decoded, trace = decode_host_tail(boundaries, trace=True)
        expected_trace, expected, onnx_version, ort_version = reference_tail(MODEL, boundaries)
        stages = {stage: compare(trace[stage], expected_trace[stage]) for stage in STAGES}
        final = compare(decoded, expected)
        if not all(entry["allclose_atol_0.0001_rtol_0.00001"] for entry in stages.values()) \
                or not final["allclose_atol_0.0001_rtol_0.00001"]:
            raise AssertionError(f"{name}: host tail differs from pinned ONNX tail")
        report["cases"][name] = {
            "provenance": provenance,
            "input_hwc8_sha256": sha256(physical_input),
            "input_int8_min": int(source.min()), "input_int8_max": int(source.max()),
            "output_sha256": sha256(output_data),
            "materialized_boundaries": len(comparisons),
            "exact_values_compared": sum(item["values"] for item in comparisons),
            "boundary_comparisons": comparisons,
            "host_stage_comparisons": stages,
            "host_decoded_comparison": final,
            "operational_postprocessing": postprocess_comparison(decoded, expected, letterbox, 0.25),
            "evaluation_postprocessing": postprocess_comparison(decoded, expected, letterbox, 0.001),
            "simulator_stats": stats,
            "environment": {"onnx": onnx_version, "onnxruntime": ort_version},
            "simulator_executed_in_this_invocation": not args.reuse_output,
            "elapsed_seconds_total": round(time.monotonic() - started, 3),
        }
        if not args.reuse_output:
            runtime.reset()
        report["status"] = "complete" if set(report["requested_cases"]) <= set(report["cases"]) else "in_progress"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"PASS {name}: {report['cases'][name]['exact_values_compared']:,} exact values, "
              f"{len(report['cases'][name]['boundary_comparisons'])} boundaries, "
              f"{time.monotonic() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
