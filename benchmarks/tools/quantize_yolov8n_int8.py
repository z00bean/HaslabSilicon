#!/usr/bin/env python3
"""Calibrate the pinned YOLOv8n workload and emit an INT8 candidate package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reference"))
from haslab_ref import build_silu_lut  # noqa: E402


EXPECTED_MODEL_SHA256 = "5b232bf21720cac4264463896ace02b919d8bec3f3fd1e4b9d36695493fa1718"
EXPECTED_SELECTION_SHA256 = "cf40f74b98f22fd76f974be06357a2fc3c06bf422cade84752f6105eb73e3e7c"
EXPECTED_IMAGE_COLLECTION_SHA256 = "c6661e43d1d8fe20a3cc62b52a0de095a53773bbb0dbdebee3f5f61b3dc2c760"
HOST_TAIL_START = 218


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to decode {path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = min(320 / height, 320 / width, 1.0)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
    pad_width = 320 - resized_width
    pad_height = 320 - resized_height
    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    tensor = image.transpose(2, 0, 1)[None].astype(np.float32)
    tensor /= np.float32(255.0)
    return np.ascontiguousarray(tensor)


class ImageReader(CalibrationDataReader):
    def __init__(self, paths: list[Path], limit: int | None = None):
        self.paths = paths[:limit]
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.paths):
            return None
        path = self.paths[self.index]
        self.index += 1
        return {"images": preprocess(path)}

    def rewind(self) -> None:
        self.index = 0


def fixed_point_ratio(value: float) -> tuple[int, int]:
    if not math.isfinite(value) or value < 0:
        raise ValueError("fixed-point ratio must be finite and nonnegative")
    for shift in range(62, -1, -1):
        multiplier = round(value * (1 << shift))
        if 0 <= multiplier <= (1 << 31) - 1:
            return multiplier, shift
    raise ValueError(f"fixed-point ratio is outside the v0 range: {value}")


def initializer_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-luts", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    model_hash = sha256_file(args.model)
    if model_hash != EXPECTED_MODEL_SHA256:
        raise SystemExit(f"model hash mismatch: {model_hash}")
    root = args.dataset_root.resolve()
    selection_path = args.selection_manifest or (
        root / "calibration-train2017-512.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selection"]["ordered_filename_sha256"] != EXPECTED_SELECTION_SHA256:
        raise SystemExit("calibration filename selection hash mismatch")
    if selection["selection"]["image_collection_sha256"] != EXPECTED_IMAGE_COLLECTION_SHA256:
        raise SystemExit("calibration image collection hash mismatch")
    image_dir = root / "images" / "train2017-calibration-512"
    image_paths = [image_dir / item["filename"] for item in selection["selection"]["images"]]
    if any(not path.is_file() for path in image_paths):
        raise SystemExit("one or more selected calibration images are missing")

    float_model = onnx.load(args.model)
    nodes_to_quantize = [node.name for node in float_model.graph.node[:HOST_TAIL_START]]
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        args.model,
        args.output_model,
        ImageReader(image_paths, args.limit),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=nodes_to_quantize,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "CalibTensorRangeSymmetric": True,
            # Keep calibration memory bounded. Use a chunk size that does not
            # divide 512 because ORT 1.20.1 rejects an empty final chunk.
            "CalibMaxIntermediateOutputs": 17,
            "ForceQuantizeNoInputCheck": True,
        },
    )

    quantized = onnx.load(args.output_model)
    onnx.checker.check_model(quantized, full_check=True)
    quantized_nodes = {node.name: node for node in quantized.graph.node}
    original_initializers = {item.name for item in float_model.graph.initializer}
    values = initializer_values(quantized)
    scales: list[dict[str, object]] = []
    scale_values: dict[str, np.ndarray] = {}
    for name, array in sorted(values.items()):
        if not name.endswith("_scale"):
            continue
        array32 = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(array32)) or np.any(array32 <= 0):
            raise SystemExit(f"invalid quantization scale: {name}")
        scale_values[name] = array32
        base_name = name[: -len("_scale")]
        scales.append(
            {
                "name": name,
                "kind": "weight_per_output_channel" if base_name in original_initializers else "activation_per_tensor",
                "shape": list(array32.shape),
                "values": array32.reshape(-1).astype(float).tolist(),
                "binary32_sha256": hashlib.sha256(array32.tobytes()).hexdigest(),
            }
        )
    zero_points = [
        np.asarray(array)
        for name, array in values.items()
        if name.endswith("_zero_point")
    ]
    if any(np.any(array != 0) for array in zero_points):
        raise SystemExit("quantized model contains a nonzero zero point")

    producers = {
        output: index
        for index, node in enumerate(float_model.graph.node)
        for output in node.output
    }
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(float_model.graph.node):
        for name in node.input:
            consumers.setdefault(name, []).append(index)

    lut_blob = bytearray()
    lut_records: list[dict[str, object]] = []
    for sigmoid_index, sigmoid in enumerate(float_model.graph.node[:HOST_TAIL_START]):
        if sigmoid.op_type != "Sigmoid" or len(sigmoid.input) != 1:
            continue
        uses = consumers.get(sigmoid.output[0], [])
        if len(uses) != 1:
            continue
        mul_index = uses[0]
        mul = float_model.graph.node[mul_index]
        if mul.op_type != "Mul" or sorted(mul.input) != sorted((sigmoid.input[0], sigmoid.output[0])):
            continue
        conv_index = producers.get(sigmoid.input[0])
        if conv_index is None or float_model.graph.node[conv_index].op_type != "Conv":
            raise SystemExit(f"SiLU input is not produced by Conv: {sigmoid.name}")
        conv = float_model.graph.node[conv_index]
        quantized_conv = quantized_nodes[f"{conv.name}_quant"]
        quantized_mul = quantized_nodes[f"{mul.name}_quant"]
        input_scale = values[quantized_conv.input[1]].reshape(-1)
        weight_scale = values[quantized_conv.input[4]].reshape(-1)
        pre_scale = float(values[quantized_conv.input[6]].reshape(-1)[0])
        output_scale = float(values[quantized_mul.input[6]].reshape(-1)[0])
        delta = float(np.float32(pre_scale * (127.0 / 511.0)))
        lut = build_silu_lut(delta, output_scale)
        offset = len(lut_blob)
        lut_bytes = lut.tobytes()
        lut_blob.extend(lut_bytes)
        accumulator_scales = input_scale * weight_scale
        ratios = [float(scale) / delta for scale in accumulator_scales]
        fixed = [fixed_point_ratio(ratio) for ratio in ratios]
        ratio_errors = [
            abs(ratio - multiplier / float(1 << shift))
            for ratio, (multiplier, shift) in zip(ratios, fixed)
        ]
        lut_records.append(
            {
                "conv_node_index": conv_index,
                "conv_node_name": conv.name,
                "sigmoid_node_index": sigmoid_index,
                "mul_node_index": mul_index,
                "input_tensor": sigmoid.input[0],
                "output_tensor": mul.output[0],
                "grid": {
                    "index_min": -512,
                    "index_max": 511,
                    "delta_binary32": delta,
                    "real_min": float(np.float32(-512 * delta)),
                    "real_max": float(np.float32(511 * delta)),
                },
                "output_scale_binary32": output_scale,
                "table_offset": offset,
                "table_bytes": len(lut_bytes),
                "table_sha256": hashlib.sha256(lut_bytes).hexdigest(),
                "accumulator_to_grid": {
                    "multipliers": [item[0] for item in fixed],
                    "shifts": [item[1] for item in fixed],
                    "maximum_absolute_ratio_error": max(ratio_errors),
                },
            }
        )

    args.output_luts.parent.mkdir(parents=True, exist_ok=True)
    args.output_luts.write_bytes(lut_blob)
    calibration_count = args.limit if args.limit is not None else len(image_paths)
    report = {
        "schema_version": 1,
        "status": "candidate",
        "source_model": {
            "filename": args.model.name,
            "sha256": model_hash,
        },
        "calibration": {
            "images": calibration_count,
            "selection_manifest": selection_path.name,
            "ordered_filename_sha256": EXPECTED_SELECTION_SHA256,
            "image_collection_sha256": EXPECTED_IMAGE_COLLECTION_SHA256,
            "preprocessing": "pinned RGB letterbox 320x320, scale-up disabled, OpenCV INTER_LINEAR, pad 114, NCHW FLOAT32 divided by FLOAT32(255)",
            "observer": "ONNX Runtime MinMax",
            "symmetric_range": True,
        },
        "profile": {
            "quantized_node_range": [0, HOST_TAIL_START - 1],
            "format": "ONNX Runtime QOperator software proxy",
            "activations": "signed symmetric INT8 per tensor, zero point 0",
            "weights": "signed symmetric INT8 per Conv output channel, zero point 0",
            "accumulation": "signed INT32 in QLinearConv",
            "host_tail": "FLOAT beginning at node 218",
        },
        "quantized_model": {
            "filename": args.output_model.name,
            "bytes": args.output_model.stat().st_size,
            "sha256": sha256_file(args.output_model),
            "onnx_node_count": len(quantized.graph.node),
        },
        "scales": scales,
        "scale_counts": {
            "activation_per_tensor": sum(
                record["kind"] == "activation_per_tensor" for record in scales
            ),
            "weight_per_output_channel": sum(
                record["kind"] == "weight_per_output_channel" for record in scales
            ),
        },
        "zero_point_count": len(zero_points),
        "all_zero_points_are_zero": True,
        "silu": {
            "tables": len(lut_records),
            "grid_policy": "delta = binary32(pre-SiLU symmetric MinMax scale * 127 / 511), covering the calibrated symmetric preactivation range with indices -512..511",
            "binary_filename": args.output_luts.name,
            "binary_bytes": len(lut_blob),
            "binary_sha256": hashlib.sha256(lut_blob).hexdigest(),
            "records": lut_records,
        },
        "limitations": [
            "The ONNX Runtime QOperator model is an executable software proxy for calibrated graph-level accuracy; QLinearSigmoid/QLinearMul do not execute the HASLAB 1024-entry fused SiLU LUT.",
            "Exact HASLAB command-level execution, accumulator overflow checks, local-memory tiling, and fixed-point epilogues require the M6 compiler and simulated runtime.",
            "Clipping/saturation counts and layerwise error are produced by a separate diagnostic pass.",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv_python": cv2.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "calibration_images": calibration_count,
                "quantized_model_sha256": report["quantized_model"]["sha256"],
                "scales": len(scales),
                "silu_tables": len(lut_records),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
