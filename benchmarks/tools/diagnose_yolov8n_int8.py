#!/usr/bin/env python3
"""Measure layerwise error and saturation for the pinned INT8 proxy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

from quantize_yolov8n_int8 import preprocess, sha256_file


HOST_TAIL_START = 218


def value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    inferred = onnx.shape_inference.infer_shapes(model)
    return {
        item.name: item
        for item in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }


def add_outputs(
    model: onnx.ModelProto,
    names: list[str],
    information: dict[str, onnx.ValueInfoProto],
) -> onnx.ModelProto:
    augmented = copy.deepcopy(model)
    existing = {item.name for item in augmented.graph.output}
    for name in names:
        if name not in existing:
            augmented.graph.output.append(copy.deepcopy(information[name]))
    return augmented


def tensor_scale_names(model: onnx.ModelProto) -> dict[str, str]:
    scales: dict[str, str] = {}
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            scales[node.output[0]] = node.input[1]
        elif node.op_type == "QLinearConv":
            scales[node.output[0]] = node.input[6]
        elif node.op_type == "QLinearSigmoid":
            scales[node.output[0]] = node.input[3]
        elif node.op_type in {"QLinearAdd", "QLinearMul"}:
            scales[node.output[0]] = node.input[6]
        elif node.op_type == "QLinearConcat":
            scales[node.output[0]] = node.input[0]
        elif node.input and node.input[0] in scales:
            for output in node.output:
                scales[output] = scales[node.input[0]]
    return scales


def new_stats() -> dict[str, int | float]:
    return {
        "values": 0,
        "clipped_low": 0,
        "clipped_high": 0,
        "saturated_low": 0,
        "saturated_high": 0,
        "local_absolute_error_sum": 0.0,
        "local_squared_error_sum": 0.0,
        "local_max_absolute_error": 0.0,
        "cumulative_absolute_error_sum": 0.0,
        "cumulative_squared_error_sum": 0.0,
        "cumulative_max_absolute_error": 0.0,
    }


def update_stats(
    stats: dict[str, int | float],
    float_values: np.ndarray,
    quantized_values: np.ndarray,
    scale: float,
) -> None:
    source = np.asarray(float_values, dtype=np.float32)
    quantized = np.asarray(quantized_values, dtype=np.int8)
    if source.shape != quantized.shape:
        raise ValueError(f"diagnostic shape mismatch: {source.shape} versus {quantized.shape}")
    rounded = np.rint(source.astype(np.float64) / float(np.float32(scale)))
    local_quantized = np.clip(rounded, -128, 127).astype(np.int8)
    local_dequantized = local_quantized.astype(np.float32) * np.float32(scale)
    cumulative_dequantized = quantized.astype(np.float32) * np.float32(scale)
    local_error = np.abs(source - local_dequantized).astype(np.float64)
    cumulative_error = np.abs(source - cumulative_dequantized).astype(np.float64)
    stats["values"] += source.size
    stats["clipped_low"] += int(np.count_nonzero(rounded < -128))
    stats["clipped_high"] += int(np.count_nonzero(rounded > 127))
    stats["saturated_low"] += int(np.count_nonzero(quantized == -128))
    stats["saturated_high"] += int(np.count_nonzero(quantized == 127))
    stats["local_absolute_error_sum"] += float(local_error.sum())
    stats["local_squared_error_sum"] += float(np.square(local_error).sum())
    stats["local_max_absolute_error"] = max(
        float(stats["local_max_absolute_error"]), float(local_error.max(initial=0.0))
    )
    stats["cumulative_absolute_error_sum"] += float(cumulative_error.sum())
    stats["cumulative_squared_error_sum"] += float(np.square(cumulative_error).sum())
    stats["cumulative_max_absolute_error"] = max(
        float(stats["cumulative_max_absolute_error"]),
        float(cumulative_error.max(initial=0.0)),
    )


def finish_stats(stats: dict[str, int | float]) -> dict[str, int | float]:
    count = int(stats["values"])
    result = dict(stats)
    result["local_mean_absolute_error"] = float(stats["local_absolute_error_sum"]) / count
    result["local_root_mean_squared_error"] = (
        float(stats["local_squared_error_sum"]) / count
    ) ** 0.5
    result["cumulative_mean_absolute_error"] = (
        float(stats["cumulative_absolute_error_sum"]) / count
    )
    result["cumulative_root_mean_squared_error"] = (
        float(stats["cumulative_squared_error_sum"]) / count
    ) ** 0.5
    result["clipped_fraction"] = (
        int(stats["clipped_low"]) + int(stats["clipped_high"])
    ) / count
    result["saturated_fraction"] = (
        int(stats["saturated_low"]) + int(stats["saturated_high"])
    ) / count
    for key in (
        "local_absolute_error_sum",
        "local_squared_error_sum",
        "cumulative_absolute_error_sum",
        "cumulative_squared_error_sum",
    ):
        del result[key]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("float_model", type=Path)
    parser.add_argument("quantized_model", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    selection = json.loads((root / "calibration-train2017-512.json").read_text())
    image_dir = root / "images" / "train2017-calibration-512"
    image_paths = [
        image_dir / item["filename"]
        for item in selection["selection"]["images"][: args.limit]
    ]

    float_model = onnx.load(args.float_model)
    quantized_model = onnx.load(args.quantized_model)
    float_initializers = {
        item.name: numpy_helper.to_array(item) for item in float_model.graph.initializer
    }
    float_nodes = {node.name: node for node in float_model.graph.node[:HOST_TAIL_START]}
    quantized_nodes = {
        node.name[: -len("_quant")]: node
        for node in quantized_model.graph.node
        if node.name.endswith("_quant")
    }
    scale_names = tensor_scale_names(quantized_model)
    quantized_initializers = {
        item.name: numpy_helper.to_array(item) for item in quantized_model.graph.initializer
    }

    pairs: list[dict[str, object]] = []
    for node_name, float_node in float_nodes.items():
        quantized_node = quantized_nodes.get(node_name)
        if quantized_node is None:
            continue
        for float_output, quantized_output in zip(float_node.output, quantized_node.output):
            scale_name = scale_names.get(quantized_output)
            if scale_name is None:
                continue
            scale_array = np.asarray(quantized_initializers[scale_name]).reshape(-1)
            if scale_array.size != 1:
                continue
            pairs.append(
                {
                    "node_name": node_name,
                    "op_type": float_node.op_type,
                    "float_tensor": float_output,
                    "quantized_tensor": quantized_output,
                    "scale_name": scale_name,
                    "scale": float(scale_array[0]),
                }
            )

    float_info = value_info_map(float_model)
    pairs = [
        pair
        for pair in pairs
        if pair["float_tensor"] in float_info
        and float_info[str(pair["float_tensor"])].type.tensor_type.elem_type == TensorProto.FLOAT
    ]
    float_names = [str(pair["float_tensor"]) for pair in pairs]
    quantized_names = [str(pair["quantized_tensor"]) for pair in pairs]
    quantized_info: dict[str, onnx.ValueInfoProto] = {}
    for pair in pairs:
        value_info = copy.deepcopy(float_info[str(pair["float_tensor"])])
        value_info.name = str(pair["quantized_tensor"])
        value_info.type.tensor_type.elem_type = TensorProto.INT8
        quantized_info[value_info.name] = value_info

    float_augmented = args.float_model.with_name(args.float_model.stem + "-diagnostic.onnx")
    quantized_augmented = args.quantized_model.with_name(
        args.quantized_model.stem + "-diagnostic.onnx"
    )
    onnx.save(add_outputs(float_model, float_names, float_info), float_augmented)
    onnx.save(
        add_outputs(quantized_model, quantized_names, quantized_info), quantized_augmented
    )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    float_session = ort.InferenceSession(
        str(float_augmented), sess_options=options, providers=["CPUExecutionProvider"]
    )
    quantized_session = ort.InferenceSession(
        str(quantized_augmented), sess_options=options, providers=["CPUExecutionProvider"]
    )
    tensor_stats = [new_stats() for _ in pairs]
    for image_index, path in enumerate(image_paths):
        feed = {"images": preprocess(path)}
        float_outputs = float_session.run(float_names, feed)
        quantized_outputs = quantized_session.run(quantized_names, feed)
        for pair, stats, float_values, quantized_values in zip(
            pairs, tensor_stats, float_outputs, quantized_outputs
        ):
            update_stats(stats, float_values, quantized_values, float(pair["scale"]))
        if (image_index + 1) % 32 == 0:
            print(f"diagnosed {image_index + 1}/{len(image_paths)} images", flush=True)

    records = []
    aggregate = new_stats()
    for pair, stats in zip(pairs, tensor_stats):
        records.append({**pair, **finish_stats(stats)})
        for key, value in stats.items():
            if key.endswith("max_absolute_error"):
                aggregate[key] = max(float(aggregate[key]), float(value))
            else:
                aggregate[key] += value

    weight_records = []
    weight_aggregate = new_stats()
    for node_name, float_node in float_nodes.items():
        if float_node.op_type != "Conv":
            continue
        quantized_node = quantized_nodes[node_name]
        float_weight = np.asarray(float_initializers[float_node.input[1]], dtype=np.float32)
        quantized_weight = np.asarray(
            quantized_initializers[quantized_node.input[3]], dtype=np.int8
        )
        weight_scales = np.asarray(
            quantized_initializers[quantized_node.input[4]], dtype=np.float32
        ).reshape(-1)
        stats = new_stats()
        for channel, scale in enumerate(weight_scales):
            update_stats(
                stats,
                float_weight[channel],
                quantized_weight[channel],
                float(scale),
            )
        weight_records.append(
            {
                "node_name": node_name,
                "float_tensor": float_node.input[1],
                "quantized_tensor": quantized_node.input[3],
                "scale_tensor": quantized_node.input[4],
                "output_channels": len(weight_scales),
                **finish_stats(stats),
            }
        )
        for key, value in stats.items():
            if key.endswith("max_absolute_error"):
                weight_aggregate[key] = max(float(weight_aggregate[key]), float(value))
            else:
                weight_aggregate[key] += value

    report = {
        "schema_version": 1,
        "status": "complete",
        "scope": "all quantized accelerator-region node outputs and all convolution weights in the ONNX Runtime QOperator proxy",
        "images": len(image_paths),
        "ordered_prefix": True,
        "float_model_sha256": sha256_file(args.float_model),
        "quantized_model_sha256": sha256_file(args.quantized_model),
        "tensor_count": len(records),
        "definitions": {
            "local_error": "FLOAT tensor versus independently RNE-quantized and dequantized FLOAT tensor at the recorded scale",
            "cumulative_error": "FLOAT tensor versus the corresponding dequantized tensor produced after all preceding QOperator quantization",
            "clipped": "pre-clamp RNE integer is outside [-128,127]",
            "saturated": "executed QOperator tensor equals -128 or 127",
        },
        "activations": {
            "aggregate": finish_stats(aggregate),
            "tensors": records,
        },
        "weights": {
            "aggregate": finish_stats(weight_aggregate),
            "tensors": weight_records,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["activations"]["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
