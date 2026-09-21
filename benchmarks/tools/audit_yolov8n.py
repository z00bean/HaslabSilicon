#!/usr/bin/env python3
"""Audit the pinned YOLOv8n ONNX graph against the HASLAB v0 contract.

The script deliberately operates on a locally supplied model. It never downloads
artifacts. The generated JSON is an inventory and feasibility check, not a
compiler output or proof of quantized accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper, numpy_helper


BOUNDARY_TENSORS = (
    "/model.22/Concat_output_0",
    "/model.22/Concat_1_output_0",
    "/model.22/Concat_2_output_0",
)

CAPACITY = {
    "input": 16_384,
    "weight": 16_384,
    "acc": 8_192,
    "output": 8_192,
    "param": 3_072,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shape_of(value: onnx.ValueInfoProto) -> list[int | str | None]:
    dims: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def json_attribute(attribute: onnx.AttributeProto) -> Any:
    value = helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, TensorProto):
        return tensor_record(value, include_values=True)
    return value


def tensor_record(tensor: TensorProto, *, include_values: bool) -> dict[str, Any]:
    array = numpy_helper.to_array(tensor)
    data = array.tobytes(order="C")
    record: dict[str, Any] = {
        "name": tensor.name,
        "dtype": TensorProto.DataType.Name(tensor.data_type),
        "shape": list(array.shape),
        "byte_length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if include_values and array.size <= 32:
        record["values"] = array.reshape(-1).tolist()
    return record


def round_up(value: int, quantum: int) -> int:
    return quantum * math.ceil(value / quantum)


def nchw_int8_bytes(shape: list[int | str | None]) -> int | None:
    if len(shape) != 4 or not all(isinstance(item, int) for item in shape):
        return None
    n, channels, height, width = shape
    assert isinstance(n, int) and isinstance(channels, int)
    assert isinstance(height, int) and isinstance(width, int)
    return n * height * width * round_up(channels, 8)


def attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {attribute.name: json_attribute(attribute) for attribute in node.attribute}


def exact_silu_pairs(nodes: list[onnx.NodeProto]) -> dict[int, int]:
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers[name].append(index)

    pairs: dict[int, int] = {}
    for sigmoid_index, node in enumerate(nodes):
        if node.op_type != "Sigmoid" or len(node.input) != 1 or len(node.output) != 1:
            continue
        uses = consumers[node.output[0]]
        if len(uses) != 1:
            continue
        mul_index = uses[0]
        mul = nodes[mul_index]
        if mul.op_type == "Mul" and sorted(mul.input) == sorted((node.input[0], node.output[0])):
            pairs[sigmoid_index] = mul_index
            pairs[mul_index] = sigmoid_index
    return pairs


def classify_node(
    index: int,
    node: onnx.NodeProto,
    host_tail_start: int,
    silu_pairs: dict[int, int],
    shapes: dict[str, list[int | str | None]],
) -> tuple[str, str, list[str]]:
    if index >= host_tail_start:
        return "host_tail", "terminal DFL, box decode, and class probability graph", []

    attrs = attributes(node)
    problems: list[str] = []
    if node.op_type == "Conv":
        allowed = (
            attrs.get("group", 1) == 1
            and attrs.get("dilations", [1, 1]) == [1, 1]
            and attrs.get("kernel_shape") in ([1, 1], [3, 3])
            and attrs.get("strides") in ([1, 1], [2, 2])
            and attrs.get("pads")
            in ([0, 0, 0, 0], [1, 1, 1, 1])
        )
        if allowed:
            return "native", "tiled INT8 convolution with INT32 accumulation", []
        problems.append("convolution attributes are outside the v0 profile")
    elif node.op_type in {"Sigmoid", "Mul"} and index in silu_pairs:
        return "compiler_fused", "exact x * sigmoid(x) pattern lowered to the SiLU epilogue", []
    elif node.op_type == "Add":
        input_shapes = [shapes.get(name) for name in node.input]
        if len(input_shapes) == 2 and input_shapes[0] == input_shapes[1]:
            return "native", "same-shape residual add", []
        problems.append("Add requires broadcasting or mismatched shapes")
    elif node.op_type == "MaxPool":
        expected = {
            "ceil_mode": 0,
            "dilations": [1, 1],
            "kernel_shape": [5, 5],
            "pads": [2, 2, 2, 2],
            "strides": [1, 1],
        }
        if all(attrs.get(key) == value for key, value in expected.items()):
            return "native", "5x5 stride-one SPPF max pool", []
        problems.append("MaxPool attributes are outside the v0 profile")
    elif node.op_type == "Resize":
        expected = {
            "coordinate_transformation_mode": "asymmetric",
            "mode": "nearest",
            "nearest_mode": "floor",
        }
        output_shape = shapes.get(node.output[0])
        input_shape = shapes.get(node.input[0])
        spatial_2x = (
            input_shape is not None
            and output_shape is not None
            and len(input_shape) == len(output_shape) == 4
            and output_shape[2] == 2 * input_shape[2]
            and output_shape[3] == 2 * input_shape[3]
        )
        if spatial_2x and all(attrs.get(key) == value for key, value in expected.items()):
            return "native", "fixed nearest-neighbor 2x spatial upsample", []
        problems.append("Resize is not the fixed asymmetric/floor 2x v0 form")
    elif node.op_type == "Split":
        output_channels = [shapes.get(name, [None, None])[1] for name in node.output]
        if attrs.get("axis", 0) == 1 and all(
            isinstance(channels, int) and channels % 8 == 0 for channels in output_channels
        ):
            return "compiler_view", "static channel split with eight-aligned boundaries", []
        problems.append("Split has a non-channel or non-eight-aligned boundary")
    elif node.op_type == "Concat":
        input_channels = [shapes.get(name, [None, None])[1] for name in node.input]
        if attrs.get("axis", 0) == 1 and all(
            isinstance(channels, int) and channels % 8 == 0 for channels in input_channels
        ):
            return "native", "channel concat materialized by copy/rescale schedule", []
        problems.append("Concat has a non-channel or non-eight-aligned boundary")
    elif node.op_type == "Constant":
        return "compiler_folded", "static shape/scale constant", []
    else:
        problems.append("operator is not supported before the declared host tail")

    return "unsupported", "outside HASLAB v0", problems


def convolution_audit(
    index: int,
    node: onnx.NodeProto,
    shapes: dict[str, list[int | str | None]],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    weight = numpy_helper.to_array(initializers[node.input[1]])
    cout, cin, kernel_h, kernel_w = weight.shape
    output_shape = shapes[node.output[0]]
    attrs = attributes(node)
    out_h, out_w = int(output_shape[2]), int(output_shape[3])
    tile_h, tile_w = min(8, out_h), min(8, out_w)
    stride_h, stride_w = attrs["strides"]
    physical_chunk = min(32, round_up(int(cin), 8))
    input_h = (tile_h - 1) * stride_h + int(kernel_h)
    input_w = (tile_w - 1) * stride_w + int(kernel_w)
    byte_use = {
        "input": input_h * input_w * physical_chunk,
        "weight": int(kernel_h) * int(kernel_w) * physical_chunk * 8,
        "acc": tile_h * tile_w * 8 * 4,
        "output": tile_h * tile_w * 8,
        "param": 8 * 16 + 1_024,
    }
    output_tiles = math.ceil(out_h / 8) * math.ceil(out_w / 8)
    output_groups = math.ceil(int(cout) / 8)
    input_chunks = math.ceil(int(cin) / 32)
    return {
        "node_index": index,
        "node_name": node.name,
        "input_shape_nchw": shapes[node.input[0]],
        "weight_shape_oihw": list(weight.shape),
        "output_shape_nchw": output_shape,
        "spatial_tile": [tile_h, tile_w],
        "input_channel_chunk": min(int(cin), 32),
        "physical_input_channel_chunk": physical_chunk,
        "local_bytes": byte_use,
        "fits": {name: byte_use[name] <= CAPACITY[name] for name in byte_use},
        "reference_schedule": {
            "output_tiles": output_tiles,
            "output_channel_groups": output_groups,
            "input_channel_chunks": input_chunks,
            "conv_commands": output_tiles * output_groups * input_chunks,
            "epilogue_commands": output_tiles * output_groups,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    model_bytes_hash = sha256_file(args.model)
    if args.expected_sha256 and model_bytes_hash != args.expected_sha256:
        raise SystemExit(
            f"model hash mismatch: expected {args.expected_sha256}, got {model_bytes_hash}"
        )

    model = onnx.load(args.model)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {
        value.name: shape_of(value)
        for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }
    producers = {
        output: index
        for index, node in enumerate(inferred.graph.node)
        for output in node.output
    }
    missing_boundary = [name for name in BOUNDARY_TENSORS if name not in producers]
    if missing_boundary:
        raise SystemExit(f"missing pinned boundary tensors: {missing_boundary}")
    host_tail_start = max(producers[name] for name in BOUNDARY_TENSORS) + 1
    nodes = list(inferred.graph.node)
    silu_pairs = exact_silu_pairs(nodes[:host_tail_start])
    initializers = {item.name: item for item in inferred.graph.initializer}

    node_records: list[dict[str, Any]] = []
    classifications: Counter[str] = Counter()
    unsupported: list[dict[str, Any]] = []
    convolution_records: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        classification, lowering, problems = classify_node(
            index, node, host_tail_start, silu_pairs, shapes
        )
        classifications[classification] += 1
        record = {
            "index": index,
            "name": node.name,
            "domain": node.domain,
            "op_type": node.op_type,
            "inputs": [
                {"name": name, "shape": shapes.get(name)} for name in node.input
            ],
            "outputs": [
                {"name": name, "shape": shapes.get(name)} for name in node.output
            ],
            "attributes": attributes(node),
            "classification": classification,
            "lowering": lowering,
        }
        if problems:
            record["problems"] = problems
            unsupported.append(record)
        node_records.append(record)
        if index < host_tail_start and node.op_type == "Conv":
            convolution_records.append(
                convolution_audit(index, node, shapes, initializers)
            )

    constant_records: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if node.op_type == "Constant":
            tensor = next(
                helper.get_attribute_value(attr)
                for attr in node.attribute
                if attr.name == "value"
            )
            record = tensor_record(tensor, include_values=True)
            record.update({"node_index": index, "node_name": node.name})
            constant_records.append(record)

    conv_commands = sum(
        item["reference_schedule"]["conv_commands"] for item in convolution_records
    )
    epilogue_commands = sum(
        item["reference_schedule"]["epilogue_commands"] for item in convolution_records
    )
    worst_local = {
        space: max(item["local_bytes"][space] for item in convolution_records)
        for space in CAPACITY
    }

    report = {
        "schema_version": 1,
        "audit_scope": "FLOAT export structure and v0 static feasibility; not an INT8 accuracy result",
        "model": {
            "path": args.model.name,
            "byte_length": args.model.stat().st_size,
            "sha256": model_bytes_hash,
            "ir_version": model.ir_version,
            "opsets": [
                {"domain": item.domain, "version": item.version}
                for item in model.opset_import
            ],
            "inputs": [
                {
                    "name": value.name,
                    "dtype": TensorProto.DataType.Name(value.type.tensor_type.elem_type),
                    "shape": shape_of(value),
                }
                for value in model.graph.input
            ],
            "outputs": [
                {
                    "name": value.name,
                    "dtype": TensorProto.DataType.Name(value.type.tensor_type.elem_type),
                    "shape": shape_of(value),
                }
                for value in model.graph.output
            ],
        },
        "partition": {
            "accelerator_node_range": [0, host_tail_start - 1],
            "host_tail_node_range": [host_tail_start, len(nodes) - 1],
            "boundary_tensors": [
                {
                    "name": name,
                    "shape_nchw": shapes[name],
                    "int8_hwc8_bytes": nchw_int8_bytes(shapes[name]),
                    "int32_hwc8_bytes": 4 * nchw_int8_bytes(shapes[name]),
                }
                for name in BOUNDARY_TENSORS
            ],
        },
        "inventory": {
            "node_count": len(nodes),
            "initializer_count": len(initializers),
            "operator_counts": dict(sorted(Counter(node.op_type for node in nodes).items())),
            "classification_counts": dict(sorted(classifications.items())),
            "nodes": node_records,
            "initializers": [
                tensor_record(item, include_values=False)
                for item in inferred.graph.initializer
            ],
            "constants": constant_records,
        },
        "local_memory": {
            "capacities": CAPACITY,
            "policy": "8x8 spatial output tile, eight output lanes, at most 32 input channels per reduction chunk",
            "worst_case_bytes": worst_local,
            "all_convolution_tiles_fit": all(
                all(item["fits"].values()) for item in convolution_records
            ),
            "convolutions": convolution_records,
        },
        "command_pressure": {
            "scope": "subtotal for the documented 8x8 spatial and at-most-32-input-channel reference tiling; larger legal tiles may reduce it, while DMA/fill/utility/end commands increase total traffic",
            "conv_commands": conv_commands,
            "epilogue_commands": epilogue_commands,
            "reference_tiling_subtotal": conv_commands + epilogue_commands,
        },
        "result": {
            "unsupported_accelerator_nodes": [
                {"index": item["index"], "name": item["name"], "problems": item["problems"]}
                for item in unsupported
            ],
            "passes_structural_v0_audit": not unsupported
            and all(all(item["fits"].values()) for item in convolution_records),
        },
    }

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if report["result"]["passes_structural_v0_audit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
