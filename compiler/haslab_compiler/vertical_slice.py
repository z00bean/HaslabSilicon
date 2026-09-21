"""Lower the pinned first Conv-SiLU tile to the experimental M6 package schema."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from haslab_ref import nchw_to_hwc8, oihw_to_khwci8, quantize_int8
from haslab_sim import ABI_MAJOR, ABI_MINOR, Command, MemorySpace, Opcode

from .hxb import Section, SectionType, build_hxb

SCHEMA = "haslab.vertical-slice.v1"
MODEL_SHA256 = "5b232bf21720cac4264463896ace02b919d8bec3f3fd1e4b9d36695493fa1718"
FIRST_FLAG = 1
LAST_FLAG = 2
TILE_H = 8
TILE_W = 8
TILE_Y = 1
TILE_X = 1
KERNEL = 3
STRIDE = 2
INPUT_CHANNELS = 3
OUTPUT_LANES = 8


class CompileError(ValueError):
    """A source graph or calibration record is outside the vertical-slice profile."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _align(value: int, alignment: int = 8) -> int:
    return (value + alignment - 1) // alignment * alignment


def _bias_i32(bias: np.ndarray, input_scale: float, weight_scales: np.ndarray) -> np.ndarray:
    denominator = np.float32(input_scale).astype(np.float64) * weight_scales.astype(np.float64)
    rounded = np.rint(bias.astype(np.float64) / denominator)
    if np.any(rounded < -(1 << 31)) or np.any(rounded > (1 << 31) - 1):
        raise CompileError("quantized bias exceeds signed INT32")
    return rounded.astype(np.int32)


def _parameter_records(
    bias: np.ndarray, multipliers: np.ndarray, shifts: np.ndarray
) -> bytes:
    return b"".join(
        struct.pack(
            "<iIII",
            int(bias[lane]),
            int(multipliers[lane]),
            int(shifts[lane]),
            0,
        )
        for lane in range(OUTPUT_LANES)
    )


def extract_input_patch(full_input_i8: object) -> np.ndarray:
    """Extract the interior 17x17 NCHW halo used by the fixed 8x8 output tile."""

    source = np.asarray(full_input_i8)
    if source.shape != (1, INPUT_CHANNELS, 320, 320) or source.dtype != np.int8:
        raise CompileError("vertical-slice input must be INT8 NCHW [1,3,320,320]")
    start_y = TILE_Y * STRIDE - 1
    start_x = TILE_X * STRIDE - 1
    patch_h = (TILE_H - 1) * STRIDE + KERNEL
    patch_w = (TILE_W - 1) * STRIDE + KERNEL
    return np.ascontiguousarray(
        source[:, :, start_y : start_y + patch_h, start_x : start_x + patch_w]
    )


def compile_conv_silu_slice(
    *,
    weights: object,
    bias: object,
    input_scale: float,
    weight_scales: object,
    output_scale: float,
    multipliers: object,
    shifts: object,
    lut: object,
    source_model_sha256: str,
    node_names: tuple[str, str, str] = (
        "/model.0/conv/Conv",
        "/model.0/act/Sigmoid",
        "/model.0/act/Mul",
    ),
) -> bytes:
    """Compile output channels 0..7 of one 8x8 first-layer Conv-SiLU tile."""

    weight_values = np.asarray(weights, dtype=np.float32)
    bias_values = np.asarray(bias, dtype=np.float32)
    scales = np.asarray(weight_scales, dtype=np.float32)
    multiplier_values = np.asarray(multipliers, dtype=np.int64)
    shift_values = np.asarray(shifts, dtype=np.int64)
    table = np.asarray(lut, dtype=np.int8).reshape(-1)
    if weight_values.shape != (16, 3, 3, 3):
        raise CompileError("first Conv weights must have shape [16,3,3,3]")
    if bias_values.shape != (16,) or scales.shape != (16,):
        raise CompileError("first Conv bias and weight scales must have 16 channels")
    if multiplier_values.shape != (16,) or shift_values.shape != (16,):
        raise CompileError("first SiLU coefficients must have 16 channels")
    if table.size != 1024:
        raise CompileError("first SiLU LUT must contain 1024 bytes")
    if not all(np.isfinite(x) and x > 0 for x in (input_scale, output_scale)):
        raise CompileError("activation scales must be finite and positive")
    if np.any(~np.isfinite(weight_values)) or np.any(~np.isfinite(bias_values)):
        raise CompileError("weights and bias must be finite")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise CompileError("weight scales must be finite and positive")
    if np.any(multiplier_values < 0) or np.any(multiplier_values > (1 << 31) - 1):
        raise CompileError("SiLU multiplier is outside the v0 range")
    if np.any(shift_values < 0) or np.any(shift_values > 62):
        raise CompileError("SiLU shift is outside the v0 range")
    if len(node_names) != 3 or not all(node_names):
        raise CompileError("three nonempty source node names are required")

    selected_weights = weight_values[:OUTPUT_LANES]
    selected_scales = scales[:OUTPUT_LANES]
    quantized_weights = quantize_int8(
        selected_weights, selected_scales.reshape(OUTPUT_LANES, 1, 1, 1)
    )
    packed_weights = np.ascontiguousarray(oihw_to_khwci8(quantized_weights))
    quantized_bias = _bias_i32(
        bias_values[:OUTPUT_LANES], float(input_scale), selected_scales
    )
    params = _parameter_records(
        quantized_bias, multiplier_values[:OUTPUT_LANES], shift_values[:OUTPUT_LANES]
    )
    weight_bytes = packed_weights.tobytes()
    lut_bytes = table.tobytes()
    parameter_offset = _align(len(weight_bytes))
    lut_offset = _align(parameter_offset + len(params))
    constants = bytearray(lut_offset + len(lut_bytes))
    constants[: len(weight_bytes)] = weight_bytes
    constants[parameter_offset : parameter_offset + len(params)] = params
    constants[lut_offset : lut_offset + len(lut_bytes)] = lut_bytes

    patch_bytes = 17 * 17 * 8
    output_bytes = TILE_H * TILE_W * 8
    commands = [
        Command.build(Opcode.DMA_COPY2D, 1, [MemorySpace.EXT, 0, MemorySpace.INPUT, 0, patch_bytes, 1, patch_bytes, patch_bytes, 0]),
        Command.build(Opcode.DMA_COPY2D, 2, [MemorySpace.EXT, 0, MemorySpace.WEIGHT, 0, len(weight_bytes), 1, len(weight_bytes), len(weight_bytes), 0]),
        Command.build(Opcode.DMA_COPY2D, 3, [MemorySpace.EXT, 0, MemorySpace.PARAM, 0, len(params), 1, len(params), len(params), 0]),
        Command.build(Opcode.DMA_COPY2D, 4, [MemorySpace.EXT, 0, MemorySpace.PARAM, 128, len(lut_bytes), 1, len(lut_bytes), len(lut_bytes), 0]),
        Command.build(Opcode.CONV_I8, 5, [0, 0, 0, TILE_H, TILE_W, OUTPUT_LANES, INPUT_CHANNELS, 0, INPUT_CHANNELS, KERNEL, STRIDE], flags=FIRST_FLAG | LAST_FLAG),
        Command.build(Opcode.EPILOGUE, 6, [0, 0, 0, 2, 128]),
        Command.build(Opcode.DMA_COPY2D, 7, [MemorySpace.OUTPUT, 0, MemorySpace.EXT, 0, output_bytes, 1, output_bytes, output_bytes, 0]),
        Command.build(Opcode.END, 8),
    ]
    command_bytes = b"".join(command.to_bytes() for command in commands)

    manifest = {
        "abi": {"major": ABI_MAJOR, "minor": ABI_MINOR},
        "buffers": {
            "constants": {"alignment": 64, "bytes": len(constants), "role": "constant"},
            "input": {"alignment": 64, "bytes": patch_bytes, "role": "input"},
            "output": {"alignment": 64, "bytes": output_bytes, "role": "output"},
        },
        "commands": {"bytes": len(command_bytes), "count": len(commands), "record_bytes": 128},
        "constants": [
            {"addend": 0, "bytes": len(weight_bytes), "kind": "KHWCI8_INT8", "name": "conv.weight"},
            {"addend": parameter_offset, "bytes": len(params), "kind": "EPILOGUE_RECORDS", "name": "conv.epilogue"},
            {"addend": lut_offset, "bytes": len(lut_bytes), "kind": "SILU_LUT_INT8", "name": "silu.lut"},
        ],
        "input": {"dtype": "int8", "layout": "HWC8", "logical_shape": [1, 3, 17, 17], "physical_shape": [17, 17, 1, 8], "scale_binary32": float(np.float32(input_scale)), "tile_origin_yx": [TILE_Y, TILE_X]},
        "output": {"dtype": "int8", "layout": "HWC8", "logical_shape": [1, 8, 8, 8], "physical_shape": [8, 8, 1, 8], "scale_binary32": float(np.float32(output_scale))},
        "profile": "M6_FIRST_CONV_SILU_TILE",
        "relocations": [
            {"addend": 0, "buffer": "input", "command_index": 0, "payload_word": 1},
            {"addend": 0, "buffer": "constants", "command_index": 1, "payload_word": 1},
            {"addend": parameter_offset, "buffer": "constants", "command_index": 2, "payload_word": 1},
            {"addend": lut_offset, "buffer": "constants", "command_index": 3, "payload_word": 1},
            {"addend": 0, "buffer": "output", "command_index": 6, "payload_word": 3},
        ],
        "schema": SCHEMA,
        "source": {"model_sha256": source_model_sha256, "nodes": list(node_names)},
    }
    debug = {
        "command_to_node": {
            "1": "runtime input upload",
            "2": node_names[0],
            "3": node_names[0],
            "4": f"{node_names[1]} + {node_names[2]}",
            "5": node_names[0],
            "6": f"{node_names[1]} + {node_names[2]}",
            "7": "runtime output download",
            "8": "frame end",
        },
        "derived": {
            "bias_i32": quantized_bias.astype(int).tolist(),
            "constant_data_sha256": _sha256(bytes(constants)),
            "weight_i8_sha256": _sha256(weight_bytes),
        },
        "schema": SCHEMA,
    }
    return build_hxb(
        [
            Section(SectionType.MANIFEST_UTF8_JSON, _canonical_json(manifest)),
            Section(SectionType.COMMANDS, command_bytes),
            Section(SectionType.CONSTANT_DATA, bytes(constants)),
            Section(SectionType.DEBUG_UTF8_JSON, _canonical_json(debug)),
        ]
    )


def compile_pinned_first_block(
    model_path: str | Path,
    calibration_path: str | Path,
    lut_path: str | Path,
) -> bytes:
    """Validate and compile the first block of the pinned YOLOv8n ONNX graph."""

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised by integration environment
        raise CompileError("ONNX is required to import the pinned graph") from exc

    model_bytes = Path(model_path).read_bytes()
    model_hash = _sha256(model_bytes)
    if model_hash != MODEL_SHA256:
        raise CompileError(f"pinned model hash mismatch: {model_hash}")
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if calibration.get("source_model", {}).get("sha256") != model_hash:
        raise CompileError("calibration source hash does not match the ONNX model")
    model = onnx.load_from_string(model_bytes)
    if model.ir_version != 7 or [(item.domain, item.version) for item in model.opset_import] != [("", 13)]:
        raise CompileError("pinned graph must use ONNX IR 7 and default-domain opset 13")
    if len(model.graph.node) < 3:
        raise CompileError("graph does not contain the required first Conv-Sigmoid-Mul block")
    conv, sigmoid, mul = model.graph.node[:3]
    if [conv.op_type, sigmoid.op_type, mul.op_type] != ["Conv", "Sigmoid", "Mul"]:
        raise CompileError("nodes 0..2 must be Conv, Sigmoid, and Mul")
    if list(sigmoid.input) != [conv.output[0]] or sorted(mul.input) != sorted([conv.output[0], sigmoid.output[0]]):
        raise CompileError("nodes 0..2 do not form an exact SiLU dataflow")
    attrs = {item.name: onnx.helper.get_attribute_value(item) for item in conv.attribute}
    expected = {"dilations": [1, 1], "group": 1, "kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2]}
    if attrs != expected:
        raise CompileError(f"unsupported first Conv attributes: {attrs!r}")
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    try:
        weights, bias = (initializers[name] for name in conv.input[1:3])
    except (KeyError, ValueError) as exc:
        raise CompileError("first Conv requires constant weight and bias") from exc
    scales_by_name = {item["name"]: item["values"] for item in calibration["scales"]}
    try:
        input_scale = scales_by_name["images_scale"][0]
        weight_scales = scales_by_name["model.0.conv.weight_scale"]
        record = calibration["silu"]["records"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise CompileError("calibration lacks the first-block scales or SiLU record") from exc
    if record.get("conv_node_index") != 0 or record.get("conv_node_name") != conv.name:
        raise CompileError("first SiLU calibration record does not match graph node 0")
    lut_blob = Path(lut_path).read_bytes()
    if _sha256(lut_blob) != calibration["silu"]["binary_sha256"]:
        raise CompileError("SiLU LUT binary hash mismatch")
    offset, length = int(record["table_offset"]), int(record["table_bytes"])
    lut = np.frombuffer(lut_blob[offset : offset + length], dtype=np.int8).copy()
    if _sha256(lut.tobytes()) != record["table_sha256"]:
        raise CompileError("first SiLU LUT table hash mismatch")
    return compile_conv_silu_slice(
        weights=weights,
        bias=bias,
        input_scale=input_scale,
        weight_scales=weight_scales,
        output_scale=record["output_scale_binary32"],
        multipliers=record["accumulator_to_grid"]["multipliers"],
        shifts=record["accumulator_to_grid"]["shifts"],
        lut=lut,
        source_model_sha256=model_hash,
        node_names=(conv.name, sigmoid.name, mul.name),
    )
