"""Reusable sequential Conv-SiLU scheduling for the HASLAB v0 M6 profile."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from haslab_ref import oihw_to_khwci8, quantize_int8
from haslab_sim import ABI_MAJOR, ABI_MINOR, Command, MemorySpace, Opcode

from .hxb import Section, SectionType, build_hxb
from .vertical_slice import (
    FIRST_FLAG,
    LAST_FLAG,
    MODEL_SHA256,
    CompileError,
    _align,
    _bias_i32,
    _canonical_json,
    _parameter_records,
    _sha256,
)

SCHEMA = "haslab.conv-silu-pipeline.v1"
TILE_H = 8
TILE_W = 8
KERNEL = 3
STRIDE = 2
PADDING = 1
LANES = 8
PATCH_H = (TILE_H - 1) * STRIDE + KERNEL
PATCH_W = (TILE_W - 1) * STRIDE + KERNEL
CHUNK_BYTES = PATCH_H * PATCH_W * LANES
WEIGHT_CHUNK_BYTES = KERNEL * KERNEL * LANES * LANES


@dataclass(frozen=True)
class ConvSiluLayerSpec:
    """Numerical inputs and graph identity for one fused 3x3s2 Conv-SiLU."""

    name: str
    input_h: int
    input_w: int
    input_channels: int
    weights: object
    bias: object
    input_scale: float
    weight_scales: object
    output_scale: float
    multipliers: object
    shifts: object
    lut: object
    node_names: tuple[str, str, str]


@dataclass(frozen=True)
class _PreparedLayer:
    spec: ConvSiluLayerSpec
    weights_i8: np.ndarray
    bias_i32: np.ndarray
    weight_scales: np.ndarray
    multipliers: np.ndarray
    shifts: np.ndarray
    lut: np.ndarray
    output_h: int
    output_w: int
    output_channels: int
    input_groups: int
    input_chunks: int
    output_groups: int


def _prepare(spec: ConvSiluLayerSpec) -> _PreparedLayer:
    weights = np.asarray(spec.weights, dtype=np.float32)
    bias = np.asarray(spec.bias, dtype=np.float32)
    scales = np.asarray(spec.weight_scales, dtype=np.float32)
    multipliers = np.asarray(spec.multipliers, dtype=np.int64)
    shifts = np.asarray(spec.shifts, dtype=np.int64)
    lut = np.asarray(spec.lut, dtype=np.int8).reshape(-1)
    if not spec.name or len(spec.node_names) != 3 or not all(spec.node_names):
        raise CompileError("each pipeline layer requires a name and three source nodes")
    if spec.input_h <= 0 or spec.input_w <= 0 or spec.input_channels <= 0:
        raise CompileError("pipeline input dimensions must be positive")
    if weights.ndim != 4 or weights.shape[1:] != (spec.input_channels, 3, 3):
        raise CompileError(f"{spec.name} weights do not match its input channels and 3x3 kernel")
    output_channels = int(weights.shape[0])
    if output_channels % LANES:
        raise CompileError(f"{spec.name} output channels must be a multiple of eight")
    if bias.shape != (output_channels,) or scales.shape != (output_channels,):
        raise CompileError(f"{spec.name} bias and weight scales must match output channels")
    if multipliers.shape != (output_channels,) or shifts.shape != (output_channels,):
        raise CompileError(f"{spec.name} SiLU coefficients must match output channels")
    if lut.size != 1024:
        raise CompileError(f"{spec.name} SiLU LUT must contain 1024 bytes")
    if not all(np.isfinite(value) and value > 0 for value in (spec.input_scale, spec.output_scale)):
        raise CompileError(f"{spec.name} activation scales must be finite and positive")
    if np.any(~np.isfinite(weights)) or np.any(~np.isfinite(bias)):
        raise CompileError(f"{spec.name} weights and bias must be finite")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise CompileError(f"{spec.name} weight scales must be finite and positive")
    if np.any(multipliers < 0) or np.any(multipliers > (1 << 31) - 1):
        raise CompileError(f"{spec.name} SiLU multiplier is outside the v0 range")
    if np.any(shifts < 0) or np.any(shifts > 62):
        raise CompileError(f"{spec.name} SiLU shift is outside the v0 range")
    output_h = (spec.input_h + 2 * PADDING - KERNEL) // STRIDE + 1
    output_w = (spec.input_w + 2 * PADDING - KERNEL) // STRIDE + 1
    if output_h % TILE_H or output_w % TILE_W:
        raise CompileError(f"{spec.name} output must divide exactly into 8x8 tiles")
    input_groups = (spec.input_channels + LANES - 1) // LANES
    output_groups = output_channels // LANES
    if input_groups * CHUNK_BYTES > 16_384:
        raise CompileError(f"{spec.name} input chunks exceed v0 input SRAM")
    if input_groups * output_groups * WEIGHT_CHUNK_BYTES > 16_384:
        raise CompileError(f"{spec.name} resident weights exceed v0 weight SRAM")
    if output_groups * 128 + 1024 > 3_072:
        raise CompileError(f"{spec.name} parameters and LUT exceed v0 parameter SRAM")
    weights_i8 = quantize_int8(weights, scales.reshape(output_channels, 1, 1, 1))
    return _PreparedLayer(
        spec=spec,
        weights_i8=weights_i8,
        bias_i32=_bias_i32(bias, float(spec.input_scale), scales),
        weight_scales=scales,
        multipliers=multipliers,
        shifts=shifts,
        lut=lut,
        output_h=output_h,
        output_w=output_w,
        output_channels=output_channels,
        input_groups=input_groups,
        input_chunks=input_groups,
        output_groups=output_groups,
    )


def compile_conv_silu_pipeline(
    *, layers: Sequence[ConvSiluLayerSpec], source_model_sha256: str
) -> bytes:
    """Compile a connected sequence of locally resident 3x3s2 Conv-SiLU layers."""

    prepared = [_prepare(layer) for layer in layers]
    if not prepared:
        raise CompileError("a pipeline must contain at least one layer")
    for previous, current in zip(prepared, prepared[1:]):
        if (
            current.spec.input_h != previous.output_h
            or current.spec.input_w != previous.output_w
            or current.spec.input_channels != previous.output_channels
        ):
            raise CompileError(f"{current.spec.name} is not shape-connected to {previous.spec.name}")
        if np.float32(current.spec.input_scale).tobytes() != np.float32(
            previous.spec.output_scale
        ).tobytes():
            raise CompileError(f"{current.spec.name} input scale does not match prior output")

    constant_blob = bytearray()
    constant_records: list[dict[str, object]] = []
    constant_offsets: dict[str, int] = {}
    layer_constants: list[dict[str, bytes]] = []
    for layer in prepared:
        packed = np.ascontiguousarray(oihw_to_khwci8(layer.weights_i8))
        weight_parts = []
        for output_group in range(layer.output_groups):
            for input_chunk in range(layer.input_chunks):
                start = input_chunk * LANES
                weight_parts.append(
                    np.ascontiguousarray(
                        packed[:, :, start : start + LANES, output_group, :]
                    ).tobytes()
                )
        parts = {
            "weights": b"".join(weight_parts),
            "parameters": b"".join(
                _parameter_records(
                    layer.bias_i32[group * LANES : (group + 1) * LANES],
                    layer.multipliers[group * LANES : (group + 1) * LANES],
                    layer.shifts[group * LANES : (group + 1) * LANES],
                )
                for group in range(layer.output_groups)
            ),
            "lut": layer.lut.tobytes(),
        }
        layer_constants.append(parts)
        for suffix, kind in (
            ("weights", "KHWCI8_CHUNKED_INT8"),
            ("parameters", "EPILOGUE_RECORDS"),
            ("lut", "SILU_LUT_INT8"),
        ):
            name = f"{layer.spec.name}.{suffix}"
            raw = parts[suffix]
            offset = _align(len(constant_blob))
            constant_blob.extend(bytes(offset - len(constant_blob)))
            constant_offsets[name] = offset
            constant_blob.extend(raw)
            constant_records.append(
                {"addend": offset, "bytes": len(raw), "kind": kind, "name": name}
            )

    tensor_allocations: list[dict[str, object]] = []
    output_buffer_bytes = 0
    for layer in prepared:
        output_buffer_bytes = _align(output_buffer_bytes, 64)
        tensor_bytes = layer.output_h * layer.output_w * layer.output_groups * LANES
        tensor_allocations.append(
            {
                "addend": output_buffer_bytes,
                "bytes": tensor_bytes,
                "dtype": "int8",
                "layout": "HWC8",
                "logical_shape": [
                    1,
                    layer.output_channels,
                    layer.output_h,
                    layer.output_w,
                ],
                "name": layer.spec.name,
                "physical_shape": [
                    layer.output_h,
                    layer.output_w,
                    layer.output_groups,
                    LANES,
                ],
                "scale_binary32": float(np.float32(layer.spec.output_scale)),
            }
        )
        output_buffer_bytes += tensor_bytes

    commands: list[Command] = []
    relocations: list[dict[str, object]] = []
    cumulative_opcodes = Counter()
    cumulative_dma = Counter()
    layer_schedules: list[dict[str, object]] = []

    def emit(opcode: Opcode, payload: Sequence[int] = (), *, flags: int = 0) -> int:
        index = len(commands)
        commands.append(Command.build(opcode, index + 1, payload, flags=flags))
        cumulative_opcodes[opcode.name] += 1
        return index

    def relocate(index: int, word: int, buffer: str, addend: int) -> None:
        relocations.append(
            {
                "addend": addend,
                "buffer": buffer,
                "command_index": index,
                "payload_word": word,
            }
        )

    for layer_index, layer in enumerate(prepared):
        start_command = len(commands)
        opcodes = Counter()
        dma = Counter()
        boundary_tiles = 0
        source_buffer = "input" if layer_index == 0 else "output"
        source_tensor_addend = 0 if layer_index == 0 else int(
            tensor_allocations[layer_index - 1]["addend"]
        )
        output_tensor_addend = int(tensor_allocations[layer_index]["addend"])

        def layer_emit(opcode: Opcode, payload: Sequence[int] = (), *, flags: int = 0) -> int:
            opcodes[opcode.name] += 1
            return emit(opcode, payload, flags=flags)

        for suffix, destination, destination_offset in (
            ("weights", MemorySpace.WEIGHT, 0),
            ("parameters", MemorySpace.PARAM, 0),
            ("lut", MemorySpace.PARAM, layer.output_groups * 128),
        ):
            name = f"{layer.spec.name}.{suffix}"
            raw = layer_constants[layer_index][suffix]
            command_index = layer_emit(
                Opcode.DMA_COPY2D,
                [
                    MemorySpace.EXT,
                    0,
                    destination,
                    destination_offset,
                    len(raw),
                    1,
                    len(raw),
                    len(raw),
                    0,
                ],
            )
            relocate(command_index, 1, "constants", constant_offsets[name])
            dma["constants"] += len(raw)

        tiles_y = layer.output_h // TILE_H
        tiles_x = layer.output_w // TILE_W
        for tile_y in range(tiles_y):
            output_y = tile_y * TILE_H
            input_start_y = output_y * STRIDE - PADDING
            valid_y0 = max(0, input_start_y)
            valid_y1 = min(layer.spec.input_h, input_start_y + PATCH_H)
            valid_h = valid_y1 - valid_y0
            destination_y = valid_y0 - input_start_y
            for tile_x in range(tiles_x):
                output_x = tile_x * TILE_W
                input_start_x = output_x * STRIDE - PADDING
                valid_x0 = max(0, input_start_x)
                valid_x1 = min(layer.spec.input_w, input_start_x + PATCH_W)
                valid_w = valid_x1 - valid_x0
                destination_x = valid_x0 - input_start_x
                is_boundary = valid_h != PATCH_H or valid_w != PATCH_W
                if is_boundary:
                    layer_emit(
                        Opcode.FILL8,
                        [MemorySpace.INPUT, 0, layer.input_chunks * CHUNK_BYTES, 0],
                    )
                    boundary_tiles += 1

                for chunk in range(layer.input_chunks):
                    chunk_base = chunk * CHUNK_BYTES
                    if layer.input_groups == 1:
                        source_addend = source_tensor_addend + (
                            valid_y0 * layer.spec.input_w + valid_x0
                        ) * LANES
                        destination_offset = chunk_base + (
                            destination_y * PATCH_W + destination_x
                        ) * LANES
                        row_bytes = valid_w * LANES
                        command_index = layer_emit(
                            Opcode.DMA_COPY2D,
                            [
                                MemorySpace.EXT,
                                0,
                                MemorySpace.INPUT,
                                destination_offset,
                                row_bytes,
                                valid_h,
                                layer.spec.input_w * LANES,
                                PATCH_W * LANES,
                                0,
                            ],
                        )
                        relocate(command_index, 1, source_buffer, source_addend)
                    else:
                        for row in range(valid_h):
                            source_addend = source_tensor_addend + (
                                (
                                    (valid_y0 + row) * layer.spec.input_w
                                    + valid_x0
                                )
                                * layer.input_groups
                                + chunk
                            ) * LANES
                            destination_offset = chunk_base + (
                                (destination_y + row) * PATCH_W + destination_x
                            ) * LANES
                            command_index = layer_emit(
                                Opcode.DMA_COPY2D,
                                [
                                    MemorySpace.EXT,
                                    0,
                                    MemorySpace.INPUT,
                                    destination_offset,
                                    LANES,
                                    valid_w,
                                    layer.input_groups * LANES,
                                    LANES,
                                    0,
                                ],
                            )
                            relocate(command_index, 1, source_buffer, source_addend)
                    dma["input"] += valid_h * valid_w * LANES

                for output_group in range(layer.output_groups):
                    for chunk in range(layer.input_chunks):
                        chunk_start = chunk * LANES
                        chunk_channels = min(LANES, layer.spec.input_channels - chunk_start)
                        flags = (FIRST_FLAG if chunk == 0 else 0) | (
                            LAST_FLAG if chunk == layer.input_chunks - 1 else 0
                        )
                        layer_emit(
                            Opcode.CONV_I8,
                            [
                                chunk * CHUNK_BYTES,
                                (output_group * layer.input_chunks + chunk)
                                * WEIGHT_CHUNK_BYTES,
                                0,
                                TILE_H,
                                TILE_W,
                                LANES,
                                chunk_channels,
                                chunk_start,
                                layer.spec.input_channels,
                                KERNEL,
                                STRIDE,
                            ],
                            flags=flags,
                        )
                    layer_emit(
                        Opcode.EPILOGUE,
                        [0, 0, output_group * 128, 2, layer.output_groups * 128],
                    )
                    for row in range(TILE_H):
                        output_addend = output_tensor_addend + (
                            (
                                (output_y + row) * layer.output_w + output_x
                            )
                            * layer.output_groups
                            + output_group
                        ) * LANES
                        command_index = layer_emit(
                            Opcode.DMA_COPY2D,
                            [
                                MemorySpace.OUTPUT,
                                row * TILE_W * LANES,
                                MemorySpace.EXT,
                                0,
                                LANES,
                                TILE_W,
                                LANES,
                                layer.output_groups * LANES,
                                0,
                            ],
                        )
                        relocate(command_index, 3, "output", output_addend)
                        dma["output"] += TILE_W * LANES

        cumulative_dma.update(dma)
        layer_schedules.append(
            {
                "boundary_fill_bytes": boundary_tiles
                * layer.input_chunks
                * CHUNK_BYTES,
                "boundary_fill_commands": boundary_tiles,
                "channel_groups": layer.output_groups,
                "command_count": len(commands) - start_command,
                "command_range": [start_command + 1, len(commands)],
                "dma_bytes": {
                    "constants": dma["constants"],
                    "input": dma["input"],
                    "output": dma["output"],
                    "total": sum(dma.values()),
                },
                "input_channel_chunks": layer.input_chunks,
                "macs": layer.output_h
                * layer.output_w
                * layer.output_channels
                * KERNEL
                * KERNEL
                * layer.spec.input_channels,
                "name": layer.spec.name,
                "opcode_counts": dict(sorted(opcodes.items())),
                "spatial_tiles": tiles_y * tiles_x,
                "tile_executions": tiles_y * tiles_x * layer.output_groups,
                "tile_shape": [TILE_H, TILE_W],
            }
        )

    emit(Opcode.END)
    command_bytes = b"".join(command.to_bytes() for command in commands)
    first = prepared[0]
    input_bytes = first.spec.input_h * first.spec.input_w * first.input_groups * LANES
    manifest = {
        "abi": {"major": ABI_MAJOR, "minor": ABI_MINOR},
        "buffers": {
            "constants": {"alignment": 64, "bytes": len(constant_blob), "role": "constant"},
            "input": {"alignment": 64, "bytes": input_bytes, "role": "input"},
            "output": {"alignment": 64, "bytes": output_buffer_bytes, "role": "output"},
        },
        "commands": {
            "bytes": len(command_bytes),
            "count": len(commands),
            "record_bytes": 128,
        },
        "constants": constant_records,
        "input": {
            "dtype": "int8",
            "layout": "HWC8",
            "logical_shape": [1, first.spec.input_channels, first.spec.input_h, first.spec.input_w],
            "physical_shape": [first.spec.input_h, first.spec.input_w, first.input_groups, LANES],
            "scale_binary32": float(np.float32(first.spec.input_scale)),
        },
        "output": {
            "allocation": "all pipeline intermediates retained in declaration order",
            "tensors": tensor_allocations,
        },
        "profile": "M6_SEQUENTIAL_CONV_SILU_PIPELINE",
        "relocations": relocations,
        "schedule": {
            "dma_bytes": {
                "constants": cumulative_dma["constants"],
                "input": cumulative_dma["input"],
                "output": cumulative_dma["output"],
                "total": sum(cumulative_dma.values()),
            },
            "layers": layer_schedules,
            "macs": sum(int(item["macs"]) for item in layer_schedules),
            "opcode_counts": dict(sorted(cumulative_opcodes.items())),
        },
        "schema": SCHEMA,
        "source": {
            "model_sha256": source_model_sha256,
            "nodes": [name for layer in prepared for name in layer.spec.node_names],
        },
    }
    debug = {
        "constant_data_sha256": _sha256(bytes(constant_blob)),
        "layers": [
            {
                "bias_i32": layer.bias_i32.astype(int).tolist(),
                "name": layer.spec.name,
                "weight_i8_sha256": _sha256(layer.weights_i8.tobytes()),
            }
            for layer in prepared
        ],
        "schema": SCHEMA,
    }
    return build_hxb(
        [
            Section(SectionType.MANIFEST_UTF8_JSON, _canonical_json(manifest)),
            Section(SectionType.COMMANDS, command_bytes),
            Section(SectionType.CONSTANT_DATA, bytes(constant_blob)),
            Section(SectionType.DEBUG_UTF8_JSON, _canonical_json(debug)),
        ]
    )


def load_pinned_first_two_layers(
    model_path: str | Path, calibration_path: str | Path, lut_path: str | Path
) -> tuple[list[ConvSiluLayerSpec], str]:
    """Validate and extract nodes 0..5 of the pinned YOLOv8n graph."""

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised by the integration tool.
        raise CompileError("ONNX is required to compile the pinned model") from exc

    model_bytes = Path(model_path).read_bytes()
    model_hash = _sha256(model_bytes)
    if model_hash != MODEL_SHA256:
        raise CompileError(f"pinned model hash mismatch: {model_hash}")
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if calibration.get("source_model", {}).get("sha256") != model_hash:
        raise CompileError("calibration source model hash does not match the ONNX model")
    model = onnx.load_from_string(model_bytes)
    if model.ir_version != 7 or [
        (item.domain, item.version) for item in model.opset_import
    ] != [("", 13)]:
        raise CompileError("pinned graph must use ONNX IR 7 and default-domain opset 13")
    if len(model.graph.node) < 6:
        raise CompileError("graph does not contain the required first two Conv-SiLU blocks")
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    scales = {item["name"]: item["values"] for item in calibration["scales"]}
    lut_blob = Path(lut_path).read_bytes()
    if _sha256(lut_blob) != calibration["silu"]["binary_sha256"]:
        raise CompileError("SiLU LUT binary hash mismatch")

    dimensions = [(320, 320, 3), (160, 160, 16)]
    prefixes = ["model.0", "model.1"]
    result: list[ConvSiluLayerSpec] = []
    prior_scale = float(scales["images_scale"][0])
    for layer_index, (height, width, channels) in enumerate(dimensions):
        base = layer_index * 3
        conv, sigmoid, mul = model.graph.node[base : base + 3]
        if [conv.op_type, sigmoid.op_type, mul.op_type] != ["Conv", "Sigmoid", "Mul"]:
            raise CompileError(f"nodes {base}..{base + 2} must form Conv, Sigmoid, Mul")
        if list(sigmoid.input) != [conv.output[0]] or sorted(mul.input) != sorted(
            [conv.output[0], sigmoid.output[0]]
        ):
            raise CompileError(f"nodes {base}..{base + 2} do not form exact SiLU")
        attrs = {item.name: onnx.helper.get_attribute_value(item) for item in conv.attribute}
        expected = {
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [2, 2],
        }
        if attrs != expected:
            raise CompileError(f"unsupported Conv attributes for {conv.name}: {attrs!r}")
        try:
            weights = initializers[conv.input[1]]
            bias = initializers[conv.input[2]]
            weight_scales = scales[f"{prefixes[layer_index]}.conv.weight_scale"]
            record = calibration["silu"]["records"][layer_index]
        except (KeyError, IndexError) as exc:
            raise CompileError(f"calibration lacks {prefixes[layer_index]} data") from exc
        if record.get("conv_node_index") != base or record.get("conv_node_name") != conv.name:
            raise CompileError(f"SiLU calibration does not match {conv.name}")
        offset, length = int(record["table_offset"]), int(record["table_bytes"])
        lut = np.frombuffer(lut_blob[offset : offset + length], dtype=np.int8).copy()
        if _sha256(lut.tobytes()) != record["table_sha256"]:
            raise CompileError(f"SiLU LUT hash does not match {conv.name}")
        result.append(
            ConvSiluLayerSpec(
                name=prefixes[layer_index],
                input_h=height,
                input_w=width,
                input_channels=channels,
                weights=weights,
                bias=bias,
                input_scale=prior_scale,
                weight_scales=weight_scales,
                output_scale=record["output_scale_binary32"],
                multipliers=record["accumulator_to_grid"]["multipliers"],
                shifts=record["accumulator_to_grid"]["shifts"],
                lut=lut,
                node_names=(conv.name, sigmoid.name, mul.name),
            )
        )
        prior_scale = float(record["output_scale_binary32"])
    return result, model_hash


def compile_pinned_first_two_layers(
    model_path: str | Path, calibration_path: str | Path, lut_path: str | Path
) -> bytes:
    """Compile the first two connected Conv-SiLU blocks of pinned YOLOv8n."""

    layers, model_hash = load_pinned_first_two_layers(model_path, calibration_path, lut_path)
    return compile_conv_silu_pipeline(layers=layers, source_model_sha256=model_hash)
