"""Schedule the complete pinned YOLOv8n first Conv-SiLU layer."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from haslab_ref import oihw_to_khwci8, quantize_int8
from haslab_sim import ABI_MAJOR, ABI_MINOR, Command, MemorySpace, Opcode

from .hxb import Section, SectionType, build_hxb
from .vertical_slice import (
    FIRST_FLAG,
    INPUT_CHANNELS,
    KERNEL,
    LAST_FLAG,
    OUTPUT_LANES,
    STRIDE,
    TILE_H,
    TILE_W,
    CompileError,
    _align,
    _bias_i32,
    _canonical_json,
    _load_pinned_first_block_data,
    _parameter_records,
    _sha256,
)

SCHEMA = "haslab.first-layer.v1"
INPUT_H = 320
INPUT_W = 320
OUTPUT_H = 160
OUTPUT_W = 160
OUTPUT_CHANNELS = 16
CHANNEL_GROUPS = 2
TILES_Y = OUTPUT_H // TILE_H
TILES_X = OUTPUT_W // TILE_W
PATCH_H = (TILE_H - 1) * STRIDE + KERNEL
PATCH_W = (TILE_W - 1) * STRIDE + KERNEL
INPUT_PIXEL_BYTES = 8
OUTPUT_PIXEL_BYTES = CHANNEL_GROUPS * OUTPUT_LANES


def _validated_arrays(
    weights: object,
    bias: object,
    input_scale: float,
    weight_scales: object,
    output_scale: float,
    multipliers: object,
    shifts: object,
    lut: object,
    node_names: tuple[str, str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weight_values = np.asarray(weights, dtype=np.float32)
    bias_values = np.asarray(bias, dtype=np.float32)
    scales = np.asarray(weight_scales, dtype=np.float32)
    multiplier_values = np.asarray(multipliers, dtype=np.int64)
    shift_values = np.asarray(shifts, dtype=np.int64)
    table = np.asarray(lut, dtype=np.int8).reshape(-1)
    if weight_values.shape != (OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL, KERNEL):
        raise CompileError("first Conv weights must have shape [16,3,3,3]")
    if bias_values.shape != (OUTPUT_CHANNELS,) or scales.shape != (OUTPUT_CHANNELS,):
        raise CompileError("first Conv bias and weight scales must have 16 channels")
    if multiplier_values.shape != (OUTPUT_CHANNELS,) or shift_values.shape != (
        OUTPUT_CHANNELS,
    ):
        raise CompileError("first SiLU coefficients must have 16 channels")
    if table.size != 1024:
        raise CompileError("first SiLU LUT must contain 1024 bytes")
    if not all(np.isfinite(value) and value > 0 for value in (input_scale, output_scale)):
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
    return (
        weight_values,
        bias_values,
        scales,
        multiplier_values,
        shift_values,
        table,
    )


def compile_conv_silu_first_layer(
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
    """Compile all 400 tiles and both output groups of the first layer."""

    (
        weight_values,
        bias_values,
        scales,
        multiplier_values,
        shift_values,
        table,
    ) = _validated_arrays(
        weights,
        bias,
        input_scale,
        weight_scales,
        output_scale,
        multipliers,
        shifts,
        lut,
        node_names,
    )

    quantized_weights = quantize_int8(
        weight_values, scales.reshape(OUTPUT_CHANNELS, 1, 1, 1)
    )
    packed = np.ascontiguousarray(oihw_to_khwci8(quantized_weights))
    weight_groups = [
        np.ascontiguousarray(packed[:, :, :, group, :]).tobytes()
        for group in range(CHANNEL_GROUPS)
    ]
    quantized_bias = _bias_i32(bias_values, float(input_scale), scales)
    parameter_groups = [
        _parameter_records(
            quantized_bias[group * OUTPUT_LANES : (group + 1) * OUTPUT_LANES],
            multiplier_values[group * OUTPUT_LANES : (group + 1) * OUTPUT_LANES],
            shift_values[group * OUTPUT_LANES : (group + 1) * OUTPUT_LANES],
        )
        for group in range(CHANNEL_GROUPS)
    ]
    lut_bytes = table.tobytes()

    constant_parts: list[tuple[str, str, bytes]] = [
        ("conv.weight.group0", "KHWCI8_INT8", weight_groups[0]),
        ("conv.weight.group1", "KHWCI8_INT8", weight_groups[1]),
        ("conv.epilogue.group0", "EPILOGUE_RECORDS", parameter_groups[0]),
        ("conv.epilogue.group1", "EPILOGUE_RECORDS", parameter_groups[1]),
        ("silu.lut", "SILU_LUT_INT8", lut_bytes),
    ]
    constant_records: list[dict[str, object]] = []
    constant_offsets: dict[str, int] = {}
    constant_blob = bytearray()
    for name, kind, raw in constant_parts:
        offset = _align(len(constant_blob))
        constant_blob.extend(bytes(offset - len(constant_blob)))
        constant_offsets[name] = offset
        constant_blob.extend(raw)
        constant_records.append(
            {"addend": offset, "bytes": len(raw), "kind": kind, "name": name}
        )

    commands: list[Command] = []
    relocations: list[dict[str, object]] = []
    dma_bytes = Counter()
    opcode_counts = Counter()
    boundary_tiles = 0

    def emit(opcode: Opcode, payload: list[int] | tuple[int, ...] = (), *, flags: int = 0) -> int:
        index = len(commands)
        commands.append(Command.build(opcode, index + 1, payload, flags=flags))
        opcode_counts[opcode.name] += 1
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

    def upload_constant(name: str, destination: MemorySpace, destination_offset: int) -> None:
        record = next(item for item in constant_records if item["name"] == name)
        length = int(record["bytes"])
        index = emit(
            Opcode.DMA_COPY2D,
            [
                MemorySpace.EXT,
                0,
                destination,
                destination_offset,
                length,
                1,
                length,
                length,
                0,
            ],
        )
        relocate(index, 1, "constants", constant_offsets[name])
        dma_bytes["constants"] += length

    for group in range(CHANNEL_GROUPS):
        upload_constant(f"conv.weight.group{group}", MemorySpace.WEIGHT, 0)
        upload_constant(f"conv.epilogue.group{group}", MemorySpace.PARAM, 0)
        if group == 0:
            upload_constant("silu.lut", MemorySpace.PARAM, 128)

        for tile_y in range(TILES_Y):
            output_y = tile_y * TILE_H
            input_start_y = output_y * STRIDE - 1
            valid_y0 = max(0, input_start_y)
            valid_y1 = min(INPUT_H, input_start_y + PATCH_H)
            valid_h = valid_y1 - valid_y0
            destination_y = valid_y0 - input_start_y

            for tile_x in range(TILES_X):
                output_x = tile_x * TILE_W
                input_start_x = output_x * STRIDE - 1
                valid_x0 = max(0, input_start_x)
                valid_x1 = min(INPUT_W, input_start_x + PATCH_W)
                valid_w = valid_x1 - valid_x0
                destination_x = valid_x0 - input_start_x
                is_boundary = valid_h != PATCH_H or valid_w != PATCH_W
                if is_boundary:
                    emit(Opcode.FILL8, [MemorySpace.INPUT, 0, PATCH_H * PATCH_W * 8, 0])
                    boundary_tiles += 1

                source_addend = (valid_y0 * INPUT_W + valid_x0) * INPUT_PIXEL_BYTES
                destination_offset = (
                    destination_y * PATCH_W + destination_x
                ) * INPUT_PIXEL_BYTES
                row_bytes = valid_w * INPUT_PIXEL_BYTES
                input_dma = emit(
                    Opcode.DMA_COPY2D,
                    [
                        MemorySpace.EXT,
                        0,
                        MemorySpace.INPUT,
                        destination_offset,
                        row_bytes,
                        valid_h,
                        INPUT_W * INPUT_PIXEL_BYTES,
                        PATCH_W * INPUT_PIXEL_BYTES,
                        0,
                    ],
                )
                relocate(input_dma, 1, "input", source_addend)
                dma_bytes["input"] += row_bytes * valid_h

                emit(
                    Opcode.CONV_I8,
                    [
                        0,
                        0,
                        0,
                        TILE_H,
                        TILE_W,
                        OUTPUT_LANES,
                        INPUT_CHANNELS,
                        0,
                        INPUT_CHANNELS,
                        KERNEL,
                        STRIDE,
                    ],
                    flags=FIRST_FLAG | LAST_FLAG,
                )
                emit(Opcode.EPILOGUE, [0, 0, 0, 2, 128])

                for row in range(TILE_H):
                    output_addend = (
                        ((output_y + row) * OUTPUT_W + output_x) * CHANNEL_GROUPS
                        + group
                    ) * OUTPUT_LANES
                    output_dma = emit(
                        Opcode.DMA_COPY2D,
                        [
                            MemorySpace.OUTPUT,
                            row * TILE_W * OUTPUT_LANES,
                            MemorySpace.EXT,
                            0,
                            OUTPUT_LANES,
                            TILE_W,
                            OUTPUT_LANES,
                            OUTPUT_PIXEL_BYTES,
                            0,
                        ],
                    )
                    relocate(output_dma, 3, "output", output_addend)
                    dma_bytes["output"] += OUTPUT_LANES * TILE_W

    emit(Opcode.END)
    command_bytes = b"".join(command.to_bytes() for command in commands)
    total_dma_bytes = sum(dma_bytes.values())
    expected_boundary_tiles = (TILES_X + TILES_Y - 1) * CHANNEL_GROUPS
    if boundary_tiles != expected_boundary_tiles:
        raise AssertionError("first-layer boundary-tile accounting changed")

    manifest = {
        "abi": {"major": ABI_MAJOR, "minor": ABI_MINOR},
        "buffers": {
            "constants": {
                "alignment": 64,
                "bytes": len(constant_blob),
                "role": "constant",
            },
            "input": {
                "alignment": 64,
                "bytes": INPUT_H * INPUT_W * INPUT_PIXEL_BYTES,
                "role": "input",
            },
            "output": {
                "alignment": 64,
                "bytes": OUTPUT_H * OUTPUT_W * OUTPUT_PIXEL_BYTES,
                "role": "output",
            },
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
            "logical_shape": [1, INPUT_CHANNELS, INPUT_H, INPUT_W],
            "physical_shape": [INPUT_H, INPUT_W, 1, 8],
            "scale_binary32": float(np.float32(input_scale)),
        },
        "output": {
            "dtype": "int8",
            "layout": "HWC8",
            "logical_shape": [1, OUTPUT_CHANNELS, OUTPUT_H, OUTPUT_W],
            "physical_shape": [OUTPUT_H, OUTPUT_W, CHANNEL_GROUPS, OUTPUT_LANES],
            "scale_binary32": float(np.float32(output_scale)),
        },
        "profile": "M6_FIRST_CONV_SILU_LAYER",
        "relocations": relocations,
        "schedule": {
            "boundary_fill_bytes": boundary_tiles * PATCH_H * PATCH_W * 8,
            "boundary_fill_commands": boundary_tiles,
            "channel_groups": CHANNEL_GROUPS,
            "dma_bytes": {
                "constants": dma_bytes["constants"],
                "input": dma_bytes["input"],
                "output": dma_bytes["output"],
                "total": total_dma_bytes,
            },
            "macs": OUTPUT_H * OUTPUT_W * OUTPUT_CHANNELS * KERNEL * KERNEL * INPUT_CHANNELS,
            "opcode_counts": dict(sorted(opcode_counts.items())),
            "spatial_tiles": TILES_Y * TILES_X,
            "tile_executions": TILES_Y * TILES_X * CHANNEL_GROUPS,
            "tile_shape": [TILE_H, TILE_W],
        },
        "schema": SCHEMA,
        "source": {"model_sha256": source_model_sha256, "nodes": list(node_names)},
    }
    debug = {
        "constant_data_sha256": _sha256(bytes(constant_blob)),
        "node_command_range": {
            "first_sequence": 1,
            "last_sequence": len(commands) - 1,
            "nodes": list(node_names),
        },
        "padding": {
            "bottom_tiles": 0,
            "left_tiles_per_group": TILES_Y,
            "right_tiles": 0,
            "top_tiles_per_group": TILES_X,
            "unique_boundary_tiles_per_group": TILES_X + TILES_Y - 1,
        },
        "quantized": {
            "bias_i32": quantized_bias.astype(int).tolist(),
            "weight_i8_sha256": _sha256(quantized_weights.tobytes()),
        },
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


def compile_pinned_first_layer(
    model_path: str | Path,
    calibration_path: str | Path,
    lut_path: str | Path,
) -> bytes:
    """Validate and compile the complete first layer of the pinned graph."""

    return compile_conv_silu_first_layer(
        **_load_pinned_first_block_data(model_path, calibration_path, lut_path)
    )
