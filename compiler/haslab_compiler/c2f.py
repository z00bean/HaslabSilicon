"""First branch-sensitive YOLOv8n C2f lowering for the HASLAB v0 profile."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Sequence

import numpy as np

from haslab_ref import oihw_to_khwci8, quantize_int8
from haslab_sim import ABI_MAJOR, ABI_MINOR, Command, MemorySpace, Opcode

from .hxb import Section, SectionType, build_hxb
from .pipeline import ConvSiluLayerSpec, compile_conv_silu_pipeline, load_pinned_first_two_layers
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

SCHEMA = "haslab.first-c2f.v1"
TILE_H = 8
TILE_W = 8
LANES = 8
MULTIPLIER_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class C2fConvSpec:
    """One fused Conv-SiLU inside the first C2f block."""

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
    kernel: int
    stride: int
    padding: int


@dataclass(frozen=True)
class C2fBlockSpec:
    """The exact cv1/split/bottleneck/add/concat/cv2 structure of model.2."""

    cv1: C2fConvSpec
    bottleneck_cv1: C2fConvSpec
    bottleneck_cv2: C2fConvSpec
    cv2: C2fConvSpec
    residual_scale: float
    concat_scale: float
    split_nodes: tuple[str, str]
    add_node: str
    concat_node: str


@dataclass(frozen=True)
class _PreparedConv:
    spec: C2fConvSpec
    weights_i8: np.ndarray
    bias_i32: np.ndarray
    multipliers: np.ndarray
    shifts: np.ndarray
    lut: np.ndarray
    output_h: int
    output_w: int
    output_channels: int
    input_groups: int
    output_groups: int


@dataclass(frozen=True)
class _TensorRef:
    name: str
    addend: int
    height: int
    width: int
    channels: int
    physical_groups: int
    group_offset: int
    scale: float


def _same_scale(left: float, right: float) -> bool:
    return np.float32(left).tobytes() == np.float32(right).tobytes()


def _fixed_ratio(ratio: float) -> tuple[int, int]:
    if not np.isfinite(ratio) or ratio < 0:
        raise CompileError("scale ratio must be finite and nonnegative")
    for shift in range(62, -1, -1):
        multiplier = round(ratio * (1 << shift))
        if 0 <= multiplier <= MULTIPLIER_MAX:
            return int(multiplier), shift
    raise CompileError("scale ratio cannot be represented by the v0 parameter format")


def _fixed_add(left_scale: float, right_scale: float, output_scale: float) -> tuple[int, int, int]:
    ratios = (
        float(np.float32(left_scale)) / float(np.float32(output_scale)),
        float(np.float32(right_scale)) / float(np.float32(output_scale)),
    )
    for shift in range(62, -1, -1):
        multipliers = [round(ratio * (1 << shift)) for ratio in ratios]
        if all(0 <= value <= MULTIPLIER_MAX for value in multipliers):
            return int(multipliers[0]), int(multipliers[1]), shift
    raise CompileError("residual scales cannot share a v0 ADD shift")


def _utility_records(multiplier: int, shift: int, groups: int) -> bytes:
    return b"".join(
        _parameter_records(
            np.zeros(LANES, dtype=np.int32),
            np.full(LANES, multiplier, dtype=np.int64),
            np.full(LANES, shift, dtype=np.int64),
        )
        for _ in range(groups)
    )


def _extract_hxb(source: bytes) -> tuple[dict[str, object], bytes, bytes, dict[str, object]]:
    """Read compiler-produced sections without adding a compiler/runtime dependency."""

    _, _, _, _, _, count, table_offset, _, _, _ = struct.unpack_from(
        "<8sHHIIIQQI20s", source, 0
    )
    sections: dict[int, bytes] = {}
    for index in range(count):
        kind, _, offset, length, _, _ = struct.unpack_from(
            "<IIQQ32s8s", source, table_offset + index * 64
        )
        sections[kind] = source[offset : offset + length]
    return (
        json.loads(sections[int(SectionType.MANIFEST_UTF8_JSON)]),
        sections[int(SectionType.COMMANDS)],
        sections[int(SectionType.CONSTANT_DATA)],
        json.loads(sections[int(SectionType.DEBUG_UTF8_JSON)]),
    )


def _prepare(spec: C2fConvSpec) -> _PreparedConv:
    weights = np.asarray(spec.weights, dtype=np.float32)
    bias = np.asarray(spec.bias, dtype=np.float32)
    scales = np.asarray(spec.weight_scales, dtype=np.float32)
    multipliers = np.asarray(spec.multipliers, dtype=np.int64)
    shifts = np.asarray(spec.shifts, dtype=np.int64)
    lut = np.asarray(spec.lut, dtype=np.int8).reshape(-1)
    if spec.kernel not in (1, 3) or spec.stride not in (1, 2):
        raise CompileError(f"{spec.name} uses an unsupported kernel or stride")
    if spec.padding not in (0, 1) or spec.padding != (spec.kernel - 1) // 2:
        raise CompileError(f"{spec.name} uses unsupported padding")
    if weights.ndim != 4 or weights.shape[1:] != (
        spec.input_channels,
        spec.kernel,
        spec.kernel,
    ):
        raise CompileError(f"{spec.name} weights do not match its declared convolution")
    output_channels = int(weights.shape[0])
    if output_channels % LANES or spec.input_channels % LANES:
        raise CompileError(f"{spec.name} channels must be multiples of eight")
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
    if np.any(multipliers < 0) or np.any(multipliers > MULTIPLIER_MAX):
        raise CompileError(f"{spec.name} SiLU multiplier is outside the v0 range")
    if np.any(shifts < 0) or np.any(shifts > 62):
        raise CompileError(f"{spec.name} SiLU shift is outside the v0 range")
    output_h = (spec.input_h + 2 * spec.padding - spec.kernel) // spec.stride + 1
    output_w = (spec.input_w + 2 * spec.padding - spec.kernel) // spec.stride + 1
    if output_h % TILE_H or output_w % TILE_W:
        raise CompileError(f"{spec.name} output must divide exactly into 8x8 tiles")
    input_groups = spec.input_channels // LANES
    output_groups = output_channels // LANES
    patch_h = (TILE_H - 1) * spec.stride + spec.kernel
    patch_w = (TILE_W - 1) * spec.stride + spec.kernel
    chunk_bytes = patch_h * patch_w * LANES
    weight_chunk_bytes = spec.kernel * spec.kernel * LANES * LANES
    if input_groups * chunk_bytes > 16_384:
        raise CompileError(f"{spec.name} input chunks exceed v0 input SRAM")
    if input_groups * output_groups * weight_chunk_bytes > 16_384:
        raise CompileError(f"{spec.name} resident weights exceed v0 weight SRAM")
    if output_groups * 128 + 1024 > 3_072:
        raise CompileError(f"{spec.name} parameters and LUT exceed v0 parameter SRAM")
    weights_i8 = quantize_int8(weights, scales.reshape(output_channels, 1, 1, 1))
    return _PreparedConv(
        spec,
        weights_i8,
        _bias_i32(bias, float(spec.input_scale), scales),
        multipliers,
        shifts,
        lut,
        output_h,
        output_w,
        output_channels,
        input_groups,
        output_groups,
    )


class _Builder:
    def __init__(
        self,
        manifest: dict[str, object],
        command_bytes: bytes,
        constant_bytes: bytes,
        debug: dict[str, object],
    ) -> None:
        self.manifest = manifest
        self.commands = [
            Command.from_bytes(command_bytes[offset : offset + 128])
            for offset in range(0, len(command_bytes), 128)
        ]
        if not self.commands or self.commands[-1].opcode != Opcode.END:
            raise CompileError("base pipeline must terminate with END")
        self.commands.pop()
        self.constants = bytearray(constant_bytes)
        self.debug = debug
        self.relocations: list[dict[str, object]] = manifest["relocations"]
        self.tensors: list[dict[str, object]] = manifest["output"]["tensors"]
        self.views: list[dict[str, object]] = []
        self.output_bytes = int(manifest["buffers"]["output"]["bytes"])
        self.opcodes = Counter(command.opcode for command in self.commands)
        self.dma = Counter(manifest["schedule"]["dma_bytes"])
        self.operations: list[dict[str, object]] = []

    def emit(self, opcode: Opcode, payload: Sequence[int] = (), *, flags: int = 0) -> int:
        index = len(self.commands)
        self.commands.append(Command.build(opcode, index + 1, payload, flags=flags))
        self.opcodes[int(opcode)] += 1
        return index

    def relocate(self, index: int, word: int, buffer: str, addend: int) -> None:
        self.relocations.append(
            {
                "addend": addend,
                "buffer": buffer,
                "command_index": index,
                "payload_word": word,
            }
        )

    def add_constant(self, name: str, kind: str, raw: bytes) -> int:
        offset = _align(len(self.constants))
        self.constants.extend(bytes(offset - len(self.constants)))
        self.constants.extend(raw)
        self.manifest["constants"].append(
            {"addend": offset, "bytes": len(raw), "kind": kind, "name": name}
        )
        return offset

    def load_constant(self, offset: int, length: int, space: MemorySpace, local: int) -> None:
        index = self.emit(
            Opcode.DMA_COPY2D,
            [MemorySpace.EXT, 0, space, local, length, 1, length, length, 0],
        )
        self.relocate(index, 1, "constants", offset)
        self.dma["constants"] += length

    def allocate_tensor(self, name: str, h: int, w: int, channels: int, scale: float) -> _TensorRef:
        self.output_bytes = _align(self.output_bytes, 64)
        groups = channels // LANES
        length = h * w * groups * LANES
        record = {
            "addend": self.output_bytes,
            "bytes": length,
            "dtype": "int8",
            "layout": "HWC8",
            "logical_shape": [1, channels, h, w],
            "name": name,
            "physical_shape": [h, w, groups, LANES],
            "scale_binary32": float(np.float32(scale)),
        }
        self.tensors.append(record)
        result = _TensorRef(name, self.output_bytes, h, w, channels, groups, 0, scale)
        self.output_bytes += length
        return result

    def add_view(self, name: str, source: _TensorRef, group_offset: int, channels: int) -> _TensorRef:
        if group_offset < 0 or channels % LANES or group_offset + channels // LANES > source.physical_groups:
            raise CompileError("split view must have an eight-aligned in-range channel extent")
        view = _TensorRef(
            name,
            source.addend,
            source.height,
            source.width,
            channels,
            source.physical_groups,
            source.group_offset + group_offset,
            source.scale,
        )
        self.views.append(
            {
                "alias_of": source.name,
                "byte_allocation": 0,
                "channel_group_offset": view.group_offset,
                "dtype": "int8",
                "layout": "HWC8_VIEW",
                "logical_shape": [1, channels, source.height, source.width],
                "name": name,
                "scale_binary32": float(np.float32(source.scale)),
            }
        )
        return view

    def _load_patch_group(
        self,
        source: _TensorRef,
        relative_group: int,
        valid_y0: int,
        valid_x0: int,
        valid_h: int,
        valid_w: int,
        destination_offset: int,
        destination_y: int,
        destination_x: int,
        patch_w: int,
    ) -> None:
        group = source.group_offset + relative_group
        if source.physical_groups == 1:
            addend = source.addend + (valid_y0 * source.width + valid_x0) * LANES
            local = destination_offset + (destination_y * patch_w + destination_x) * LANES
            index = self.emit(
                Opcode.DMA_COPY2D,
                [
                    MemorySpace.EXT,
                    0,
                    MemorySpace.INPUT,
                    local,
                    valid_w * LANES,
                    valid_h,
                    source.width * LANES,
                    patch_w * LANES,
                    0,
                ],
            )
            self.relocate(index, 1, "output", addend)
        else:
            for row in range(valid_h):
                addend = source.addend + (
                    ((valid_y0 + row) * source.width + valid_x0) * source.physical_groups
                    + group
                ) * LANES
                local = destination_offset + (
                    (destination_y + row) * patch_w + destination_x
                ) * LANES
                index = self.emit(
                    Opcode.DMA_COPY2D,
                    [
                        MemorySpace.EXT,
                        0,
                        MemorySpace.INPUT,
                        local,
                        LANES,
                        valid_w,
                        source.physical_groups * LANES,
                        LANES,
                        0,
                    ],
                )
                self.relocate(index, 1, "output", addend)
        self.dma["input"] += valid_h * valid_w * LANES

    def _load_dense_group(self, source: _TensorRef, relative_group: int, y: int, x: int, local: int) -> None:
        self._load_patch_group(
            source,
            relative_group,
            y,
            x,
            TILE_H,
            TILE_W,
            local,
            0,
            0,
            TILE_W,
        )

    def _store_group(self, destination: _TensorRef, group: int, y: int, x: int) -> None:
        for row in range(TILE_H):
            addend = destination.addend + (
                ((y + row) * destination.width + x) * destination.physical_groups + group
            ) * LANES
            index = self.emit(
                Opcode.DMA_COPY2D,
                [
                    MemorySpace.OUTPUT,
                    row * TILE_W * LANES,
                    MemorySpace.EXT,
                    0,
                    LANES,
                    TILE_W,
                    LANES,
                    destination.physical_groups * LANES,
                    0,
                ],
            )
            self.relocate(index, 3, "output", addend)
        self.dma["output"] += TILE_H * TILE_W * LANES

    def schedule_conv(self, source: _TensorRef, destination: _TensorRef, layer: _PreparedConv) -> None:
        if (source.height, source.width, source.channels) != (
            layer.spec.input_h,
            layer.spec.input_w,
            layer.spec.input_channels,
        ):
            raise CompileError(f"{layer.spec.name} source shape does not match its declaration")
        if not _same_scale(source.scale, layer.spec.input_scale):
            raise CompileError(f"{layer.spec.name} input scale does not match its source")
        packed = np.ascontiguousarray(oihw_to_khwci8(layer.weights_i8))
        weight_parts = []
        for output_group in range(layer.output_groups):
            for input_group in range(layer.input_groups):
                start = input_group * LANES
                weight_parts.append(
                    np.ascontiguousarray(
                        packed[:, :, start : start + LANES, output_group, :]
                    ).tobytes()
                )
        weights = b"".join(weight_parts)
        parameters = b"".join(
            _parameter_records(
                layer.bias_i32[group * LANES : (group + 1) * LANES],
                layer.multipliers[group * LANES : (group + 1) * LANES],
                layer.shifts[group * LANES : (group + 1) * LANES],
            )
            for group in range(layer.output_groups)
        )
        weight_offset = self.add_constant(f"{layer.spec.name}.weights", "KHWCI8_CHUNKED_INT8", weights)
        parameter_offset = self.add_constant(f"{layer.spec.name}.parameters", "EPILOGUE_RECORDS", parameters)
        lut_offset = self.add_constant(f"{layer.spec.name}.lut", "SILU_LUT_INT8", layer.lut.tobytes())
        start = len(self.commands)
        before_opcodes = self.opcodes.copy()
        before_dma = self.dma.copy()
        self.load_constant(weight_offset, len(weights), MemorySpace.WEIGHT, 0)
        self.load_constant(parameter_offset, len(parameters), MemorySpace.PARAM, 0)
        self.load_constant(lut_offset, 1024, MemorySpace.PARAM, layer.output_groups * 128)
        patch_h = (TILE_H - 1) * layer.spec.stride + layer.spec.kernel
        patch_w = (TILE_W - 1) * layer.spec.stride + layer.spec.kernel
        chunk_bytes = patch_h * patch_w * LANES
        weight_chunk_bytes = layer.spec.kernel * layer.spec.kernel * LANES * LANES
        boundary_tiles = 0
        for output_y in range(0, layer.output_h, TILE_H):
            input_start_y = output_y * layer.spec.stride - layer.spec.padding
            valid_y0 = max(0, input_start_y)
            valid_y1 = min(source.height, input_start_y + patch_h)
            valid_h = valid_y1 - valid_y0
            destination_y = valid_y0 - input_start_y
            for output_x in range(0, layer.output_w, TILE_W):
                input_start_x = output_x * layer.spec.stride - layer.spec.padding
                valid_x0 = max(0, input_start_x)
                valid_x1 = min(source.width, input_start_x + patch_w)
                valid_w = valid_x1 - valid_x0
                destination_x = valid_x0 - input_start_x
                if valid_h != patch_h or valid_w != patch_w:
                    self.emit(Opcode.FILL8, [MemorySpace.INPUT, 0, layer.input_groups * chunk_bytes, 0])
                    boundary_tiles += 1
                for input_group in range(layer.input_groups):
                    self._load_patch_group(
                        source,
                        input_group,
                        valid_y0,
                        valid_x0,
                        valid_h,
                        valid_w,
                        input_group * chunk_bytes,
                        destination_y,
                        destination_x,
                        patch_w,
                    )
                for output_group in range(layer.output_groups):
                    for input_group in range(layer.input_groups):
                        flags = (FIRST_FLAG if input_group == 0 else 0) | (
                            LAST_FLAG if input_group == layer.input_groups - 1 else 0
                        )
                        self.emit(
                            Opcode.CONV_I8,
                            [
                                input_group * chunk_bytes,
                                (output_group * layer.input_groups + input_group) * weight_chunk_bytes,
                                0,
                                TILE_H,
                                TILE_W,
                                LANES,
                                LANES,
                                input_group * LANES,
                                layer.spec.input_channels,
                                layer.spec.kernel,
                                layer.spec.stride,
                            ],
                            flags=flags,
                        )
                    self.emit(
                        Opcode.EPILOGUE,
                        [0, 0, output_group * 128, 2, layer.output_groups * 128],
                    )
                    self._store_group(destination, output_group, output_y, output_x)
        opcode_delta = self.opcodes - before_opcodes
        dma_delta = self.dma - before_dma
        self.operations.append(
            {
                "boundary_fill_commands": boundary_tiles,
                "command_count": len(self.commands) - start,
                "dma_bytes": {key: dma_delta[key] for key in ("constants", "input", "output")}
                | {"total": sum(dma_delta[key] for key in ("constants", "input", "output"))},
                "input_channel_chunks": layer.input_groups,
                "kind": "conv_silu",
                "macs": layer.output_h * layer.output_w * layer.output_channels * layer.spec.kernel * layer.spec.kernel * layer.spec.input_channels,
                "name": layer.spec.name,
                "opcode_counts": {
                    Opcode(key).name: value for key, value in sorted(opcode_delta.items())
                },
                "output_group_tiles": (layer.output_h // TILE_H) * (layer.output_w // TILE_W) * layer.output_groups,
            }
        )
        self.debug.setdefault("layers", []).append(
            {
                "bias_i32": layer.bias_i32.astype(int).tolist(),
                "name": layer.spec.name,
                "weight_i8_sha256": _sha256(layer.weights_i8.tobytes()),
            }
        )

    def schedule_add(
        self,
        left: _TensorRef,
        right: _TensorRef,
        destination: _TensorRef,
        node_name: str,
    ) -> None:
        if (left.height, left.width, left.channels) != (right.height, right.width, right.channels):
            raise CompileError("residual inputs must have equal shapes")
        if (destination.height, destination.width, destination.channels) != (
            left.height,
            left.width,
            left.channels,
        ):
            raise CompileError("residual output shape must match both inputs")
        left_multiplier, right_multiplier, shift = _fixed_add(
            left.scale, right.scale, destination.scale
        )
        groups = left.channels // LANES
        left_raw = _utility_records(left_multiplier, shift, groups)
        right_raw = _utility_records(right_multiplier, shift, groups)
        left_offset = self.add_constant(f"{destination.name}.left_parameters", "ADD_PARAMETERS", left_raw)
        right_offset = self.add_constant(f"{destination.name}.right_parameters", "ADD_PARAMETERS", right_raw)
        start = len(self.commands)
        before_opcodes = self.opcodes.copy()
        before_dma = self.dma.copy()
        self.load_constant(left_offset, len(left_raw), MemorySpace.PARAM, 0)
        self.load_constant(right_offset, len(right_raw), MemorySpace.PARAM, len(left_raw))
        for y in range(0, left.height, TILE_H):
            for x in range(0, left.width, TILE_W):
                for group in range(groups):
                    self._load_dense_group(left, group, y, x, 0)
                    self._load_dense_group(right, group, y, x, TILE_H * TILE_W * LANES)
                    self.emit(
                        Opcode.ADD_I8,
                        [
                            0,
                            TILE_H * TILE_W * LANES,
                            0,
                            TILE_H,
                            TILE_W,
                            LANES,
                            group * 128,
                            len(left_raw) + group * 128,
                        ],
                    )
                    self._store_group(destination, group, y, x)
        opcode_delta = self.opcodes - before_opcodes
        dma_delta = self.dma - before_dma
        self.operations.append(
            {
                "command_count": len(self.commands) - start,
                "dma_bytes": {key: dma_delta[key] for key in ("constants", "input", "output")}
                | {"total": sum(dma_delta[key] for key in ("constants", "input", "output"))},
                "fixed_point": {
                    "left_multiplier": left_multiplier,
                    "right_multiplier": right_multiplier,
                    "shift": shift,
                },
                "kind": "residual_add",
                "name": node_name,
                "opcode_counts": {
                    Opcode(key).name: value for key, value in sorted(opcode_delta.items())
                },
            }
        )

    def schedule_concat(
        self,
        sources: Sequence[_TensorRef],
        destination: _TensorRef,
        node_name: str,
    ) -> None:
        if not sources or any((item.height, item.width) != (destination.height, destination.width) for item in sources):
            raise CompileError("concat sources must share the destination spatial shape")
        if sum(item.channels for item in sources) != destination.channels:
            raise CompileError("concat source channels do not match the destination")
        records = bytearray()
        conversions: list[dict[str, object]] = []
        for source in sources:
            multiplier, shift = _fixed_ratio(
                float(np.float32(source.scale)) / float(np.float32(destination.scale))
            )
            groups = source.channels // LANES
            records.extend(_utility_records(multiplier, shift, groups))
            conversions.append(
                {
                    "input_scale_binary32": float(np.float32(source.scale)),
                    "multiplier": multiplier,
                    "output_scale_binary32": float(np.float32(destination.scale)),
                    "shift": shift,
                    "source": source.name,
                }
            )
        parameter_offset = self.add_constant(
            f"{destination.name}.map_parameters", "MAP_PARAMETERS", bytes(records)
        )
        start = len(self.commands)
        before_opcodes = self.opcodes.copy()
        before_dma = self.dma.copy()
        self.load_constant(parameter_offset, len(records), MemorySpace.PARAM, 0)
        destination_group = 0
        parameter_group = 0
        for source in sources:
            groups = source.channels // LANES
            for y in range(0, source.height, TILE_H):
                for x in range(0, source.width, TILE_W):
                    for group in range(groups):
                        self._load_dense_group(source, group, y, x, 0)
                        self.emit(
                            Opcode.MAP_I8,
                            [0, 0, TILE_H, TILE_W, LANES, (parameter_group + group) * 128],
                        )
                        self._store_group(destination, destination_group + group, y, x)
            destination_group += groups
            parameter_group += groups
        opcode_delta = self.opcodes - before_opcodes
        dma_delta = self.dma - before_dma
        self.operations.append(
            {
                "command_count": len(self.commands) - start,
                "dma_bytes": {key: dma_delta[key] for key in ("constants", "input", "output")}
                | {"total": sum(dma_delta[key] for key in ("constants", "input", "output"))},
                "fixed_point": conversions,
                "kind": "channel_concat",
                "name": node_name,
                "opcode_counts": {
                    Opcode(key).name: value for key, value in sorted(opcode_delta.items())
                },
            }
        )

    def finish(self, source_nodes: Sequence[str]) -> bytes:
        self.emit(Opcode.END)
        command_bytes = b"".join(command.to_bytes() for command in self.commands)
        self.manifest["schema"] = SCHEMA
        self.manifest["profile"] = "M6_FIRST_C2F_GRAPH"
        self.manifest["commands"] = {
            "bytes": len(command_bytes),
            "count": len(self.commands),
            "record_bytes": 128,
        }
        self.manifest["buffers"]["constants"]["bytes"] = len(self.constants)
        self.manifest["buffers"]["output"]["bytes"] = self.output_bytes
        self.manifest["output"]["allocation"] = "retained operator tensors plus zero-allocation split views"
        self.manifest["output"]["views"] = self.views
        self.manifest["source"]["nodes"] = list(source_nodes)
        self.manifest["schedule"]["c2f_operations"] = self.operations
        self.manifest["schedule"]["dma_bytes"] = {
            key: self.dma[key] for key in ("constants", "input", "output")
        } | {"total": sum(self.dma[key] for key in ("constants", "input", "output"))}
        self.manifest["schedule"]["macs"] = int(self.manifest["schedule"]["macs"]) + sum(
            int(item.get("macs", 0)) for item in self.operations
        )
        self.manifest["schedule"]["opcode_counts"] = {
            Opcode(key).name: value for key, value in sorted(self.opcodes.items())
        }
        self.debug["schema"] = SCHEMA
        self.debug["constant_data_sha256"] = _sha256(bytes(self.constants))
        return build_hxb(
            [
                Section(SectionType.MANIFEST_UTF8_JSON, _canonical_json(self.manifest)),
                Section(SectionType.COMMANDS, command_bytes),
                Section(SectionType.CONSTANT_DATA, bytes(self.constants)),
                Section(SectionType.DEBUG_UTF8_JSON, _canonical_json(self.debug)),
            ]
        )


def compile_first_c2f(
    *, stem_layers: Sequence[ConvSiluLayerSpec], block: C2fBlockSpec, source_model_sha256: str
) -> bytes:
    """Compile the two-layer stem and complete first C2f block into one package."""

    if len(stem_layers) != 2:
        raise CompileError("the first C2f profile requires exactly two stem layers")
    base = compile_conv_silu_pipeline(layers=stem_layers, source_model_sha256=source_model_sha256)
    manifest, commands, constants, debug = _extract_hxb(base)
    builder = _Builder(manifest, commands, constants, debug)
    stem_record = builder.tensors[-1]
    stem = _TensorRef(
        stem_record["name"],
        int(stem_record["addend"]),
        int(stem_record["physical_shape"][0]),
        int(stem_record["physical_shape"][1]),
        int(stem_record["logical_shape"][1]),
        int(stem_record["physical_shape"][2]),
        0,
        float(stem_record["scale_binary32"]),
    )
    cv1 = _prepare(block.cv1)
    bottleneck_cv1 = _prepare(block.bottleneck_cv1)
    bottleneck_cv2 = _prepare(block.bottleneck_cv2)
    cv2 = _prepare(block.cv2)
    if (cv1.output_channels, cv1.output_h, cv1.output_w) != (32, stem.height, stem.width):
        raise CompileError("first C2f cv1 must produce 32 channels without changing spatial size")
    if (bottleneck_cv1.output_channels, bottleneck_cv2.output_channels) != (16, 16):
        raise CompileError("first C2f bottleneck must keep 16 channels")
    if cv2.spec.input_channels != 48 or cv2.output_channels != 32:
        raise CompileError("first C2f cv2 must map 48 concatenated channels to 32")

    cv1_tensor = builder.allocate_tensor(block.cv1.name, cv1.output_h, cv1.output_w, 32, block.cv1.output_scale)
    builder.schedule_conv(stem, cv1_tensor, cv1)
    split0 = builder.add_view("model.2.split0", cv1_tensor, 0, 16)
    split1 = builder.add_view("model.2.split1", cv1_tensor, 2, 16)
    bottleneck1_tensor = builder.allocate_tensor(
        block.bottleneck_cv1.name,
        bottleneck_cv1.output_h,
        bottleneck_cv1.output_w,
        16,
        block.bottleneck_cv1.output_scale,
    )
    builder.schedule_conv(split1, bottleneck1_tensor, bottleneck_cv1)
    bottleneck2_tensor = builder.allocate_tensor(
        block.bottleneck_cv2.name,
        bottleneck_cv2.output_h,
        bottleneck_cv2.output_w,
        16,
        block.bottleneck_cv2.output_scale,
    )
    builder.schedule_conv(bottleneck1_tensor, bottleneck2_tensor, bottleneck_cv2)
    residual = builder.allocate_tensor(
        "model.2.m.0.add", stem.height, stem.width, 16, block.residual_scale
    )
    builder.schedule_add(split1, bottleneck2_tensor, residual, block.add_node)
    concat = builder.allocate_tensor(
        "model.2.concat", stem.height, stem.width, 48, block.concat_scale
    )
    builder.schedule_concat((split0, split1, residual), concat, block.concat_node)
    output = builder.allocate_tensor(
        "model.2", cv2.output_h, cv2.output_w, 32, block.cv2.output_scale
    )
    builder.schedule_conv(concat, output, cv2)
    # Preserve ONNX execution order in the manifest.
    source_nodes = [name for layer in stem_layers for name in layer.node_names] + [
        block.cv1.node_names[0],
        block.cv1.node_names[1],
        block.cv1.node_names[2],
        block.split_nodes[0],
        block.split_nodes[1],
        block.bottleneck_cv1.node_names[0],
        block.bottleneck_cv1.node_names[1],
        block.bottleneck_cv1.node_names[2],
        block.bottleneck_cv2.node_names[0],
        block.bottleneck_cv2.node_names[1],
        block.bottleneck_cv2.node_names[2],
        block.add_node,
        block.concat_node,
        block.cv2.node_names[0],
        block.cv2.node_names[1],
        block.cv2.node_names[2],
    ]
    return builder.finish(source_nodes)


def _conv_from_nodes(
    *,
    model: object,
    initializers: dict[str, np.ndarray],
    scales: dict[str, list[float]],
    records: dict[int, dict[str, object]],
    lut_blob: bytes,
    node_index: int,
    prefix: str,
    input_h: int,
    input_w: int,
    input_channels: int,
    input_scale: float,
) -> C2fConvSpec:
    import onnx

    conv, sigmoid, mul = model.graph.node[node_index : node_index + 3]
    if [conv.op_type, sigmoid.op_type, mul.op_type] != ["Conv", "Sigmoid", "Mul"]:
        raise CompileError(f"nodes {node_index}..{node_index + 2} must form Conv, Sigmoid, Mul")
    if list(sigmoid.input) != [conv.output[0]] or sorted(mul.input) != sorted(
        [conv.output[0], sigmoid.output[0]]
    ):
        raise CompileError(f"nodes {node_index}..{node_index + 2} do not form exact SiLU")
    attrs = {item.name: onnx.helper.get_attribute_value(item) for item in conv.attribute}
    kernel = int(attrs.get("kernel_shape", [0, 0])[0])
    stride = int(attrs.get("strides", [0, 0])[0])
    padding = int(attrs.get("pads", [0, 0, 0, 0])[0])
    expected = {
        "dilations": [1, 1],
        "group": 1,
        "kernel_shape": [kernel, kernel],
        "pads": [padding, padding, padding, padding],
        "strides": [stride, stride],
    }
    if attrs != expected or (kernel, stride, padding) not in {(1, 1, 0), (3, 1, 1)}:
        raise CompileError(f"unsupported first-C2f Conv attributes for {conv.name}: {attrs!r}")
    try:
        record = records[node_index]
        weights = initializers[conv.input[1]]
        bias = initializers[conv.input[2]]
        weight_scales = scales[f"{prefix}.conv.weight_scale"]
    except KeyError as exc:
        raise CompileError(f"calibration lacks {prefix} data") from exc
    if record.get("conv_node_name") != conv.name:
        raise CompileError(f"SiLU calibration does not match {conv.name}")
    offset, length = int(record["table_offset"]), int(record["table_bytes"])
    lut = np.frombuffer(lut_blob[offset : offset + length], dtype=np.int8).copy()
    if _sha256(lut.tobytes()) != record["table_sha256"]:
        raise CompileError(f"SiLU LUT hash does not match {conv.name}")
    return C2fConvSpec(
        prefix,
        input_h,
        input_w,
        input_channels,
        weights,
        bias,
        input_scale,
        weight_scales,
        float(record["output_scale_binary32"]),
        record["accumulator_to_grid"]["multipliers"],
        record["accumulator_to_grid"]["shifts"],
        lut,
        (conv.name, sigmoid.name, mul.name),
        kernel,
        stride,
        padding,
    )


def load_pinned_first_c2f(
    model_path: str | Path, calibration_path: str | Path, lut_path: str | Path
) -> tuple[list[ConvSiluLayerSpec], C2fBlockSpec, str]:
    """Validate and extract nodes 0..21 of the pinned YOLOv8n graph."""

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - integration-only dependency.
        raise CompileError("ONNX is required to compile the pinned model") from exc
    stem, model_hash = load_pinned_first_two_layers(model_path, calibration_path, lut_path)
    if model_hash != MODEL_SHA256:
        raise CompileError("pinned model hash mismatch")
    model = onnx.load(model_path)
    if len(model.graph.node) < 22:
        raise CompileError("graph does not contain the complete first C2f block")
    expected_types = [
        "Conv", "Sigmoid", "Mul", "Constant", "Split", "Conv", "Sigmoid", "Mul",
        "Conv", "Sigmoid", "Mul", "Add", "Concat", "Conv", "Sigmoid", "Mul",
    ]
    if [node.op_type for node in model.graph.node[6:22]] != expected_types:
        raise CompileError("nodes 6..21 do not match the pinned first C2f structure")
    constant, split = model.graph.node[9:11]
    add, concat = model.graph.node[17:19]
    if {item.name: onnx.helper.get_attribute_value(item) for item in split.attribute} != {"axis": 1}:
        raise CompileError("first C2f split must use channel axis 1")
    if {item.name: onnx.helper.get_attribute_value(item) for item in concat.attribute} != {"axis": 1}:
        raise CompileError("first C2f concat must use channel axis 1")
    split_values = onnx.helper.get_attribute_value(constant.attribute[0])
    if np.asarray(numpy_helper.to_array(split_values)).tolist() != [16, 16]:
        raise CompileError("first C2f split must be the static 16/16 partition")
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    scales = {item["name"]: item["values"] for item in calibration["scales"]}
    records = {int(item["conv_node_index"]): item for item in calibration["silu"]["records"]}
    lut_blob = Path(lut_path).read_bytes()
    if _sha256(lut_blob) != calibration["silu"]["binary_sha256"]:
        raise CompileError("SiLU LUT binary hash mismatch")
    cv1 = _conv_from_nodes(
        model=model, initializers=initializers, scales=scales, records=records, lut_blob=lut_blob,
        node_index=6, prefix="model.2.cv1", input_h=80, input_w=80, input_channels=32,
        input_scale=stem[-1].output_scale,
    )
    bottleneck_cv1 = _conv_from_nodes(
        model=model, initializers=initializers, scales=scales, records=records, lut_blob=lut_blob,
        node_index=11, prefix="model.2.m.0.cv1", input_h=80, input_w=80, input_channels=16,
        input_scale=cv1.output_scale,
    )
    bottleneck_cv2 = _conv_from_nodes(
        model=model, initializers=initializers, scales=scales, records=records, lut_blob=lut_blob,
        node_index=14, prefix="model.2.m.0.cv2", input_h=80, input_w=80, input_channels=16,
        input_scale=bottleneck_cv1.output_scale,
    )
    residual_scale = float(scales["/model.2/m.0/Add_output_0_scale"][0])
    concat_scale = float(scales["/model.2/Concat_output_0_scale"][0])
    cv2 = _conv_from_nodes(
        model=model, initializers=initializers, scales=scales, records=records, lut_blob=lut_blob,
        node_index=19, prefix="model.2.cv2", input_h=80, input_w=80, input_channels=48,
        input_scale=concat_scale,
    )
    if list(model.graph.node[6].input[:1]) != [model.graph.node[5].output[0]]:
        raise CompileError("first C2f cv1 is not connected to the stem output")
    if list(split.input) != [model.graph.node[8].output[0], constant.output[0]]:
        raise CompileError("first C2f split inputs do not match the pinned graph")
    if list(model.graph.node[11].input[:1]) != [split.output[1]]:
        raise CompileError("first bottleneck convolution is not connected to split1")
    if list(model.graph.node[14].input[:1]) != [model.graph.node[13].output[0]]:
        raise CompileError("second bottleneck convolution is not connected to the first")
    if list(add.input) != [split.output[1], model.graph.node[16].output[0]]:
        raise CompileError("first C2f residual inputs do not match the pinned graph")
    if list(concat.input) != [split.output[0], split.output[1], add.output[0]]:
        raise CompileError("first C2f concat inputs do not match the pinned graph")
    if list(model.graph.node[19].input[:1]) != [concat.output[0]]:
        raise CompileError("first C2f cv2 is not connected to concat")
    block = C2fBlockSpec(
        cv1,
        bottleneck_cv1,
        bottleneck_cv2,
        cv2,
        residual_scale,
        concat_scale,
        (constant.name, split.name),
        add.name,
        concat.name,
    )
    return stem, block, model_hash


def compile_pinned_first_c2f(
    model_path: str | Path, calibration_path: str | Path, lut_path: str | Path
) -> bytes:
    stem, block, model_hash = load_pinned_first_c2f(model_path, calibration_path, lut_path)
    return compile_first_c2f(stem_layers=stem, block=block, source_model_sha256=model_hash)
