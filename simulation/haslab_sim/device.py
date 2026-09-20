"""Sequential functional simulator for HASLAB v0 command behavior."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from haslab_ref import (
    ArithmeticOverflowError,
    add_i8,
    map_i8,
    maxpool5_i8,
    requantize_int32,
    silu_lut_i32,
    upsample2_nearest_i8,
)

from .abi import ABI_MAJOR, ABI_MINOR, UINT32_MAX, Command, MemorySpace, Opcode
from .errors import ErrorCode, ExecutionFault, FaultRecord
from .memory import MemoryMap

FIFO_DEPTH = 8
FIRST_FLAG = 1 << 0
LAST_FLAG = 1 << 1


class DeviceState(IntEnum):
    RESETTING = 0
    IDLE = 1
    RUNNING = 2
    FAULT = 3


class SubmitResult(IntEnum):
    ACCEPTED = 0
    BUSY = 1
    BLOCKED = 2
    INVALID = 3


@dataclass
class _AccumulatorContext:
    acc_offset: int
    output_h: int
    output_w: int
    valid_lanes: int
    total_channels: int
    kernel: int
    stride: int
    next_channel: int
    ready: bool = False


class HaslabDevice:
    """One-context, in-order functional device.

    Execution is immediate when ``run_next`` is called. Bus latency and cycle
    timing are intentionally not modeled.
    """

    def __init__(self, ext_bytes: int = 1 << 20) -> None:
        self.memory = MemoryMap(ext_bytes)
        self.reset_generation = 0
        self._queue: deque[Command] = deque()
        self._context: _AccumulatorContext | None = None
        self.state = DeviceState.IDLE
        self.last_accepted = 0
        self.last_completed = 0
        self.last_end = 0
        self.submit_result = SubmitResult.ACCEPTED
        self.fault = FaultRecord()

    @property
    def fifo_free(self) -> int:
        return FIFO_DEPTH - len(self._queue)

    @property
    def queued_commands(self) -> int:
        return len(self._queue)

    @property
    def accumulator_open(self) -> bool:
        return self._context is not None

    def reset(self) -> None:
        self.state = DeviceState.RESETTING
        self._queue.clear()
        self._context = None
        self.memory.clear_local()
        self.last_accepted = 0
        self.last_completed = 0
        self.last_end = 0
        self.submit_result = SubmitResult.ACCEPTED
        self.fault = FaultRecord()
        self.reset_generation = (self.reset_generation + 1) & UINT32_MAX
        self.state = DeviceState.IDLE

    def submit(self, command: Command | bytes | bytearray | memoryview) -> SubmitResult:
        if self.state in (DeviceState.FAULT, DeviceState.RESETTING):
            self.submit_result = SubmitResult.BLOCKED
            return self.submit_result
        if len(self._queue) >= FIFO_DEPTH:
            self.submit_result = SubmitResult.BUSY
            return self.submit_result
        try:
            parsed = command if isinstance(command, Command) else Command.from_bytes(command)
        except (TypeError, ValueError):
            self.submit_result = SubmitResult.INVALID
            return self.submit_result

        expected_sequence = self.last_accepted + 1
        valid_flags = FIRST_FLAG | LAST_FLAG if parsed.opcode == Opcode.CONV_I8 else 0
        if (
            parsed.abi_major != ABI_MAJOR
            or parsed.abi_minor != ABI_MINOR
            or parsed.reserved != 0
            or parsed.sequence == 0
            or parsed.sequence != expected_sequence
            or parsed.flags & ~valid_flags
        ):
            self.submit_result = SubmitResult.INVALID
            return self.submit_result

        self._queue.append(parsed)
        self.last_accepted = parsed.sequence
        self.submit_result = SubmitResult.ACCEPTED
        if self.state is DeviceState.IDLE:
            self.state = DeviceState.RUNNING
        return self.submit_result

    def run_next(self) -> bool:
        if self.state is DeviceState.FAULT or not self._queue:
            return False
        if self.state is DeviceState.IDLE:
            self.state = DeviceState.RUNNING
        command = self._queue.popleft()
        try:
            self._execute(command)
        except ExecutionFault as exc:
            self._set_fault(command, exc)
            return False
        except ArithmeticOverflowError as exc:
            self._set_fault(command, ExecutionFault(ErrorCode.ARITH_OVERFLOW, str(exc)))
            return False
        except Exception as exc:  # An unexpected model bug is architectural INTERNAL.
            self._set_fault(command, ExecutionFault(ErrorCode.INTERNAL, str(exc)))
            return False
        self.last_completed = command.sequence
        return True

    def run_all(self) -> int:
        retired = 0
        while self._queue and self.state is not DeviceState.FAULT:
            if not self.run_next():
                break
            retired += 1
        return retired

    def _set_fault(self, command: Command, exc: ExecutionFault) -> None:
        self.state = DeviceState.FAULT
        self.fault = FaultRecord(
            exc.code,
            command.sequence,
            exc.space,
            exc.offset,
            exc.field,
            str(exc),
        )

    @staticmethod
    def _require_unused_zero(payload: tuple[int, ...], first_unused: int) -> None:
        for index in range(first_unused, len(payload)):
            if payload[index] != 0:
                raise ExecutionFault(
                    ErrorCode.BAD_FIELD,
                    "unused payload word must be zero",
                    field=index + 4,
                )

    @staticmethod
    def _require_alignment(value: int, field: int) -> None:
        if value % 8 != 0:
            raise ExecutionFault(
                ErrorCode.ALIGNMENT,
                "field must be eight-byte aligned",
                offset=value,
                field=field,
            )

    @staticmethod
    def _space(value: int, field: int) -> MemorySpace:
        try:
            return MemorySpace(value)
        except ValueError as exc:
            raise ExecutionFault(
                ErrorCode.BAD_SPACE, "unknown memory space", space=value, field=field
            ) from exc

    def _execute(self, command: Command) -> None:
        payload = command.padded_payload
        try:
            opcode = Opcode(command.opcode)
        except ValueError as exc:
            raise ExecutionFault(ErrorCode.BAD_OPCODE, "unknown opcode") from exc

        dispatch = {
            Opcode.DMA_COPY2D: self._execute_dma,
            Opcode.FILL8: self._execute_fill,
            Opcode.CONV_I8: self._execute_conv,
            Opcode.EPILOGUE: self._execute_epilogue,
            Opcode.MAP_I8: self._execute_map,
            Opcode.ADD_I8: self._execute_add,
            Opcode.MAXPOOL5_I8: self._execute_pool,
            Opcode.UPSAMPLE2_I8: self._execute_upsample,
            Opcode.COPY2D: self._execute_copy,
            Opcode.FENCE: self._execute_fence,
            Opcode.END: self._execute_end,
        }
        dispatch[opcode](command, payload)

    def _execute_dma(self, _command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 9)
        source_space = self._space(p[0], 4)
        destination_space = self._space(p[2], 6)
        source_offset, destination_offset = p[1], p[3]
        row_bytes, rows, source_stride, destination_stride = p[4:8]
        if p[8] != 0:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "DMA reserved word must be zero", field=12)
        local_spaces = {
            MemorySpace.INPUT,
            MemorySpace.WEIGHT,
            MemorySpace.OUTPUT,
            MemorySpace.PARAM,
        }
        if not (
            (source_space is MemorySpace.EXT and destination_space in local_spaces)
            or (destination_space is MemorySpace.EXT and source_space in local_spaces)
        ):
            raise ExecutionFault(ErrorCode.BAD_SPACE, "DMA requires one EXT and one legal local endpoint")
        for value, field in [
            (source_offset, 5),
            (destination_offset, 7),
            (row_bytes, 8),
            (source_stride, 10),
            (destination_stride, 11),
        ]:
            self._require_alignment(value, field)
        if row_bytes == 0 or rows == 0 or source_stride < row_bytes or destination_stride < row_bytes:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "invalid DMA dimensions or strides")
        self.memory.check(source_space, source_offset + (rows - 1) * source_stride, row_bytes)
        self.memory.check(
            destination_space, destination_offset + (rows - 1) * destination_stride, row_bytes
        )
        for row in range(rows):
            data = self.memory.read(source_space, source_offset + row * source_stride, row_bytes)
            self.memory.write(destination_space, destination_offset + row * destination_stride, data)

    def _execute_fill(self, _command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 4)
        space = self._space(p[0], 4)
        if space not in {
            MemorySpace.INPUT,
            MemorySpace.WEIGHT,
            MemorySpace.OUTPUT,
            MemorySpace.PARAM,
        }:
            raise ExecutionFault(ErrorCode.BAD_SPACE, "FILL8 cannot target this memory", space=int(space))
        offset, length, value = p[1:4]
        self._require_alignment(offset, 5)
        self._require_alignment(length, 6)
        if length == 0 or value > 255:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "invalid FILL8 length or byte value")
        self.memory.fill(space, offset, length, value)

    def _execute_conv(self, command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 11)
        input_offset, weight_offset, acc_offset = p[0:3]
        output_h, output_w, valid_lanes = p[3:6]
        chunk_channels, chunk_start, total_channels, kernel, stride = p[6:11]
        for value, field in [(input_offset, 4), (weight_offset, 5), (acc_offset, 6)]:
            self._require_alignment(value, field)
        if not 1 <= output_h <= 8 or not 1 <= output_w <= 8:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "CONV output dimensions must be 1..8")
        if not 1 <= valid_lanes <= 8 or not 1 <= chunk_channels <= 32:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "invalid CONV lane or chunk count")
        if not 1 <= total_channels <= 65_535 or kernel not in (1, 3) or stride not in (1, 2):
            raise ExecutionFault(ErrorCode.BAD_FIELD, "invalid CONV total, kernel, or stride")
        if chunk_start % 8 or chunk_start + chunk_channels > total_channels:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "invalid CONV chunk position")
        chunk_end = chunk_start + chunk_channels
        if chunk_end < total_channels and chunk_channels % 8:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "nonfinal CONV chunks must be eight-aligned")
        first = bool(command.flags & FIRST_FLAG)
        last = bool(command.flags & LAST_FLAG)
        if first != (chunk_start == 0) or last != (chunk_end == total_channels):
            raise ExecutionFault(ErrorCode.BAD_STATE, "CONV FIRST/LAST flags do not match chunk range")

        if first:
            if self._context is not None:
                raise ExecutionFault(ErrorCode.BAD_STATE, "another accumulator context is open")
            context = _AccumulatorContext(
                acc_offset,
                output_h,
                output_w,
                valid_lanes,
                total_channels,
                kernel,
                stride,
                0,
            )
            acc = np.zeros((output_h, output_w, 1, 8), dtype=np.int32)
        else:
            context = self._context
            if context is None or context.ready:
                raise ExecutionFault(ErrorCode.BAD_STATE, "CONV continuation has no open context")
            expected = (
                context.acc_offset,
                context.output_h,
                context.output_w,
                context.valid_lanes,
                context.total_channels,
                context.kernel,
                context.stride,
                context.next_channel,
            )
            actual = (
                acc_offset,
                output_h,
                output_w,
                valid_lanes,
                total_channels,
                kernel,
                stride,
                chunk_start,
            )
            if actual != expected:
                raise ExecutionFault(ErrorCode.BAD_STATE, "CONV continuation does not match context")
            acc_bytes = output_h * output_w * 8 * 4
            acc = np.frombuffer(
                self.memory.read(MemorySpace.ACC, acc_offset, acc_bytes), dtype="<i4"
            ).copy().reshape(output_h, output_w, 1, 8)

        acc_bytes = output_h * output_w * 8 * 4
        self.memory.check(MemorySpace.ACC, acc_offset, acc_bytes)
        channel_padded = ((chunk_channels + 7) // 8) * 8
        patch_h = (output_h - 1) * stride + kernel
        patch_w = (output_w - 1) * stride + kernel
        input_bytes = patch_h * patch_w * channel_padded
        weight_bytes = kernel * kernel * channel_padded * 8
        activation = np.frombuffer(
            self.memory.read(MemorySpace.INPUT, input_offset, input_bytes), dtype=np.int8
        ).reshape(patch_h, patch_w, channel_padded // 8, 8)
        weights = np.frombuffer(
            self.memory.read(MemorySpace.WEIGHT, weight_offset, weight_bytes), dtype=np.int8
        ).reshape(kernel, kernel, channel_padded, 8)

        for y in range(output_h):
            for x in range(output_w):
                for ky in range(kernel):
                    for kx in range(kernel):
                        for ci in range(chunk_channels):
                            input_value = int(activation[y * stride + ky, x * stride + kx, ci // 8, ci % 8])
                            for lane in range(valid_lanes):
                                value = int(acc[y, x, 0, lane]) + input_value * int(weights[ky, kx, ci, lane])
                                if not -(1 << 31) <= value <= (1 << 31) - 1:
                                    raise ArithmeticOverflowError("signed INT32 CONV accumulation overflow")
                                acc[y, x, 0, lane] = value

        self.memory.write(MemorySpace.ACC, acc_offset, acc.astype("<i4", copy=False).tobytes())
        context.next_channel = chunk_end
        context.ready = last
        self._context = context

    def _read_parameters(
        self, offset: int, field: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_alignment(offset, field)
        raw = self.memory.read(MemorySpace.PARAM, offset, 8 * 16)
        bias = np.empty(8, dtype=np.int32)
        multipliers = np.empty(8, dtype=np.int64)
        shifts = np.empty(8, dtype=np.int64)
        for lane in range(8):
            lane_bias, multiplier, shift, reserved = struct.unpack_from("<iIII", raw, lane * 16)
            if reserved != 0 or multiplier > (1 << 31) - 1 or shift > 62:
                raise ExecutionFault(ErrorCode.BAD_FIELD, "invalid parameter record")
            bias[lane] = lane_bias
            multipliers[lane] = multiplier
            shifts[lane] = shift
        return bias, multipliers, shifts

    @staticmethod
    def _validate_padding_parameters(
        valid_lanes: int,
        bias: np.ndarray,
        multipliers: np.ndarray,
        shifts: np.ndarray,
    ) -> None:
        if np.any(bias[valid_lanes:] != 0) or np.any(multipliers[valid_lanes:] != 0) or np.any(
            shifts[valid_lanes:] != 0
        ):
            raise ExecutionFault(ErrorCode.BAD_FIELD, "padding-lane parameters must be zero")

    def _execute_epilogue(self, _command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 5)
        acc_offset, output_offset, parameter_offset, mode, lut_offset = p[:5]
        for value, field in [(acc_offset, 4), (output_offset, 5), (parameter_offset, 6)]:
            self._require_alignment(value, field)
        context = self._context
        if context is None or not context.ready or acc_offset != context.acc_offset:
            raise ExecutionFault(ErrorCode.BAD_STATE, "EPILOGUE requires the ready accumulator context")
        if mode not in (0, 1, 2):
            raise ExecutionFault(ErrorCode.BAD_FIELD, "unknown EPILOGUE mode", field=7)
        bias, multipliers, shifts = self._read_parameters(parameter_offset, 6)
        self._validate_padding_parameters(context.valid_lanes, bias, multipliers, shifts)
        if mode == 0 and (np.any(multipliers != 0) or np.any(shifts != 0) or lut_offset != 0):
            raise ExecutionFault(ErrorCode.BAD_FIELD, "raw EPILOGUE requires zero scale fields")
        if mode == 1 and lut_offset != 0:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "linear EPILOGUE requires zero LUT offset")

        count = context.output_h * context.output_w * 8
        acc = np.frombuffer(
            self.memory.read(MemorySpace.ACC, acc_offset, count * 4), dtype="<i4"
        ).copy().reshape(context.output_h, context.output_w, 1, 8)
        biased = acc.astype(np.int64) + bias.reshape(1, 1, 1, 8).astype(np.int64)
        if np.any(biased < -(1 << 31)) or np.any(biased > (1 << 31) - 1):
            raise ArithmeticOverflowError("signed INT32 EPILOGUE bias overflow")
        biased32 = biased.astype(np.int32)
        biased32[..., context.valid_lanes :] = 0

        if mode == 0:
            output = biased32.astype("<i4", copy=False).tobytes()
        elif mode == 1:
            output = requantize_int32(biased32, multipliers, shifts)
            output[..., context.valid_lanes :] = 0
            output = output.tobytes()
        else:
            self._require_alignment(lut_offset, 8)
            parameter_end = parameter_offset + 128
            lut_end = lut_offset + 1024
            if max(parameter_offset, lut_offset) < min(parameter_end, lut_end):
                raise ExecutionFault(ErrorCode.BAD_FIELD, "LUT overlaps parameter records")
            lut = np.frombuffer(
                self.memory.read(MemorySpace.PARAM, lut_offset, 1024), dtype=np.int8
            ).copy()
            output_array = silu_lut_i32(biased32, multipliers, shifts, lut)
            output_array[..., context.valid_lanes :] = 0
            output = output_array.tobytes()
        self.memory.write(MemorySpace.OUTPUT, output_offset, output)
        self._context = None

    def _utility_shape(self, h: int, w: int, lanes: int) -> None:
        if not 1 <= h <= 64 or not 1 <= w <= 64 or not 1 <= lanes <= 8:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "invalid utility dimensions or lane count")

    def _execute_map(self, _command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 6)
        if self._context is not None:
            raise ExecutionFault(ErrorCode.BAD_STATE, "utility operation while accumulator is open")
        input_offset, output_offset, h, w, lanes, parameter_offset = p[:6]
        for value, field in [(input_offset, 4), (output_offset, 5), (parameter_offset, 9)]:
            self._require_alignment(value, field)
        self._utility_shape(h, w, lanes)
        bias, multipliers, shifts = self._read_parameters(parameter_offset, 9)
        if np.any(bias != 0):
            raise ExecutionFault(ErrorCode.BAD_FIELD, "MAP parameter bias must be zero")
        self._validate_padding_parameters(lanes, bias, multipliers, shifts)
        count = h * w * 8
        source = np.frombuffer(
            self.memory.read(MemorySpace.INPUT, input_offset, count), dtype=np.int8
        ).copy().reshape(h, w, 1, 8)
        output = map_i8(source, multipliers, shifts)
        output[..., lanes:] = 0
        self.memory.write(MemorySpace.OUTPUT, output_offset, output.tobytes())

    def _execute_add(self, _command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 8)
        if self._context is not None:
            raise ExecutionFault(ErrorCode.BAD_STATE, "utility operation while accumulator is open")
        a_offset, b_offset, output_offset, h, w, lanes, a_param, b_param = p[:8]
        for value, field in [(a_offset, 4), (b_offset, 5), (output_offset, 6), (a_param, 10), (b_param, 11)]:
            self._require_alignment(value, field)
        self._utility_shape(h, w, lanes)
        bias_a, multiplier_a, shift_a = self._read_parameters(a_param, 10)
        bias_b, multiplier_b, shift_b = self._read_parameters(b_param, 11)
        if np.any(bias_a != 0) or np.any(bias_b != 0):
            raise ExecutionFault(ErrorCode.BAD_FIELD, "ADD parameter bias must be zero")
        self._validate_padding_parameters(lanes, bias_a, multiplier_a, shift_a)
        self._validate_padding_parameters(lanes, bias_b, multiplier_b, shift_b)
        if np.any(shift_a[:lanes] != shift_b[:lanes]):
            raise ExecutionFault(ErrorCode.BAD_FIELD, "ADD inputs must share each lane shift")
        count = h * w * 8
        a = np.frombuffer(self.memory.read(MemorySpace.INPUT, a_offset, count), dtype=np.int8).copy()
        b = np.frombuffer(self.memory.read(MemorySpace.INPUT, b_offset, count), dtype=np.int8).copy()
        output = add_i8(
            a.reshape(h, w, 1, 8),
            b.reshape(h, w, 1, 8),
            multiplier_a,
            multiplier_b,
            shift_a,
        )
        output[..., lanes:] = 0
        self.memory.write(MemorySpace.OUTPUT, output_offset, output.tobytes())

    def _execute_pool(self, _command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 5)
        if self._context is not None:
            raise ExecutionFault(ErrorCode.BAD_STATE, "utility operation while accumulator is open")
        input_offset, output_offset, h, w, lanes = p[:5]
        self._require_alignment(input_offset, 4)
        self._require_alignment(output_offset, 5)
        self._utility_shape(h, w, lanes)
        input_count = (h + 4) * (w + 4) * 8
        source = np.frombuffer(
            self.memory.read(MemorySpace.INPUT, input_offset, input_count), dtype=np.int8
        ).copy().reshape(h + 4, w + 4, 1, 8)
        output = maxpool5_i8(source, lanes)
        self.memory.write(MemorySpace.OUTPUT, output_offset, output.tobytes())

    def _execute_upsample(self, _command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 5)
        if self._context is not None:
            raise ExecutionFault(ErrorCode.BAD_STATE, "utility operation while accumulator is open")
        input_offset, output_offset, h, w, lanes = p[:5]
        self._require_alignment(input_offset, 4)
        self._require_alignment(output_offset, 5)
        self._utility_shape(h, w, lanes)
        source = np.frombuffer(
            self.memory.read(MemorySpace.INPUT, input_offset, h * w * 8), dtype=np.int8
        ).copy().reshape(h, w, 1, 8)
        output = upsample2_nearest_i8(source, lanes)
        self.memory.write(MemorySpace.OUTPUT, output_offset, output.tobytes())

    def _execute_copy(self, _command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 9)
        if self._context is not None:
            raise ExecutionFault(ErrorCode.BAD_STATE, "utility operation while accumulator is open")
        source_space = self._space(p[0], 4)
        destination_space = self._space(p[2], 6)
        if source_space not in {MemorySpace.INPUT, MemorySpace.OUTPUT} or destination_space not in {
            MemorySpace.INPUT,
            MemorySpace.OUTPUT,
        }:
            raise ExecutionFault(ErrorCode.BAD_SPACE, "COPY2D supports INPUT/OUTPUT only")
        source_offset, destination_offset = p[1], p[3]
        row_bytes, rows, source_stride, destination_stride = p[4:8]
        if p[8] != 0:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "COPY2D reserved word must be zero")
        for value, field in [
            (source_offset, 5),
            (destination_offset, 7),
            (row_bytes, 8),
            (source_stride, 10),
            (destination_stride, 11),
        ]:
            self._require_alignment(value, field)
        if row_bytes == 0 or rows == 0 or source_stride < row_bytes or destination_stride < row_bytes:
            raise ExecutionFault(ErrorCode.BAD_FIELD, "invalid COPY2D dimensions or strides")
        self.memory.check(source_space, source_offset + (rows - 1) * source_stride, row_bytes)
        self.memory.check(destination_space, destination_offset + (rows - 1) * destination_stride, row_bytes)
        if source_space is destination_space:
            source_ranges = [
                (source_offset + row * source_stride, source_offset + row * source_stride + row_bytes)
                for row in range(rows)
            ]
            destination_ranges = [
                (
                    destination_offset + row * destination_stride,
                    destination_offset + row * destination_stride + row_bytes,
                )
                for row in range(rows)
            ]
            if any(max(a0, b0) < min(a1, b1) for a0, a1 in source_ranges for b0, b1 in destination_ranges):
                raise ExecutionFault(ErrorCode.BAD_FIELD, "COPY2D source and destination overlap")
        for row in range(rows):
            data = self.memory.read(source_space, source_offset + row * source_stride, row_bytes)
            self.memory.write(destination_space, destination_offset + row * destination_stride, data)

    def _execute_fence(self, _command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 0)

    def _execute_end(self, command: Command, p: tuple[int, ...]) -> None:
        self._require_unused_zero(p, 0)
        if self._context is not None:
            raise ExecutionFault(ErrorCode.BAD_STATE, "END while accumulator context is open")
        self.last_end = command.sequence
        self.state = DeviceState.IDLE
