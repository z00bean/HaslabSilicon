"""Application-facing lifecycle for executing an HXB on the functional simulator."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum

from haslab_sim import Command, DeviceState, HaslabDevice, MemorySpace, Opcode, SubmitResult

from .package import HxbPackage, load_hxb


class RuntimeErrorCode(str, Enum):
    BAD_PACKAGE = "BAD_PACKAGE"
    BAD_BINDING = "BAD_BINDING"
    BAD_STATE = "BAD_STATE"
    DEVICE_FAULT = "DEVICE_FAULT"
    STALE_TOKEN = "STALE_TOKEN"


class RuntimeFailure(RuntimeError):
    def __init__(self, code: RuntimeErrorCode, message: str):
        super().__init__(message)
        self.code = code


class RuntimeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class SubmissionToken:
    session: int
    reset_generation: int
    final_sequence: int


@dataclass(frozen=True)
class Completion:
    status: RuntimeStatus
    output: bytes | None
    error: dict[str, object] | None


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class SimulatorRuntime:
    """One-model synchronous runtime using the byte-level simulator backend."""

    def __init__(self, ext_bytes: int = 1 << 20) -> None:
        self.device = HaslabDevice(ext_bytes)
        self._package: HxbPackage | None = None
        self._bases: dict[str, int] = {}
        self._bound = False
        self._session = 1
        self._completion: Completion | None = None

    @property
    def package(self) -> HxbPackage | None:
        return self._package

    def load_model(self, source: bytes | bytearray | memoryview) -> HxbPackage:
        try:
            package = load_hxb(source)
            manifest = package.manifest
            if manifest.get("schema") != "haslab.vertical-slice.v1":
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "unsupported manifest schema")
            if manifest.get("abi") != {"major": 0, "minor": 1}:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "package ABI is incompatible")
            commands = manifest.get("commands")
            if not isinstance(commands, dict) or commands.get("record_bytes") != 128:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "invalid command metadata")
            if commands.get("bytes") != len(package.commands) or len(package.commands) % 128:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "command section length mismatch")
            if commands.get("count") != len(package.commands) // 128:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "command count mismatch")
            buffers = manifest.get("buffers")
            if not isinstance(buffers, dict) or set(buffers) != {"input", "constants", "output"}:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "invalid external buffer declaration")
            cursor = 0
            bases: dict[str, int] = {}
            for name in ("input", "constants", "output"):
                record = buffers[name]
                if not isinstance(record, dict):
                    raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "invalid buffer record")
                size, alignment = record.get("bytes"), record.get("alignment")
                if not isinstance(size, int) or size <= 0 or not isinstance(alignment, int) or alignment < 8 or alignment & (alignment - 1):
                    raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "invalid buffer size or alignment")
                cursor = _align(cursor, alignment)
                bases[name] = cursor
                cursor += size
            if cursor > self.device.memory.capacity(MemorySpace.EXT):
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "package exceeds EXT capacity")
            if buffers["constants"]["bytes"] != len(package.constants):
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "constant section length mismatch")
            self._validate_commands_and_relocations(package, bases)
        except RuntimeFailure:
            raise
        except Exception as exc:
            raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, str(exc)) from exc

        self._package = package
        self._bases = bases
        self._bound = False
        self._completion = None
        self.device.memory.write(MemorySpace.EXT, bases["constants"], package.constants)
        return package

    def _validate_commands_and_relocations(
        self, package: HxbPackage, bases: dict[str, int]
    ) -> None:
        commands = [
            Command.from_bytes(package.commands[offset : offset + 128])
            for offset in range(0, len(package.commands), 128)
        ]
        if [command.sequence for command in commands] != list(range(1, len(commands) + 1)):
            raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "relative sequences are not contiguous")
        relocations = package.manifest.get("relocations")
        if not isinstance(relocations, list):
            raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "relocations must be a list")
        seen: set[tuple[int, int]] = set()
        buffers = package.manifest["buffers"]
        for relocation in relocations:
            if not isinstance(relocation, dict):
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "invalid relocation")
            try:
                index = relocation["command_index"]
                word = relocation["payload_word"]
                name = relocation["buffer"]
                addend = relocation["addend"]
            except KeyError as exc:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "incomplete relocation") from exc
            if not all(isinstance(value, int) for value in (index, word, addend)) or name not in bases:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "invalid relocation fields")
            if (index, word) in seen or not 0 <= index < len(commands) or word not in (1, 3):
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "duplicate or invalid relocation target")
            seen.add((index, word))
            command = commands[index]
            payload = command.padded_payload
            if command.opcode != Opcode.DMA_COPY2D:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "relocation does not target DMA")
            expected_word = 1 if payload[0] == MemorySpace.EXT else 3 if payload[2] == MemorySpace.EXT else -1
            if word != expected_word or payload[word] != 0:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "relocation does not target an EXT placeholder")
            row_bytes, rows = payload[4], payload[5]
            stride = payload[6] if word == 1 else payload[7]
            extent = (rows - 1) * stride + row_bytes
            if addend < 0 or addend + extent > buffers[name]["bytes"]:
                raise RuntimeFailure(RuntimeErrorCode.BAD_PACKAGE, "relocation exceeds its declared buffer")

    def bind_input(self, data: bytes | bytearray | memoryview) -> None:
        if self._package is None:
            raise RuntimeFailure(RuntimeErrorCode.BAD_STATE, "load a model before binding input")
        raw = bytes(data)
        expected = self._package.manifest["buffers"]["input"]["bytes"]
        if len(raw) != expected:
            raise RuntimeFailure(
                RuntimeErrorCode.BAD_BINDING,
                f"input binding has {len(raw)} bytes; expected {expected}",
            )
        self.device.memory.write(MemorySpace.EXT, self._bases["input"], raw)
        self._bound = True

    def _relocated_commands(self) -> list[Command]:
        assert self._package is not None
        records = [
            bytearray(self._package.commands[offset : offset + 128])
            for offset in range(0, len(self._package.commands), 128)
        ]
        for relocation in self._package.manifest["relocations"]:
            value = self._bases[relocation["buffer"]] + relocation["addend"]
            struct.pack_into("<I", records[relocation["command_index"]], 16 + 4 * relocation["payload_word"], value)
        base_sequence = self.device.last_accepted
        for index, record in enumerate(records):
            struct.pack_into("<I", record, 8, base_sequence + index + 1)
        return [Command.from_bytes(record) for record in records]

    def submit(self) -> SubmissionToken:
        if self._package is None or not self._bound:
            raise RuntimeFailure(RuntimeErrorCode.BAD_STATE, "model and exact input binding are required")
        if self.device.state is DeviceState.FAULT:
            raise RuntimeFailure(RuntimeErrorCode.BAD_STATE, "reset the faulted device before submit")
        output_size = self._package.manifest["buffers"]["output"]["bytes"]
        self.device.memory.fill(MemorySpace.EXT, self._bases["output"], output_size, 0)
        commands = self._relocated_commands()
        for command in commands:
            while self.device.submit(command) is SubmitResult.BUSY:
                self.device.run_next()
            if self.device.submit_result is not SubmitResult.ACCEPTED:
                raise RuntimeFailure(RuntimeErrorCode.BAD_STATE, "device rejected a validated command")
        self.device.run_all()
        final_sequence = commands[-1].sequence
        token = SubmissionToken(self._session, self.device.reset_generation, final_sequence)
        if self.device.state is DeviceState.FAULT:
            fault = self.device.fault
            self._completion = Completion(
                RuntimeStatus.FAILURE,
                None,
                {
                    "runtime_code": RuntimeErrorCode.DEVICE_FAULT.value,
                    "device_code": int(fault.code),
                    "sequence": fault.sequence,
                    "space": fault.space,
                    "offset": fault.offset,
                    "field": fault.field,
                    "message": fault.message,
                },
            )
        else:
            output = self.device.memory.read(MemorySpace.EXT, self._bases["output"], output_size)
            self._completion = Completion(RuntimeStatus.SUCCESS, output, None)
        return token

    def wait(self, token: SubmissionToken) -> Completion:
        if token.session != self._session or token.reset_generation != self.device.reset_generation:
            raise RuntimeFailure(RuntimeErrorCode.STALE_TOKEN, "submission token is stale")
        if self._completion is None:
            raise RuntimeFailure(RuntimeErrorCode.BAD_STATE, "submission has no matching completion")
        if (
            self._completion.status is RuntimeStatus.SUCCESS
            and token.final_sequence != self.device.last_completed
        ):
            raise RuntimeFailure(RuntimeErrorCode.BAD_STATE, "submission has no matching completion")
        return self._completion

    def reset(self) -> None:
        self.device.reset()
        self._bound = False
        self._completion = None
        self._session += 1
        if self._package is not None:
            self.device.memory.write(MemorySpace.EXT, self._bases["constants"], self._package.constants)
