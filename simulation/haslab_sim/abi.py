"""HASLAB v0 command encoding."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

ABI_MAJOR = 0
ABI_MINOR = 1
COMMAND_WORDS = 32
PAYLOAD_WORDS = 28
COMMAND_BYTES = COMMAND_WORDS * 4
UINT32_MAX = (1 << 32) - 1


class MemorySpace(IntEnum):
    EXT = 0
    INPUT = 1
    WEIGHT = 2
    ACC = 3
    OUTPUT = 4
    PARAM = 5


class Opcode(IntEnum):
    DMA_COPY2D = 0x0001
    FILL8 = 0x0002
    CONV_I8 = 0x0010
    EPILOGUE = 0x0011
    MAP_I8 = 0x0020
    ADD_I8 = 0x0021
    MAXPOOL5_I8 = 0x0022
    UPSAMPLE2_I8 = 0x0023
    COPY2D = 0x0024
    FENCE = 0x0030
    END = 0x0031


@dataclass(frozen=True)
class Command:
    """One 128-byte v0 command record.

    Opcode remains an integer so an unknown opcode can pass framing and fault
    at execution, as required by the contract.
    """

    opcode: int
    sequence: int
    payload: tuple[int, ...] = ()
    flags: int = 0
    abi_major: int = ABI_MAJOR
    abi_minor: int = ABI_MINOR
    reserved: int = 0

    def __post_init__(self) -> None:
        scalar_fields = {
            "opcode": self.opcode,
            "sequence": self.sequence,
            "flags": self.flags,
            "abi_major": self.abi_major,
            "abi_minor": self.abi_minor,
            "reserved": self.reserved,
        }
        for name, value in scalar_fields.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= self.opcode <= 0xFFFF:
            raise ValueError("opcode must fit 16 bits")
        if not 0 <= self.flags <= UINT32_MAX:
            raise ValueError("flags must fit u32")
        if not 0 <= self.sequence <= UINT32_MAX:
            raise ValueError("sequence must fit u32")
        if not 0 <= self.abi_major <= 0xFF or not 0 <= self.abi_minor <= 0xFF:
            raise ValueError("ABI major/minor must fit eight bits")
        if not 0 <= self.reserved <= UINT32_MAX:
            raise ValueError("reserved word must fit u32")
        if len(self.payload) > PAYLOAD_WORDS:
            raise ValueError(f"payload may contain at most {PAYLOAD_WORDS} words")
        for value in self.payload:
            if not isinstance(value, int) or not 0 <= value <= UINT32_MAX:
                raise ValueError("payload words must be u32 integers")

    @classmethod
    def build(
        cls,
        opcode: Opcode | int,
        sequence: int,
        payload: Iterable[int] = (),
        *,
        flags: int = 0,
    ) -> "Command":
        return cls(int(opcode), sequence, tuple(int(word) for word in payload), flags)

    @property
    def padded_payload(self) -> tuple[int, ...]:
        return self.payload + (0,) * (PAYLOAD_WORDS - len(self.payload))

    def to_words(self) -> tuple[int, ...]:
        header = (self.abi_major << 24) | (self.abi_minor << 16) | self.opcode
        return (header, self.flags, self.sequence, self.reserved, *self.padded_payload)

    def to_bytes(self) -> bytes:
        return struct.pack("<32I", *self.to_words())

    @classmethod
    def from_bytes(cls, record: bytes | bytearray | memoryview) -> "Command":
        data = bytes(record)
        if len(data) != COMMAND_BYTES:
            raise ValueError(f"command record must contain exactly {COMMAND_BYTES} bytes")
        words = struct.unpack("<32I", data)
        header = words[0]
        return cls(
            opcode=header & 0xFFFF,
            sequence=words[2],
            payload=tuple(words[4:]),
            flags=words[1],
            abi_major=(header >> 24) & 0xFF,
            abi_minor=(header >> 16) & 0xFF,
            reserved=words[3],
        )
