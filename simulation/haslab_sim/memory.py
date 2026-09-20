"""Bounded byte-addressed memories used by the functional simulator."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

from collections.abc import Mapping

from .abi import MemorySpace
from .errors import ErrorCode, ExecutionFault

LOCAL_CAPACITIES: Mapping[MemorySpace, int] = {
    MemorySpace.INPUT: 16_384,
    MemorySpace.WEIGHT: 16_384,
    MemorySpace.ACC: 8_192,
    MemorySpace.OUTPUT: 8_192,
    MemorySpace.PARAM: 3_072,
}


class MemoryMap:
    def __init__(self, ext_bytes: int) -> None:
        if not isinstance(ext_bytes, int) or not 0 < ext_bytes <= (1 << 32):
            raise ValueError("EXT capacity must be in the range 1..2^32")
        self._regions: dict[MemorySpace, bytearray] = {
            MemorySpace.EXT: bytearray(ext_bytes),
            **{space: bytearray(size) for space, size in LOCAL_CAPACITIES.items()},
        }

    def capacity(self, space: MemorySpace | int) -> int:
        return len(self._region(space))

    def _region(self, space: MemorySpace | int) -> bytearray:
        try:
            resolved = MemorySpace(space)
        except ValueError as exc:
            raise ExecutionFault(ErrorCode.BAD_SPACE, "unknown memory space", space=int(space)) from exc
        return self._regions[resolved]

    def check(self, space: MemorySpace | int, offset: int, length: int) -> None:
        region = self._region(space)
        if offset < 0 or length < 0 or offset + length > len(region):
            raise ExecutionFault(
                ErrorCode.BOUNDS,
                "memory access is outside its region",
                space=int(space),
                offset=max(0, offset),
            )

    def read(self, space: MemorySpace | int, offset: int, length: int) -> bytes:
        self.check(space, offset, length)
        region = self._region(space)
        return bytes(region[offset : offset + length])

    def write(self, space: MemorySpace | int, offset: int, data: bytes | bytearray | memoryview) -> None:
        raw = bytes(data)
        self.check(space, offset, len(raw))
        region = self._region(space)
        region[offset : offset + len(raw)] = raw

    def fill(self, space: MemorySpace | int, offset: int, length: int, value: int) -> None:
        self.check(space, offset, length)
        self._region(space)[offset : offset + length] = bytes([value]) * length

    def clear_local(self) -> None:
        for space in LOCAL_CAPACITIES:
            self._regions[space][:] = bytes(len(self._regions[space]))
