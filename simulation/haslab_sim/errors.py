"""Architectural execution-fault records."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ErrorCode(IntEnum):
    NONE = 0
    BAD_OPCODE = 1
    BAD_FIELD = 2
    ALIGNMENT = 3
    BOUNDS = 4
    BAD_SPACE = 5
    BAD_STATE = 6
    ARITH_OVERFLOW = 7
    DMA_BUS = 8
    INTERNAL = 9


@dataclass(frozen=True)
class FaultRecord:
    code: ErrorCode = ErrorCode.NONE
    sequence: int = 0
    space: int = 0xFFFFFFFF
    offset: int = 0
    field: int = 0xFFFFFFFF
    message: str = ""


class ExecutionFault(Exception):
    """Internal control-flow exception carrying an architectural fault."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        space: int = 0xFFFFFFFF,
        offset: int = 0,
        field: int = 0xFFFFFFFF,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.space = space
        self.offset = offset
        self.field = field
