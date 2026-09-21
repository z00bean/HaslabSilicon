"""Deterministic writer for the proposed HASLAB v0 .hxb container."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import IntEnum

MAGIC = b"HSLBHXB0"
FORMAT_MAJOR = 0
FORMAT_MINOR = 1
HEADER_BYTES = 64
SECTION_ENTRY_BYTES = 64
ALIGNMENT = 64


class SectionType(IntEnum):
    MANIFEST_UTF8_JSON = 1
    COMMANDS = 2
    CONSTANT_DATA = 3
    HOST_TAIL_ONNX = 4
    DEBUG_UTF8_JSON = 5


@dataclass(frozen=True)
class Section:
    kind: SectionType
    data: bytes


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def build_hxb(sections: list[Section]) -> bytes:
    """Serialize sections with canonical ordering, padding, and SHA-256 entries."""

    if not sections:
        raise ValueError("an HXB package requires sections")
    kinds = [section.kind for section in sections]
    required = {
        SectionType.MANIFEST_UTF8_JSON,
        SectionType.COMMANDS,
        SectionType.CONSTANT_DATA,
    }
    if not required.issubset(kinds) or len(kinds) != len(set(kinds)):
        raise ValueError("HXB requires one manifest, commands, and constants section")
    canonical = sorted(sections, key=lambda section: int(section.kind))
    manifest_index = next(
        index for index, section in enumerate(canonical)
        if section.kind is SectionType.MANIFEST_UTF8_JSON
    )

    image = bytearray(HEADER_BYTES)
    entries: list[tuple[Section, int]] = []
    for section in canonical:
        offset = _align(len(image))
        image.extend(bytes(offset - len(image)))
        raw = bytes(section.data)
        entries.append((Section(section.kind, raw), offset))
        image.extend(raw)

    table_offset = _align(len(image))
    image.extend(bytes(table_offset - len(image)))
    for section, offset in entries:
        image.extend(
            struct.pack(
                "<IIQQ32s8s",
                int(section.kind),
                0,
                offset,
                len(section.data),
                hashlib.sha256(section.data).digest(),
                bytes(8),
            )
        )

    total_bytes = len(image)
    header = struct.pack(
        "<8sHHIIIQQI20s",
        MAGIC,
        FORMAT_MAJOR,
        FORMAT_MINOR,
        HEADER_BYTES,
        0,
        len(entries),
        table_offset,
        total_bytes,
        manifest_index,
        bytes(20),
    )
    image[:HEADER_BYTES] = header
    return bytes(image)
