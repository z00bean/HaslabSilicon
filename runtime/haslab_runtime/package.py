"""Strict reader for the proposed HASLAB v0 .hxb container."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass

MAGIC = b"HSLBHXB0"
HEADER_BYTES = 64
ENTRY_BYTES = 64
ALIGNMENT = 64
MAX_PACKAGE_BYTES = 64 << 20
KNOWN_TYPES = {1, 2, 3, 4, 5}


class HxbFormatError(ValueError):
    """The package is malformed, corrupt, or incompatible."""


@dataclass(frozen=True)
class HxbPackage:
    manifest: dict[str, object]
    commands: bytes
    constants: bytes
    debug: dict[str, object] | None
    raw_sha256: str


def _object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HxbFormatError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HxbFormatError(f"nonfinite JSON number in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HxbFormatError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise HxbFormatError(f"{label} JSON must be an object")
    return value


def load_hxb(source: bytes | bytearray | memoryview) -> HxbPackage:
    """Validate all framing, hashes, padding, and required package sections."""

    data = bytes(source)
    if len(data) < HEADER_BYTES or len(data) > MAX_PACKAGE_BYTES:
        raise HxbFormatError("package size is outside the runtime limit")
    try:
        magic, major, minor, header_bytes, flags, count, table_offset, total, manifest_index, reserved = struct.unpack_from(
            "<8sHHIIIQQI20s", data, 0
        )
    except struct.error as exc:
        raise HxbFormatError("truncated HXB header") from exc
    if magic != MAGIC or (major, minor) != (0, 1):
        raise HxbFormatError("unsupported HXB magic or format version")
    if header_bytes != HEADER_BYTES or flags != 0 or any(reserved):
        raise HxbFormatError("invalid HXB header fields")
    if total != len(data) or count < 3 or count > 5 or manifest_index >= count:
        raise HxbFormatError("invalid HXB size or section count")
    if table_offset % ALIGNMENT or table_offset < HEADER_BYTES:
        raise HxbFormatError("section table is not aligned")
    table_end = table_offset + count * ENTRY_BYTES
    if table_end != len(data):
        raise HxbFormatError("section table must terminate the package")

    sections: dict[int, bytes] = {}
    table_kinds: list[int] = []
    ranges: list[tuple[int, int]] = [(0, HEADER_BYTES), (table_offset, table_end)]
    for index in range(count):
        try:
            kind, section_flags, offset, length, digest, entry_reserved = struct.unpack_from(
                "<IIQQ32s8s", data, table_offset + index * ENTRY_BYTES
            )
        except struct.error as exc:
            raise HxbFormatError("truncated section table") from exc
        if kind not in KNOWN_TYPES or kind in sections or section_flags != 0 or any(entry_reserved):
            raise HxbFormatError("unknown, duplicate, or flagged HXB section")
        if offset % ALIGNMENT or offset < HEADER_BYTES or offset + length > table_offset:
            raise HxbFormatError("invalid HXB section bounds")
        current = (offset, offset + length)
        if any(max(current[0], start) < min(current[1], end) for start, end in ranges):
            raise HxbFormatError("overlapping HXB sections")
        payload = data[offset : offset + length]
        if hashlib.sha256(payload).digest() != digest:
            raise HxbFormatError("HXB section hash mismatch")
        sections[kind] = payload
        table_kinds.append(kind)
        ranges.append(current)
    if set((1, 2, 3)) - sections.keys() or table_kinds[manifest_index] != 1:
        raise HxbFormatError("required HXB section or manifest index is invalid")

    occupied = sorted(ranges)
    for (_, end), (start, _) in zip(occupied, occupied[1:]):
        if any(data[end:start]):
            raise HxbFormatError("HXB alignment padding must be zero")
    manifest = _decode_json(sections[1], "manifest")
    debug = _decode_json(sections[5], "debug") if 5 in sections else None
    return HxbPackage(
        manifest=manifest,
        commands=sections[2],
        constants=sections[3],
        debug=debug,
        raw_sha256=hashlib.sha256(data).hexdigest(),
    )
