# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import hashlib
import struct
import unittest

from haslab_compiler.hxb import Section, SectionType, build_hxb


class HxbWriterTests(unittest.TestCase):
    def test_writer_is_deterministic_and_uses_documented_header(self) -> None:
        sections = [
            Section(SectionType.CONSTANT_DATA, b"constants"),
            Section(SectionType.MANIFEST_UTF8_JSON, b"{}\n"),
            Section(SectionType.COMMANDS, bytes(128)),
        ]
        first = build_hxb(sections)
        second = build_hxb(list(reversed(sections)))
        self.assertEqual(first, second)
        self.assertEqual(hashlib.sha256(first).digest(), hashlib.sha256(second).digest())
        fields = struct.unpack_from("<8sHHIIIQQI20s", first)
        self.assertEqual(fields[:6], (b"HSLBHXB0", 0, 1, 64, 0, 3))
        self.assertEqual(fields[7], len(first))
        self.assertEqual(fields[8], 0)

    def test_writer_rejects_missing_or_duplicate_required_sections(self) -> None:
        with self.assertRaises(ValueError):
            build_hxb([Section(SectionType.MANIFEST_UTF8_JSON, b"{}")])
        with self.assertRaises(ValueError):
            build_hxb(
                [
                    Section(SectionType.MANIFEST_UTF8_JSON, b"{}"),
                    Section(SectionType.MANIFEST_UTF8_JSON, b"{}"),
                    Section(SectionType.COMMANDS, bytes(128)),
                    Section(SectionType.CONSTANT_DATA, b"x"),
                ]
            )


if __name__ == "__main__":
    unittest.main()
