# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import struct
import unittest

from haslab_sim import ABI_MAJOR, ABI_MINOR, Command, Opcode


class CommandEncodingTests(unittest.TestCase):
    def test_command_is_exactly_128_little_endian_bytes(self) -> None:
        command = Command.build(Opcode.FILL8, 7, [1, 8, 16, 0xAB])
        encoded = command.to_bytes()
        self.assertEqual(len(encoded), 128)
        words = struct.unpack("<32I", encoded)
        self.assertEqual(words[0], (ABI_MAJOR << 24) | (ABI_MINOR << 16) | Opcode.FILL8)
        self.assertEqual(words[1:8], (0, 7, 0, 1, 8, 16, 0xAB))
        self.assertTrue(all(word == 0 for word in words[8:]))

    def test_round_trip_preserves_every_word(self) -> None:
        original = Command.build(Opcode.CONV_I8, 99, range(28), flags=3)
        decoded = Command.from_bytes(original.to_bytes())
        self.assertEqual(decoded.to_words(), original.to_words())

    def test_unknown_opcode_can_be_framed(self) -> None:
        command = Command.build(0xF00D, 1)
        self.assertEqual(Command.from_bytes(command.to_bytes()).opcode, 0xF00D)

    def test_wrong_record_size_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Command.from_bytes(bytes(127))
        with self.assertRaises(ValueError):
            Command.from_bytes(bytes(129))

    def test_oversized_payload_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Command.build(Opcode.FILL8, 1, range(29))


if __name__ == "__main__":
    unittest.main()
