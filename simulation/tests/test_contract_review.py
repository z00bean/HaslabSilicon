"""Regression evidence for the v0 contract review; not the M4 fixture suite."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import struct
import unittest
from unittest.mock import patch

from haslab_sim import Command, DeviceState, ErrorCode, HaslabDevice, MemorySpace, Opcode, SubmitResult
from haslab_sim.memory import MemoryMap


class ContractReviewTests(unittest.TestCase):
    def test_payload_is_snapshotted_before_submission(self):
        words = [1, 0, 8, 7]
        command = Command(2, 1, words)
        device = HaslabDevice(ext_bytes=64)
        self.assertEqual(device.submit(command), SubmitResult.ACCEPTED)
        words[3] = 99
        self.assertTrue(device.run_next())
        self.assertEqual(device.memory.read(MemorySpace.INPUT, 0, 8), b"\x07" * 8)
        self.assertIsInstance(command.payload, tuple)

    def test_builder_does_not_silently_truncate_noninteger_fields(self):
        for word in (1.5, "1", -1, 1 << 32):
            with self.subTest(word=word), self.assertRaises((ValueError, TypeError)):
                Command.build(Opcode.FILL8, 1, [1, 0, 8, word])
        with self.assertRaises(TypeError):
            Command.build(2.5, 1)

    def test_ext_capacity_must_fit_register_without_allocating_four_gib(self):
        with self.assertRaises(ValueError):
            MemoryMap(1 << 32)
        # Exercise the representable upper boundary without a multi-GiB allocation.
        with patch("haslab_sim.memory.bytearray", return_value=bytearray()) as allocate:
            MemoryMap((1 << 32) - 1)
            self.assertEqual(allocate.call_args_list[0].args, ((1 << 32) - 1,))

    def test_single_row_dma_and_copy_require_canonical_strides(self):
        for opcode, source, destination in (
            (Opcode.DMA_COPY2D, MemorySpace.EXT, MemorySpace.INPUT),
            (Opcode.COPY2D, MemorySpace.INPUT, MemorySpace.OUTPUT),
        ):
            for strides in ((16, 8), (8, 16), (16, 16)):
                with self.subTest(opcode=opcode, strides=strides):
                    device = HaslabDevice(ext_bytes=64)
                    device.memory.write(destination, 0, b"sentinel")
                    device.submit(Command.build(opcode, 1, [source, 0, destination, 0, 8, 1, *strides, 0]))
                    self.assertFalse(device.run_next())
                    self.assertEqual(device.fault.code, ErrorCode.BAD_FIELD)
                    self.assertEqual(device.memory.read(destination, 0, 8), b"sentinel")

    def test_widened_dma_bounds_never_wrap_into_a_valid_address(self):
        device = HaslabDevice(ext_bytes=64)
        device.memory.write(MemorySpace.INPUT, 0, b"sentinel")
        device.submit(Command.build(Opcode.DMA_COPY2D, 1, [0, 0xFFFFFFF8, 1, 0, 8, 2, 8, 8, 0]))
        self.assertFalse(device.run_next())
        self.assertEqual(device.fault.code, ErrorCode.BOUNDS)
        self.assertEqual(device.fault.space, 0)
        self.assertEqual(device.fault.offset, 0)  # Derived 2^32 address cannot fit ERROR_OFFSET.
        self.assertEqual(device.memory.read(MemorySpace.INPUT, 0, 8), b"sentinel")

    def test_unused_payload_words_fault_for_every_opcode(self):
        for opcode in Opcode:
            with self.subTest(opcode=opcode):
                device = HaslabDevice(ext_bytes=64)
                device.submit(Command.build(opcode, 1, [0] * 27 + [1]))
                self.assertFalse(device.run_next())
                self.assertEqual(device.fault.code, ErrorCode.BAD_FIELD)
                self.assertEqual(device.fault.field, 31)

    def test_submission_priority_and_retry_preserve_queue(self):
        device = HaslabDevice(ext_bytes=64)
        for sequence in range(1, 9):
            device.submit(Command.build(Opcode.FENCE, sequence))
        self.assertEqual(device.submit(b"malformed"), SubmitResult.BUSY)
        self.assertTrue(device.run_next())
        self.assertEqual(device.submit(b"malformed"), SubmitResult.INVALID)
        self.assertEqual(device.last_accepted, 8)
        self.assertEqual(device.submit(Command.build(Opcode.END, 9)), SubmitResult.ACCEPTED)
        self.assertEqual(device.run_all(), 8)
        self.assertEqual(device.last_end, 9)

    def test_version_match_is_exact(self):
        for major, minor in ((0, 0), (0, 2), (1, 1)):
            with self.subTest(version=(major, minor)):
                device = HaslabDevice(ext_bytes=64)
                self.assertEqual(device.submit(Command(0x30, 1, abi_major=major, abi_minor=minor)), SubmitResult.INVALID)
                self.assertEqual(device.state, DeviceState.IDLE)
                self.assertEqual(device.last_accepted, 0)

    def test_sequence_exhaustion_requires_reset(self):
        device = HaslabDevice(ext_bytes=64)
        # Testbench injection represents the end of a long session.
        device.last_accepted = 0xFFFFFFFE
        device.submit(Command.build(Opcode.END, 0xFFFFFFFF))
        self.assertTrue(device.run_next())
        for sequence in (0, 1):
            self.assertEqual(device.submit(Command.build(Opcode.FENCE, sequence)), SubmitResult.INVALID)
        device.reset()
        self.assertEqual(device.submit(Command.build(Opcode.END, 1)), SubmitResult.ACCEPTED)

    def test_fault_is_sticky_and_later_work_has_no_effect(self):
        device = HaslabDevice(ext_bytes=64)
        device.memory.write(MemorySpace.INPUT, 0, b"sentinel")
        for command in (Command.build(Opcode.FENCE, 1), Command.build(0xFFFF, 2),
                        Command.build(Opcode.FILL8, 3, [1, 0, 8, 7])):
            device.submit(command)
        self.assertEqual(device.run_all(), 1)
        fault = device.fault
        self.assertEqual((fault.code, fault.sequence), (ErrorCode.BAD_OPCODE, 2))
        self.assertEqual((device.last_accepted, device.last_completed), (3, 1))
        self.assertFalse(device.run_next())
        self.assertEqual(device.submit(b"malformed"), SubmitResult.BLOCKED)
        self.assertEqual(device.fault, fault)
        self.assertEqual(device.memory.read(MemorySpace.INPUT, 0, 8), b"sentinel")
        device.reset()
        self.assertEqual(device.fifo_free, 8)
        self.assertEqual(device.fault.code, ErrorCode.NONE)

    def test_reset_generation_wrap_is_modulo_u32(self):
        device = HaslabDevice(ext_bytes=64)
        device.reset_generation = 0xFFFFFFFF
        device.reset()
        self.assertEqual(device.reset_generation, 0)
        self.assertEqual(device.state, DeviceState.IDLE)

    def test_copy_allows_disjoint_rows_with_overlapping_envelopes(self):
        device = HaslabDevice(ext_bytes=64)
        device.memory.write(MemorySpace.INPUT, 0, b"abcdefgh........ijklmnop........")
        device.submit(Command.build(Opcode.COPY2D, 1, [1, 0, 1, 8, 8, 2, 16, 16, 0]))
        self.assertTrue(device.run_next())
        self.assertEqual(device.memory.read(MemorySpace.INPUT, 0, 32), b"abcdefghabcdefghijklmnopijklmnop")

    def test_copy_rejects_cross_row_overlap_before_any_write(self):
        device = HaslabDevice(ext_bytes=64)
        original = bytes(range(64))
        device.memory.write(MemorySpace.INPUT, 0, original)
        device.submit(Command.build(Opcode.COPY2D, 1, [1, 0, 1, 16, 8, 2, 16, 16, 0]))
        self.assertFalse(device.run_next())
        self.assertEqual(device.fault.code, ErrorCode.BAD_FIELD)
        self.assertEqual(device.memory.read(MemorySpace.INPUT, 0, 64), original)

    def test_epilogue_preflight_precedes_arithmetic_overflow(self):
        for mode, output_offset, lut_offset, error in (
            (0, 8192, 0, ErrorCode.BOUNDS),
            (2, 0, 1, ErrorCode.ALIGNMENT),
            (2, 0, 0, ErrorCode.BAD_FIELD),
            (2, 0, 3072, ErrorCode.BOUNDS),
        ):
            with self.subTest(mode=mode, output=output_offset, lut=lut_offset):
                device = HaslabDevice(ext_bytes=64)
                device.memory.write(MemorySpace.INPUT, 0, b"\x01" + bytes(7))
                device.memory.write(MemorySpace.WEIGHT, 0, b"\x01" + bytes(63))
                device.memory.write(MemorySpace.PARAM, 0, struct.pack("<iIII", 0x7FFFFFFF, 0, 0, 0) + bytes(112))
                device.memory.write(MemorySpace.OUTPUT, 0, b"sentinel")
                device.submit(Command.build(Opcode.CONV_I8, 1, [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1], flags=3))
                self.assertTrue(device.run_next())
                device.submit(Command.build(Opcode.EPILOGUE, 2, [0, output_offset, 0, mode, lut_offset]))
                self.assertFalse(device.run_next())
                self.assertEqual(device.fault.code, error)
                self.assertEqual(device.last_completed, 1)
                self.assertEqual(device.memory.read(MemorySpace.OUTPUT, 0, 8), b"sentinel")

    def test_utility_output_bounds_are_checked_before_compute(self):
        for opcode, payload, helper in (
            (Opcode.MAP_I8, [0, 8192, 1, 1, 1, 0], "map_i8"),
            (Opcode.ADD_I8, [0, 0, 8192, 1, 1, 1, 0, 0], "add_i8"),
            (Opcode.MAXPOOL5_I8, [0, 8192, 1, 1, 1], "maxpool5_i8"),
            (Opcode.UPSAMPLE2_I8, [0, 8192, 1, 1, 1], "upsample2_nearest_i8"),
        ):
            with self.subTest(opcode=opcode):
                device = HaslabDevice(ext_bytes=64)
                device.memory.write(MemorySpace.INPUT, 0, bytes(200))
                device.memory.write(MemorySpace.PARAM, 0, bytes(128))
                device.submit(Command.build(opcode, 1, payload))
                with patch("haslab_sim.device." + helper) as compute:
                    self.assertFalse(device.run_next())
                    compute.assert_not_called()
                self.assertEqual(device.fault.code, ErrorCode.BOUNDS)
                self.assertEqual(device.fault.space, MemorySpace.OUTPUT)

    def test_fill_and_fence_are_allowed_with_open_accumulator(self):
        device = HaslabDevice(ext_bytes=64)
        device.memory.write(MemorySpace.INPUT, 0, bytes(8))
        device.memory.write(MemorySpace.WEIGHT, 0, bytes(64))
        commands = (
            Command.build(Opcode.CONV_I8, 1, [0, 0, 0, 1, 1, 1, 8, 0, 16, 1, 1], flags=1),
            Command.build(Opcode.FILL8, 2, [1, 0, 8, 1]),
            Command.build(Opcode.FENCE, 3),
        )
        for command in commands:
            device.submit(command)
            self.assertTrue(device.run_next())
            self.assertTrue(device.accumulator_open)
        self.assertEqual(device.last_completed, 3)
        self.assertEqual(device.memory.read(MemorySpace.INPUT, 0, 8), b"\x01" * 8)


if __name__ == "__main__":
    unittest.main()
