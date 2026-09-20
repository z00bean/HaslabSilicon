# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

from haslab_sim import (
    Command,
    DeviceState,
    ErrorCode,
    HaslabDevice,
    MemorySpace,
    Opcode,
    SubmitResult,
)


class SubmissionTests(unittest.TestCase):
    def test_sequences_are_contiguous_and_versioned(self) -> None:
        device = HaslabDevice()
        self.assertEqual(device.submit(Command.build(Opcode.FENCE, 1)), SubmitResult.ACCEPTED)
        self.assertEqual(device.submit(Command.build(Opcode.FENCE, 3)), SubmitResult.INVALID)
        self.assertEqual(
            device.submit(Command(Opcode.FENCE, 2, abi_minor=2)), SubmitResult.INVALID
        )
        self.assertEqual(device.last_accepted, 1)

    def test_fifo_has_eight_atomic_slots(self) -> None:
        device = HaslabDevice()
        for sequence in range(1, 9):
            self.assertEqual(
                device.submit(Command.build(Opcode.FENCE, sequence)), SubmitResult.ACCEPTED
            )
        self.assertEqual(device.fifo_free, 0)
        self.assertEqual(device.submit(Command.build(Opcode.FENCE, 9)), SubmitResult.BUSY)
        self.assertEqual(device.last_accepted, 8)

    def test_invalid_flags_rejected_before_enqueue(self) -> None:
        device = HaslabDevice()
        self.assertEqual(
            device.submit(Command.build(Opcode.FILL8, 1, flags=1)), SubmitResult.INVALID
        )
        self.assertEqual(device.queued_commands, 0)

    def test_unknown_opcode_faults_at_execution_and_blocks_submission(self) -> None:
        device = HaslabDevice()
        self.assertEqual(device.submit(Command.build(0xF00D, 1)), SubmitResult.ACCEPTED)
        self.assertFalse(device.run_next())
        self.assertEqual(device.state, DeviceState.FAULT)
        self.assertEqual(device.fault.code, ErrorCode.BAD_OPCODE)
        self.assertEqual(device.fault.sequence, 1)
        self.assertEqual(device.last_completed, 0)
        self.assertEqual(device.submit(Command.build(Opcode.FENCE, 2)), SubmitResult.BLOCKED)

    def test_end_marks_success_and_reset_advances_generation(self) -> None:
        device = HaslabDevice(ext_bytes=256)
        device.memory.write(MemorySpace.EXT, 0, b"persistent")
        device.memory.write(MemorySpace.INPUT, 0, b"local!!!")
        self.assertEqual(device.submit(Command.build(Opcode.END, 1)), SubmitResult.ACCEPTED)
        self.assertEqual(device.run_all(), 1)
        self.assertEqual(device.state, DeviceState.IDLE)
        self.assertEqual(device.last_end, 1)
        device.reset()
        self.assertEqual(device.reset_generation, 1)
        self.assertEqual(device.last_end, 0)
        self.assertEqual(device.memory.read(MemorySpace.EXT, 0, 10), b"persistent")
        self.assertEqual(device.memory.read(MemorySpace.INPUT, 0, 8), bytes(8))

    def test_queued_command_after_end_starts_new_running_interval(self) -> None:
        device = HaslabDevice()
        device.submit(Command.build(Opcode.END, 1))
        device.submit(Command.build(Opcode.FILL8, 2, [MemorySpace.INPUT, 0, 8, 7]))
        self.assertTrue(device.run_next())
        self.assertEqual(device.state, DeviceState.IDLE)
        self.assertTrue(device.run_next())
        self.assertEqual(device.state, DeviceState.RUNNING)
        self.assertEqual(device.memory.read(MemorySpace.INPUT, 0, 8), bytes([7]) * 8)

    def test_end_rejects_open_accumulator(self) -> None:
        device = HaslabDevice()
        activation = bytes([1] + [0] * 7)
        weights = bytes([1] + [0] * 63)
        device.memory.write(MemorySpace.INPUT, 0, activation)
        device.memory.write(MemorySpace.WEIGHT, 0, weights)
        conv = Command.build(
            Opcode.CONV_I8,
            1,
            [0, 0, 0, 1, 1, 1, 8, 0, 9, 1, 1],
            flags=1,
        )
        self.assertEqual(device.submit(conv), SubmitResult.ACCEPTED)
        self.assertTrue(device.run_next())
        self.assertTrue(device.accumulator_open)
        self.assertEqual(device.submit(Command.build(Opcode.END, 2)), SubmitResult.ACCEPTED)
        self.assertFalse(device.run_next())
        self.assertEqual(device.fault.code, ErrorCode.BAD_STATE)


if __name__ == "__main__":
    unittest.main()
