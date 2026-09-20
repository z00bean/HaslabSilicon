# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import unittest

from haslab_sim import Command, DeviceState, ErrorCode, HaslabDevice, MemorySpace, Opcode


class MemoryCommandTests(unittest.TestCase):
    def _run(self, device: HaslabDevice, command: Command) -> None:
        self.assertEqual(int(device.submit(command)), 0)
        self.assertTrue(device.run_next())

    def test_dma_two_dimensional_copy_in_both_directions(self) -> None:
        device = HaslabDevice(ext_bytes=256)
        device.memory.write(MemorySpace.EXT, 0, bytes(range(64)))
        self._run(
            device,
            Command.build(
                Opcode.DMA_COPY2D,
                1,
                [MemorySpace.EXT, 0, MemorySpace.INPUT, 0, 8, 2, 16, 8, 0],
            ),
        )
        self.assertEqual(device.memory.read(MemorySpace.INPUT, 0, 16), bytes(range(8)) + bytes(range(16, 24)))
        self._run(
            device,
            Command.build(
                Opcode.DMA_COPY2D,
                2,
                [MemorySpace.INPUT, 0, MemorySpace.EXT, 128, 8, 2, 8, 16, 0],
            ),
        )
        self.assertEqual(device.memory.read(MemorySpace.EXT, 128, 8), bytes(range(8)))
        self.assertEqual(device.memory.read(MemorySpace.EXT, 144, 8), bytes(range(16, 24)))

    def test_fill_and_local_copy(self) -> None:
        device = HaslabDevice()
        self._run(device, Command.build(Opcode.FILL8, 1, [MemorySpace.INPUT, 0, 16, 0xA5]))
        self._run(
            device,
            Command.build(
                Opcode.COPY2D,
                2,
                [MemorySpace.INPUT, 0, MemorySpace.OUTPUT, 0, 8, 2, 8, 8, 0],
            ),
        )
        self.assertEqual(device.memory.read(MemorySpace.OUTPUT, 0, 16), bytes([0xA5]) * 16)

    def test_unaligned_dma_faults_without_completion(self) -> None:
        device = HaslabDevice()
        command = Command.build(
            Opcode.DMA_COPY2D,
            1,
            [MemorySpace.EXT, 1, MemorySpace.INPUT, 0, 8, 1, 8, 8, 0],
        )
        device.submit(command)
        self.assertFalse(device.run_next())
        self.assertEqual(device.state, DeviceState.FAULT)
        self.assertEqual(device.fault.code, ErrorCode.ALIGNMENT)
        self.assertEqual(device.last_completed, 0)

    def test_dma_illegal_endpoints_fault(self) -> None:
        device = HaslabDevice()
        command = Command.build(
            Opcode.DMA_COPY2D,
            1,
            [MemorySpace.INPUT, 0, MemorySpace.OUTPUT, 0, 8, 1, 8, 8, 0],
        )
        device.submit(command)
        self.assertFalse(device.run_next())
        self.assertEqual(device.fault.code, ErrorCode.BAD_SPACE)

    def test_overlapping_local_copy_faults(self) -> None:
        device = HaslabDevice()
        command = Command.build(
            Opcode.COPY2D,
            1,
            [MemorySpace.INPUT, 0, MemorySpace.INPUT, 8, 16, 1, 16, 16, 0],
        )
        device.submit(command)
        self.assertFalse(device.run_next())
        self.assertEqual(device.fault.code, ErrorCode.BAD_FIELD)

    def test_bounds_error_reports_space_and_offset(self) -> None:
        device = HaslabDevice(ext_bytes=64)
        command = Command.build(
            Opcode.DMA_COPY2D,
            1,
            [MemorySpace.EXT, 64, MemorySpace.INPUT, 0, 8, 1, 8, 8, 0],
        )
        device.submit(command)
        self.assertFalse(device.run_next())
        self.assertEqual(device.fault.code, ErrorCode.BOUNDS)
        self.assertEqual(device.fault.space, MemorySpace.EXT)
        self.assertEqual(device.fault.offset, 64)


if __name__ == "__main__":
    unittest.main()
