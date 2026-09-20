# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import struct
import unittest

import numpy as np

from haslab_sim import Command, ErrorCode, HaslabDevice, MemorySpace, Opcode


def parameter_records(
    bias: list[int] | None = None,
    multiplier: list[int] | None = None,
    shift: list[int] | None = None,
) -> bytes:
    bias = bias or [0] * 8
    multiplier = multiplier or [0] * 8
    shift = shift or [0] * 8
    return b"".join(
        struct.pack("<iIII", bias[lane], multiplier[lane], shift[lane], 0)
        for lane in range(8)
    )


class ComputeCommandTests(unittest.TestCase):
    def _submit_run(self, device: HaslabDevice, command: Command) -> None:
        self.assertEqual(int(device.submit(command)), 0)
        self.assertTrue(device.run_next(), device.fault)

    def test_single_chunk_convolution_raw_epilogue_and_end(self) -> None:
        device = HaslabDevice()
        activation = np.zeros((1, 1, 1, 8), dtype=np.int8)
        activation[0, 0, 0, :3] = [1, 2, 3]
        weights = np.zeros((1, 1, 8, 8), dtype=np.int8)
        weights[0, 0, :3, 0] = [2, 3, 4]
        weights[0, 0, :3, 1] = [1, 1, 1]
        device.memory.write(MemorySpace.INPUT, 0, activation.tobytes())
        device.memory.write(MemorySpace.WEIGHT, 0, weights.tobytes())
        device.memory.write(
            MemorySpace.PARAM,
            0,
            parameter_records(bias=[5, -1, 0, 0, 0, 0, 0, 0]),
        )

        self._submit_run(
            device,
            Command.build(
                Opcode.CONV_I8,
                1,
                [0, 0, 0, 1, 1, 2, 3, 0, 3, 1, 1],
                flags=3,
            ),
        )
        self.assertTrue(device.accumulator_open)
        self._submit_run(device, Command.build(Opcode.EPILOGUE, 2, [0, 0, 0, 0, 0]))
        result = np.frombuffer(device.memory.read(MemorySpace.OUTPUT, 0, 32), dtype="<i4")
        np.testing.assert_array_equal(result, [25, 5, 0, 0, 0, 0, 0, 0])
        self.assertFalse(device.accumulator_open)
        self._submit_run(device, Command.build(Opcode.END, 3))
        self.assertEqual(device.last_end, 3)

    def test_two_chunk_convolution_preserves_partial_sum(self) -> None:
        device = HaslabDevice()
        first_activation = np.ones((1, 1, 1, 8), dtype=np.int8)
        first_weights = np.zeros((1, 1, 8, 8), dtype=np.int8)
        first_weights[0, 0, :, 0] = 1
        device.memory.write(MemorySpace.INPUT, 0, first_activation.tobytes())
        device.memory.write(MemorySpace.WEIGHT, 0, first_weights.tobytes())
        self._submit_run(
            device,
            Command.build(
                Opcode.CONV_I8,
                1,
                [0, 0, 0, 1, 1, 1, 8, 0, 10, 1, 1],
                flags=1,
            ),
        )

        second_activation = np.zeros((1, 1, 1, 8), dtype=np.int8)
        second_activation[0, 0, 0, :2] = [2, 3]
        second_weights = np.zeros((1, 1, 8, 8), dtype=np.int8)
        second_weights[0, 0, :2, 0] = [4, 5]
        device.memory.write(MemorySpace.INPUT, 0, second_activation.tobytes())
        device.memory.write(MemorySpace.WEIGHT, 0, second_weights.tobytes())
        self._submit_run(
            device,
            Command.build(
                Opcode.CONV_I8,
                2,
                [0, 0, 0, 1, 1, 1, 2, 8, 10, 1, 1],
                flags=2,
            ),
        )
        device.memory.write(MemorySpace.PARAM, 0, parameter_records())
        self._submit_run(device, Command.build(Opcode.EPILOGUE, 3, [0, 0, 0, 0, 0]))
        result = np.frombuffer(device.memory.read(MemorySpace.OUTPUT, 0, 32), dtype="<i4")
        self.assertEqual(int(result[0]), 31)

    def test_linear_epilogue_requantizes_and_zeroes_padding(self) -> None:
        device = HaslabDevice()
        activation = np.zeros((1, 1, 1, 8), dtype=np.int8)
        activation[0, 0, 0, 0] = 5
        weights = np.zeros((1, 1, 8, 8), dtype=np.int8)
        weights[0, 0, 0, 0] = 1
        device.memory.write(MemorySpace.INPUT, 0, activation.tobytes())
        device.memory.write(MemorySpace.WEIGHT, 0, weights.tobytes())
        device.memory.write(
            MemorySpace.PARAM,
            0,
            parameter_records(multiplier=[1, 0, 0, 0, 0, 0, 0, 0], shift=[1, 0, 0, 0, 0, 0, 0, 0]),
        )
        self._submit_run(
            device,
            Command.build(Opcode.CONV_I8, 1, [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1], flags=3),
        )
        self._submit_run(device, Command.build(Opcode.EPILOGUE, 2, [0, 0, 0, 1, 0]))
        self.assertEqual(device.memory.read(MemorySpace.OUTPUT, 0, 8), bytes([2, 0, 0, 0, 0, 0, 0, 0]))

    def test_silu_epilogue_uses_serialized_lut(self) -> None:
        device = HaslabDevice()
        activation = np.zeros((1, 1, 1, 8), dtype=np.int8)
        activation[0, 0, 0, 0] = 4
        weights = np.zeros((1, 1, 8, 8), dtype=np.int8)
        weights[0, 0, 0, 0] = 1
        lut = np.arange(256, dtype=np.uint8).repeat(4).view(np.int8)
        device.memory.write(MemorySpace.INPUT, 0, activation.tobytes())
        device.memory.write(MemorySpace.WEIGHT, 0, weights.tobytes())
        device.memory.write(
            MemorySpace.PARAM,
            0,
            parameter_records(multiplier=[1, 0, 0, 0, 0, 0, 0, 0]),
        )
        device.memory.write(MemorySpace.PARAM, 128, lut.tobytes())
        self._submit_run(
            device,
            Command.build(Opcode.CONV_I8, 1, [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1], flags=3),
        )
        self._submit_run(device, Command.build(Opcode.EPILOGUE, 2, [0, 0, 0, 2, 128]))
        self.assertEqual(device.memory.read(MemorySpace.OUTPUT, 0, 1), bytes([int(lut[516]) & 0xFF]))

    def test_epilogue_overflow_becomes_architectural_fault(self) -> None:
        device = HaslabDevice()
        activation = np.zeros((1, 1, 1, 8), dtype=np.int8)
        activation[0, 0, 0, 0] = 1
        weights = np.zeros((1, 1, 8, 8), dtype=np.int8)
        weights[0, 0, 0, 0] = 1
        device.memory.write(MemorySpace.INPUT, 0, activation.tobytes())
        device.memory.write(MemorySpace.WEIGHT, 0, weights.tobytes())
        device.memory.write(
            MemorySpace.PARAM,
            0,
            parameter_records(bias=[(1 << 31) - 1, 0, 0, 0, 0, 0, 0, 0]),
        )
        self._submit_run(
            device,
            Command.build(Opcode.CONV_I8, 1, [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1], flags=3),
        )
        device.submit(Command.build(Opcode.EPILOGUE, 2, [0, 0, 0, 0, 0]))
        self.assertFalse(device.run_next())
        self.assertEqual(device.fault.code, ErrorCode.ARITH_OVERFLOW)
        self.assertEqual(device.last_completed, 1)

    def test_map_and_add_utility_commands(self) -> None:
        device = HaslabDevice()
        source_a = np.array([1, 2, -3, 0, 0, 0, 0, 0], dtype=np.int8)
        source_b = np.array([3, -2, 1, 0, 0, 0, 0, 0], dtype=np.int8)
        device.memory.write(MemorySpace.INPUT, 0, source_a.tobytes())
        device.memory.write(MemorySpace.INPUT, 8, source_b.tobytes())
        device.memory.write(
            MemorySpace.PARAM,
            0,
            parameter_records(
                multiplier=[2, 2, 2, 0, 0, 0, 0, 0],
                shift=[1, 1, 1, 0, 0, 0, 0, 0],
            ),
        )
        device.memory.write(
            MemorySpace.PARAM,
            128,
            parameter_records(
                multiplier=[1, 1, 1, 0, 0, 0, 0, 0],
                shift=[1, 1, 1, 0, 0, 0, 0, 0],
            ),
        )
        self._submit_run(device, Command.build(Opcode.MAP_I8, 1, [0, 0, 1, 1, 3, 0]))
        np.testing.assert_array_equal(
            np.frombuffer(device.memory.read(MemorySpace.OUTPUT, 0, 8), dtype=np.int8),
            source_a,
        )
        self._submit_run(device, Command.build(Opcode.ADD_I8, 2, [0, 8, 8, 1, 1, 3, 0, 128]))
        result = np.frombuffer(device.memory.read(MemorySpace.OUTPUT, 8, 8), dtype=np.int8)
        np.testing.assert_array_equal(result, [2, 1, -2, 0, 0, 0, 0, 0])

    def test_pool_and_upsample_commands(self) -> None:
        device = HaslabDevice()
        padded = np.full((5, 5, 1, 8), -128, dtype=np.int8)
        padded[2, 2, 0, 0] = 9
        device.memory.write(MemorySpace.INPUT, 0, padded.tobytes())
        self._submit_run(device, Command.build(Opcode.MAXPOOL5_I8, 1, [0, 0, 1, 1, 1]))
        self.assertEqual(device.memory.read(MemorySpace.OUTPUT, 0, 8), bytes([9, 0, 0, 0, 0, 0, 0, 0]))

        source = np.array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        device.memory.write(MemorySpace.INPUT, 0, source.tobytes())
        self._submit_run(device, Command.build(Opcode.UPSAMPLE2_I8, 2, [0, 0, 1, 2, 1]))
        result = np.frombuffer(device.memory.read(MemorySpace.OUTPUT, 0, 64), dtype=np.int8).reshape(2, 4, 8)
        np.testing.assert_array_equal(result[:, :, 0], [[1, 1, 2, 2], [1, 1, 2, 2]])

    def test_bad_continuation_faults(self) -> None:
        device = HaslabDevice()
        device.memory.write(MemorySpace.INPUT, 0, bytes(8))
        device.memory.write(MemorySpace.WEIGHT, 0, bytes(64))
        first = Command.build(
            Opcode.CONV_I8, 1, [0, 0, 0, 1, 1, 1, 8, 0, 16, 1, 1], flags=1
        )
        self._submit_run(device, first)
        bad = Command.build(
            Opcode.CONV_I8, 2, [0, 0, 0, 1, 1, 1, 8, 0, 16, 1, 1], flags=0
        )
        device.submit(bad)
        self.assertFalse(device.run_next())
        self.assertEqual(device.fault.code, ErrorCode.BAD_STATE)


if __name__ == "__main__":
    unittest.main()
