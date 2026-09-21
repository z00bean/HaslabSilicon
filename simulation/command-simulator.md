# HASLAB v0 functional command simulator

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: CERN-OHL-S-2.0

`haslab_sim` executes the proposed v0 128-byte command records against byte-addressed memory regions. It is the behavioral bridge between the arithmetic golden model and future RTL. The simulator follows document revision 0.2 of `docs/haslab-v0-contract.md`, using experimental command ABI 0.1. This is a freeze candidate awaiting independent conformance evidence; document and ABI versions are distinct.

## Modeled behavior

- Commands are exactly 32 little-endian u32 words. Framing checks ABI 0.1, a zero reserved word, legal flags, and contiguous nonzero sequence numbers before enqueue.
- The FIFO has eight atomic records. A full FIFO returns `BUSY`; framing errors return `INVALID`; a faulted/resetting device returns `BLOCKED`.
- Commands retire in order. An execution fault is sticky, prevents later commands from executing, records the first failing sequence and available field/space/offset information, and requires reset.
- EXT and the five local spaces have the capacities in the v0 contract. Reset clears modeled local bytes and preserves EXT; the architectural meaning is that local contents become invalid, so software must reload them.
- DMA and copy are aligned two-dimensional byte transfers. DMA requires exactly one EXT endpoint. Local copy supports INPUT/OUTPUT only and rejects every overlap between source and destination rows.
- CONV implements the FIRST/LAST accumulator context, validates consecutive channel chunks, consumes HWC8/KHWCI8 bytes, and checks every INT32 addition.
- EPILOGUE validates eight 16-byte parameter records and supports raw biased INT32, linear INT8, and SiLU-LUT INT8 output modes.
- MAP, ADD, MAXPOOL5, UPSAMPLE2, FILL, FENCE, and END implement their bounded v0 rules. Utilities are rejected while an accumulator context is open.
- END rejects an unfinished accumulator, records `LAST_END`, and returns the device to IDLE. If already queued work follows END, execution of the next record starts a new RUNNING interval.

## Contract-review corrections

The [revision 0.2 review](../docs/reviews/v0-contract-review.md) adds regressions for these rules:

- Command payloads are copied to immutable tuples, including when constructed from caller-owned lists. Builder methods reject fractional or string-valued fields instead of silently coercing them.
- EXT capacity is limited to 2^32−1 bytes so its literal value fits EXT_BYTES. Wide failed-address calculations do not wrap into valid addresses or leak oversized values into ERROR_OFFSET.
- Single-row DMA/COPY2D requires both strides to equal row length.
- Structural output/LUT checks precede arithmetic. Malformed epilogues cannot hide an invalid destination behind a bias-overflow result.
- Submission precedence is BLOCKED, BUSY, INVALID, ACCEPTED. Rejected framing never poisons already accepted work.
- Sequences cannot wrap; hardware reset generation does wrap modulo 2^32. Runtime session/token protection is still future work.
- COPY2D checks actual row intersections rather than rejecting every overlapping bounding span.

`test_contract_review.py` is regression evidence, not the independently serialized M4 conformance package. Fixtures for portable targets must mask optional fault diagnostics and unspecified post-arithmetic/bus-fault memory. The simulator retains queued records after an execution fault, but hardware queue contents in FAULT are not a portable result.

## Deliberately unmodeled behavior

This is not a cycle-accurate or bus-signal simulator. It does not assign cycle counts, split bursts, model backpressure, represent host cache maintenance, emulate MMIO staging registers, inject DMA bus errors, model interrupts, or enforce immutable EXT buffers after submission. `DMA_BUS` remains reserved for a future platform/bus model.

Memory writes performed directly through the simulator API are testbench setup and observation facilities, analogous to loading board memory. They are not accelerator commands and must not be confused with runtime behavior.

Unexpected Python exceptions become architectural `INTERNAL` faults so a test cannot accidentally treat a partially executed command as successful. During development, the fault message should be investigated as a simulator defect.

## Running tests

From the repository root:

```sh
make sim
```

`make test` runs both the arithmetic golden model and command simulator suites. Passing these tests establishes consistency with this Python implementation, not independent proof that the proposed ABI is correct or suitable for a selected FPGA.
