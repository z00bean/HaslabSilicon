# ADR 0001: v0 ABI candidate and compatibility boundary

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: CERN-OHL-S-2.0

Date: 2026-09-20

Status: accepted for experimental development; stable ABI freeze pending independent conformance evidence.

## Context

The numerical model and command simulator exist, but neither alone proves an independently implementable contract. The initial review found ambiguous MMIO/reset behavior and mismatches in aperture size, single-row strides, immutable submission, and validation ordering. No released hardware or compiler package depends on the experimental ABI.

## Decision

1. Keep the current 128-byte encoding and numeric allocations as experimental command ABI 0.1. Publish the reviewed contract as document revision 0.2; document version does not alter header bytes.
2. Require exact major/minor matching. Do not assume forward or backward minor compatibility. A semantic or encoding change after this candidate needs a new experimental minor plus a decision record. A later stable ABI will be allocated explicitly after independent review.
3. Allow M4 fixture design against this pinned candidate, recording the contract revision or hash. Independent expected values and transition review are required to close M3; generating expected results from the simulator does not satisfy that gate.
4. Separate command semantics from the `.hxb` package schema, calibrated workload, transport, and future FP8 profile. Each has its own release gates. Accepting the candidate does not close board/reset/accuracy/package verification work.
5. Define deterministic submission precedence, immutable snapshots, no sequence wrap, modulo-u32 reset generation, and reset acknowledgement through generation plus IDLE. Remove the undefined RESET_DONE reference.
6. Limit EXT to the literal representable u32 capacity. Use widened extent arithmetic and never truncate unrepresentable fault offsets.
7. Require structural validation before arithmetic and bus access. Multiple structural violations may return any applicable structural error; optional fault details are masked in portable fixtures. Arithmetic/bus-fault destinations remain invalid.

The normative tables remain in the contract. Existing executable command/opcode/space constants live in `simulation/haslab_sim/abi.py`; error, state, and capacity definitions remain in their current simulator modules. Before a second compiler/RTL consumer is added, M4 must introduce one versioned machine-readable definition and validate or generate those consumers from it. This ADR does not claim a shared schema already exists.

## Alternatives

- Declaring ABI 1.0 now would overstate the independent evidence and constrain corrections prematurely.
- Generating conformance outputs from the simulator alone would preserve common implementation mistakes.
- Inventing a RESET_DONE register is unnecessary when generation and state already provide acknowledgement.
- Encoding 4 GiB with a sentinel zero capacity adds a special case without a workload need.

## Consequences and verification

Existing valid ABI 0.1 command bytes remain unchanged. Invalid single-row stride encodings are now rejected; a 4 GiB simulator allocation is rejected before allocation; command construction no longer accepts lossy field conversions. Commands passed as mutable payload lists are snapshotted.

Regression evidence is in `simulation/tests/test_contract_review.py`. The [review record](../reviews/v0-contract-review.md) lists coverage and open gates. MMIO staging, reset races, bus faults, and cycle-count semantics remain specification-only until a transport/cycle model or RTL verifies them.
