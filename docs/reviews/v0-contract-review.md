# v0 contract review record

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: CERN-OHL-S-2.0

Date: 2026-09-20

Result: document revision 0.2 is an ABI 0.1 freeze candidate. This review compared the contract with the numerical model, command simulator, and existing tests. It is not an independent external review, complete conformance proof, or hardware sign-off. M3 remains open for independent evidence; M4 fixture development is now the next action.

Follow-up, 2026-09-20: [M4's candidate corpus](../../conformance/README.md) now supplies 46 independently authored command fixtures plus an ABI registry, schema, hashes, and CI runner. Fixture development described in the original review is complete at candidate scope. Independent review of the contract/derivations and stable ABI allocation remain open; M5 workload auditing is the next implementation task.

Follow-up, 2026-09-21: [M5's pinned workload](../../benchmarks/manifests/yolov8n-320-opset13/README.md) now supplies the exact artifact and graph inventory, preprocessing and host-tail boundary, reproducible FLOAT baseline, deterministic INT8 calibration package, LUT/coefficient records, full calibration diagnostics, and proxy accuracy inside the adopted one-percentage-point budget. Exact HASLAB command-level agreement remains an M6 differential gate; this evidence does not close the independent M3/M4 review.

Follow-up, 2026-09-21: [M6's first vertical slice](../../benchmarks/manifests/yolov8n-320-opset13/m6-first-conv-silu-slice.json) now compiles one first-block tile into the proposed `.hxb` framing, loads and relocates it through the simulator runtime, and matches all 512 output bytes against the independent integer model. This validates a narrow package/command path; full-layer scheduling, whole-graph allocation, stable manifest limits, host-tail execution, and independent M3/M4 review remain open.

Follow-up, 2026-09-22: [M6's complete first-layer schedule](../../benchmarks/manifests/yolov8n-320-opset13/m6-first-conv-silu-layer.json) now covers all 400 spatial tiles, both output groups, boundary halos, and canonical HWC8 output assembly. All 409,600 values match the integer model. The generated 8,884-command schedule and actual simulator-runtime FIFO refill counts provide first measured traffic evidence; they are not transport timing or whole-model throughput evidence.

Follow-up, 2026-09-22: [M6's first multi-layer schedule](../../benchmarks/manifests/yolov8n-320-opset13/m6-first-two-conv-silu-layers.json) now executes nodes 0–5 in one package, preserves both HWC8 intermediates, and exercises two input-channel accumulation chunks in the second convolution. All 614,400 retained values match exactly. The 16,245-command schedule records cumulative allocation, DMA traffic, and FIFO behavior; branched graph liveness and transport timing remain open.

Follow-up, 2026-09-23: [M6's first C2f schedule](../../benchmarks/manifests/yolov8n-320-opset13/m6-first-c2f.json) now executes nodes 0–21 and exercises pointwise and stride-one convolution, two zero-allocation split views, explicit unequal-scale residual addition, and common-scale concat materialization. All 1,843,200 values across ten boundaries match exactly. The 59,052-command retain-all package proves the branch semantics while exposing command-dispatch and allocation pressure that reusable liveness scheduling must address before whole-graph release.

## Scope and review results

| Area | Reviewed behavior | Outcome / evidence |
|---|---|---|
| Command framing | 32 little-endian words; version/opcode header; flags; sequence; reserved and unused words | Exact version matching clarified; all opcodes tested for nonzero trailing payload rejection |
| Submission | Eight slots, snapshot, backpressure, rejection, sequence exhaustion | Fixed caller-owned mutable payload; specified and tested BLOCKED → BUSY → INVALID precedence and no sequence wrap |
| Addressing and DMA | Spaces, capacities, alignment, row extents, endpoints, overlap | Fixed 4 GiB capacity acceptance and single-row stride mismatch; wide bounds diagnostics remain u32; cross-row overlap tests added |
| Convolution | HWC8/KHWCI8, lane/chunk bounds, FIRST/LAST, ACC ownership | Reviewed existing tests and implementation; reduction and padding responsibilities clarified; exhaustive independent vectors pending |
| Epilogue | Bias, INT32 overflow, records, raw/linear/SiLU modes, LUT disjointness | Moved output/LUT structural validation before arithmetic; tested malformed epilogues with overflowing bias |
| Utilities | MAP, ADD, pool, upsample, copy, and context exclusion | Added output preflight; clarified permitted read aliasing and FILL exemption; existing operation tests retained |
| Numerical rules | INT8/INT32, ties-to-even, multiplier/shift, padding, accumulation order | Device rules retained; calibrated coefficient/LUT generation and accuracy budgets belong to workload/compiler gates |
| Completion/fault | END/FENCE, counters, sticky faults, subsequent queue effects | Tested sticky faults/no later writes; diagnostic masks and partial-result rules specified |
| Reset/MMIO | Register accesses, commit/reset races, generation, cycles | Undefined RESET_DONE removed; acknowledgement and ordering specified; no hardware/bus verification claimed |
| Host metadata and package | Tensor manifest, scale arrays, `.hxb`, relocation, host-tail boundary | Remain proposals; required schema and workload decisions assigned below |

## Corrected defects

| Defect | Correction | Regression |
|---|---|---|
| Mutable payload could change accepted work | Copy payload to tuple at construction | `test_payload_is_snapshotted_before_submission` |
| Builder silently truncated fractions/coerced strings | Preserve types and reject invalid fields | `test_builder_does_not_silently_truncate_noninteger_fields` |
| 2^32 capacity did not fit EXT_BYTES | Require 1…2^32−1, reject before allocation | `test_ext_capacity_must_fit_register_without_allocating_four_gib` |
| Single-row strides disagreed with the contract | Require both strides to equal row length for DMA and COPY2D | `test_single_row_dma_and_copy_require_canonical_strides` |
| Derived out-of-range address exceeded fault register width | Report zero when diagnostic address is not representable; retain BOUNDS and space | `test_widened_dma_bounds_never_wrap_into_a_valid_address` |
| Bias overflow could obscure malformed epilogue operands | Check destination and LUT structure before arithmetic | `test_epilogue_preflight_precedes_arithmetic_overflow` |

Regressions are in [test_contract_review.py](../../simulation/tests/test_contract_review.py). Additional cases cover exact versions, unused payloads, FIFO retry, sequence exhaustion, reset-generation wrap, sticky faults, disjoint interleaved copy rows, and cross-row overlap. Existing tests continue to cover numerical behavior and compute commands. These are tests of selected rules, not exhaustive validation of all combinations.

Validation on 2026-09-20: `make test` passed 43 numerical-model tests and 42 simulator tests (85 total, including 16 new review regressions). `make check` and `git diff --check` also passed. No RTL, FPGA, bus, or full-model test was run because those implementations do not exist yet.

## Pre-RTL gate ownership

| Gate | Resolution / owner milestone | Status |
|---|---|---|
| ABI allocation and command/state review | M3 candidate + ADR 0001; independent M4 evidence required for stable freeze | Candidate ready; independent review pending |
| Single definition for multiple implementation consumers | M4 machine-readable ABI definition and schema consistency checks | Candidate registry implemented and checked against simulator constants; new consumers still require integration |
| Workload artifact, exporter, IR, opset, weights, hashes | M5 pinned manifest | Complete for the first YOLOv8n workload |
| Detection-tail boundary and preprocessing/output conventions | M5 graph inventory and reproducible baseline | Complete for the pinned workload |
| INT8 accuracy budget and calibration | M5 evaluation; one-percentage-point mAP50–95 budget | Calibrated proxy passes; exact command-level comparison remains in M6 |
| Scale approximation and reproducible coefficient/LUT bytes | M5/M6 coefficient-generation policy and accuracy/error report | Candidate bytes and approximation errors recorded; M6 must execute them exactly |
| Residual/concat scales, split boundaries, chunk/halo layouts | M5 every-layer audit, then M6 lowering tests | First C2f passes with explicit residual/concat coefficients and zero-allocation split views; repeated/general graph forms remain pending |
| Package schema, limits, relocations, parser behavior | M6 schema freeze before compiler/runtime implementation | Pending |
| Commands per frame, host refill, patch copies, spill traffic | M5/M6 schedule estimates; M9 measurements | Nodes 0–21 measured cumulatively in the functional runtime; whole-model, optimized liveness, and transport timing pending |
| Board, aperture, bus ordering, cache maintenance, reset quiescence | M9 transport adapter and fault-injection evidence before integrated RTL freeze | Pending |
| Physical SRAM mapping and synchronous read timing | M7 prototypes/M9 target probes | Pending |
| Native FP8 | M11 separate numeric contract and arithmetic feasibility | Deferred; does not block v0 |
| ASIC SRAM/PHY/DFT/process | M12 feasibility | Deferred; does not block v0 |

Small isolated RTL experiments may follow reviewed independent vectors, but full v0 RTL freeze still requires the workload/numerical and relevant transport gates. A documentation decision alone cannot satisfy a measurement gate.

## Next action and closure criteria

The M4 fixture schema and first independently derived corpus now exist against **ABI 0.1 / contract revision 0.2**, and the M5 workload package is pinned. M6 now executes nodes 0–21 differentially through the first C2f block. The next implementation action is a reusable graph IR and liveness allocator followed by nodes 22–47: the next downsampling Conv-SiLU and the repeated-bottleneck second C2f block.

Before marking M3 complete, independently review the command tables and transitions, run the fixtures against another implementation or derivation, resolve discrepancies with a decision record, and explicitly allocate the frozen ABI. M6 candidate work may proceed, but a stable compiler/package release and full RTL commitment still depend on that independent review.
