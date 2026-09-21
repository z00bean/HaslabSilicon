# HASLAB development plan

This is the tracked source of truth for HASLAB implementation progress. The top-level README contains a short public snapshot; this document contains the work order, dependencies, and evidence required to call a milestone complete.

Last reviewed: 2026-09-20

## Status vocabulary

| Status | Meaning |
|---|---|
| `DONE` | Exit criteria are satisfied and supporting evidence is checked in |
| `ACTIVE` | Current work; scope and exit criteria are defined |
| `NEXT` | Ready after the active milestone or able to proceed without changing frozen behavior |
| `BLOCKED` | Waiting for a named decision, dependency, artifact, or measurement |
| `LATER` | Intentionally deferred |

A feature is not `DONE` because it has placeholder files, compiles once, or produces plausible output. Completion requires the evidence listed for its milestone.

## Current focus

The M3 implementation review is complete and documented in the [contract review record](reviews/v0-contract-review.md). Document revision 0.2 is a freeze candidate for experimental command ABI 0.1. Stable freeze remains open until independent command/transition review and conformance evidence are available.

Next: build M4's independent conformance-vector format against the pinned candidate before accelerator RTL begins. This evidence feeds back into M3; candidate readiness is not a stable ABI release.

The next implementation artifact should be a small, versioned conformance package containing serialized commands, initial memory, expected memory, expected completion or error state, and human-readable metadata. These vectors will be shared by the functional simulator and future RTL tests.

## Milestone plan

| ID | Milestone | Status | Depends on | Completion evidence |
|---|---|---|---|---|
| M0 | Repository foundation | `DONE` | — | Project structure, build entry points, documentation skeletons, CI skeleton, and licensing are tracked |
| M1 | Numerical golden model | `DONE` | M0 | Documented FP8/INT8 semantics and passing numerical, layout, activation, and operation tests |
| M2 | Functional command simulator | `DONE` | M1 | Versioned 128-byte command encoding, memory/device behavior, architectural errors, and passing simulator tests |
| M3 | Freeze the v0 contract | `ACTIVE` | M1, M2; independent M4 evidence for closure | Candidate review and compatibility policy documented; final freeze awaits independent command/transition review and conformance evidence |
| M4 | v0 conformance package | `NEXT` | M3 candidate (ready); frozen M3 for final corpus release | Checked-in independent positive and negative vectors with hashes, schema documentation, and automated simulator validation |
| M5 | Pin the first YOLO workload | `NEXT` | M3 candidate (ready) | Exact model/export hashes, license, ONNX profile, preprocessing, operator inventory, host partition, floating baseline, and INT8 acceptance budget are tracked |
| M6 | Minimal compiler and simulated runtime | `BLOCKED` | M4, M5 | Deterministic model package generation and complete execution through the simulator with layerwise comparisons |
| M7 | First RTL vertical slice | `BLOCKED` | Frozen M3, M4 | Command decode through one INT8 MAC path, INT32 accumulation, requantization, completion/error behavior, and differential tests |
| M8 | Incremental RTL operation coverage | `BLOCKED` | M7 | Each added DMA, convolution, or utility operation passes its conformance and assertion gates |
| M9 | FPGA selection and v0 bring-up | `BLOCKED` | M5, M6, M8 | Measured resource probes inform board selection; reproducible build, timing, utilization, accuracy, transfer, and end-to-end latency reports are published |
| M10 | v1 RISC-V and native FP8 research | `LATER` | Measured v0 | RISC-V control and FP8 are added only with workload evidence and independent numerical/implementation validation |
| M11 | ASIC feasibility and test core | `LATER` | Stable measured FPGA design | Process, SRAM, IO, clock, power, DFT, physical verification, packaging, and fabrication collateral are explicitly resolved |

M4 candidate fixtures and M5 workload auditing can proceed against M3's explicitly pinned candidate. Candidate fixtures provide the independent evidence needed to close M3; only then is the final M4 corpus tied to a frozen ABI. This staged gate avoids requiring a freeze before gathering the evidence needed to justify it. M7 may begin with reviewed M4 vectors after M3 freeze, before the full model compiler is finished, but no complete v0 RTL claim can be made before M5 and M6 establish the real workload path.

## M3 — freeze the v0 contract

- [x] Review command fields, flags, descriptors, alignment, and address calculations against the implementation; document remaining evidence gaps.
- [x] Select the candidate ABI identifier and define exact version rejection rules.
- [ ] Freeze the stable ABI identifier after independent evidence review.
- [x] Specify invalid-field behavior, commit/reset precedence, sequence handling, partial DMA failure, and post-fault behavior; transport verification remains a later gate.
- [x] Review accumulator lifecycle, chunk continuation, overflow, padding, and zero-fill rules against existing tests.
- [x] Separate device guarantees from compiler, runtime, transport, and host-tail responsibilities in the review record.
- [x] Record ABI and control decisions in [ADR 0001](decisions/0001-v0-abi-freeze-candidate.md).
- [x] Assign unresolved workload-dependent items to explicit M5/M6 gates.
- [ ] Complete an independent specification review before declaring the ABI frozen.
- [x] Fix simulator discrepancies found in the review and add targeted regressions.

The v0 contract can freeze without selecting v1 FP8 payloads, an ASIC SRAM macro, or a RISC-V integration. Those remain outside the v0 ABI.

## M4 — build the conformance package

Planned location: `conformance/`.

- [ ] Specify a versioned fixture manifest and directory layout.
- [ ] Introduce one machine-readable ABI definition with checks against executable constants before adding compiler/RTL consumers.
- [ ] Store serialized 128-byte commands as the actual bytes consumed by an implementation.
- [ ] Store initial and expected memory images without deriving expected results during the test.
- [ ] Store expected completion records, sequence counters, reset generation, and structured errors.
- [ ] Mask optional fault diagnostics and unspecified memory after arithmetic/bus faults; use isolated violations or allowed error sets for invalid commands.
- [ ] Include provenance, ABI version, endianness, dimensions, tensor layout, and file hashes.
- [ ] Add normal cases for DMA, fill, copy, convolution, epilogue, map, add, pool, upsample, and end/reset behavior.
- [ ] Add boundary cases for rounding ties, saturation, overflow, channel padding, halos, and chunk continuation.
- [ ] Add negative cases for malformed framing, unknown opcodes, bounds, alignment, overlap, FIFO full, sequence gaps, stale tokens, and illegal accumulator state.
- [ ] Keep the expected-value source independent from the implementation being tested.
- [ ] Validate every fixture against the functional simulator in CI.
- [ ] Minimize every discovered failure into a permanent regression fixture.

Exit criterion: another implementation can consume the fixture specification without importing private helpers from the Python reference model or simulator.

## M5 — pin the first YOLO-class workload

- [ ] Select the exact model revision and confirm redistribution and weight licenses.
- [ ] Pin input shape, exporter version, ONNX IR version, opset, export command, and graph/weight hashes.
- [ ] Freeze preprocessing: color order, resize/letterbox, interpolation, padding, normalization, layout, and input quantization.
- [ ] Inventory every graph node, shape, attribute, and constant.
- [ ] Classify each node as native HASLAB, compiler-fused, declared host-tail, or unsupported.
- [ ] Freeze the learned-head/host-tail boundary and output coordinate/order conventions.
- [ ] Record a reproducible floating-point output and accuracy baseline.
- [ ] Calibrate INT8 and agree the accuracy-loss budget before treating quantization as acceptable.
- [ ] Check every layer against local-memory, tiling, channel-group, convolution-chunk, and command-traffic limits.
- [ ] Publish a machine-readable workload manifest under `benchmarks/manifests/`.

Exit criterion: no operator, preprocessing step, fallback, or accuracy comparison is implicit.

## M6 — minimal compiler and simulated runtime

- [ ] Validate only the pinned static ONNX profile and fail unsupported graphs with node-level diagnostics.
- [ ] Perform shape inference, constant extraction, layout conversion, quantization lowering, tiling, and SRAM planning.
- [ ] Emit target-specific commands and a versioned HASLAB package with graph, weight, and configuration hashes.
- [ ] Guarantee byte-identical package output for identical inputs and tool versions.
- [ ] Load packages, bind tensors, submit commands, wait, reset, and report structured errors through a simulator backend.
- [ ] Compare intermediate tensors layer by layer against an independent ONNX or framework reference.
- [ ] Execute the declared host tail explicitly; never use silent per-layer fallback.

## M7–M9 — RTL and FPGA rules

The first RTL slice is intentionally narrow:

```text
command decoder
    → local SRAM model
    → one INT8 MAC path
    → INT32 accumulator
    → requantization
    → completion/error record
```

Every new hardware operation must pass this gate:

- [ ] Specification and reserved behavior are frozen.
- [ ] Independent positive, boundary, and negative vectors are checked in.
- [ ] Functional-simulator tests pass.
- [ ] RTL matches the same vectors bit for bit at architectural boundaries.
- [ ] Assertions cover bounds, protocol state, accumulator lifecycle, and illegal states.
- [ ] Random tests record their seeds and minimized failures become regressions.
- [ ] CI passes with pinned tool versions.

FPGA selection follows small synthesis probes for SRAM, MAC, and DMA structures on at least two plausible targets. The choice must be based on measured memory mapping, DSP use, external-memory support, tool availability, cost, and accessibility to university teams.

## Project-wide controls

- Maintain one source of truth for opcodes, fields, layouts, enums, error codes, and ABI versions.
- Generate language-specific constants from that source when practical.
- Pin dependencies, exporters, simulators, synthesis tools, and relevant command lines.
- Keep deterministic tests; record random seeds and artifact hashes.
- Test error paths with the same care as successful execution.
- Compare intermediate tensors instead of relying only on final detections.
- Treat silent fallback, ignored errors, unreported saturation, and accidental approximation as failures.
- Keep the golden oracle independent from the implementation under test.
- Require reproducible evidence before changing a public status to implemented or tested.
- Record interface and numerical changes as architecture decision records.

## Updating this record

When work changes status:

1. Update the milestone table and its checklist in the same change as the evidence.
2. Add a dated entry below with links to tests, specifications, decisions, or measured reports.
3. Update the shorter progress snapshot in the top-level README.
4. Do not mark downstream milestones active if their named dependencies remain unresolved; respect the candidate-versus-final distinction for M3/M4.

## Progress log

| Date | Change | Evidence |
|---|---|---|
| 2026-09-20 | Established the tracked implementation plan; recorded repository foundation, numerical model, and functional simulator as complete; made contract freeze the active milestone | `README.md`, `reference/numerical-semantics.md`, `reference/tests/`, `simulation/command-simulator.md`, `simulation/tests/`, `docs/haslab-v0-contract.md` |
| 2026-09-20 | Completed M3 implementation review; published document revision 0.2 / ABI 0.1 candidate and ADR 0001; corrected simulator snapshot, validation, aperture, stride, and diagnostics behavior; enabled M4 candidate fixtures while retaining independent review as the stable-freeze gate | [Review](reviews/v0-contract-review.md), [decision](decisions/0001-v0-abi-freeze-candidate.md), [regressions](../simulation/tests/test_contract_review.py) |
| 2026-09-20 | Review validation passed: 43 numerical tests + 42 simulator tests = 85 total; repository and whitespace checks passed | `make test`, `make check`, `git diff --check`; details in the [review record](reviews/v0-contract-review.md) |
