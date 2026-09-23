# HASLAB development plan

This is the tracked source of truth for HASLAB implementation progress. The top-level README contains a short public snapshot; this document contains the work order, dependencies, and evidence required to call a milestone complete.

Last reviewed: 2026-09-22

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

The M3 implementation review is complete and documented in the [contract review record](reviews/v0-contract-review.md). Document revision 0.2 is a freeze candidate for experimental command ABI 0.1. Stable freeze remains open until independent command/transition review of the candidate and its conformance evidence is complete.

M4's candidate package is implemented in [conformance/](../conformance/README.md): 46 independently authored binary fixtures, schema version 1, ABI registry with consistency checks, strict artifact validation, simulator runner, and CI integration. These expectations are independent of implementation helpers, but have not received independent external review. M3 freeze and final M4 release remain open.

M5 is complete as a workload-pinning milestone. The exact YOLOv8n artifact, graph, preprocessing, host boundary, FLOAT baseline, deterministic 512-image calibration set, all candidate INT8 scales, 57 SiLU tables, and layerwise diagnostics are recorded. The signed-symmetric INT8 software proxy measured 0.27610 bbox mAP50–95, a 0.887-point loss from FLOAT and inside the one-point budget. This proxy uses ONNX Runtime QLinear sigmoid/multiply operations and requantized learned-head outputs, so it does not claim whole-model HASLAB command or hardware agreement.

M6 is active. A reusable scheduler now executes the first two Conv-SiLU blocks as one package. It retains the `16×160×160` and `32×80×80` HWC8 tensors, handles two input-channel accumulation chunks in the second convolution, and reports 16,245 commands, 1,999,320 DMA bytes, 40,550,400 MACs, and 1,442,176 externally allocated bytes. All 614,400 retained INT8 values match the independent golden path, and runtime submission records the exact eight-entry FIFO refill behavior. The next slice is the first C2f block, including 1×1 convolution, split/view, concat liveness, and residual-scale handling. Independent M3/M4 review remains required before a stable ABI/corpus or package release.

The scope priority is vision and perception: complete the pinned detector, prove the same path in portable RTL, bring it up on an FPGA, and measure a live camera-to-detection pipeline. A second independently pinned vision workload should then demonstrate that changing the model does not require changing the hardware image. Small policy or other inference models may follow when they fit the measured architecture and operator set without delaying the detector path.

## Milestone plan

| ID | Milestone | Status | Depends on | Completion evidence |
|---|---|---|---|---|
| M0 | Repository foundation | `DONE` | — | Project structure, build entry points, documentation skeletons, CI skeleton, and licensing are tracked |
| M1 | Numerical golden model | `DONE` | M0 | Documented FP8/INT8 semantics and passing numerical, layout, activation, and operation tests |
| M2 | Functional command simulator | `DONE` | M1 | Versioned 128-byte command encoding, memory/device behavior, architectural errors, and passing simulator tests |
| M3 | Freeze the v0 contract | `ACTIVE` | M1, M2; independent M4 review for closure | Candidate and conformance evidence available; final freeze awaits independent command/transition review |
| M4 | v0 conformance package | `ACTIVE` | Candidate implemented; frozen M3 for final corpus release | 46 binary fixtures, hashes/schema, ABI consistency, simulator runner, and CI implemented; final release awaits independent review and stable ABI |
| M5 | Pin the first YOLO workload | `DONE` | M3 candidate | Exact artifact, graph, preprocessing, partition, FLOAT baseline, deterministic INT8 calibration package, proxy accuracy, and diagnostic evidence are tracked; exact target execution belongs to M6 |
| M6 | Minimal compiler and simulated runtime | `ACTIVE` | M5 complete; M3/M4 candidate usable, stable release still gated by independent review | Reusable lowering through the first two Conv-SiLU blocks passes exactly; completion still requires branched whole-model lowering, host tail, and layerwise comparisons |
| M7 | First RTL vertical slice | `BLOCKED` | Frozen M3, M4 | Command decode through one INT8 MAC path, INT32 accumulation, requantization, completion/error behavior, and differential tests |
| M8 | Incremental RTL operation coverage | `BLOCKED` | M7 | Each added DMA, convolution, or utility operation passes its conformance and assertion gates |
| M9 | FPGA selection, v0 bring-up, and camera measurement | `BLOCKED` | M5, M6, M8 | Measured probes inform board selection; reproducible build, accuracy, resources, timing, power, transfers, and camera-to-box latency are published |
| M10 | Second perception workload and reprogramming proof | `LATER` | Complete measured M9 detector | A separately pinned vision/perception graph runs on the same bitstream; operator coverage, compile time, changed package bytes, accuracy, and performance are published |
| M11 | v1 RISC-V and native FP8 research | `LATER` | Measured v0 and M10 evidence | RISC-V control and FP8 are added only with workload evidence and independent numerical/implementation validation |
| M12 | ASIC feasibility and test core | `LATER` | Stable measured FPGA design | Process, SRAM, IO, clock, power, DFT, physical verification, packaging, and fabrication collateral are explicitly resolved |

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

Location: [conformance/](../conformance/README.md). Candidate schema/runner decision: [ADR 0002](decisions/0002-conformance-corpus-format.md).

- [x] Specify a versioned fixture manifest and directory layout.
- [x] Introduce one machine-readable ABI definition with checks against executable constants before adding compiler/RTL consumers.
- [x] Store serialized 128-byte commands as the actual bytes consumed by an implementation.
- [x] Store initial and expected memory images without deriving expected results during device execution.
- [x] Store expected completion counters, reset generation, and structured errors at checkpoints.
- [x] Mask optional fault diagnostics and omit unspecified memory after arithmetic faults; document the same rule for future bus-fault cases.
- [x] Include provenance, ABI version, endianness, tensor metadata, and file hashes.
- [x] Add normal cases for DMA, fill, copy, convolution, epilogue, map, add, pool, upsample, and end/reset behavior.
- [x] Add boundary cases for rounding ties, saturation, overflow, channel padding, halos, and chunk continuation.
- [x] Add negative cases for malformed framing, unknown opcodes, bounds, alignment, overlap, FIFO full, sequence gaps, and illegal accumulator state.
- [x] Add stale-token tests through the M6 simulator runtime; transport reset with outstanding work remains assigned to M9.
- [ ] Add DMA bus failure/reset-with-outstanding-transfer cases when the transport model exists (M9).
- [x] Keep expected-value recipes independent from the implementation being tested.
- [x] Validate every fixture against the functional simulator through `make conformance`, included in `make test` and CI.
- [x] Test the runner using corrupted artifacts, malformed schemas, and deliberately wrong expectations.
- [ ] Complete independent review and stable-ABI corpus release after M3 closes.
- [ ] Publish a short external-review quickstart that identifies the ABI registry, fixture derivations, reproduction commands, unsupported-operation diagnostics, and how to submit a minimized failing case.

Ongoing rule: minimize each newly discovered failure into a permanent regression. The initial corpus reproduced the candidate simulator results without requiring a new arithmetic or command-semantic change. Runner discrepancies found during future use are bugs to investigate, not reasons to copy the implementation's output into golden files.

Exit criterion: another implementation can consume the fixture specification without importing private helpers from the Python reference model or simulator.

## M5 — pin the first YOLO-class workload

- [x] Select the exact model revision and confirm redistribution and weight licenses.
- [x] Pin input shape, exporter version and commit, ONNX IR version, opset, export command, and graph/weight hashes.
- [x] Freeze preprocessing: UINT8 source, RGB letterbox, normalization, NCHW reference layout, HWC8 device layout, and binary32 INT8 input scale are pinned.
- [x] Inventory every graph node, shape, attribute, initializer, and graph constant.
- [x] Classify each node as native HASLAB, compiler-fused, compiler view/fold, declared host-tail, or unsupported.
- [x] Freeze the learned-head/host-tail boundary and output coordinate/order conventions. The boundary uses three INT32 HWC8 tensors with per-channel dequantization scales.
- [x] Record a reproducible floating-point output and accuracy baseline. The synthetic PyTorch/ONNX output comparison passes, and two full COCO 2017 validation runs produced identical aggregate metrics, all 80 per-class AP pairs, and prediction JSON hash.
- [x] Calibrate signed-symmetric INT8 with the deterministic 512-image train2017 subset and measure the candidate against the one-percentage-point budget. The QOperator proxy loses 0.887 points mAP50–95 and passes; exact HASLAB execution remains an M6 differential gate.
- [x] Check every layer against local-memory, tiling, channel-group, and convolution-chunk limits. All 63 learned convolutions and graph alignment constraints pass. The 38,696-command CONV/EPILOGUE reference subtotal is only a warning; optimized whole-graph traffic and FIFO demand must be generated in M6.
- [x] Publish a machine-readable workload manifest under `benchmarks/manifests/`.

Exit criterion: no operator, preprocessing step, fallback, or accuracy comparison is implicit.

## M6 — minimal compiler and simulated runtime

- [ ] Validate only the pinned static ONNX profile and fail unsupported graphs with node-level diagnostics.
- [ ] Perform shape inference, constant extraction, layout conversion, quantization lowering, tiling, and SRAM planning.
- [ ] Emit target-specific commands and a versioned HASLAB package with graph, weight, and configuration hashes.
- [ ] Guarantee byte-identical package output for identical inputs and tool versions.
- [ ] Load packages, bind tensors, submit commands, wait, reset, and report structured errors through a simulator backend.
- [ ] Compare intermediate tensors layer by layer against an independent ONNX or framework reference.
- [ ] Execute the declared host tail explicitly; never use silent per-layer fallback.

First vertical-slice evidence:

- [x] Validate the pinned model/hash/opset and exact first `Conv → Sigmoid → Mul` dataflow with node-level failures.
- [x] Extract constants, quantize weights and bias, pack KHWCI8, and lower one interior 8×8 tile for output channels 0–7.
- [x] Emit eight ABI 0.1 commands in the proposed sectioned `.hxb` framing with section hashes, debug provenance, and validated EXT relocations.
- [x] Recompile the same inputs to a byte-identical 5,056-byte package.
- [x] Strictly load, bind, submit, wait, reset, reject corruption/invalid binding, and invalidate stale tokens through the simulator backend.
- [x] Match all 512 tile output bytes against `haslab_ref` and compare the dequantized intermediate with ONNX Runtime.
- [x] Schedule the full first layer: boundary halos, 400 spatial tiles, two output groups, output assembly, and generated command/DMA/FIFO accounting.
- [x] Match all 409,600 first-layer outputs exactly; record 8,884 commands, 2,250,768 DMA bytes, 11,059,200 MACs, and actual eight-entry FIFO refill behavior.
- [x] Replace the first-layer-specific scheduler with a reusable layer plan and execute the second Conv-SiLU block, preserving comparisons at both layer boundaries.
- [ ] Lower the first C2f block with 1×1 convolution, split/view aliases, concat liveness, residual addition, and explicit scale checks.
- [ ] Generalize from the first layer to every accelerator-region node and the declared host tail before closing M6.

## M7–M9 — RTL, FPGA, and camera rules

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

Portable RTL development starts with Verilator on macOS/Linux and a second simulator where the supported SystemVerilog subset permits it. Vendor simulation, synthesis, implementation, and programming are separate board-specific gates; passing the portable simulator does not establish timing or FPGA compatibility.

FPGA selection follows small synthesis probes for SRAM, MAC, DMA, and host-interface structures on at least two plausible targets. Run these probes before purchasing a board. The choice must be based on measured memory mapping, DSP use, external-memory and camera support, host/software integration, tool and license availability, cost, lead time, and accessibility to university teams.

M9 bring-up order:

- [ ] Pin the portable RTL simulator versions and run the M4 binary conformance cases through the RTL harness.
- [ ] Synthesize resource probes for at least two candidate FPGA families and publish tool versions, commands, inferred memories, DSP use, timing, and warnings.
- [ ] Select and purchase a board only after the probes show that the v0 core, external-memory interface, and debug instrumentation fit with margin.
- [ ] Bring up register access, reset, DMA, and a stored-image inference before adding a live camera; preserve each failure as a regression.
- [ ] Integrate one documented USB or direct sensor path and state exactly which capture, resize, color conversion, normalization, and postprocessing steps run on the host or FPGA.
- [ ] Measure layerwise correctness and COCO accuracy from the actual HASLAB command path before presenting live detections as evidence.
- [ ] Measure cold and warm camera-to-box latency, sustained throughput, dropped frames, transfer traffic, thermals, power, and accuracy under the [hardware measurement protocol](../benchmarks/MEASUREMENT.md).

## M10 — second perception workload and reprogramming proof

- [ ] Choose a separately licensed and independently pinned vision/perception workload after the first detector is complete; prefer a graph that exposes a useful new operator or memory pattern.
- [ ] Keep the FPGA bitstream unchanged. Record compiler/package hashes, compilation time, changed constant/package bytes, required host code, and unsupported operations.
- [ ] Publish task-appropriate accuracy, layerwise agreement, end-to-end latency, throughput, power, and memory traffic using the same measurement discipline as the detector.
- [ ] If a small policy or non-vision inference graph is added, treat it as an additional workload and do not replace the second perception proof or delay the complete detector.

Exit criterion: the evidence shows model reprogramming on one hardware image rather than a second fixed-function demonstration.

## Project-wide controls

- Maintain one source of truth for opcodes, fields, layouts, enums, error codes, and ABI versions.
- Generate language-specific constants from that source when practical.
- Pin dependencies, exporters, simulators, synthesis tools, and relevant command lines.
- Keep deterministic tests; record random seeds and artifact hashes.
- Test error paths with the same care as successful execution.
- Compare intermediate tensors instead of relying only on final detections.
- Keep the public workload order explicit: detector first, live perception second, then compatible policy or other inference experiments.
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
| 2026-09-20 | Implemented M4 candidate package: 46 independently authored binary fixtures, schema v1, pinned contract/registry hashes, strict loader, simulator runner, and ABI consistency checks. Added 19 infrastructure tests and integrated the corpus into CI. Stable M3/M4 release remains pending independent review; next implementation task is M5 workload pinning/audit. | [Package](../conformance/README.md), [format](../conformance/FORMAT.md), [ADR 0002](decisions/0002-conformance-corpus-format.md) |
| 2026-09-20 | M4 candidate validation passed: 104 unit tests (43 numerical + 42 simulator + 19 conformance infrastructure), all 46 binary fixture cases, byte-for-byte corpus reproduction, repository checks, and whitespace checks. | `make test`, `make check`, `git diff --check` |
| 2026-09-21 | Activated M5 with a pinned YOLOv8n v8.3.0 weight, reproducible 320×320 opset-13 export, complete 261-node/149-constant inventory, explicit learned-head boundary, deterministic FLOAT smoke baseline, and per-convolution v0 memory audit. Structural coverage passes; full COCO FLOAT accuracy, INT8 calibration, and total schedule/traffic remain open. | [Manifest and report](../benchmarks/manifests/yolov8n-320-opset13/README.md), [audit tool](../benchmarks/tools/audit_yolov8n.py) |
| 2026-09-21 | Completed the pinned FLOAT accuracy baseline over all 5,000 COCO val2017 images: bbox mAP50–95 0.284969 and mAP50 0.413595. A second full run reproduced the aggregate metrics, every per-class AP pair, and the 79,449,100-byte prediction JSON hash exactly. INT8 calibration is the next M5 step. | [Evaluation report](../benchmarks/manifests/yolov8n-320-opset13/float-coco-baseline.json), [reproduction guide](../benchmarks/manifests/yolov8n-320-opset13/README.md) |
| 2026-09-21 | Completed M5 calibration evidence with a deterministic 512-image train2017 subset, 205 activation scales, 63 per-output-channel weight-scale vectors, 57 SiLU tables, and diagnostics over 6.93 billion activation values. The INT8 QOperator proxy measured 0.276098 mAP50–95, a 0.887-point loss that passes the one-point budget. Calibration artifacts and two full evaluations reproduced exactly. Exact HASLAB command semantics remain the first M6 differential target. | [Calibration](../benchmarks/manifests/yolov8n-320-opset13/int8-calibration.json), [diagnostics](../benchmarks/manifests/yolov8n-320-opset13/int8-diagnostics.json), [accuracy](../benchmarks/manifests/yolov8n-320-opset13/int8-coco-baseline.json) |
| 2026-09-21 | Activated M6 with a strict first-block compiler/runtime slice. One 8×8 tile and output channels 0–7 compile to a deterministic 5,056-byte `.hxb`, execute as eight ABI commands, and match all 512 integer golden values; the dequantized tile is also compared with ONNX Runtime. Package binaries remain ignored because they contain derived third-party weights. | [Slice report](../benchmarks/manifests/yolov8n-320-opset13/m6-first-conv-silu-slice.json), [compiler](../compiler/README.md), [runtime](../runtime/README.md) |
| 2026-09-22 | Completed the full first-layer M6 schedule: 400 spatial tiles, two channel groups, explicit boundary halos, canonical HWC8 assembly, 8,884 commands, and 2,250,768 DMA bytes. All 409,600 outputs match the integer golden model; the runtime records 8,876 FIFO `BUSY`/refill events and an eight-command final drain. | [First-layer report](../benchmarks/manifests/yolov8n-320-opset13/m6-first-conv-silu-layer.json), [schedule format](../compiler/first-layer-format.md) |
| 2026-09-22 | Generalized M6 to a reusable sequential Conv-SiLU scheduler and executed nodes 0–5 as one package. Both HWC8 intermediates remain allocated and all 614,400 values match exactly. The schedule contains 16,245 commands, 1,999,320 DMA bytes, two second-layer input chunks, and 40,550,400 MACs; the runtime records 16,237 FIFO refill events. | [Two-layer report](../benchmarks/manifests/yolov8n-320-opset13/m6-first-two-conv-silu-layers.json), [schedule format](../compiler/multi-layer-format.md) |
