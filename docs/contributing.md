# Contributing to HASLAB

HASLAB welcomes focused reviews, reproducibility reports, failing test cases, and bounded implementation changes. The v0 command ABI and package format are candidates, so proposals that change them need extra review. Please discuss broad interface changes before building on an assumed stable API.

## Useful work now

| Area | Concrete contribution | Evidence to include |
|---|---|---|
| Contract and conformance | Independently review an opcode, state transition, or fixture derivation; add a minimal positive, boundary, or negative case | Contract rule, hand-derived expected bytes/status, and `make conformance` result |
| Compiler and runtime | Lower nodes 120–136 through the second top-down neck branch | Exact `UPSAMPLE2_I8`, concat with the long-lived `model.4` skip, non-residual `model.15` C2f lowering, diagnostic versus release allocation, exact boundary comparisons, and `make test` result |
| Workload reproduction | Repeat the pinned export, calibration, or COCO evaluation on a documented environment | Artifact hashes, versions, commands, accuracy results, and any mismatch from checked-in reports |
| Hardware preparation | Prototype a small SRAM/MAC/DMA structure or review a board constraint without claiming v0 FPGA support | Source, tool versions, synthesis or simulation output, and limitations |
| Documentation | Correct a stale status statement or make an experiment easier to reproduce | Link to the implementation or report that supports the change |

The [development plan](development-plan.md) is the status and dependency record. The [v0 contract](haslab-v0-contract.md), [conformance format](../conformance/FORMAT.md), and [reusable graph schedule](../compiler/reusable-graph-format.md) are good starting points. Run `make check`, `make test`, and `git diff --check` for changes that affect executable behavior. Documentation-only changes need `make check` and link/status review.

## Submission expectations

- State the affected contract section, command, operator, or workload node and the reason for the change.
- Pin any model, dataset, tool, PDK, or vendor-IP version used to produce evidence. Do not commit third-party weights or datasets without an explicit redistribution grant.
- Keep expected conformance values independent of the implementation under test. Explain how they were derived.
- Report unsupported behavior and host fallbacks explicitly. Do not imply simulator counts are hardware timing, or proxy accuracy is exact HASLAB execution.
- Include the smallest test that would fail before a behavioral fix. Record random seeds and artifact hashes when relevant.
- Propose interface or numerical changes through a short decision record under [decisions/](decisions/). A candidate ABI change must update the registry, specification, fixtures, and consumers together.
- Follow the [licensing guidance](licensing.md) and add file-level SPDX identifiers to new source files. Contributors retain their copyright under the applicable project license; no assignment or relicensing agreement is assumed.

Before opening a broad feature branch, check the milestone dependencies. RTL work can start as an isolated experiment, while a stable v0 RTL release waits for independent ABI and fixture review. Hardware performance claims require the [measurement protocol](../benchmarks/MEASUREMENT.md).
