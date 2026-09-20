# HASLAB

**Open silicon for edge intelligence.**

HASLAB is an open-source physical-AI inference-engine project for computer vision, robotics, and edge AI. The project aims to develop a programmable, inspectable path from exported neural-network models to open hardware: software reference model → RTL simulation → FPGA → open ASIC flow → fabricated silicon.

Physical-AI systems need perception results in a form that can feed a controller or policy with predictable, measured behavior. HASLAB focuses on the inference portion of that path: moving sensor or host data through explicit memory, running quantized neural-network operations, and returning outputs to a host or future action pipeline. It is not a complete robot-control stack, an ISP, or a training system.

## Status

The repository is in the **reference-model and architecture stage**. It contains specifications, an executable Python numerical golden model, and a functional v0 command simulator. No tensor accelerator RTL, compiler, runtime, FPGA design, cycle-accurate simulator, or supported model execution exists yet.

## Target workloads

- Modern YOLO-class object detection and contemporary CNN vision models
- Small policy networks and robotics perception-to-action pipelines
- Later, selected compact transformers where the required operations and memory traffic are demonstrated feasible

The proposed v0 workload is a pinned YOLOv8n detector at batch one with 320×320 RGB input. This is a proposed validation target, not an implemented feature or performance claim.

## Proposed architecture

The planned system separates model format from accelerator commands:

```text
Framework model → ONNX → HASLAB compiler → target-specific package
                                             ↓
Application ← HASLAB runtime ← FPGA / future ASIC
```

The long-term design includes an INT8/FP8 tensor engine, explicit SRAM and DMA, vector/utility operations, and a RISC-V control core. The proposed v0 FPGA profile deliberately starts smaller: host-controlled INT8 convolution, INT32 accumulation, bounded utility operations, and serialized DMA. Native FP8 and RISC-V integration are planned for a later FPGA stage only after the INT8 path is validated.

| Stage | Precision and control | Scope |
|---|---|---|
| v0 FPGA | INT8 inputs/weights, INT32 accumulation; host control | One complete, explicitly partitioned detector pipeline |
| v1 FPGA | Expanded INT8 and native FP8 candidate; RISC-V control | Measured programmable inference prototype |
| ASIC | Configuration chosen from measured PPA and memory traffic | Physically viable implementation, subject to flow and fabrication validation |

Read the [revised architecture](docs/haslab-v0-revised-architecture.md) and [v0 software/hardware contract](docs/haslab-v0-contract.md) before adding implementation work. The [original architecture plan](docs/architecture-plan.md) remains background material.

## Repository map

| Area | Purpose today |
|---|---|
| `hardware/rtl/` | Future portable synthesizable RTL; intentionally empty of logic |
| `hardware/testbenches/` | Future RTL testbenches and test vectors |
| `simulation/` | Functional v0 command/memory simulator; future RTL harnesses |
| `fpga/` | Future board targets, constraints, and build wrappers |
| `software/` | Future host-facing software and firmware support |
| `compiler/` | Future graph lowering, scheduling, and package generation |
| `runtime/` | Future C-compatible runtime and platform backends |
| `onnx/` | ONNX profile, importer, and operator-coverage work |
| `reference/` | Python numerical golden model and its unit tests |
| `benchmarks/` | Workload manifests, evaluation recipes, and results schema |
| `docs/` | Architecture, contracts, decisions, and contributor documentation |
| `scripts/` | Reproducible developer and CI helpers |
| `.github/` | Continuous-integration workflows and issue templates |

The root [Makefile](Makefile) provides repository checks and Python reference/simulator tests. It does not build accelerator hardware or a model runtime.

## FPGA and ASIC direction

The first FPGA target will use an existing board memory/host transport and keep vendor-specific wrappers outside portable accelerator logic. The future ASIC path will require separately validated SRAM macros, IO, clocking, power, test, packaging, and fabrication collateral. Open RTL-to-GDS tools are useful infrastructure but are not alone evidence of tapeout readiness.

## Contributions

Contributions are welcome once the project begins accepting implementation work. Proposed changes should preserve the documented command and numerical contracts, state their verification method, avoid unmeasured performance claims, and keep third-party model, dataset, and tool licenses explicit. The contribution process, code of conduct, and issue templates are placeholders to be completed before broad implementation contributions are requested.

## Licensing

Original hardware designs and design documentation are licensed under **CERN-OHL-S-2.0**. Original software is licensed under **GPL-3.0-or-later**. Both allow commercial use under their terms. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [licensing notes](docs/licensing.md).

The local `haslab-site/` directory is reserved for website assets and intentionally ignored by Git. Use a separate repository or direct deployment workflow when a public site is ready.
