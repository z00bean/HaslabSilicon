# HASLAB compiler

The compiler area now executes pinned YOLOv8n nodes 0–21 through the complete first C2f block. It validates fused `Conv → Sigmoid → Mul` patterns, 1×1 and 3×3 convolutions, zero-allocation split views, scaled residual addition, and materialized channel concat. It consumes the recorded calibration scales and LUTs, emits deterministic fixed-point parameters, and retains every operator boundary needed for differential validation.

The implementation lives in `haslab_compiler/`. `hxb.py` writes the sectioned container proposed by the v0 contract, `vertical_slice.py` retains the initial one-tile proof, `first_layer.py` retains the complete first-layer proof, `pipeline.py` implements sequential layer planning, and `c2f.py` adds branch-aware allocation and scheduling. Identical inputs produce byte-identical packages. The [first-C2f format note](first-c2f-format.md) records the latest schedule, numerical evidence, and scaling limit. The tracked workload reports record package hashes without redistributing the third-party weights embedded in ignored local packages.

This is not a general ONNX compiler. Its branch lowering is pinned to the first C2f topology; it does not yet have a reusable graph IR, release-oriented liveness allocation, whole-graph scheduling, all audited convolution forms, or the declared host tail. The experimental manifest schema is intentionally narrow and must be reviewed before it becomes a stable package interface.

Run the source-independent compiler tests through `make test`. Reproduce the pinned real-model layer in the workload guide under `benchmarks/manifests/yolov8n-320-opset13/`.
