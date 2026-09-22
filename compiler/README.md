# HASLAB compiler

The compiler area now has a reusable sequential Conv-SiLU scheduler and executes the first two M6 layers. It validates nodes 0–5 of the pinned YOLOv8n opset-13 graph as connected `Conv → Sigmoid → Mul` SiLU patterns, extracts constants, consumes the recorded calibration scales and LUTs, packs channel-chunked INT8 weights, derives INT32 bias records, and emits one deterministic `.hxb` that retains both layer outputs.

The implementation lives in `haslab_compiler/`. `hxb.py` writes the sectioned container proposed by the v0 contract, `vertical_slice.py` retains the initial one-tile proof, `first_layer.py` retains the complete first-layer proof, and `pipeline.py` implements connected layer planning, input-channel reduction chunks, intermediate allocation, tile reuse, canonical HWC8 assembly, and cumulative accounting. Identical inputs produce byte-identical packages. The [multi-layer format note](multi-layer-format.md) records the generated schedule and current limits. The tracked workload reports record package hashes without redistributing the third-party weights embedded in ignored local packages.

This is not a general ONNX compiler. It does not yet lower C2f split/concat/residual paths, allocate the whole graph, support all audited convolution forms, or emit the declared host tail. The experimental manifest schema is intentionally narrow and must be reviewed before it becomes a stable package interface.

Run the source-independent compiler tests through `make test`. Reproduce the pinned real-model layer in the workload guide under `benchmarks/manifests/yolov8n-320-opset13/`.
