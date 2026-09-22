# HASLAB compiler

The compiler area now schedules the complete first M6 layer. It validates nodes 0–2 of the pinned YOLOv8n opset-13 graph as an exact `Conv → Sigmoid → Mul` SiLU pattern, extracts constant weights and bias, consumes the recorded calibration scales and LUT, packs INT8 data as KHWCI8, derives INT32 bias records, and emits a deterministic `.hxb` for all 400 spatial tiles and both output-channel groups.

The implementation lives in `haslab_compiler/`. `hxb.py` writes the sectioned container proposed by the v0 contract, `vertical_slice.py` retains the initial one-tile proof, and `first_layer.py` generates boundary halos, weight/parameter reuse, canonical HWC8 output assembly, and schedule accounting. Identical inputs produce byte-identical packages. The [first-layer format note](first-layer-format.md) records the generated schedule and remaining schema work. The tracked workload reports record package hashes without redistributing the third-party weights embedded in ignored local packages.

This is not a general ONNX compiler. It does not yet lower the second or later layers, allocate a whole graph, handle input-channel reduction chunks and residual/concat paths across layers, or emit the declared host tail. The experimental manifest schema is intentionally narrow and must be reviewed before it becomes a stable package interface.

Run the source-independent compiler tests through `make test`. Reproduce the pinned real-model layer in the workload guide under `benchmarks/manifests/yolov8n-320-opset13/`.
