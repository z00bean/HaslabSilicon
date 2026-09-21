# HASLAB compiler

The compiler area now contains the first M6 vertical slice. It validates nodes 0–2 of the pinned YOLOv8n opset-13 graph as an exact `Conv → Sigmoid → Mul` SiLU pattern, extracts constant weights and bias, consumes the recorded calibration scales and LUT, packs INT8 data as KHWCI8, derives INT32 bias records, and emits an eight-command `.hxb` package for one 8×8 tile and output channels 0–7.

The implementation lives in `haslab_compiler/`. `hxb.py` writes the sectioned container proposed by the v0 contract, and `vertical_slice.py` performs the bounded import and lowering. Identical inputs produce byte-identical packages. The [experimental format note](vertical-slice-format.md) records the exact sections, command schedule, relocations, and remaining schema work. The tracked workload report records the generated package hash without redistributing the third-party weights embedded in the ignored local package.

This is not a general ONNX compiler. It does not yet schedule the complete first layer, handle boundary halos, allocate a whole graph, lower residual/concat paths, or emit the declared host tail. The experimental manifest schema is intentionally narrow and must be reviewed before it becomes a stable package interface.

Run the source-independent compiler tests through `make test`. Reproduce the pinned real-model slice in the workload guide under `benchmarks/manifests/yolov8n-320-opset13/`.
