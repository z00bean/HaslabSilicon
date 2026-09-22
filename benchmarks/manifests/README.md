# Workload manifests

Each manifest pins a third-party model identity, exporter, input and preprocessing contract, graph partition, output conventions, audit evidence, and the remaining accuracy/calibration gates. Model weights, exported graphs, and datasets are not tracked here.

Current candidate:

- [`yolov8n-320-opset13/`](yolov8n-320-opset13/README.md) — exact FLOAT artifact, graph inventory, host-tail boundary, v0 structural audit, reproducible FLOAT accuracy, deterministic 512-image INT8 calibration, layerwise diagnostics, a software proxy inside the stated accuracy budget, and exact compiler/runtime results through the first two Conv-SiLU blocks. Remaining-layer and whole-model command execution remain pending.
