# Workload manifests

Each manifest pins a third-party model identity, exporter, input and preprocessing contract, graph partition, output conventions, audit evidence, and the remaining accuracy/calibration gates. Model weights, exported graphs, and datasets are not tracked here.

Current candidate:

- [`yolov8n-320-opset13/`](yolov8n-320-opset13/README.md) — exact FLOAT artifact, graph inventory, host-tail boundary, deterministic smoke comparison, v0 structural feasibility audit, and reproducible full COCO val2017 FLOAT baseline; INT8 calibration and full schedule evidence pending.
