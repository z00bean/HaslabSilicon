# Benchmarks

This directory holds workload manifests, export/calibration recipes, evaluation protocols, and machine-readable results. It must not contain unlicensed model weights, datasets, or unverifiable performance claims.

The first workload candidate is [YOLOv8n at batch one, 320×320](manifests/yolov8n-320-opset13/README.md). Its exact weight and export are pinned, its FLOAT ONNX graph has been structurally audited against HASLAB v0, and its full 5,000-image COCO val2017 FLOAT baseline is reproducible. INT8 calibration, compilation, and hardware measurements remain pending.

Preparation, audit, and evaluation tools live under `tools/`. The generated reports are checked in so graph and accuracy conclusions can be reviewed without committing third-party weights or datasets. Local model artifacts, predictions, and datasets stay in the ignored `models/` and `datasets/` directories.
