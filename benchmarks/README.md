# Benchmarks

This directory holds workload manifests, export/calibration recipes, evaluation protocols, and machine-readable results. It must not contain unlicensed model weights, datasets, or unverifiable performance claims.

The first workload candidate is [YOLOv8n at batch one, 320×320](manifests/yolov8n-320-opset13/README.md). Its exact weight and export are pinned, its FLOAT ONNX graph has been structurally audited against HASLAB v0, and its FLOAT and calibrated signed-symmetric INT8 software baselines are reproducible over all 5,000 COCO val2017 images. One first-layer tile now compiles and executes through exact HASLAB command semantics; full-layer, whole-model, RTL, and hardware measurements remain pending.

Preparation, audit, and evaluation tools live under `tools/`. The generated reports are checked in so graph and accuracy conclusions can be reviewed without committing third-party weights or datasets. Local model artifacts, predictions, and datasets stay in the ignored `models/` and `datasets/` directories.
