# YOLOv8n 320×320 workload candidate

This directory pins the first concrete HASLAB workload candidate. The FLOAT export has been reproduced, checked by ONNX, inventoried node by node, partitioned at the learned-head boundary, checked against the proposed v0 local memories, and evaluated twice over all 5,000 COCO 2017 validation images with identical predictions. The model has **not** been calibrated to INT8, compiled to HASLAB commands, or run on RTL.

## Third-party artifact and license

The weight file comes from the [Ultralytics v8.3.0 asset release](https://github.com/ultralytics/assets/releases/tag/v8.3.0). Ultralytics publishes its code and models under its [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/21162bd870444550286983a601afbfb142f4c198/LICENSE) and offers separate enterprise terms. HASLAB does not redistribute or relicense the weight or exported ONNX file. Review the upstream terms for the intended use before downloading either artifact.

Expected local paths are ignored by Git:

```text
models/downloads/yolov8n.pt
models/generated/yolov8n-320-opset13.onnx
```

## Reproduce the export

Create an isolated Python 3.11 environment from `export-requirements.txt`, then run:

```sh
mkdir -p models/downloads models/generated
curl --fail --location \
  --output models/downloads/yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
shasum -a 256 models/downloads/yolov8n.pt

python -c "from ultralytics import YOLO; YOLO('models/downloads/yolov8n.pt').export(format='onnx', imgsz=320, batch=1, dynamic=False, simplify=False, opset=13, nms=False, half=False, device='cpu')"
python benchmarks/tools/canonicalize_onnx.py \
  models/downloads/yolov8n.onnx \
  models/generated/yolov8n-320-opset13.onnx
shasum -a 256 models/generated/yolov8n-320-opset13.onnx
```

Ultralytics 8.3.40 places the wall-clock export time and a filename-derived description in ONNX metadata, so its raw export is not byte reproducible. `canonicalize_onnx.py` removes only that volatility, checks the graph, sorts metadata, and writes deterministic protobuf bytes. Two exports with identical nodes and initializers must then match the canonical hash in `manifest.json`. A different canonical hash is a different workload artifact even when its predictions appear similar. Record the platform and resolved dependency artifacts before accepting a new hash.

## Reproduce the audit

```sh
python benchmarks/tools/audit_yolov8n.py \
  models/generated/yolov8n-320-opset13.onnx \
  --expected-sha256 5b232bf21720cac4264463896ace02b919d8bec3f3fd1e4b9d36695493fa1718 \
  --output benchmarks/manifests/yolov8n-320-opset13/graph-audit.json

python benchmarks/tools/yolov8n_float_baseline.py \
  models/generated/yolov8n-320-opset13.onnx \
  --weights models/downloads/yolov8n.pt \
  --output benchmarks/manifests/yolov8n-320-opset13/float-smoke-baseline.json
```

`graph-audit.json` records every node, inferred shape, attribute, initializer, graph constant, classification, and convolution tile calculation. The audit assigns nodes 0–217 to the accelerator and nodes 218–260 to the host tail. Its three boundary tensors contain 64 DFL regression logits followed by 80 class logits at strides 8, 16, and 32.

The structural audit found no unsupported accelerator nodes and all proposed convolution tiles fit local memory. The documented 8×8×32 reference tiling produces a subtotal of 38,696 `CONV` plus `EPILOGUE` commands per frame. Larger legal tiles may reduce that subtotal, while DMA, fill, utility, concat, fence, and end commands add to it. This is a feasibility warning, not a throughput estimate; M6 must optimize and generate the real schedule, and M9 must measure command refill behavior.

`float-smoke-baseline.json` uses a deterministic synthetic tensor to check the export against PyTorch. It is an export-equivalence check; detection quality is measured separately below.

## Reproduce the COCO FLOAT baseline

Download the official [COCO 2017 validation images and annotations](https://cocodataset.org/#download), and review the [COCO terms of use](https://cocodataset.org/#termsofuse). The dataset remains third-party material and is ignored by Git.

```sh
mkdir -p datasets/downloads datasets/coco
curl --fail --location \
  --output datasets/downloads/val2017.zip \
  http://images.cocodataset.org/zips/val2017.zip
curl --fail --location \
  --output datasets/downloads/annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip datasets/downloads/val2017.zip -d datasets/coco/images
unzip datasets/downloads/annotations_trainval2017.zip \
  annotations/instances_val2017.json -d datasets/coco

python benchmarks/tools/prepare_coco_val2017.py \
  --dataset-root datasets/coco \
  --image-archive datasets/downloads/val2017.zip \
  --annotation-archive datasets/downloads/annotations_trainval2017.zip

python benchmarks/tools/evaluate_yolov8n_coco.py \
  models/generated/yolov8n-320-opset13.onnx \
  --dataset-root datasets/coco \
  --output benchmarks/manifests/yolov8n-320-opset13/float-coco-baseline.json
```

The evaluator uses the static FLOAT32 ONNX model through ONNX Runtime's CPU provider, 320×320 square batches of one, confidence threshold 0.001, class-aware NMS at IoU 0.7 with at most 300 candidate detections, and the standard COCO bbox AP cap of 100 detections. The machine-readable report includes all 80 per-class AP pairs and exact tool versions.

| Measurement | Recorded result |
|---|---:|
| COCO bbox mAP50–95 | 0.2849691922 |
| COCO bbox mAP50 | 0.4135947784 |
| Ultralytics matched precision | 0.5715859845 |
| Ultralytics matched recall | 0.3857629412 |
| Validation images | 5,000 |
| Kept non-crowd boxes | 36,335 |
| Prediction JSON SHA-256 | `4272b93765f6d6a7000d28193f9219ba56e48eb8842ba716000c90965a5a3ed3` |

Two complete runs produced the same aggregate metrics, every per-class AP value, and prediction-file hash. Runtime measurements are recorded for diagnostics but are not treated as a performance result. See [`float-coco-baseline.json`](float-coco-baseline.json) for the full evidence.

M5 remains open for calibrated INT8 comparison and optimized whole-graph scheduling/traffic evidence. The next implementation step is to pin the COCO train2017 calibration subset, generate all quantization parameters, and compare INT8 accuracy with this FLOAT baseline.
