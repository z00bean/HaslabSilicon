# YOLOv8n 320×320 workload candidate

This directory pins the first concrete HASLAB workload candidate. The FLOAT export has been reproduced, checked by ONNX, inventoried node by node, partitioned at the learned-head boundary, checked against the proposed v0 local memories, and evaluated twice over all 5,000 COCO 2017 validation images with identical predictions. A deterministic 512-image train2017 subset has also been calibrated to signed symmetric INT8 and an executable software proxy passes the one-percentage-point accuracy budget. A reusable scheduler now compiles the first two Conv-SiLU blocks, retains both outputs, and matches all 614,400 INT8 values exactly in the functional simulator. The remaining model, RTL, and hardware remain unimplemented.

## Third-party artifact and license

The weight file comes from the [Ultralytics v8.3.0 asset release](https://github.com/ultralytics/assets/releases/tag/v8.3.0). Ultralytics publishes its code and models under its [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/21162bd870444550286983a601afbfb142f4c198/LICENSE) and offers separate enterprise terms. HASLAB does not redistribute or relicense the weight or exported ONNX file. Review the upstream terms for the intended use before downloading either artifact.

Expected local paths are ignored by Git:

```text
models/downloads/yolov8n.pt
models/generated/yolov8n-320-opset13.onnx
models/generated/yolov8n-320-opset13-int8-qoperator.onnx
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
| COCO bbox mAP50–95 | 28.4969 (`0.2849691922`) |
| COCO bbox mAP50 | 41.3595 (`0.4135947784`) |
| Ultralytics matched precision | 0.5715859845 |
| Ultralytics matched recall | 0.3857629412 |
| Validation images | 5,000 |
| Kept non-crowd boxes | 36,335 |
| Prediction JSON SHA-256 | `4272b93765f6d6a7000d28193f9219ba56e48eb8842ba716000c90965a5a3ed3` |

Two complete runs produced the same aggregate metrics, every per-class AP value, and prediction-file hash. Runtime measurements are recorded for diagnostics but are not treated as a performance result. See [`float-coco-baseline.json`](float-coco-baseline.json) for the full evidence.

## Reproduce INT8 calibration and evaluation

Extract the train annotation file from the already pinned annotation archive. The selection tool sorts every train2017 filename by the SHA-256 of its UTF-8 filename, uses the filename as an explicit tie-breaker, and takes the first 512. It downloads only that 81.6 MB subset rather than requiring the full train image archive.

```sh
unzip datasets/downloads/annotations_trainval2017.zip \
  annotations/instances_train2017.json -d datasets/coco

python benchmarks/tools/prepare_coco_train_calibration.py \
  --dataset-root datasets/coco \
  --annotation-archive datasets/downloads/annotations_trainval2017.zip \
  --download \
  --output benchmarks/manifests/yolov8n-320-opset13/calibration-selection.json

python benchmarks/tools/quantize_yolov8n_int8.py \
  models/generated/yolov8n-320-opset13.onnx \
  --dataset-root datasets/coco \
  --selection-manifest benchmarks/manifests/yolov8n-320-opset13/calibration-selection.json \
  --output-model models/generated/yolov8n-320-opset13-int8-qoperator.onnx \
  --output-report benchmarks/manifests/yolov8n-320-opset13/int8-calibration.json \
  --output-luts benchmarks/manifests/yolov8n-320-opset13/int8-silu-luts.bin

python benchmarks/tools/diagnose_yolov8n_int8.py \
  models/generated/yolov8n-320-opset13.onnx \
  models/generated/yolov8n-320-opset13-int8-qoperator.onnx \
  --dataset-root datasets/coco \
  --output benchmarks/manifests/yolov8n-320-opset13/int8-diagnostics.json

python benchmarks/tools/evaluate_yolov8n_coco.py \
  models/generated/yolov8n-320-opset13-int8-qoperator.onnx \
  --dataset-root datasets/coco \
  --expected-sha256 48f967b5a756f47163176195c0efb8bd4e2467e658c0f49fb89e9d8479d15d3e \
  --run-name yolov8n-320-opset13-int8-qoperator \
  --profile INT8-QOperator-proxy \
  --output benchmarks/manifests/yolov8n-320-opset13/int8-coco-baseline.json
```

Calibration uses ONNX Runtime 1.20.1 MinMax ranges forced symmetric around zero. Activations use one signed INT8 scale per tensor; all 63 convolution weight tensors use signed INT8 scales per output channel; all 268 zero points are zero. The package records 205 activation scales, 63 weight-scale vectors, and 57 independently generated 1,024-byte SiLU tables. Each LUT uses `delta = binary32(pre-SiLU scale × 127 / 511)`, so indices −512 through 511 cover the calibrated symmetric preactivation range; per-channel accumulator-to-grid ratios are serialized as v0 multipliers and shifts with their maximum approximation error. Two complete calibrations produced byte-identical quantized models and LUT binaries.

| Measurement | FLOAT | INT8 proxy | Change |
|---|---:|---:|---:|
| COCO bbox mAP50–95 | 28.4969 | 27.6098 | −0.8871 points |
| COCO bbox mAP50 | 41.3595 | 40.4444 | −0.9151 points |
| Ultralytics matched precision | 0.5715859845 | 0.5825443393 | +0.0109583548 |
| Ultralytics matched recall | 0.3857629412 | 0.3742943314 | −0.0114686098 |

The mAP50–95 loss is **0.887 percentage points**, inside the specified maximum of 1.0 percentage point. Two complete INT8 evaluations produced identical aggregate metrics, all 80 per-class AP pairs, and prediction JSON SHA-256 `2793fd75bf77628a9af92fabadc428f6b3b36c2944f687ecc00fc33e57e73216`.

The full diagnostic covers 215 quantized accelerator-region tensors over all 512 calibration images: 6,927,155,200 activation values with a 0.00852% clipping fraction, plus 3,146,160 weights with zero clipping. See [`int8-calibration.json`](int8-calibration.json), [`int8-diagnostics.json`](int8-diagnostics.json), and [`int8-coco-baseline.json`](int8-coco-baseline.json).

This accuracy result is a calibrated **ONNX Runtime QOperator software proxy**. Its QLinear sigmoid/multiply path does not execute the HASLAB fused SiLU LUT, and it requantizes final learned-head outputs where the v0 architecture preserves an INT32 boundary. It therefore establishes that the calibration candidate meets the budget, but it is not evidence of exact command-level or hardware agreement.

## Reproduce the first exact HASLAB command slice

Use the pinned export environment, with the repository packages on `PYTHONPATH`:

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_vertical_slice.py
```

The tool validates the model and calibration hashes, compiles nodes 0–2, emits a deterministic package under ignored `artifacts/m6/`, executes all eight commands through the simulator runtime, and compares 512 INT8 output values against `haslab_ref`. It also exposes the same intermediate tensor as an ONNX Runtime output and records dequantized error for a deterministic 320×320 input.

| Slice measurement | Result |
|---|---:|
| Package bytes | 5,056 |
| Package SHA-256 | `a2e420402f27fc5af59970c5e9140e85794d896324e6d2017e203d4d60c338ab` |
| Commands | 8 |
| Exact integer values compared | 512 |
| Integer mismatches | 0 |
| FLOAT-reference mean absolute error | 0.095207 |
| INT8 saturation count | 0 |

The checked-in [`m6-first-conv-silu-slice.json`](m6-first-conv-silu-slice.json) preserves the initial narrow proof. It does not establish package-schema stability.

## Reproduce the complete first layer

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_first_layer.py
```

The tool compiles all 400 spatial tiles and both output-channel groups, generates top/left zero halos, assembles the complete canonical 160×160×2×8 HWC8 output, and compares all 409,600 values with the independent integer model. It separately compares the dequantized layer with ONNX Runtime. Model-derived package, input, and output binaries remain under ignored `artifacts/m6/`.

| Complete first-layer measurement | Result |
|---|---:|
| Package bytes | 1,671,680 |
| Package SHA-256 | `2984ff498715f4430bc63f2585c68e0465b5b4b83643c18fb5b509c0b1aba0c6` |
| Commands | 8,884 |
| DMA bytes | 2,250,768 |
| INT8 MACs | 11,059,200 |
| Boundary fill commands | 78 |
| Exact integer values compared | 409,600 |
| Integer mismatches | 0 |
| FLOAT-reference mean absolute error | 0.071028 |
| INT8 saturation count | 0 |
| FIFO `BUSY`/refill events | 8,876 |
| FIFO high-water mark | 8 |

The checked-in [`m6-first-conv-silu-layer.json`](m6-first-conv-silu-layer.json) contains the exact command mix, byte traffic, hashes, comparison, and runtime FIFO counts. These FIFO counts describe the synchronous functional runtime; they are not cycle timing or FPGA throughput.

## Reproduce the first two layers

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_first_two_layers.py
```

The reusable scheduler validates nodes 0–5, retains the first `16×160×160` tensor, gathers its two HWC8 input groups into separate eight-channel reduction chunks, and produces the second `32×80×80` tensor across four output groups. The integration compares both boundaries independently with `haslab_ref` and with ONNX Runtime.

| Two-layer measurement | Result |
|---|---:|
| Package bytes | 3,083,456 |
| Package SHA-256 | `2178e93d7d5277b0cdce00a3c1ae5f0712f4be46ec0b2621266ef26e906ec9d8` |
| Retained external tensors | 409,600 + 204,800 bytes |
| Total declared external allocation | 1,442,176 bytes |
| Commands | 16,245 |
| DMA bytes | 1,999,320 |
| INT8 MACs | 40,550,400 |
| Exact integer values compared | 614,400 |
| Integer mismatches | 0 |
| Second-layer FLOAT-reference mean absolute error | 0.197399 |
| Second-layer INT8 saturation count | 0 |
| FIFO `BUSY`/refill events | 16,237 |
| FIFO high-water mark | 8 |

The checked-in [`m6-first-two-conv-silu-layers.json`](m6-first-two-conv-silu-layers.json) records per-layer and cumulative allocation, command mix, traffic, numerical comparisons, hashes, and FIFO behavior. The command reduction relative to concatenating two first-layer-style schedules comes from loading each spatial input tile once and reusing it across output groups.

The next implementation step is the first C2f block beginning at model node 6. It requires 1×1 convolution lowering, split/view handling, concat liveness, residual addition with explicit scales, and cumulative allocation across a branched subgraph.
