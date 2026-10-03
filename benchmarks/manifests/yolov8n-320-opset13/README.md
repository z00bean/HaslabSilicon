# YOLOv8n 320×320 workload candidate

This directory pins the first concrete HASLAB workload candidate. The FLOAT export has been reproduced, checked by ONNX, inventoried node by node, partitioned at the learned-head boundary, checked against the proposed v0 local memories, and evaluated twice over all 5,000 COCO 2017 validation images with identical predictions. A deterministic 512-image train2017 subset has also been calibrated to signed symmetric INT8 and an executable software proxy passes the one-percentage-point accuracy budget. The command path executes accelerator nodes 0–217, including all three learned detection-head branches. Diagnostic execution matches 6,931,200 values across 106 materialized or aliased boundaries, and a lifetime-reuse package exactly matches all 302,400 values in the three raw INT32 HWC8 host-boundary tensors. The explicit Python host tail executes nodes 218–260 and matches an isolated ONNX Runtime tail on the saved synthetic command output. Full command-path COCO evaluation, RTL, and hardware remain unimplemented.

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

## Reproduce the first C2f block

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_first_c2f.py
```

The compiler validates and executes nodes 0–21. The first C2f increment adds 1×1 and stride-one 3×3 convolutions, two zero-allocation 16-channel split views, a scaled residual add, six-group concat materialization, and a final 48-to-32-channel pointwise convolution. Every operator boundary remains available for differential checking.

| First-C2f measurement | Result |
|---|---:|
| Package bytes | 11,295,040 |
| Package SHA-256 | `c70702182333b77b8a108ff197cd7ed687f66771dfc6288ef27bd989718a9f5c` |
| Commands | 59,052 |
| DMA bytes | 4,368,728 |
| INT8 MACs | 86,425,600 |
| Materialized output allocation | 1,638,400 bytes |
| Total declared external allocation | 2,480,256 bytes |
| Zero-allocation split views | 2 |
| Exact integer values compared | 1,843,200 |
| Integer mismatches | 0 |
| FIFO `BUSY`/refill events | 59,044 |
| FIFO high-water mark | 8 |

The checked-in [`m6-first-c2f.json`](m6-first-c2f.json) records per-operation command and DMA counts, residual and concat coefficients, allocations, hashes, exact comparisons, and FLOAT-reference errors. The [format note](../../../compiler/first-c2f-format.md) explains the lowering and its current scaling limit.

## Reproduce the reusable graph through the second C2f

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_second_c2f.py
```

The compiler validates and executes nodes 0–47. The existing first C2f produces an `80×80×32` tensor in NCHW notation; the reusable extension adds the stride-two `model.3` convolution and the two-bottleneck `model.4` block, ending with a `40×40×64` tensor. Diagnostic mode retains every boundary. Release mode uses an auditable deterministic lifetime plan and exposes only `model.4`.

| Nodes 0–47 measurement | Diagnostic | Release |
|---|---:|---:|
| Package bytes | 18,596,288 | 18,530,880 |
| Peak output allocation | 2,457,600 | 614,400 |
| Declared external bytes | 3,383,424 | 1,540,224 |
| Commands | 97,670 | 97,670 |
| DMA bytes | 6,951,768 | 6,951,768 |
| INT8 MACs | 194,560,000 | 194,560,000 |
| Exact values compared | 2,764,800 | 102,400 final values |
| Integer mismatches | 0 | 0 |
| FIFO `BUSY`/refill events | 97,662 | 97,662 |

The checked-in [`m6-through-second-c2f.json`](m6-through-second-c2f.json) records both package hashes, tensor lifetimes, all operation counts, exact comparisons, FLOAT-reference errors, and FIFO behavior. The [format note](../../../compiler/reusable-graph-format.md) describes the IR, validation, and allocation rules.

## Reproduce the reusable graph through the third C2f

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_third_c2f.py
```

The compiler validates and executes nodes 0–73. The extension adds the stride-two `model.5` convolution and two-bottleneck `model.6` block, ending with a `20×20×128` tensor. The scheduler selects 5×5 output tiles for this region because 8×8 would exceed input SRAM or fail to divide 20 exactly. Wide convolution weights stream one output group per spatial tile, and the 4,096-byte concat mapping table streams one source at a time through the 3 KiB parameter SRAM.

| Nodes 0–73 measurement | Diagnostic | Release |
|---|---:|---:|
| Package bytes | 26,651,136 | 26,543,616 |
| Peak output allocation | 2,867,200 | 614,400 |
| Declared external bytes | 4,088,960 | 1,836,160 |
| Commands | 143,156 | 143,156 |
| DMA bytes | 12,116,376 | 12,116,376 |
| INT8 MACs | 302,694,400 | 302,694,400 |
| Exact values compared | 3,225,600 | 51,200 final values |
| Integer mismatches | 0 | 0 |
| FIFO `BUSY`/refill events | 143,148 | 143,148 |

The checked-in [`m6-through-third-c2f.json`](m6-through-third-c2f.json) records both package hashes, tensor lifetimes, dynamic tile choices, weight and parameter residency, exact comparisons, FLOAT-reference errors, and FIFO behavior.

## Reproduce the complete backbone through SPPF

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_backbone.py
```

The compiler validates and executes nodes 0–102. This extension adds the stride-two `model.7` convolution, the one-bottleneck `model.8` C2f, and `model.9` SPPF, ending with a `10×10×256` tensor. The three chained pools use `MAXPOOL5_I8` with explicit `-128` border halos. The 256-channel convolutions stream one epilogue record per output group while retaining the SiLU LUT, and the 128-channel residual streams one left/right parameter pair per group; both stay within the unchanged 3 KiB parameter SRAM.

| Nodes 0–102 measurement | Diagnostic | Release |
|---|---:|---:|
| Package bytes | 34,599,168 | 34,453,504 |
| Peak output allocation | 3,148,800 | 614,400 |
| Declared external bytes | 5,336,192 | 2,801,792 |
| Commands | 186,928 | 186,928 |
| DMA bytes | 16,602,136 | 16,602,136 |
| INT8 MACs | 394,444,800 | 394,444,800 |
| `MAXPOOL5_I8` commands | 192 | 192 |
| Exact values compared | 3,532,800 | 25,600 final values |
| Integer mismatches | 0 | 0 |
| FIFO `BUSY`/refill events | 186,920 | 186,920 |

The checked-in [`m6-through-backbone.json`](m6-through-backbone.json) records both package hashes, pooled tensor lifetimes, wide-parameter residency, exact comparisons, FLOAT-reference errors, and FIFO behavior.

## Reproduce the first top-down neck stage

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_first_neck.py
```

The compiler validates and executes nodes 0–119. This extension lowers the pinned asymmetric nearest-neighbor `model.10` Resize to exact `UPSAMPLE2_I8`, concatenates it with the earlier `model.6` backbone tensor, and executes the non-residual `model.12` C2f block. The lifetime plan retains `model.6` from operation 25 through its skip use at operation 40. The 256-channel upsample source requires 4 KiB of concat mapping records, so its records stream one channel group at a time within the unchanged 3 KiB parameter SRAM.

| Nodes 0–119 measurement | Diagnostic | Release |
|---|---:|---:|
| Package bytes | 42,995,264 | 42,819,200 |
| Peak output allocation | 3,635,200 | 614,400 |
| Declared external bytes | 5,989,504 | 2,968,704 |
| Commands | 235,474 | 235,474 |
| DMA bytes | 20,101,656 | 20,101,656 |
| INT8 MACs | 453,427,200 | 453,427,200 |
| `UPSAMPLE2_I8` commands | 128 | 128 |
| Exact values compared | 4,070,400 | 51,200 final values |
| Integer mismatches | 0 | 0 |
| FIFO `BUSY`/refill events | 235,466 | 235,466 |

The checked-in [`m6-through-first-neck.json`](m6-through-first-neck.json) records both package hashes, the long-lived skip allocation, upsample and wide-concat residency, non-residual branch comparisons, FLOAT-reference errors, and FIFO behavior.

## Reproduce the second top-down neck stage

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_second_neck.py
```

The compiler validates and executes nodes 0–136. This extension reuses the generic top-down neck lowering for exact `model.13` upsampling, `model.14` concat with the earlier `model.4` tensor, and the non-residual `model.15` C2f block. The release lifetime plan retains `model.4` from operation 15 through operation 47. This longer skip lifetime raises the measured release peak from 614,400 to 665,600 bytes.

| Nodes 0–136 measurement | Diagnostic | Release |
|---|---:|---:|
| Package bytes | 50,900,736 | 50,720,384 |
| Peak output allocation | 4,608,000 | 665,600 |
| Declared external bytes | 7,010,944 | 3,068,544 |
| Commands | 277,436 | 277,436 |
| DMA bytes | 22,243,352 | 22,243,352 |
| INT8 MACs | 512,409,600 | 512,409,600 |
| `UPSAMPLE2_I8` commands | 384 | 384 |
| Exact values compared | 5,145,600 | 102,400 final values |
| Integer mismatches | 0 | 0 |
| FIFO `BUSY`/refill events | 277,428 | 277,428 |

The checked-in [`m6-through-second-neck.json`](m6-through-second-neck.json) records both package hashes, the `model.4` skip lifetime, both upsample stages, exact intermediate comparisons, FLOAT-reference errors, and FIFO behavior.

## Reproduce the first bottom-up neck stage

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_first_bottom_up_neck.py
```

The compiler validates and executes nodes 0–154. The reusable bottom-up extension lowers stride-two `model.16`, concatenates its `20×20×64` output with the retained `20×20×128` `model.12` tensor, and executes the non-residual `model.18` C2f block. Release liveness retains `model.12` from operation 45 through operation 54 without increasing the 665,600-byte peak reached by the previous stage.

| Nodes 0–154 measurement | Diagnostic | Release |
|---|---:|---:|
| Package bytes | 57,364,608 | 57,150,976 |
| Peak output allocation | 4,940,800 | 665,600 |
| Declared external bytes | 7,521,920 | 3,246,720 |
| Commands | 314,439 | 314,439 |
| Command bytes | 40,248,192 | 40,248,192 |
| DMA bytes | 25,662,552 | 25,662,552 |
| INT8 MACs | 576,307,200 | 576,307,200 |
| Exact values compared | 5,529,600 | 51,200 final values |
| Integer mismatches | 0 | 0 |
| FIFO `BUSY`/refill events | 314,431 | 314,431 |

The checked-in [`m6-through-first-bottom-up-neck.json`](m6-through-first-bottom-up-neck.json) records both package hashes, the retained `model.12` lifetime, exact intermediate comparisons, FLOAT-reference errors, and FIFO behavior.

## Reproduce the second bottom-up neck stage

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_second_bottom_up_neck.py
```

The compiler validates and executes nodes 0–172, completing the pinned backbone and neck. The second bottom-up extension lowers stride-two `model.19`, concatenates its `10×10×128` output with the retained `10×10×256` `model.9` tensor, and executes the non-residual `model.21` C2f block. Release liveness retains `model.9` from operation 38 through operation 61, increasing the measured peak by 25,600 bytes to 691,200 bytes.

| Nodes 0–172 measurement | Diagnostic | Release |
|---|---:|---:|
| Package bytes | 62,233,792 | 61,998,464 |
| Peak output allocation | 5,107,200 | 691,200 |
| Declared external bytes | 8,359,040 | 3,943,040 |
| Commands | 340,926 | 340,926 |
| Command bytes | 43,638,528 | 43,638,528 |
| DMA bytes | 28,700,376 | 28,700,376 |
| INT8 MACs | 640,204,800 | 640,204,800 |
| Exact values compared | 5,721,600 | 25,600 final values |
| Integer mismatches | 0 | 0 |
| FIFO `BUSY`/refill events | 340,918 | 340,918 |

The checked-in [`m6-through-second-bottom-up-neck.json`](m6-through-second-bottom-up-neck.json) records both package hashes, the retained `model.9` lifetime, exact intermediate comparisons, FLOAT-reference errors, and FIFO behavior.

## Reproduce the learned detection head

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/compile_yolov8n_detection_head.py
```

The compiler validates and executes nodes 0–217. At each 40×40, 20×20, and 10×10 scale it runs two fused Conv-SiLU layers for box regression, a raw 1×1 box convolution, two fused Conv-SiLU layers for classification, and a raw 1×1 class convolution. EPILOGUE mode 0 preserves biased INT32 accumulators. The boundary concatenates 64 DFL box channels before 80 class channels and records one binary32 dequantization scale per logical channel.

| Nodes 0–217 measurement | Diagnostic | Release |
|---|---:|---:|
| Package bytes | 75,688,384 | 75,409,728 |
| Peak output allocation | 8,131,200 | 2,355,200 |
| Commands | 412,092 | 412,092 |
| Command bytes | 52,747,776 | 52,747,776 |
| DMA bytes | 44,598,232 | 44,598,232 |
| INT8 MACs | 1,092,864,000 | 1,092,864,000 |
| Exact values compared | 6,931,200 | 302,400 final values |
| Integer mismatches | 0 | 0 |
| FIFO `BUSY`/refill events | 412,084 | 412,084 |

The 256-input-channel 3×3 branch requires 18 KiB of weights for one output group, so the scheduler streams one 8×8 weight chunk per reduction step within the 16 KiB weight SRAM. The package exceeds the earlier provisional 64 MiB loader cap because commands and relocations remain uncompressed; the pre-freeze cap is now 128 MiB, and compaction is an explicit follow-up rather than a hidden format change.

The checked-in [`m6-through-detection-head.json`](m6-through-detection-head.json) records both package hashes, exact comparisons, FLOAT-reference errors, per-operation residency, allocation lifetimes, and FIFO behavior.

## Reproduce the explicit host tail

After generating the ignored release package and output above, run in the same pinned export environment:

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/verify_yolov8n_host_tail.py
```

The host implementation in [`runtime/haslab_runtime/yolov8n_tail.py`](../../../runtime/haslab_runtime/yolov8n_tail.py) reads all three INT32 HWC8 output records and applies the stored binary32 scale for each channel. It reshapes candidates in stride 8/16/32 row-major order, performs four-way 16-bin DFL softmax and expectation, builds grid anchors, converts side distances to center/size boxes, multiplies by stride, and applies sigmoid to all 80 class logits. These steps correspond to ONNX nodes 218–260; they run in NumPy without calling ONNX Runtime. The separately declared postprocessing reverses the 320×320 letterbox, chooses each candidate's best class, applies a strict confidence threshold, and performs deterministic class-aware greedy NMS. Operational defaults are confidence 0.25, IoU 0.7, and 300 maximum detections. The COCO evaluation profile uses confidence 0.001 and must be validated as a separate end-to-end run.

The verification tool copies only nodes 218–260 into an independent ONNX Runtime subgraph and feeds it the **same dequantized HASLAB boundary tensors**. The tracked [`m6-host-tail.json`](m6-host-tail.json) records comparisons at DFL, anchor, box, stride, class-score, and final output stages. All 176,400 final values agree within `atol=0.0001, rtol=0.00001`; maximum absolute error is `0.000091553`. A 480×640 letterbox example yields one detection at the 0.25 operational threshold and 35 at the 0.001 COCO evaluation threshold, each with the same class order as postprocessing the ONNX-tail output. Edge-case unit tests cover HWC8 unpacking, scale validation, uniform and extreme DFL/class logits, mapping, same-class suppression, cross-class retention, confidence equality, and tie behavior. The postprocessing comparison uses the same selection routine on two independently decoded tensors; matching the upstream Ultralytics validation pipeline on real images remains part of the COCO gate. This is one synthetic-input host-tail validation, not a COCO mAP measurement.

## Reproduce the multi-input command-path check

With the pinned ONNX model, calibration package, ignored diagnostic and release `.hxb` packages, and selected train2017 calibration images present, run:

```sh
PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/verify_yolov8n_multi_input.py

PYTHONPATH=reference:simulation:compiler:runtime \
python benchmarks/tools/verify_yolov8n_multi_input.py \
  --allocation-mode release --cases calibration-0 \
  --report benchmarks/manifests/yolov8n-320-opset13/m6-multi-input-release.json
```

The checker binds each input to the **same pinned command package**, compares all 90 stored diagnostic tensors bit for bit against an independent vectorized integer oracle, and compares DFL, boxes, scores, and the full `[1,84,2100]` host output against an isolated ONNX Runtime tail fed the same INT32 boundaries. It then compares coordinate-mapped and NMS-filtered detections at both the 0.25 operational and 0.001 COCO evaluation thresholds. The vectorized oracle is separately checked against scalar `haslab_ref` arithmetic and against all 90 tensors from the previously saved synthetic run. Its FLOAT64 dot products are exact integer carriers because a conservative bound keeps every partial sum inside signed INT32.

| Input | Exact stored values | 0.25 detections | 0.001 detections |
|---|---:|---:|---:|
| INT8 zero | 6,316,800 | 0 | 1 |
| INT8 +127 | 6,316,800 | 0 | 0 |
| INT8 −128 | 6,316,800 | 2 | 3 |
| Seeded full-range INT8 | 6,316,800 | 0 | 3 |
| Pinned calibration image 0 | 6,316,800 | 7 | 300 cap |
| Pinned calibration image 1 | 6,316,800 | 3 | 39 |

The six diagnostic runs compare **37,900,800 exact stored values** across 540 boundaries and **1,058,400 decoded FLOAT32 values** against the isolated ONNX tail. All stage comparisons pass `atol=0.0001, rtol=0.00001`; the largest absolute decoded difference is `0.000122071` at a coordinate where the relative allowance applies. The separate release run of calibration image 0 compares all 302,400 INT32 boundary values exactly and produces byte-identical final tensors to diagnostic mode. See the [diagnostic report](m6-multi-input.json) and [release report](m6-multi-input-release.json) for per-boundary hashes, image/input provenance, host comparisons, and FIFO counts.

The checks do not validate Ultralytics NMS against real-image ground truth or measure COCO mAP. They compare two independently decoded outputs using the same explicit HASLAB postprocessor; COCO evaluation remains a separate gate. These Python functional runs took several minutes per image under concurrent test conditions. The next implementation step is a faster execution path that passes the existing command conformance corpus and these exact multi-input fixtures before a 5,000-image command-path evaluation is attempted.
